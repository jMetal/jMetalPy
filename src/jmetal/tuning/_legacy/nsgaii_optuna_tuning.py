import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import BLXAlphaCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator
from jmetal.core.problem import Problem

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "nsgaii_tuned_config.json"
STUDY_NAME = "nsgaii_tuning"

# PostgreSQL storage for distributed parallel execution
# Each process connects to the same DB and Optuna synchronizes trials
STORAGE_URL = "postgresql://localhost/optuna_jmetal"

# ============================================================================
# TRAINING SET CONFIGURATION
# Define the problems used for hyperparameter tuning (training phase)
# Format: (Problem instance, reference_front_filename)
# ============================================================================
TRAINING_PROBLEMS: List[Tuple[Problem, str]] = [
    (ZDT1(), "ZDT1.pf"),
    (ZDT2(), "ZDT2.pf"),
    (ZDT3(), "ZDT3.pf"),
    (ZDT4(), "ZDT4.pf"),
    (ZDT6(), "ZDT6.pf"),
]

def load_reference_fronts() -> dict:
    """Load reference fronts for all training problems."""
    fronts = {}
    for problem, ref_front_file in TRAINING_PROBLEMS:
        ref_path = ROOT_DIR / f"resources/reference_fronts/{ref_front_file}"
        solutions = read_solutions(str(ref_path))
        fronts[problem.name()] = np.array([s.objectives for s in solutions])
    return fronts

REFERENCE_FRONTS = load_reference_fronts()

# We will derive the reference point automatically from the reference front
# and apply an offset via the NormalizedHyperVolume constructor
REFERENCE_POINT_OFFSET = 0.1

POPULATION_SIZE = 100
TRAINING_EVALUATIONS = 10000  # Fewer evaluations for faster tuning
NUMBER_OF_TRIALS = 500
N_JOBS = 1  # Use n_jobs=1 per process; parallelism via multiple processes sharing PostgreSQL
# Number of independent runs per (algorithm, problem). Default 1 to keep tuning affordable.
N_REPEATS = 1
# Final aggregation method across problems: 'sum' (default), 'mean', or 'median'
FINAL_AGG = "mean"
# Base seed used to derive per-trial/per-run seeds deterministically
BASE_SEED = 1234


def objective(trial: optuna.Trial) -> float:
    """
    Evaluate a configuration across all problems in the training set.
    Returns an aggregated score (lower is better).
    """
    # Sample hyperparameters (shared across all problems in training set)
    offspring_population_size = trial.suggest_categorical(
        "offspring_population_size", [1, 10, 50, 100, 150, 200]
    )
    crossover_type = trial.suggest_categorical("crossover_type", ["sbx", "blxalpha"])
    crossover_probability = trial.suggest_float("crossover_probability", 0.7, 1.0)
    
    if crossover_type == "sbx":
        crossover_eta = trial.suggest_float("crossover_eta", 5.0, 400.0)
        crossover_description = f"SBX prob={crossover_probability:.4f}, eta_c={crossover_eta:.2f}"
    else:
        crossover_alpha = trial.suggest_float("blx_alpha", 0.0, 1.0)
        crossover_description = f"BLX prob={crossover_probability:.4f}, alpha={crossover_alpha:.2f}"
    
    # Mutation operator selection
    mutation_type = trial.suggest_categorical("mutation_type", ["polynomial", "uniform"])
    
    # Mutation probability factor: actual probability = factor * (1/n) for each problem
    # This allows the mutation rate to scale properly with problem dimensionality
    mutation_probability_factor = trial.suggest_float("mutation_probability_factor", 0.5, 2.0)
    
    if mutation_type == "polynomial":
        mutation_eta = trial.suggest_float("mutation_eta", 5.0, 400.0)
        mutation_description = f"Polynomial factor={mutation_probability_factor:.2f}, eta={mutation_eta:.2f}"
    else:
        mutation_perturbation = trial.suggest_float("mutation_perturbation", 0.1, 2.0)
        mutation_description = f"Uniform factor={mutation_probability_factor:.2f}, perturbation={mutation_perturbation:.2f}"

    print(
        f"Trial {trial.number} params -> offspring_population_size={offspring_population_size}, "
        f"{crossover_description}, {mutation_description}"
    )

    # Evaluate on each problem in the training set
    problem_scores = []
    
    for problem, _ref_front_file in TRAINING_PROBLEMS:
        reference_front = REFERENCE_FRONTS[problem.name()]
        
        # Build crossover operator
        if crossover_type == "sbx":
            crossover = SBXCrossover(probability=crossover_probability, distribution_index=crossover_eta)
        else:
            crossover = BLXAlphaCrossover(probability=crossover_probability, alpha=crossover_alpha)
        
        # Calculate effective mutation probability for this problem: factor * (1/n)
        n_variables = problem.number_of_variables()
        effective_mutation_prob = min(1.0, mutation_probability_factor / n_variables)
        
        # Build mutation operator with problem-specific probability
        if mutation_type == "polynomial":
            mutation = PolynomialMutation(probability=effective_mutation_prob, distribution_index=mutation_eta)
        else:
            mutation = UniformMutation(probability=effective_mutation_prob, perturbation=mutation_perturbation)
        
        # Create algorithm factory for this problem
        def make_algo(prob=problem, mut=mutation):
            return NSGAII(
                problem=prob,
                population_size=POPULATION_SIZE,
                offspring_population_size=offspring_population_size,
                mutation=mut,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=TRAINING_EVALUATIONS),
            )
        
        # Create indicators for this problem
        normalized_hv_indicator = NormalizedHyperVolume(
            reference_front=reference_front,
            reference_point_offset=REFERENCE_POINT_OFFSET,
        )
        normalized_hv_indicator.set_reference_front(reference_front)
        epsilon_indicator = AdditiveEpsilonIndicator(reference_front)

        # Run the algorithm N_REPEATS times and collect indicator values
        nhv_values = []
        eps_values = []
        for r in range(N_REPEATS):
            seed = BASE_SEED + trial.number * 1000 + r
            algo = make_algo()
            algo.run()
            front = get_non_dominated_solutions(algo.result())
            objectives = np.array([s.objectives for s in front])

            nhv_values.append(float(normalized_hv_indicator.compute(objectives)))
            eps_values.append(float(epsilon_indicator.compute(objectives)))

        # Aggregate across repeats
        nhv_mean = float(np.mean(nhv_values))
        eps_mean = float(np.mean(eps_values))
        
        # Composite score for this problem (sum of indicators)
        problem_composite = nhv_mean + eps_mean
        problem_scores.append(problem_composite)

    # Aggregate across all problems in training set
    if FINAL_AGG == "sum":
        overall = sum(problem_scores)
    elif FINAL_AGG == "mean":
        overall = float(np.mean(problem_scores))
    elif FINAL_AGG == "median":
        overall = float(np.median(problem_scores))
    else:
        overall = sum(problem_scores)

    return float(overall)

sampler = optuna.samplers.TPESampler(seed=42)

if __name__ == "__main__":
    # Get worker ID from environment (for logging purposes)
    worker_id = os.environ.get("WORKER_ID", "0")
    
    # PostgreSQL storage enables true parallel execution across multiple processes
    # Each process runs n_jobs=1, but multiple processes share the same study
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=STORAGE_URL,
        study_name=STUDY_NAME,
        load_if_exists=True,  # Essential for multiple processes to join the same study
    )
    
    # Calculate trials per worker (for running multiple processes)
    n_workers = int(os.environ.get("N_WORKERS", "1"))
    trials_per_worker = NUMBER_OF_TRIALS // n_workers
    
    print(f"[Worker {worker_id}] Starting {trials_per_worker} trials (total: {NUMBER_OF_TRIALS})")
    
    start = time.perf_counter()
    study.optimize(objective, n_trials=trials_per_worker, n_jobs=N_JOBS, show_progress_bar=True)
    elapsed = time.perf_counter() - start
    
    print(f"[Worker {worker_id}] Completed in {elapsed:.2f}s")
    print(f"[Worker {worker_id}] Best value so far: {study.best_value}")
    print(f"[Worker {worker_id}] Best params: {study.best_params}")
    
    # Only save results from worker 0 (or when running single process)
    if worker_id == "0":
        payload = {
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "population_size": POPULATION_SIZE,
            "training_evaluations": TRAINING_EVALUATIONS,
            "validation_evaluations": TRAINING_EVALUATIONS * 2,  # Double for validation/testing
            "training_problems": [problem.name() for problem, _ in TRAINING_PROBLEMS],
            "aggregation_method": FINAL_AGG,
            "ref_point_offset": REFERENCE_POINT_OFFSET,
            "n_trials": len(study.trials),
            "n_repeats": N_REPEATS,
            "elapsed_seconds": elapsed,
        }
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Worker {worker_id}] Configuracion guardada en {CONFIG_PATH}")
