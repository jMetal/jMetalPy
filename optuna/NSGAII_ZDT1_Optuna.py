import json
import time
from pathlib import Path

import numpy as np
import optuna
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import BLXAlphaCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, ZDT4
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "best_nsgaii_zdt1_params.json"
STORAGE_PATH = Path(__file__).resolve().parent / "optuna_nsgaii.db"
# New study name to avoid legacy trials con valores fuera del dominio actual
STUDY_NAME = "nsgaii_zdt1_discrete_offspring"

reference_front = read_solutions(str(ROOT_DIR / "resources/reference_fronts/ZDT1.pf"))
reference_front_objectives = np.array([s.objectives for s in reference_front])
# we will derive the reference point automatically from the reference front
# and apply an offset via the NormalizedHyperVolume constructor
REFERENCE_POINT_OFFSET = 0.1

POPULATION_SIZE = 100
MAXIMUM_EVALUATIONS = 25000
NUMBER_OF_TRIALS = 200
N_JOBS = 8  # parallel workers for Optuna (threads/processes)


def objective(trial: optuna.Trial) -> float:
    problem = ZDT4()
    population_size = POPULATION_SIZE
    offspring_population_size = trial.suggest_categorical(
        "offspring_population_size", [1, 10, 50, 100, 150, 200]
    )
    crossover_type = trial.suggest_categorical("crossover_type", ["sbx", "blxalpha"])
    crossover_probability = trial.suggest_float("crossover_probability", 0.7, 1.0)
    if crossover_type == "sbx":
        crossover_eta = trial.suggest_float("crossover_eta", 5.0, 400.0)
        crossover = SBXCrossover(probability=crossover_probability, distribution_index=crossover_eta)
        crossover_description = f"SBX prob={crossover_probability:.4f}, eta_c={crossover_eta:.2f}"
    else:
        crossover_alpha = trial.suggest_float("blx_alpha", 0.0, 1.0)
        crossover = BLXAlphaCrossover(probability=crossover_probability, alpha=crossover_alpha)
        crossover_description = f"BLX prob={crossover_probability:.4f}, alpha={crossover_alpha:.2f}"
    mutation_probability = trial.suggest_float(
        "mutation_probability", 1.0 / problem.number_of_variables(), 0.5
    )
    mutation_eta = trial.suggest_float("mutation_eta", 5.0, 400.0)

    print(
        f"Trial {trial.number} params -> offspring_population_size={offspring_population_size}, "
        f"{crossover_description}, mutation_probability={mutation_probability:.4f}, mutation_eta={mutation_eta:.2f}"
    )

    algo = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=offspring_population_size,
        mutation=PolynomialMutation(probability=mutation_probability, distribution_index=mutation_eta),
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=MAXIMUM_EVALUATIONS),
    )
    algo.run()
    front = get_non_dominated_solutions(algo.result())
    # Create the NormalizedHyperVolume by providing the reference front and
    # the scalar offset. Then compute and cache the reference hypervolume.
    normalized_hv_indicator = NormalizedHyperVolume(
        reference_front=reference_front_objectives,
        reference_point_offset=REFERENCE_POINT_OFFSET,
    )
    normalized_hv_indicator.set_reference_front(reference_front_objectives)
    objectives = np.array([s.objectives for s in front])
    # Compute Normalized Hypervolume (NHV)
    nhv_value = normalized_hv_indicator.compute(objectives)

    # Compute Additive Epsilon indicator between the front and reference
    epsilon_indicator = AdditiveEpsilonIndicator(reference_front_objectives)
    epsilon_value = epsilon_indicator.compute(objectives)

    # Return the sum NHV + Epsilon as the optimization objective
    return float(nhv_value + epsilon_value)

sampler = optuna.samplers.TPESampler(seed=42)

if __name__ == "__main__":
    storage_url = f"sqlite:///{STORAGE_PATH}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS, n_jobs=N_JOBS)
    elapsed = time.perf_counter() - start
    print("Best Normalized HV:", study.best_value)
    print("Best params:", study.best_params)
    print(f"Tiempo Optuna (s): {elapsed:.2f}")

    payload = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "population_size": POPULATION_SIZE,
        "max_evaluations": MAXIMUM_EVALUATIONS,
        "ref_point_offset": REFERENCE_POINT_OFFSET,
        "n_trials": NUMBER_OF_TRIALS,
        "elapsed_seconds": elapsed,
    }
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Configuracion guardada en {CONFIG_PATH}")
