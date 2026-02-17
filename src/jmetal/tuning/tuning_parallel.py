#!/usr/bin/env python
"""
Parallel hyperparameter tuning with Optuna and PostgreSQL.

This script uses PostgreSQL as a shared storage backend, allowing multiple
workers to run in parallel and share trial information. Each worker connects
to the same database and Optuna synchronizes the trials.

Prerequisites:
    1. PostgreSQL server running
    2. Database created: createdb optuna_jmetal
    3. psycopg2 installed: pip install psycopg2-binary

Usage:
    # Terminal 1 (first worker):
    WORKER_ID=0 N_WORKERS=4 python tuning_parallel.py
    
    # Terminal 2-4 (additional workers):
    WORKER_ID=1 N_WORKERS=4 python tuning_parallel.py
    WORKER_ID=2 N_WORKERS=4 python tuning_parallel.py
    WORKER_ID=3 N_WORKERS=4 python tuning_parallel.py
    
    # Or use the launcher script:
    python -m jmetal.tuning.run_parallel_tuning -w 4

Command line options:
    --trials N          Total number of trials across all workers
    --sampler tpe       Sampler type (tpe, cmaes, random)
    --mode categorical  Parameter space mode (categorical, continuous)
    --db-url URL        PostgreSQL URL (default: postgresql://localhost/optuna_jmetal)
    --study-name NAME   Study name (default: nsgaii_tuning)
    --observer TYPE     Observer type: progress, file, or none (default: progress)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Sequence

import optuna

from .algorithms import TUNERS
from .config import (
    TRAINING_PROBLEMS,
    TRAINING_EVALUATIONS,
    NUMBER_OF_TRIALS,
    N_REPEATS,
    POPULATION_SIZE,
    REFERENCE_POINT_OFFSET,
    SEED,
)
from .observers import TuningObserver, TuningProgressObserver, TuningFileObserver, TuningPlotObserver
from .tuning_config import TuningConfig, ParameterSpaceConfig


# Default PostgreSQL storage URL
DEFAULT_STORAGE_URL = "postgresql://localhost/optuna_jmetal"
DEFAULT_STUDY_NAME = "nsgaii_tuning"


def _configure_logging(quiet: bool = False) -> None:
    """Configure logging levels for tuning.
    
    Args:
        quiet: If True, suppress most logging output
    """
    if quiet:
        logging.getLogger("jmetal").setLevel(logging.WARNING)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        logging.getLogger("jmetal").setLevel(logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def create_sampler(sampler_name: str, seed: int = 42):
    """Create Optuna sampler."""
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def run_parallel_tuning(
    algorithm: str = "NSGAII",
    total_trials: int = NUMBER_OF_TRIALS,
    max_evaluations: int = TRAINING_EVALUATIONS,
    sampler_name: str = "tpe",
    mode: str = "categorical",
    seed: int = SEED,
    population_size: int = POPULATION_SIZE,
    storage_url: str = DEFAULT_STORAGE_URL,
    study_name: str = DEFAULT_STUDY_NAME,
    observers: Optional[Sequence[TuningObserver]] = None,
    output_path: Optional[str] = None,
    parameter_space: Optional[ParameterSpaceConfig] = None,
    problems: Optional[List] = None,
):
    """
    Run parallel hyperparameter tuning as one worker.
    
    This function should be called by each worker process. Workers will
    coordinate through the PostgreSQL database.
    
    Args:
        algorithm: Algorithm to tune (e.g., "NSGAII")
        total_trials: Total trials to run across all workers
        max_evaluations: Maximum function evaluations per problem per run
        sampler_name: "tpe", "cmaes", or "random"
        mode: "categorical" or "continuous"
        seed: Random seed for sampler
        population_size: Population size for the algorithm
        storage_url: PostgreSQL connection URL
        study_name: Optuna study name
        observers: List of TuningObserver instances for progress visualization
        output_path: Path for output JSON file. If None, saves to current directory
            as '{algorithm}_tuned_config.json'
        parameter_space: Custom parameter space configuration from TuningConfig
        problems: List of (Problem, reference_front) tuples to use. If None, uses defaults.
    """
    # Get worker configuration from environment
    worker_id = os.environ.get("WORKER_ID", "0")
    n_workers = int(os.environ.get("N_WORKERS", "1"))
    
    # Calculate trials for this worker
    trials_per_worker = total_trials // n_workers
    
    # Determine output mode
    use_observers = observers is not None and len(observers) > 0
    
    # Configure logging: quiet when using observers
    _configure_logging(quiet=use_observers)
    
    # Validate sampler/mode combination
    if sampler_name == "cmaes" and mode != "continuous":
        if not use_observers:
            print(f"[Worker {worker_id}] Warning: CMA-ES requires continuous mode. Switching.")
        mode = "continuous"
    
    # Get tuner
    if algorithm not in TUNERS:
        available = ", ".join(TUNERS.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    tuner = TUNERS[algorithm](population_size=population_size, parameter_space=parameter_space)
    
    # Use provided problems or defaults
    training_problems = problems if problems is not None else TRAINING_PROBLEMS
    
    # Show header (only when not using observers)
    if not use_observers:
        print("=" * 60)
        print(f"[Worker {worker_id}] {algorithm} Hyperparameter Tuning (Parallel)")
        print("=" * 60)
        print(f"[Worker {worker_id}] Worker ID: {worker_id} of {n_workers}")
        print(f"[Worker {worker_id}] Trials for this worker: {trials_per_worker}")
        print(f"[Worker {worker_id}] Total trials: {total_trials}")
        print(f"[Worker {worker_id}] Max evaluations: {max_evaluations}")
        print(f"[Worker {worker_id}] Sampler: {sampler_name}")
        print(f"[Worker {worker_id}] Mode: {mode}")
        print(f"[Worker {worker_id}] Storage: {storage_url}")
        print(f"[Worker {worker_id}] Study: {study_name}")
        print("=" * 60)
    
    # Notify observers of tuning start (include worker info)
    if use_observers and observers is not None:
        for observer in observers:
            # Pass worker-specific info through algorithm name
            observer.on_tuning_start(
                trials_per_worker, 
                f"{algorithm} [Worker {worker_id}/{n_workers}]"
            )
    
    # Create or load study from database
    sampler = create_sampler(sampler_name, seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,  # Essential for parallel execution
    )
    
    # Create objective function
    def objective(trial: optuna.Trial) -> float:
        params = tuner.sample_parameters(trial, mode=mode)
        
        if not use_observers:
            print(f"[Worker {worker_id}] Trial {trial.number}: {tuner.format_params(params)}")
        
        score = tuner.evaluate_on_problems(
            problems=training_problems,
            params=params,
            max_evaluations=max_evaluations,
            n_repeats=N_REPEATS,
        )
        return score
    
    # Prepare callbacks for Optuna (observers act as callbacks)
    optuna_callbacks = observers if use_observers and observers is not None else None
    
    # Run optimization
    start_time = time.perf_counter()
    study.optimize(
        objective, 
        n_trials=trials_per_worker, 
        n_jobs=1, 
        show_progress_bar=not use_observers,
        callbacks=optuna_callbacks,
    )
    elapsed = time.perf_counter() - start_time
    
    # Notify observers of tuning end
    if use_observers and observers is not None:
        for observer in observers:
            observer.on_tuning_end(study)
    
    # Print results (only when not using observers)
    if not use_observers:
        print(f"\n[Worker {worker_id}] " + "=" * 50)
        print(f"[Worker {worker_id}] Worker completed in {elapsed:.2f}s")
        print(f"[Worker {worker_id}] Best value so far: {study.best_value:.6f}")
        print(f"[Worker {worker_id}] Total trials in study: {len(study.trials)}")
    
    # Only save results from worker 0
    if worker_id == "0":
        # Determine output path
        if output_path is None:
            output_path = f"./{algorithm.lower()}_tuned_config.json"
        
        save_results(
            study, elapsed, sampler_name, mode, algorithm,
            population_size, max_evaluations, output_path, training_problems
        )
    
    return study


def save_results(
    study: optuna.Study, 
    elapsed: float, 
    sampler_name: str, 
    mode: str,
    algorithm: str,
    population_size: int,
    max_evaluations: int,
    output_path: str,
    training_problems: List,
):
    """Save tuning results to JSON file.
    
    Args:
        study: Completed Optuna study
        elapsed: Total elapsed time in seconds
        sampler_name: Name of the sampler used
        mode: Parameter space mode (categorical/continuous)
        algorithm: Algorithm name
        population_size: Population size used
        max_evaluations: Max evaluations per problem
        output_path: Path where to save the JSON file
        training_problems: List of (Problem, reference_front) tuples used
    """
    payload = {
        "algorithm": algorithm,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "population_size": population_size,
        "training_evaluations": max_evaluations,
        "validation_evaluations": max_evaluations * 2,
        "training_problems": [problem.name() for problem, _ in training_problems],
        "ref_point_offset": REFERENCE_POINT_OFFSET,
        "n_trials": len(study.trials),
        "n_repeats": N_REPEATS,
        "elapsed_seconds": elapsed,
        "sampler": sampler_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n[Worker 0] Configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel hyperparameter tuning with Optuna and PostgreSQL"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML configuration file (recommended way to configure tuning)"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="NSGAII",
        help="Algorithm to tune (default: NSGAII)"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=NUMBER_OF_TRIALS,
        help=f"Total number of trials (default: {NUMBER_OF_TRIALS})"
    )
    parser.add_argument(
        "--evaluations", "-e",
        type=int,
        default=TRAINING_EVALUATIONS,
        help=f"Max evaluations per problem per run (default: {TRAINING_EVALUATIONS})"
    )
    parser.add_argument(
        "--sampler", "-s",
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Sampler to use (default: tpe)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["categorical", "continuous"],
        default="categorical",
        help="Parameter space mode (default: categorical)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=POPULATION_SIZE,
        help=f"Population size (default: {POPULATION_SIZE})"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_STORAGE_URL,
        help=f"PostgreSQL URL (default: {DEFAULT_STORAGE_URL})"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=DEFAULT_STUDY_NAME,
        help=f"Study name (default: {DEFAULT_STUDY_NAME})"
    )
    parser.add_argument(
        "--observer", "-o",
        choices=["none", "progress", "plot", "file"],
        default="none",
        help="Observer type: none (verbose), progress (console), plot (matplotlib), file (CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tuning_output",
        help="Output directory for file observer (default: ./tuning_output)"
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        default=None,
        help="Output JSON file path for best configuration (default: ./{algorithm}_tuned_config.json)"
    )
    
    args = parser.parse_args()
    
    # Load configuration from file or use defaults
    if args.config:
        config = TuningConfig.from_yaml(args.config)
        parameter_space = config.parameter_space
        problems = config.get_problems_as_tuples()
    else:
        config = None
        parameter_space = None
        problems = None
    
    # Apply config values with CLI overrides
    # CLI arguments take precedence over config file
    if config is not None:
        algorithm = args.algorithm if args.algorithm != "NSGAII" else config.algorithm
        total_trials = args.trials if args.trials != NUMBER_OF_TRIALS else config.n_trials
        max_evaluations = args.evaluations if args.evaluations != TRAINING_EVALUATIONS else config.n_evaluations
        sampler = args.sampler if args.sampler != "tpe" else config.sampler
        seed = args.seed if args.seed != SEED else config.seed
        population_size = args.population_size if args.population_size != POPULATION_SIZE else config.population_size
    else:
        algorithm = args.algorithm
        total_trials = args.trials
        max_evaluations = args.evaluations
        sampler = args.sampler
        seed = args.seed
        population_size = args.population_size
    
    # Create observers based on argument
    # Note: plot observer only enabled on worker 0 to show global progress
    worker_id = os.environ.get("WORKER_ID", "0")
    observers = None
    
    if args.observer == "progress":
        observers = [TuningProgressObserver(display_frequency=5)]
    elif args.observer == "plot":
        # Only worker 0 shows the plot (shows global study progress)
        if worker_id == "0":
            observers = [
                TuningProgressObserver(display_frequency=5),
                TuningPlotObserver(title="Parallel Tuning - Global Progress"),
            ]
        else:
            # Other workers just use progress observer
            observers = [TuningProgressObserver(display_frequency=5)]
    elif args.observer == "file":
        observers = [
            TuningProgressObserver(display_frequency=5),
            TuningFileObserver(
                output_dir=args.output_dir,
                csv_file=f"tuning_history_worker{worker_id}.csv",
                json_file=f"tuning_results_worker{worker_id}.json",
            ),
        ]
    
    run_parallel_tuning(
        algorithm=algorithm,
        total_trials=total_trials,
        max_evaluations=max_evaluations,
        sampler_name=sampler,
        mode=args.mode,
        seed=seed,
        population_size=population_size,
        storage_url=args.db_url,
        study_name=args.study_name,
        observers=observers,
        output_path=args.output,
        parameter_space=parameter_space,
        problems=problems,
    )


if __name__ == "__main__":
    main()
