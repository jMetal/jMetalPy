#!/usr/bin/env python
"""
Script to run multiple Optuna workers in parallel using PostgreSQL storage.

Each worker executes a portion of the trials, all sharing the same study.
Tuning is performed over a TRAINING SET of problems (e.g., ZDT1-ZDT6).

Prerequisites:
    - PostgreSQL running with database 'optuna_jmetal' created
    - Python environment with jMetalPy and Optuna installed
    - psycopg2-binary package installed

Usage:
    python -m jmetal.tuning.cli.parallel --workers 4
    python -m jmetal.tuning.cli.parallel --config tuning_config.yaml --workers 8
    python -m jmetal.tuning.cli.parallel --workers 4 --no-clean  # Resume existing study

Examples:
    # 4 workers with YAML config
    python -m jmetal.tuning.cli.parallel --config tuning.yaml -w 4

    # 8 workers, 1000 trials, don't clean DB (resume)
    python -m jmetal.tuning.cli.parallel -w 8 -t 1000 --no-clean
    
    # 4 workers with progress observer (clean output)
    python -m jmetal.tuning.cli.parallel -w 4 --observer progress
"""

import argparse
import os
import subprocess
import sys
import time

from jmetal.tuning.config import (
    NUMBER_OF_TRIALS,
    TRAINING_EVALUATIONS,
    POPULATION_SIZE,
    SEED,
)
from jmetal.tuning.tuning_config import TuningConfig


def clean_database(db_url: str) -> bool:
    """Clean the PostgreSQL database by dropping and recreating the public schema."""
    # Extract database name from URL
    # Format: postgresql://localhost/optuna_jmetal
    db_name = db_url.split("/")[-1]
    
    try:
        result = subprocess.run(
            ["psql", db_name, "-c", "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Database cleaned.")
            return True
        else:
            print(f"Warning: Could not clean database: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Warning: psql not found. Database not cleaned.")
        return False


def launch_worker(
    worker_id: int,
    n_workers: int,
    extra_args: list,
) -> subprocess.Popen:
    """Launch a single worker process."""
    env = os.environ.copy()
    env["WORKER_ID"] = str(worker_id)
    env["N_WORKERS"] = str(n_workers)
    
    cmd = [sys.executable, "-m", "jmetal.tuning.tuning_parallel"] + extra_args
    
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel hyperparameter tuning with multiple Optuna workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config tuning.yaml -w 4    # 4 workers with YAML config
  %(prog)s -w 4                         # 4 workers, default settings
  %(prog)s -w 8 -t 1000                 # 8 workers, 1000 total trials
  %(prog)s -w 4 --no-clean              # 4 workers, resume existing study
        """,
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML configuration file (recommended way to configure tuning)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean database (resume existing study)",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://localhost/optuna_jmetal",
        help="PostgreSQL URL (default: postgresql://localhost/optuna_jmetal)",
    )
    
    # Arguments passed through to tuning_parallel.py (can override config)
    parser.add_argument(
        "--trials", "-t",
        type=int, 
        default=None,
        help=f"Total number of trials to run (default from config or {NUMBER_OF_TRIALS})"
    )
    parser.add_argument(
        "--evaluations", "-e",
        type=int,
        default=None,
        help=f"Maximum evaluations per problem (default from config or {TRAINING_EVALUATIONS})"
    )
    parser.add_argument(
        "--algorithm", "-a",
        default=None,
        help="Algorithm to tune (default: NSGAII)",
    )
    parser.add_argument(
        "--sampler", "-s",
        choices=["tpe", "cmaes", "random"],
        default=None,
        help="Sampler to use (default: tpe)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["categorical", "continuous"],
        default="categorical",
        help="Parameter space mode (default: categorical)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default from config or {SEED})",
    )
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=None,
        help=f"Population size (default from config or {POPULATION_SIZE})",
    )
    parser.add_argument(
        "--study-name",
        default="nsgaii_tuning",
        help="Study name (default: nsgaii_tuning)",
    )
    parser.add_argument(
        "--observer", "-o",
        choices=["none", "progress", "plot", "file"],
        default="none",
        help="Observer type for progress tracking (default: none). "
             "Use 'progress' for console, 'plot' for matplotlib, 'file' for CSV.",
    )
    
    args = parser.parse_args()
    
    # Load configuration from file or use defaults
    if args.config:
        config = TuningConfig.from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = TuningConfig()
    
    # Apply CLI overrides
    algorithm = args.algorithm or config.algorithm
    n_trials = args.trials if args.trials is not None else config.n_trials
    n_evaluations = args.evaluations if args.evaluations is not None else config.n_evaluations
    sampler = args.sampler or config.sampler
    seed = args.seed if args.seed is not None else config.seed
    population_size = args.population_size if args.population_size is not None else config.population_size
    
    args = parser.parse_args()
    
    # Build extra arguments to pass to workers
    extra_args = [
        "--trials", str(n_trials),
        "--evaluations", str(n_evaluations),
        "--algorithm", algorithm,
        "--sampler", sampler,
        "--mode", args.mode,
        "--seed", str(seed),
        "--population-size", str(population_size),
        "--db-url", args.db_url,
        "--study-name", args.study_name,
    ]
    
    # Pass config file to workers if specified
    if args.config:
        extra_args.extend(["--config", args.config])
    
    # Pass observer argument to workers if specified
    if args.observer != "none":
        extra_args.extend(["--observer", args.observer])
    
    print("=" * 60)
    print("Parallel Optuna Tuning")
    print("=" * 60)
    if args.config:
        print(f"Config: {args.config}")
    print(f"Workers: {args.workers}")
    print(f"Total trials: {n_trials}")
    print(f"Trials per worker: {n_trials // args.workers}")
    print(f"Max evaluations: {n_evaluations}")
    print(f"Algorithm: {algorithm}")
    print(f"Sampler: {sampler}")
    print(f"Observer: {args.observer}")
    print(f"Database: {args.db_url}")
    print(f"Study: {args.study_name}")
    print("=" * 60)
    print()
    
    # Clean database if requested
    if not args.no_clean:
        print("Cleaning database...")
        clean_database(args.db_url)
        print()
    
    # Launch workers
    workers = []
    
    print(f"Launching {args.workers} workers...")
    print()
    
    # Launch worker 0 first to initialize database schema
    print("Launching worker 0 (initializing database)...")
    workers.append(launch_worker(0, args.workers, extra_args))
    
    # Wait for database initialization
    time.sleep(3)
    
    # Launch remaining workers
    for i in range(1, args.workers):
        print(f"Launching worker {i}...")
        workers.append(launch_worker(i, args.workers, extra_args))
    
    print()
    print("All workers launched. Waiting for completion...")
    print(f"Monitor progress: psql {args.db_url.split('/')[-1]} -c \"SELECT COUNT(*) FROM trials;\"")
    print()
    
    # Wait for all workers to complete
    for worker in workers:
        worker.wait()
    
    print()
    print("=" * 60)
    print("All workers completed!")
    print("Results saved to: src/jmetal/tuning/nsgaii_tuned_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
