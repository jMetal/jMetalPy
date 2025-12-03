#!/usr/bin/env python
"""
Sequential hyperparameter tuning with Optuna.

This script runs Optuna without any external database, using in-memory storage.
It's suitable for single-machine execution or cluster jobs where each node
runs independently.

Usage:
    python -m jmetal.tuning.cli.sequential [--trials N] [--sampler tpe|cmaes] [--mode categorical|continuous]
    
Examples:
    # Run 100 trials with TPE sampler (default)
    python -m jmetal.tuning.cli.sequential --trials 100
    
    # Run with CMA-ES sampler (requires continuous mode)
    python -m jmetal.tuning.cli.sequential --trials 50 --sampler cmaes --mode continuous
    
    # Save to custom output file
    python -m jmetal.tuning.cli.sequential --output my_config.json
"""

import argparse

from jmetal.tuning.tuning import tune
from jmetal.tuning.config import NUMBER_OF_TRIALS, SEED, POPULATION_SIZE


def main():
    """Run sequential hyperparameter tuning from command line."""
    parser = argparse.ArgumentParser(
        description="Sequential hyperparameter tuning with Optuna"
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
        help=f"Number of trials (default: {NUMBER_OF_TRIALS})"
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
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: nsgaii_tuned_config.json)"
    )
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=POPULATION_SIZE,
        help=f"Population size (default: {POPULATION_SIZE})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Run tuning using the high-level API
    result = tune(
        algorithm=args.algorithm,
        n_trials=args.trials,
        sampler=args.sampler,
        mode=args.mode,
        seed=args.seed,
        population_size=args.population_size,
        output_path=args.output,
        verbose=not args.quiet,
    )
    
    return result


if __name__ == "__main__":
    main()
