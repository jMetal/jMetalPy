#!/usr/bin/env python
"""
Sequential hyperparameter tuning with Optuna.

This script runs Optuna without any external database, using in-memory storage.
It's suitable for single-machine execution or cluster jobs where each node
runs independently.

Usage:
    python -m jmetal.tuning.cli.sequential [--config CONFIG_FILE] [--trials N] [--sampler tpe|cmaes]
    
Examples:
    # Run with YAML configuration file
    python -m jmetal.tuning.cli.sequential --config tuning_config.yaml
    
    # Run 100 trials with TPE sampler (default)
    python -m jmetal.tuning.cli.sequential --trials 100
    
    # Override config file settings
    python -m jmetal.tuning.cli.sequential --config tuning_config.yaml --trials 50
    
    # Save to custom output file
    python -m jmetal.tuning.cli.sequential --output my_config.json
"""

import argparse
from pathlib import Path

from jmetal.tuning.tuning import tune
from jmetal.tuning.config import NUMBER_OF_TRIALS, SEED, POPULATION_SIZE, TRAINING_EVALUATIONS
from jmetal.tuning.tuning_config import TuningConfig


def main():
    """Run sequential hyperparameter tuning from command line."""
    parser = argparse.ArgumentParser(
        description="Sequential hyperparameter tuning with Optuna"
    )
    
    # Configuration file (primary way to configure)
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML configuration file (recommended way to configure tuning)"
    )
    
    # CLI overrides (take precedence over config file)
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default=None,
        help="Algorithm to tune (default: NSGAII)"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=None,
        help=f"Number of trials (default from config or {NUMBER_OF_TRIALS})"
    )
    parser.add_argument(
        "--evaluations", "-e",
        type=int,
        default=None,
        help=f"Max evaluations per problem (default from config or {TRAINING_EVALUATIONS})"
    )
    parser.add_argument(
        "--sampler", "-s",
        choices=["tpe", "cmaes", "random"],
        default=None,
        help="Sampler to use (default: tpe)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default from config or {SEED})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: ./nsgaii_tuned_config.json)"
    )
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=None,
        help=f"Population size (default from config or {POPULATION_SIZE})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
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
    output_path = args.output or config.output.path
    
    # Get problems from config
    problems = config.get_problems_as_tuples() if args.config else None
    
    # Get parameter space from config (only if config file provided)
    parameter_space = config.parameter_space if args.config else None
    
    # Run tuning
    result = tune(
        algorithm=algorithm,
        problems=problems,
        n_trials=n_trials,
        n_evaluations=n_evaluations,
        sampler=sampler,
        seed=seed,
        population_size=population_size,
        output_path=output_path,
        verbose=not args.quiet,
        parameter_space=parameter_space,
    )
    
    return result


if __name__ == "__main__":
    main()
