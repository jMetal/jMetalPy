#!/usr/bin/env python
"""
Basic hyperparameter tuning example.

This is the simplest example of using the tuning API to optimize
NSGA-II hyperparameters on the ZDT benchmark suite.

Usage:
    python examples/tuning/basic_tuning.py
    python examples/tuning/basic_tuning.py --trials 50
"""

import argparse

from jmetal.tuning import tune


def main():
    parser = argparse.ArgumentParser(
        description="Basic hyperparameter tuning example"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=20,
        help="Number of trials (default: 20)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for best configuration (JSON)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Basic NSGA-II Hyperparameter Tuning")
    print("=" * 60)
    print()
    
    # Simple usage: tune NSGA-II with default problems (ZDT1-6)
    result = tune(
        algorithm="NSGAII",
        n_trials=args.trials,
        output_path=args.output,
    )
    
    # Display results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best score: {result.best_score:.6f}")
    print(f"Best trial: #{result.best_trial}")
    print()
    print("Best parameters:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")
    
    if args.output:
        print(f"\nConfiguration saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
