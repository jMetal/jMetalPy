#!/usr/bin/env python
"""
Example: Tuning with custom problems.

This example shows how to tune NSGA-II on a custom set of problems
instead of using the default ZDT benchmark suite.

Usage:
    python examples/tuning/tune_custom_problems.py
    python examples/tuning/tune_custom_problems.py --trials 30
"""

import argparse

from jmetal.problem import ZDT1, ZDT4, DTLZ1, DTLZ2
from jmetal.tuning import tune, TuningProgressObserver


def main():
    parser = argparse.ArgumentParser(
        description="Tuning with custom problems example"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=15,
        help="Number of trials (default: 15)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tuning NSGA-II with Custom Problem Set")
    print("=" * 60)
    print()
    
    # Define custom problem set
    # Format: (Problem instance, reference_front_filename)
    # Reference fronts are in resources/reference_fronts/
    custom_problems = [
        (ZDT1(), "ZDT1.pf"),
        (ZDT4(), "ZDT4.pf"),
        (DTLZ1(number_of_objectives=3), "DTLZ1.3D.pf"),
        (DTLZ2(number_of_objectives=3), "DTLZ2.3D.pf"),
    ]
    
    print("Training problems:")
    for problem, ref_file in custom_problems:
        print(f"  - {problem.name()} (ref: {ref_file})")
    print()
    
    # Run tuning with custom problems
    result = tune(
        algorithm="NSGAII",
        problems=custom_problems,
        n_trials=args.trials,
        n_evaluations=5000,  # Reduced for faster demo
        observers=[TuningProgressObserver(display_frequency=3)],
    )
    
    # Display results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best score: {result.best_score:.6f}")
    print()
    print("Best parameters:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")
    
    return result


if __name__ == "__main__":
    main()
