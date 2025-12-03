#!/usr/bin/env python
"""
Example of hyperparameter tuning with progress observers.

This example demonstrates the different observer options for monitoring
the tuning process:

1. TuningProgressObserver: Clean console output with progress updates
2. TuningPlotObserver: Real-time matplotlib plot of score vs trial
3. TuningFileObserver: Log results to CSV/JSON files
4. TuningRichObserver: Rich library for beautiful terminal output

Usage:
    # Basic progress observer
    python examples/tuning/tune_with_observers.py --observer progress
    
    # Real-time plot
    python examples/tuning/tune_with_observers.py --observer plot
    
    # File logging
    python examples/tuning/tune_with_observers.py --observer file
    
    # Rich console (requires: pip install rich)
    python examples/tuning/tune_with_observers.py --observer rich
    
    # Multiple observers
    python examples/tuning/tune_with_observers.py --observer progress --observer plot
"""

import argparse

from jmetal.problem import ZDT1, ZDT2
from jmetal.tuning import tune
from jmetal.tuning.observers import (
    TuningProgressObserver,
    TuningPlotObserver,
    TuningFileObserver,
    TuningRichObserver,
    create_default_observers,
)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with observers example"
    )
    parser.add_argument(
        "--observer", "-o",
        action="append",
        choices=["progress", "plot", "file", "rich", "all"],
        default=[],
        help="Observer type(s) to use. Can be specified multiple times."
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=20,
        help="Number of trials (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        default="./tuning_output",
        help="Output directory for file observer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./nsgaii_tuned_config.json",
        help="Output file for best configuration (default: ./nsgaii_tuned_config.json)"
    )
    args = parser.parse_args()
    
    # Create observers based on arguments
    observers = []
    observer_types = args.observer if args.observer else ["progress"]
    
    if "all" in observer_types:
        observers = create_default_observers(
            console=False, plot=True, file=True, rich=True,
            output_dir=args.output_dir
        )
    else:
        if "progress" in observer_types:
            observers.append(TuningProgressObserver(display_frequency=5))
        if "rich" in observer_types:
            observers.append(TuningRichObserver())
        if "plot" in observer_types:
            observers.append(TuningPlotObserver(update_frequency=1))
        if "file" in observer_types:
            observers.append(TuningFileObserver(output_dir=args.output_dir))
    
    print(f"Using observers: {[type(o).__name__ for o in observers]}")
    print()
    
    # Define training problems (using fewer for quick demo)
    problems = [
        (ZDT1(), "ZDT1.pf"),
        (ZDT2(), "ZDT2.pf"),
    ]
    
    # Run tuning with observers
    result = tune(
        algorithm="NSGAII",
        problems=problems,
        n_trials=args.trials,
        n_evaluations=5000,  # Reduced for demo
        observers=observers,
        output_path=args.output,
    )
    
    print(f"\nFinal result: {result.best_score:.6f}")
    print(f"Configuration saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
