#!/usr/bin/env python
"""
Example: Parallel tuning with real-time matplotlib visualization.

This example demonstrates how to run parallel hyperparameter tuning
with a live plot showing optimization progress.

Prerequisites:
    - PostgreSQL running with database 'optuna_jmetal' created
    - psycopg2-binary installed: pip install psycopg2-binary
    - matplotlib installed

Usage:
    # Run with 4 workers and plot observer (single window from worker 0)
    python -m jmetal.tuning.run_parallel_tuning -w 4 -t 100 --observer plot
    
    # Or run this script directly (single worker with plot)
    python examples/tuning/parallel_tuning_with_plot.py

Note:
    When using --observer plot with multiple workers, only worker 0 opens
    the matplotlib window showing global progress. Other workers use console
    output. This prevents window overload while still showing real-time
    optimization progress.
    
    The plot shows the best value found across ALL workers (shared study),
    so you can see the combined effect of parallel optimization.
"""

import os
import sys

# Add src to path if running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from jmetal.tuning.observer import TuningProgressObserver, TuningPlotObserver
from jmetal.tuning.tuning_parallel import run_parallel_tuning


def main():
    """Run a single worker with plot visualization."""
    
    # Set worker environment (simulating worker 0 - the one with plot)
    os.environ["WORKER_ID"] = "0"
    os.environ["N_WORKERS"] = "1"
    
    # Create observers: progress + plot
    observers = [
        TuningProgressObserver(display_frequency=1),
        TuningPlotObserver(
            title="Parallel Tuning Demo - Global Progress",
            update_frequency=1,
        ),
    ]
    
    print("=" * 60)
    print("Parallel Tuning with Plot Observer Demo")
    print("=" * 60)
    print()
    print("This demo runs a single worker with real-time plot.")
    print("For multi-worker parallel tuning with plot, use:")
    print()
    print("  python -m jmetal.tuning.run_parallel_tuning -w 4 --observer plot")
    print()
    print("Only worker 0 shows the plot (global progress from all workers).")
    print("=" * 60)
    print()
    
    # Run parallel tuning (single worker mode for demo)
    # Note: This requires PostgreSQL. For in-memory demo, use tune() instead.
    try:
        run_parallel_tuning(
            algorithm="NSGAII",
            total_trials=15,
            max_evaluations=5000,
            sampler_name="tpe",
            mode="categorical",
            storage_url="postgresql://localhost/optuna_jmetal",
            study_name="plot_demo",
            observers=observers,
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires PostgreSQL.")
        print("For a simpler demo without PostgreSQL, use:")
        print("  python examples/tuning/tune_with_observers.py --observer plot")


if __name__ == "__main__":
    main()
