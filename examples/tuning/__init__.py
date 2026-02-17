"""
Tuning examples for jMetalPy.

This directory contains examples demonstrating the hyperparameter tuning
capabilities of jMetalPy using Optuna.

Examples:
---------

basic_tuning.py
    Simple example of tuning NSGA-II with default settings.
    
    Usage:
        python examples/tuning/basic_tuning.py --trials 20

tune_with_observers.py
    Demonstrates different progress observers (console, plot, file, rich).
    
    Usage:
        python examples/tuning/tune_with_observers.py --observer progress
        python examples/tuning/tune_with_observers.py --observer plot

tune_custom_problems.py
    Shows how to tune on custom problem sets (ZDT + DTLZ).
    
    Usage:
        python examples/tuning/tune_custom_problems.py

describe_parameters.py
    View tunable parameters without running tuning.
    
    Usage:
        python examples/tuning/describe_parameters.py
        python examples/tuning/describe_parameters.py --format json

compare_samplers.py
    Compare different Optuna samplers (TPE, CMA-ES, Random).
    
    Usage:
        python examples/tuning/compare_samplers.py

using_metrics.py
    Use the metrics module directly for quality indicators.
    
    Usage:
        python examples/tuning/using_metrics.py

parallel_tuning_with_plot.py
    Parallel tuning with real-time visualization (requires PostgreSQL).
    
    Usage:
        python examples/tuning/parallel_tuning_with_plot.py

Command-Line Interface:
-----------------------

Sequential tuning (in-memory):
    python -m jmetal.tuning.cli.sequential --trials 100

Parallel tuning (requires PostgreSQL):
    python -m jmetal.tuning.cli.parallel --workers 4 --trials 500
"""
