"""
Hyperparameter Tuning for Multi-Objective Optimization Algorithms.

This package provides modular tools for tuning algorithm hyperparameters
using Optuna optimization framework.

Currently Supported Algorithms:
    - NSGA-II

Package Structure:
    tuning: High-level API (recommended entry point)
    algorithms/: Algorithm-specific tuners
    observers/: Progress visualization components
    config: Configuration constants and paths
    runners/: Sequential and parallel execution (future)
    cli/: Command-line interfaces (future)

Quick Start:
    # Using the high-level API
    from jmetal.tuning import tune
    result = tune("NSGAII", n_trials=100)
    
    # With progress observers
    from jmetal.tuning import tune, TuningProgressObserver, TuningPlotObserver
    result = tune("NSGAII", n_trials=100, observers=[
        TuningProgressObserver(),
        TuningPlotObserver(),
    ])
    
    # View tunable parameters (no Python expertise required!)
    from jmetal.tuning import describe_parameters
    print(describe_parameters("NSGAII"))
    
    # Export to file
    describe_parameters("NSGAII", format="json", output_path="params.json")
    
    # Command-line sequential tuning
    python -m jmetal.tuning.tuning_sequential --trials 100
    
    # Parallel tuning (requires PostgreSQL)
    python -m jmetal.tuning.run_parallel_tuning -w 4  # 4 workers
"""

# High-level API
from .tuning import tune, describe_parameters, list_algorithms

# Algorithm tuners
from .algorithms import AlgorithmTuner, TuningResult, ParameterInfo, NSGAIITuner, TUNERS

# Observers (from new modular structure)
from .observers import (
    TuningObserver,
    TuningProgressObserver,
    TuningPlotObserver,
    TuningFileObserver,
    TuningRichObserver,
    create_default_observers,
)

# Metrics (from new modular structure)
from .metrics import (
    compute_quality_indicators,
    load_reference_front,
    aggregate_scores,
)

# Configuration
from .config import (
    POPULATION_SIZE,
    TRAINING_EVALUATIONS,
    VALIDATION_EVALUATIONS,
    NUMBER_OF_TRIALS,
    N_REPEATS,
    TRAINING_PROBLEMS,
)

__all__ = [
    # High-level API
    "tune",
    "describe_parameters",
    "list_algorithms",
    # Algorithm tuners
    "AlgorithmTuner",
    "TuningResult",
    "ParameterInfo",
    "NSGAIITuner",
    "TUNERS",
    # Observers
    "TuningObserver",
    "TuningProgressObserver",
    "TuningPlotObserver",
    "TuningFileObserver",
    "TuningRichObserver",
    "create_default_observers",
    # Metrics
    "compute_quality_indicators",
    "load_reference_front",
    "aggregate_scores",
    # Config
    "POPULATION_SIZE",
    "TRAINING_EVALUATIONS", 
    "VALIDATION_EVALUATIONS",
    "NUMBER_OF_TRIALS",
    "N_REPEATS",
    "TRAINING_PROBLEMS",
]
