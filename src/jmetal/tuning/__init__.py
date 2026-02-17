"""
Hyperparameter Tuning for Multi-Objective Optimization Algorithms.

This package provides modular tools for tuning algorithm hyperparameters
using Optuna optimization framework.

Currently Supported Algorithms:
    - NSGA-II

Package Structure:
    - tuning.py: High-level API (tune, describe_parameters)
    - algorithms/: Algorithm-specific tuners (NSGAIITuner)
    - observers/: Progress visualization (console, plot, file, rich)
    - metrics/: Quality indicators (NHV, Additive Epsilon)
    - config/: Configuration (defaults, paths, problems)
    - cli/: Command-line interfaces (sequential, parallel)
    - runners/: Execution runners (future)
    - _legacy/: Deprecated files for reference

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
    
    # View tunable parameters
    from jmetal.tuning import describe_parameters
    print(describe_parameters("NSGAII"))
    
    # Command-line sequential tuning
    python -m jmetal.tuning.cli.sequential --trials 100
    
    # Parallel tuning (requires PostgreSQL)
    python -m jmetal.tuning.cli.parallel -w 4  # 4 workers
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

# YAML Configuration
from .tuning_config import TuningConfig

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
    # YAML Configuration
    "TuningConfig",
]
