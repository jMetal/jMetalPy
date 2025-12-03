"""
Configuration module for hyperparameter tuning.

This module provides centralized configuration for the tuning package.

Submodules:
    - defaults: Core tuning parameters (evaluations, trials, seed, etc.)
    - paths: File and directory paths
    - problems: Training problem definitions

Usage:
    >>> from jmetal.tuning.config import POPULATION_SIZE, TRAINING_PROBLEMS
    >>> from jmetal.tuning.config import get_reference_front_path
    >>> from jmetal.tuning.config.problems import create_problem_set
"""

# Core tuning parameters
from .defaults import (
    POPULATION_SIZE,
    TRAINING_EVALUATIONS,
    VALIDATION_EVALUATIONS,
    NUMBER_OF_TRIALS,
    N_REPEATS,
    FINAL_AGG,
    SEED,
    REFERENCE_POINT_OFFSET,
)

# Path configuration
from .paths import (
    ROOT_DIR,
    CONFIG_PATH,
    REFERENCE_FRONTS_DIR,
    get_reference_front_path,
    get_output_path,
)

# Problem configuration
from .problems import (
    TRAINING_PROBLEMS,
    ZDT_PROBLEMS,
    get_training_problems,
    create_problem_set,
)

__all__ = [
    # Core parameters
    "POPULATION_SIZE",
    "TRAINING_EVALUATIONS",
    "VALIDATION_EVALUATIONS",
    "NUMBER_OF_TRIALS",
    "N_REPEATS",
    "FINAL_AGG",
    "SEED",
    "REFERENCE_POINT_OFFSET",
    # Paths
    "ROOT_DIR",
    "CONFIG_PATH",
    "REFERENCE_FRONTS_DIR",
    "get_reference_front_path",
    "get_output_path",
    # Problems
    "TRAINING_PROBLEMS",
    "ZDT_PROBLEMS",
    "get_training_problems",
    "create_problem_set",
]
