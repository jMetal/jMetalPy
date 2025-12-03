"""
Configuration module for hyperparameter tuning.

This module re-exports all configuration from defaults.py for backward compatibility.
"""

from .defaults import (
    # Paths
    ROOT_DIR,
    CONFIG_PATH,
    REFERENCE_FRONTS_DIR,
    get_reference_front_path,
    # Algorithm settings
    POPULATION_SIZE,
    # Tuning settings
    TRAINING_EVALUATIONS,
    VALIDATION_EVALUATIONS,
    NUMBER_OF_TRIALS,
    N_REPEATS,
    FINAL_AGG,
    SEED,
    # Quality indicator settings
    REFERENCE_POINT_OFFSET,
    # Problems
    TRAINING_PROBLEMS,
)

__all__ = [
    "ROOT_DIR",
    "CONFIG_PATH",
    "REFERENCE_FRONTS_DIR",
    "get_reference_front_path",
    "POPULATION_SIZE",
    "TRAINING_EVALUATIONS",
    "VALIDATION_EVALUATIONS",
    "NUMBER_OF_TRIALS",
    "N_REPEATS",
    "FINAL_AGG",
    "SEED",
    "REFERENCE_POINT_OFFSET",
    "TRAINING_PROBLEMS",
]
