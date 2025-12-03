"""
Default configuration constants for hyperparameter tuning.

This module contains the core tuning parameters. For path and problem
configuration, see paths.py and problems.py respectively.

All settings can be overridden at runtime via function parameters.
"""

# ============================================================================
# ALGORITHM SETTINGS
# ============================================================================
POPULATION_SIZE = 100

# ============================================================================
# TUNING SETTINGS
# ============================================================================
TRAINING_EVALUATIONS = 10000  # Evaluations per problem during tuning
VALIDATION_EVALUATIONS = 20000  # Evaluations for validation (2x training)
NUMBER_OF_TRIALS = 500  # Total Optuna trials
N_REPEATS = 1  # Independent runs per trial (increase to reduce variance)
FINAL_AGG = "mean"  # Aggregation across problems: "sum", "mean", or "median"
SEED = 42  # Random seed for reproducibility

# ============================================================================
# QUALITY INDICATOR SETTINGS
# ============================================================================
REFERENCE_POINT_OFFSET = 0.1  # Offset for hypervolume reference point

# Re-export paths and problems for backward compatibility
from .paths import (
    ROOT_DIR,
    CONFIG_PATH,
    REFERENCE_FRONTS_DIR,
    get_reference_front_path,
    get_output_path,
)

from .problems import (
    TRAINING_PROBLEMS,
    ZDT_PROBLEMS,
    get_training_problems,
    create_problem_set,
)

