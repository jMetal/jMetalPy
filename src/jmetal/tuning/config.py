"""
Configuration for hyperparameter tuning.

This module centralizes all configuration constants and training problem definitions.
Users can modify these settings to customize the tuning process.
"""

from pathlib import Path
from typing import List, Tuple

from jmetal.core.problem import Problem
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

# ============================================================================
# PATHS
# ============================================================================
# ROOT_DIR points to the jMetalPy project root (3 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "nsgaii_tuned_config.json"
REFERENCE_FRONTS_DIR = ROOT_DIR / "resources" / "reference_fronts"

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

# ============================================================================
# TRAINING PROBLEMS
# Define the problems used for hyperparameter tuning.
# Format: (Problem instance, reference_front_filename)
# The reference_front_filename is the full name of the file (with extension)
# in the REFERENCE_FRONTS_DIR directory.
# The problem name is obtained from problem.name().
# ============================================================================
TRAINING_PROBLEMS: List[Tuple[Problem, str]] = [
    (ZDT1(), "ZDT1.pf"),
    (ZDT2(), "ZDT2.pf"),
    (ZDT3(), "ZDT3.pf"),
    (ZDT4(), "ZDT4.pf"),
    (ZDT6(), "ZDT6.pf"),
]


def get_reference_front_path(reference_front_filename: str) -> Path:
    """
    Get the path to the reference front file.
    
    Args:
        reference_front_filename: Full name of the reference front file (with extension)
        
    Returns:
        Path to the reference front file
    """
    return REFERENCE_FRONTS_DIR / reference_front_filename
