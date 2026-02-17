"""
Configuration for hyperparameter tuning.

This module centralizes all configuration constants and training problem definitions.
Users can modify these settings to customize the tuning process.

Usage:
    >>> from jmetal.tuning.config import POPULATION_SIZE, TRAINING_PROBLEMS
    >>> from jmetal.tuning.config import get_reference_front_path
"""

from pathlib import Path
from typing import List, Tuple

from jmetal.core.problem import Problem
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

# ============================================================================
# PATHS
# ============================================================================
# ROOT_DIR points to the jMetalPy project root (4 levels up from this file)
# Structure: config.py -> tuning/ -> jmetal/ -> src/ -> ROOT
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Directory containing reference Pareto fronts
REFERENCE_FRONTS_DIR = ROOT_DIR / "resources" / "reference_fronts"

# Output path for tuned configuration (deprecated - use get_output_path())
CONFIG_PATH = Path(__file__).resolve().parent / "nsgaii_tuned_config.json"

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
# ============================================================================

# ZDT benchmark suite
ZDT_PROBLEMS: List[Tuple[Problem, str]] = [
    (ZDT1(), "ZDT1.pf"),
    (ZDT2(), "ZDT2.pf"),
    (ZDT3(), "ZDT3.pf"),
    (ZDT4(), "ZDT4.pf"),
    (ZDT6(), "ZDT6.pf"),
]

# Default training problem set
TRAINING_PROBLEMS: List[Tuple[Problem, str]] = ZDT_PROBLEMS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_reference_front_path(reference_front_filename: str) -> Path:
    """
    Get the path to the reference front file.
    
    Args:
        reference_front_filename: Full name of the reference front file (with extension)
        
    Returns:
        Path to the reference front file
        
    Example:
        >>> path = get_reference_front_path("ZDT1.pf")
        >>> print(path)  # /path/to/jMetalPy/resources/reference_fronts/ZDT1.pf
    """
    return REFERENCE_FRONTS_DIR / reference_front_filename


def get_output_path(filename: str = "nsgaii_tuned_config.json") -> Path:
    """
    Get the path for saving tuning results.
    
    Args:
        filename: Output filename
        
    Returns:
        Path to the output file (in current directory)
    """
    return Path.cwd() / filename


def get_training_problems() -> List[Tuple[Problem, str]]:
    """
    Get the list of training problems.
    
    Returns:
        List of (problem, reference_front_file) tuples
    """
    return TRAINING_PROBLEMS.copy()


def create_problem_set(
    problem_classes: List,
    reference_front_pattern: str = "{name}.pf",
) -> List[Tuple[Problem, str]]:
    """
    Create a problem set from problem classes.
    
    Args:
        problem_classes: List of problem classes to instantiate
        reference_front_pattern: Pattern for reference front filenames.
            Use {name} as placeholder for problem name.
            
    Returns:
        List of (problem, reference_front_file) tuples
        
    Example:
        >>> from jmetal.problem import ZDT1, ZDT2
        >>> problems = create_problem_set([ZDT1, ZDT2])
        >>> # [(ZDT1(), 'ZDT1.pf'), (ZDT2(), 'ZDT2.pf')]
    """
    problems = []
    for cls in problem_classes:
        problem = cls()
        ref_file = reference_front_pattern.format(name=problem.name())
        problems.append((problem, ref_file))
    return problems
