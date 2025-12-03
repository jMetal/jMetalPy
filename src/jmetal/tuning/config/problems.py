"""
Training problem configuration for hyperparameter tuning.

This module defines the problems used during the tuning process.
Users can modify these settings to use different problem sets.
"""

from typing import List, Tuple

from jmetal.core.problem import Problem
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

# ============================================================================
# TRAINING PROBLEMS
# Define the problems used for hyperparameter tuning.
# Format: (Problem instance, reference_front_filename)
# The reference_front_filename is the full name of the file (with extension)
# in the REFERENCE_FRONTS_DIR directory.
# ============================================================================

# Default training problems: ZDT benchmark suite
ZDT_PROBLEMS: List[Tuple[Problem, str]] = [
    (ZDT1(), "ZDT1.pf"),
    (ZDT2(), "ZDT2.pf"),
    (ZDT3(), "ZDT3.pf"),
    (ZDT4(), "ZDT4.pf"),
    (ZDT6(), "ZDT6.pf"),
]

# Default training problem set
TRAINING_PROBLEMS: List[Tuple[Problem, str]] = ZDT_PROBLEMS


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
