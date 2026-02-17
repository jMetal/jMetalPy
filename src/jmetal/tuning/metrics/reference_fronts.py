"""
Reference front utilities for algorithm tuning.

This module provides functions to load and manage reference Pareto fronts
used for computing quality indicators.
"""

from pathlib import Path
from typing import Union

import numpy as np

from jmetal.util.solution import read_solutions


def load_reference_front(
    filename: str,
    reference_fronts_dir: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Load a reference Pareto front from file.
    
    Reference fronts are typically stored as text files with one solution
    per line and objectives separated by spaces or tabs.
    
    Args:
        filename: Filename with extension (e.g., 'ZDT1.pf')
        reference_fronts_dir: Directory containing reference fronts.
            If None, uses the default path from config.
            
    Returns:
        Reference front as numpy array of shape (n_solutions, n_objectives)
        
    Raises:
        FileNotFoundError: If the reference front file doesn't exist
        
    Example:
        >>> from jmetal.tuning.metrics import load_reference_front
        >>> ref_front = load_reference_front('ZDT1.pf')
        >>> print(f"Loaded {len(ref_front)} reference points")
    """
    if reference_fronts_dir is None:
        from jmetal.tuning.config import get_reference_front_path
        ref_path = get_reference_front_path(filename)
    else:
        ref_path = Path(reference_fronts_dir) / filename
        
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference front not found: {ref_path}\n"
            f"Make sure the file '{filename}' exists in the reference fronts directory."
        )
    
    solutions = read_solutions(str(ref_path))
    return np.array([s.objectives for s in solutions])


def get_reference_point(
    reference_front: np.ndarray,
    offset: float = 0.1,
) -> np.ndarray:
    """
    Calculate the reference point for hypervolume computation.
    
    The reference point is computed as the maximum objective values
    in the reference front plus an offset.
    
    Args:
        reference_front: Reference front as numpy array
        offset: Offset to add to maximum values
        
    Returns:
        Reference point as 1D numpy array
        
    Example:
        >>> ref_point = get_reference_point(reference_front, offset=0.1)
    """
    max_values = np.max(reference_front, axis=0)
    return max_values + offset


def validate_reference_front(
    reference_front: np.ndarray,
    expected_objectives: int = 2,
) -> bool:
    """
    Validate a reference front for use in indicator computation.
    
    Args:
        reference_front: Reference front to validate
        expected_objectives: Expected number of objectives
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> if validate_reference_front(ref_front, expected_objectives=2):
        ...     # Safe to use
    """
    if reference_front.ndim != 2:
        return False
    if reference_front.shape[0] == 0:
        return False
    if reference_front.shape[1] != expected_objectives:
        return False
    return True
