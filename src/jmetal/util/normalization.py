"""
Normalization utilities for Pareto fronts.

This module provides functions to normalize Pareto fronts before applying quality indicators.
Normalization is essential to avoid bias from objectives with different scales.
"""

from typing import Tuple, List, Literal

import numpy as np

from jmetal.core.solution import Solution

# Type alias for normalization methods
NormalizationMethod = Literal["minmax", "zscore", "reference_only"]


def normalize_fronts(
    front: np.ndarray, 
    reference_front: np.ndarray, 
    method: NormalizationMethod = "reference_only"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both solution front and reference front to the same scale.
    
    This ensures that quality indicators are not biased by objectives with different scales.
    Typically normalizes to [0,1] for each objective.
    
    Args:
        front: Solution front matrix (each row is a solution, each column an objective).
        reference_front: Reference front matrix (each row is a solution, each column an objective).
        method: Normalization method. Options:
            - "minmax": Normalize to [0,1] using global min/max across both fronts
            - "zscore": Standardize using global mean and standard deviation  
            - "reference_only": Normalize using only reference front bounds (recommended, default)
    
    Returns:
        Tuple of (normalized_front, normalized_reference_front) where both matrices are normalized.
        
    Raises:
        ValueError: If fronts have different number of objectives or invalid method.
        
    Examples:
        >>> front = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 150.0]])
        >>> reference_front = np.array([[0.5, 80.0], [1.5, 120.0], [2.5, 180.0]])
        >>> norm_front, norm_ref = normalize_fronts(front, reference_front)
        >>> # Uses reference_only by default
        
        >>> # Using different methods
        >>> norm_front, norm_ref = normalize_fronts(front, reference_front, method="minmax")
        
    Note:
        For quality indicators, "reference_only" is typically recommended as it uses the 
        reference front to define the normalization bounds, which is more appropriate 
        for performance assessment.
    """
    if front.shape[1] != reference_front.shape[1]:
        raise ValueError("Fronts must have the same number of objectives")
    
    number_of_objectives = front.shape[1]
    normalized_front = np.zeros_like(front, dtype=np.float64)
    normalized_reference_front = np.zeros_like(reference_front, dtype=np.float64)
    
    if method == "minmax":
        # Use global min/max across both fronts
        combined_matrix = np.vstack([front, reference_front])
        
        for objective_index in range(number_of_objectives):
            objective_column = combined_matrix[:, objective_index]
            objective_min = np.min(objective_column)
            objective_max = np.max(objective_column)
            objective_range = objective_max - objective_min
            
            if objective_range > 0:
                # Normalize to [0,1]
                normalized_front[:, objective_index] = (front[:, objective_index] - objective_min) / objective_range
                normalized_reference_front[:, objective_index] = (reference_front[:, objective_index] - objective_min) / objective_range
            else:
                # All values are the same
                normalized_front[:, objective_index] = 0.0
                normalized_reference_front[:, objective_index] = 0.0
                
    elif method == "zscore":
        # Standardize using global mean and standard deviation
        combined_matrix = np.vstack([front, reference_front])
        
        for objective_index in range(number_of_objectives):
            objective_column = combined_matrix[:, objective_index]
            objective_mean = np.mean(objective_column)
            objective_std = np.std(objective_column, ddof=1)  # Sample standard deviation
            
            if objective_std > 0:
                normalized_front[:, objective_index] = (front[:, objective_index] - objective_mean) / objective_std
                normalized_reference_front[:, objective_index] = (reference_front[:, objective_index] - objective_mean) / objective_std
            else:
                # All values are the same
                normalized_front[:, objective_index] = 0.0
                normalized_reference_front[:, objective_index] = 0.0
                
    elif method == "reference_only":
        # Use only reference front to define normalization bounds (recommended for quality indicators)
        for objective_index in range(number_of_objectives):
            reference_column = reference_front[:, objective_index]
            reference_min = np.min(reference_column)
            reference_max = np.max(reference_column)
            reference_range = reference_max - reference_min
            
            if reference_range > 0:
                # Normalize using reference front bounds
                normalized_front[:, objective_index] = (front[:, objective_index] - reference_min) / reference_range
                normalized_reference_front[:, objective_index] = (reference_front[:, objective_index] - reference_min) / reference_range
            else:
                # All reference values are the same
                normalized_front[:, objective_index] = front[:, objective_index] - reference_min
                normalized_reference_front[:, objective_index] = 0.0
                
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'minmax', 'zscore', or 'reference_only'")
    
    return normalized_front, normalized_reference_front


def normalize_front(
    front: np.ndarray, 
    bounds_min: np.ndarray, 
    bounds_max: np.ndarray
) -> np.ndarray:
    """
    Normalize a front using predefined bounds for each objective.
    
    Args:
        front: Front matrix to normalize (each row is a solution, each column an objective).
        bounds_min: Minimum values for each objective.
        bounds_max: Maximum values for each objective.
        
    Returns:
        Normalized front matrix.
        
    Raises:
        ValueError: If dimensions don't match.
        
    Examples:
        >>> front = np.array([[1.0, 100.0], [2.0, 200.0]])
        >>> min_bounds = np.array([0.0, 50.0])
        >>> max_bounds = np.array([5.0, 250.0])
        >>> normalized = normalize_front(front, min_bounds, max_bounds)
    """
    if not (front.shape[1] == len(bounds_min) == len(bounds_max)):
        raise ValueError("Dimension mismatch between front and bounds")
    
    normalized_front = np.zeros_like(front, dtype=np.float64)
    number_of_objectives = front.shape[1]
    
    for objective_index in range(number_of_objectives):
        objective_range = bounds_max[objective_index] - bounds_min[objective_index]
        
        if objective_range > 0:
            normalized_front[:, objective_index] = (front[:, objective_index] - bounds_min[objective_index]) / objective_range
        else:
            normalized_front[:, objective_index] = front[:, objective_index] - bounds_min[objective_index]
    
    return normalized_front


def solutions_to_matrix(solutions: List[Solution]) -> np.ndarray:
    """
    Convert a list of jMetal Solution objects to a numpy matrix of objectives.
    
    Args:
        solutions: List of Solution objects.
        
    Returns:
        Matrix where each row is a solution and each column is an objective.
        
    Examples:
        >>> # Assuming solutions is a list of Solution objects
        >>> matrix = solutions_to_matrix(solutions)
        >>> normalized_matrix, _ = normalize_fronts(matrix, reference_matrix)
    """
    if not solutions:
        return np.array([]).reshape(0, 0)
    
    objectives_matrix = np.array([solution.objectives for solution in solutions])
    return objectives_matrix


def normalize_solution_fronts(
    solutions: List[Solution],
    reference_solutions: List[Solution], 
    method: NormalizationMethod = "reference_only"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize fronts from jMetal Solution objects.
    
    Convenience function that converts Solution objects to matrices and normalizes them.
    
    Args:
        solutions: List of solution objects.
        reference_solutions: List of reference solution objects.
        method: Normalization method.
        
    Returns:
        Tuple of (normalized_front, normalized_reference_front) as numpy arrays.
        
    Examples:
        >>> # Assuming solutions and reference_solutions are lists of Solution objects
        >>> norm_front, norm_ref = normalize_solution_fronts(solutions, reference_solutions)
        >>> # Can now be used with quality indicators
    """
    front_matrix = solutions_to_matrix(solutions)
    reference_matrix = solutions_to_matrix(reference_solutions)
    
    return normalize_fronts(front_matrix, reference_matrix, method)


def get_ideal_and_nadir_points(front: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get ideal and nadir points from a front.
    
    Args:
        front: Front matrix (each row is a solution, each column an objective).
        
    Returns:
        Tuple of (ideal_point, nadir_point) where:
        - ideal_point: Minimum value for each objective
        - nadir_point: Maximum value for each objective
        
    Examples:
        >>> front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        >>> ideal, nadir = get_ideal_and_nadir_points(front)
        >>> # ideal = [1.0, 1.0], nadir = [3.0, 3.0]
    """
    if front.size == 0:
        return np.array([]), np.array([])
    
    ideal_point = np.min(front, axis=0)
    nadir_point = np.max(front, axis=0)
    
    return ideal_point, nadir_point


def normalize_to_unit_hypercube(
    front: np.ndarray,
    ideal_point: np.ndarray = None,
    nadir_point: np.ndarray = None
) -> np.ndarray:
    """
    Normalize a front to the unit hypercube [0,1]^m.
    
    Args:
        front: Front matrix to normalize.
        ideal_point: Ideal point (minimum values). If None, computed from front.
        nadir_point: Nadir point (maximum values). If None, computed from front.
        
    Returns:
        Normalized front in [0,1]^m.
        
    Examples:
        >>> front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        >>> normalized = normalize_to_unit_hypercube(front)
        >>> # All values will be in [0,1]
    """
    if front.size == 0:
        return front
        
    if ideal_point is None or nadir_point is None:
        computed_ideal, computed_nadir = get_ideal_and_nadir_points(front)
        if ideal_point is None:
            ideal_point = computed_ideal
        if nadir_point is None:
            nadir_point = computed_nadir
    
    return normalize_front(front, ideal_point, nadir_point)