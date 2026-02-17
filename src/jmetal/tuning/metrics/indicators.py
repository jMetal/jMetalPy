"""
Quality indicators for algorithm tuning.

This module provides functions to compute quality indicators for
evaluating Pareto front approximations during hyperparameter tuning.
"""

from typing import List, Tuple

import numpy as np

from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator


def compute_quality_indicators(
    front: List,
    reference_front: np.ndarray,
    reference_point_offset: float = 0.1,
) -> Tuple[float, float]:
    """
    Compute quality indicators for a Pareto front approximation.
    
    This function calculates both Normalized Hypervolume (NHV) and Additive
    Epsilon (AE) indicators. Both indicators are formulated so that lower
    values indicate better quality.
    
    Args:
        front: List of non-dominated solutions with .objectives attribute
        reference_front: Reference Pareto front as numpy array of shape (n, m)
        reference_point_offset: Offset for hypervolume reference point calculation
        
    Returns:
        Tuple of (normalized_hypervolume, additive_epsilon) values
        
    Example:
        >>> from jmetal.tuning.metrics import compute_quality_indicators
        >>> nhv, ae = compute_quality_indicators(solutions, reference, 0.1)
        >>> print(f"NHV: {nhv:.4f}, AE: {ae:.4f}")
    """
    # Extract objective values as numpy array
    objectives = np.array([s.objectives for s in front])
    
    # Create Normalized Hypervolume indicator
    nhv_indicator = NormalizedHyperVolume(
        reference_front=reference_front,
        reference_point_offset=reference_point_offset,
    )
    nhv_indicator.set_reference_front(reference_front)
    
    # Create Additive Epsilon indicator
    ae_indicator = AdditiveEpsilonIndicator(reference_front)
    
    # Compute indicator values
    nhv_value = float(nhv_indicator.compute(objectives))
    ae_value = float(ae_indicator.compute(objectives))
    
    return nhv_value, ae_value


def compute_combined_score(
    nhv_value: float,
    ae_value: float,
    nhv_weight: float = 1.0,
    ae_weight: float = 1.0,
) -> float:
    """
    Compute a combined score from multiple indicators.
    
    This function aggregates multiple quality indicators into a single
    score for optimization. By default, it uses equal weights.
    
    Args:
        nhv_value: Normalized Hypervolume value (lower is better)
        ae_value: Additive Epsilon value (lower is better)
        nhv_weight: Weight for NHV in combined score
        ae_weight: Weight for AE in combined score
        
    Returns:
        Combined score (lower is better)
        
    Example:
        >>> score = compute_combined_score(0.15, 0.08)
        >>> print(f"Combined score: {score:.4f}")
    """
    return nhv_weight * nhv_value + ae_weight * ae_value


def aggregate_scores(
    scores: List[float],
    method: str = "mean",
) -> float:
    """
    Aggregate multiple scores using the specified method.
    
    Args:
        scores: List of individual scores
        method: Aggregation method - 'mean', 'median', 'min', 'max', or 'sum'
        
    Returns:
        Aggregated score
        
    Raises:
        ValueError: If method is not recognized
        
    Example:
        >>> scores = [0.23, 0.18, 0.21]
        >>> final = aggregate_scores(scores, method='mean')
    """
    if not scores:
        return float('inf')
        
    if method == "mean":
        return float(np.mean(scores))
    elif method == "median":
        return float(np.median(scores))
    elif method == "min":
        return float(np.min(scores))
    elif method == "max":
        return float(np.max(scores))
    elif method == "sum":
        return float(np.sum(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}. "
                        f"Use 'mean', 'median', 'min', 'max', or 'sum'.")
