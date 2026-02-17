"""
Metrics module for algorithm tuning.

This module provides quality indicators and reference front utilities
for evaluating Pareto front approximations during hyperparameter tuning.

Components:
    - indicators: Quality indicator computations (NHV, Additive Epsilon)
    - reference_fronts: Reference front loading and utilities

Example:
    >>> from jmetal.tuning.metrics import (
    ...     compute_quality_indicators,
    ...     load_reference_front,
    ... )
    >>> ref_front = load_reference_front('ZDT1.pf')
    >>> nhv, ae = compute_quality_indicators(solutions, ref_front)
"""

from .indicators import (
    compute_quality_indicators,
    compute_combined_score,
    aggregate_scores,
)

from .reference_fronts import (
    load_reference_front,
    get_reference_point,
    validate_reference_front,
)

__all__ = [
    # Quality indicators
    "compute_quality_indicators",
    "compute_combined_score",
    "aggregate_scores",
    # Reference front utilities
    "load_reference_front",
    "get_reference_point",
    "validate_reference_front",
]
