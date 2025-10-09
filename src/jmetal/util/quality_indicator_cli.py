#!/usr/bin/env python3
"""
Quality Indicator Command Line Interface for jMetalPy

This module provides a command-line interface for computing various quality indicators
between two fronts (solution front and reference front). It supports:
- Additive Epsilon (epsilon)
- Inverted Generational Distance (igd)
- Inverted Generational Distance Plus (igdplus)
- Hypervolume (hv)
- Normalized Hypervolume (nhv)
- All indicators at once (all)

Usage:
    python -m jmetal.util.quality_indicator_cli <front.csv> <reference.csv> <indicator> [options]

Author: jMetalPy Team
Inspired by: MetaJul qualityIndicatorCLI.jl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from jmetal.core.quality_indicator import (
    AdditiveEpsilonIndicator,
    HyperVolume,
    InvertedGenerationalDistance,
    InvertedGenerationalDistancePlus,
    NormalizedHyperVolume
)


def _load_csv(path: str) -> np.ndarray:
    """Load a CSV file and validate it's a proper 2D numeric matrix."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        data = np.loadtxt(path, delimiter=',')
    except Exception as e:
        raise ValueError(f"Error loading {path}: {e}")
    
    if data.size == 0:
        raise ValueError(f"File '{path}' is empty")
    
    # Ensure it's a 2D array
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(f"Data in {path} must be a 2D matrix")
    
    return data


def _parse_ref_point(ref_point_str: str, expected_dim: int) -> List[float]:
    """Parse reference point string 'v1,v2,...' into a list of floats."""
    try:
        values = [float(x.strip()) for x in ref_point_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid reference point format: {e}")
    
    if len(values) != expected_dim:
        raise ValueError(f"Reference point dimension mismatch: expected {expected_dim}, got {len(values)}")
    
    return values


def _auto_ref_point(reference: np.ndarray, margin: float = 0.1) -> List[float]:
    """Automatically generate a reference point based on the reference front."""
    max_vals = np.max(reference, axis=0)
    return (max_vals + margin).tolist()


def _normalize_fronts(front: np.ndarray, reference: np.ndarray, method: str = "reference_only") -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize fronts using the specified method.
    
    Args:
        front: Solution front matrix
        reference: Reference front matrix
        method: Normalization method ("reference_only" supported)
    
    Returns:
        Tuple of (normalized_front, normalized_reference)
    """
    if method == "reference_only":
        # Normalize based on reference front bounds
        min_vals = np.min(reference, axis=0)
        max_vals = np.max(reference, axis=0)
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1, ranges)
        
        # Normalize both fronts using reference bounds
        front_norm = (front - min_vals) / ranges
        reference_norm = (reference - min_vals) / ranges
        
        return front_norm, reference_norm
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def _compute_all_indicators(front: np.ndarray, reference: np.ndarray, ref_point: List[float]) -> Dict[str, float]:
    """Compute all quality indicators."""
    results = {}
    
    # Additive Epsilon
    epsilon_indicator = AdditiveEpsilonIndicator(reference)
    results["epsilon"] = epsilon_indicator.compute(front)
    
    # IGD
    igd_indicator = InvertedGenerationalDistance(reference)
    results["igd"] = igd_indicator.compute(front)
    
    # IGD+
    igdplus_indicator = InvertedGenerationalDistancePlus(reference)
    results["igdplus"] = igdplus_indicator.compute(front)
    
    # Hypervolume
    hv_indicator = HyperVolume(ref_point)
    results["hv"] = hv_indicator.compute(front)
    
    # Normalized Hypervolume
    nhv_indicator = NormalizedHyperVolume(ref_point, reference)
    results["nhv"] = nhv_indicator.compute(front)
    
    return results


def _print_results(results: Dict[str, float], output_format: str = "text") -> None:
    """Print results in the specified format."""
    if output_format == "text":
        for key in sorted(results.keys()):
            print(f"Result ({key}): {results[key]}")
    elif output_format == "json":
        print(json.dumps(results, sort_keys=True))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _validate_reference_point(ref_point: List[float], front: np.ndarray) -> None:
    """Validate that reference point is dominated by all solutions in the front."""
    ref_array = np.array(ref_point)
    
    # Check if reference point is dominated by all solutions (for minimization)
    # All solutions should have at least one objective better than ref_point
    dominated_by_all = np.all(np.any(front <= ref_array, axis=1))
    
    if not dominated_by_all:
        print("Warning: Reference point may not be dominated by all solutions in the front.")
        print(f"Reference point: {ref_point}")
        print(f"Front max values: {np.max(front, axis=0).tolist()}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Compute quality indicators between two fronts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute IGD between two fronts
  python -m jmetal.util.quality_indicator_cli front.csv reference.csv igd
  
  # Compute all indicators with custom reference point
  python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --ref-point 2.0,2.0
  
  # Normalize fronts and output as JSON
  python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --normalize --format json

Indicators:
  epsilon   - Additive Epsilon indicator
  igd       - Inverted Generational Distance
  igdplus   - Inverted Generational Distance Plus
  hv        - Hypervolume
  nhv       - Normalized Hypervolume
  all       - Compute all indicators

Notes:
  - CSV files should contain numeric data with one solution per row
  - HV and NHV require a reference point worse than all solutions
  - NHV = 1 - HV(front)/HV(reference), can be negative if front dominates reference
        """
    )
    
    parser.add_argument("front_file", help="Path to solution front CSV file")
    parser.add_argument("reference_file", help="Path to reference front CSV file")
    parser.add_argument("indicator", 
                       choices=["epsilon", "igd", "igdplus", "hv", "nhv", "all"],
                       help="Quality indicator to compute")
    
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize both fronts using reference_only strategy")
    parser.add_argument("--ref-point", type=str,
                       help="Custom reference point for HV/NHV as comma-separated values (e.g., '2.0,2.0')")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--margin", type=float, default=0.1,
                       help="Margin added when auto-building reference point (default: 0.1)")
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load data
        front = _load_csv(args.front_file)
        reference = _load_csv(args.reference_file)
        
        # Validate dimensions
        if front.shape[1] != reference.shape[1]:
            raise ValueError(f"Objective dimension mismatch: front has {front.shape[1]} objectives, "
                           f"reference has {reference.shape[1]} objectives")
        
        # Normalize if requested
        if args.normalize:
            front, reference = _normalize_fronts(front, reference)
            if args.format == "text":
                print("Fronts normalized (method=reference_only)")
        
        # Handle reference point for HV/NHV indicators
        indicators_needing_ref_point = {"hv", "nhv", "all"}
        ref_point = None
        
        if args.indicator in indicators_needing_ref_point:
            if args.ref_point is not None:
                ref_point = _parse_ref_point(args.ref_point, front.shape[1])
            else:
                if args.normalize:
                    # For normalized data, use 1.1 repeated
                    ref_point = [1.1] * front.shape[1]
                else:
                    # Auto-generate reference point
                    ref_point = _auto_ref_point(reference, args.margin)
            
            # Validate reference point
            _validate_reference_point(ref_point, front)
        
        # Compute indicators
        if args.indicator == "all":
            results = _compute_all_indicators(front, reference, ref_point)
        elif args.indicator == "epsilon":
            indicator = AdditiveEpsilonIndicator(reference)
            result = indicator.compute(front)
            results = {"epsilon": result}
        elif args.indicator == "igd":
            indicator = InvertedGenerationalDistance(reference)
            result = indicator.compute(front)
            results = {"igd": result}
        elif args.indicator == "igdplus":
            indicator = InvertedGenerationalDistancePlus(reference)
            result = indicator.compute(front)
            results = {"igdplus": result}
        elif args.indicator == "hv":
            indicator = HyperVolume(ref_point)
            result = indicator.compute(front)
            results = {"hv": result}
        elif args.indicator == "nhv":
            indicator = NormalizedHyperVolume(ref_point, reference)
            result = indicator.compute(front)
            results = {"nhv": result}
        
        # Output results
        _print_results(results, args.format)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()