#!/usr/bin/env python
"""
Example: Using metrics functions directly.

This example shows how to use the metrics module independently
to compute quality indicators for Pareto front approximations.

Usage:
    python examples/tuning/using_metrics.py
"""

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.tuning.metrics import (
    compute_quality_indicators,
    load_reference_front,
    aggregate_scores,
)


def main():
    print("=" * 60)
    print("Using Tuning Metrics Module")
    print("=" * 60)
    print()
    
    # Create and solve a problem
    problem = ZDT1()
    print(f"Problem: {problem.name()}")
    print(f"Variables: {problem.number_of_variables()}")
    print(f"Objectives: {problem.number_of_objectives()}")
    print()
    
    # Configure NSGA-II
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(),
            distribution_index=20.0,
        ),
        crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )
    
    print("Running NSGA-II...")
    algorithm.run()
    
    # Get Pareto front approximation
    solutions = get_non_dominated_solutions(algorithm.result())
    print(f"Found {len(solutions)} non-dominated solutions")
    print()
    
    # Load reference front
    print("Loading reference front...")
    ref_front = load_reference_front("ZDT1.pf")
    print(f"Reference front has {len(ref_front)} points")
    print()
    
    # Compute quality indicators
    print("Computing quality indicators...")
    nhv, ae = compute_quality_indicators(
        front=solutions,
        reference_front=ref_front,
        reference_point_offset=0.1,
    )
    
    print(f"  Normalized Hypervolume (NHV): {nhv:.6f}")
    print(f"  Additive Epsilon (AE): {ae:.6f}")
    print()
    
    # Demonstrate score aggregation
    print("Demonstrating score aggregation:")
    scores = [0.15, 0.18, 0.12, 0.20, 0.16]  # Example scores
    print(f"  Example scores: {scores}")
    print(f"  Mean: {aggregate_scores(scores, 'mean'):.4f}")
    print(f"  Median: {aggregate_scores(scores, 'median'):.4f}")
    print(f"  Min: {aggregate_scores(scores, 'min'):.4f}")
    print(f"  Max: {aggregate_scores(scores, 'max'):.4f}")


if __name__ == "__main__":
    main()
