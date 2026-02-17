#!/usr/bin/env python
"""
Example: Comparing different Optuna samplers.

This example compares the performance of different Optuna samplers
(TPE, CMA-ES, Random) for hyperparameter tuning.

Usage:
    python examples/tuning/compare_samplers.py
    python examples/tuning/compare_samplers.py --trials 30
"""

import argparse
import json

from jmetal.problem import ZDT1, ZDT2
from jmetal.tuning import tune


def main():
    parser = argparse.ArgumentParser(
        description="Compare different Optuna samplers"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=15,
        help="Number of trials per sampler (default: 15)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./sampler_comparison_best.json",
        help="Output file for best configuration (default: ./sampler_comparison_best.json)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comparing Optuna Samplers for NSGA-II Tuning")
    print("=" * 60)
    print()
    
    # Test problems
    problems = [
        (ZDT1(), "ZDT1.pf"),
        (ZDT2(), "ZDT2.pf"),
    ]
    
    # Samplers to compare
    samplers = [
        ("tpe", "categorical"),      # TPE with categorical params
        ("tpe", "continuous"),       # TPE with continuous params
        ("cmaes", "continuous"),     # CMA-ES (requires continuous)
        ("random", "categorical"),   # Random baseline
    ]
    
    results = {}
    best_result = None
    best_name = None
    
    for sampler, mode in samplers:
        name = f"{sampler.upper()} ({mode})"
        print(f"\n{'='*40}")
        print(f"Testing: {name}")
        print(f"{'='*40}")
        
        result = tune(
            algorithm="NSGAII",
            problems=problems,
            n_trials=args.trials,
            n_evaluations=5000,
            sampler=sampler,
            mode=mode,
            verbose=False,
        )
        
        results[name] = result.best_score
        print(f"Best score: {result.best_score:.6f}")
        
        # Track the best result overall
        if best_result is None or result.best_score < best_result.best_score:
            best_result = result
            best_name = name
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Sampler':<25} {'Best Score':>15}")
    print("-" * 40)
    
    # Sort by score (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, score in sorted_results:
        print(f"{name:<25} {score:>15.6f}")
    
    print()
    print(f"Winner: {best_name}")
    
    # Save best configuration
    if best_result is not None:
        output_data = best_result.to_dict()
        output_data["winning_sampler"] = best_name
        output_data["all_results"] = {name: score for name, score in sorted_results}
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nBest configuration saved to: {args.output}")


if __name__ == "__main__":
    main()
