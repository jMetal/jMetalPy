"""Example: Tuning NSGA-II with External Archive on ZDT1.

This example demonstrates how to tune NSGA-II configurations that include
external archives (CrowdingDistanceArchive or DistanceBasedArchive).

External archives maintain elite solutions throughout the optimization run,
which can improve final front quality in some cases.

Requirements:
    - optuna

Run:
    python examples/multiobjective/tuning/tune_nsgaii_with_archive.py
"""

import time
from pathlib import Path

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.tuning import (
    AlgorithmBuilderWithArchive,
    ParameterSpace,
)
from jmetal.util.solution import read_solutions, get_non_dominated_solutions


def load_reference_front(problem_name: str) -> np.ndarray:
    """Load reference front for a problem."""
    fronts_dir = Path(__file__).parent.parent.parent.parent / "resources/reference_fronts"
    csv_path = fronts_dir / f"{problem_name}.csv"
    pf_path = fronts_dir / f"{problem_name}.pf"
    
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",")
    elif pf_path.exists():
        solutions = read_solutions(str(pf_path))
        return np.array([s.objectives for s in solutions])
    else:
        raise FileNotFoundError(f"No reference front found for {problem_name}")


def run_with_archive_config(
    problem,
    archive_type: str,
    archive_size: int = 100,
    max_evaluations: int = 25000,
):
    """Run NSGA-II with a specific archive configuration.
    
    Args:
        problem: The optimization problem.
        archive_type: 'CrowdingDistance', 'DistanceBased', or 'None'.
        archive_size: Maximum archive size.
        max_evaluations: Termination criterion.
        
    Returns:
        Tuple of (front, computing_time).
    """
    builder = AlgorithmBuilderWithArchive(
        algorithm_class=NSGAII,
        problem=problem,
        fixed_params={
            "population_size": 100,
            "max_evaluations": max_evaluations,
        },
    )
    
    # Configuration with archive
    config = {
        "archive": {
            "type": archive_type,
            "maximumSize": archive_size,
            "distanceMetric": "L2_SQUARED",
            "useVectorized": True,
        },
        "offspringPopulationSize": 100,
        "variation": {
            "crossover": {
                "choice": "SBX",
                "crossoverProbability": 0.9,
                "sbxDistributionIndex": 20.0,
            },
            "mutation": {
                "choice": "Polynomial",
                "mutationProbabilityFactor": 1.0,
                "polynomialMutationDistributionIndex": 20.0,
            },
        },
    }
    
    # Build and run
    rng = np.random.default_rng(42)
    algorithm, evaluator = builder.build_with_archive(config, rng=rng)
    
    start_time = time.perf_counter()
    algorithm.run()
    elapsed = time.perf_counter() - start_time
    
    # Get front from archive if available, otherwise from algorithm result
    if evaluator is not None:
        front = evaluator.get_archive().solution_list
    else:
        front = get_non_dominated_solutions(algorithm.result())
    
    return front, elapsed


def compute_hypervolume(front, reference_point):
    """Compute hypervolume of a front."""
    try:
        from jmetal.core.quality_indicator import HyperVolume
        objectives = np.array([s.objectives for s in front])
        hv = HyperVolume(reference_point)
        return hv.compute(objectives)
    except Exception:
        return float('nan')


def main():
    print("=" * 70)
    print("NSGA-II with External Archive Comparison")
    print("=" * 70)
    
    problem = ZDT1()
    reference_point = [1.1, 1.1]
    
    # Configurations to compare
    archive_configs = [
        ("None", "Standard NSGA-II (no archive)"),
        ("CrowdingDistance", "NSGA-II + CrowdingDistanceArchive"),
        ("DistanceBased", "NSGA-II + DistanceBasedArchive"),
    ]
    
    results = []
    
    for archive_type, description in archive_configs:
        print(f"\nRunning: {description}...")
        
        front, elapsed = run_with_archive_config(
            problem=problem,
            archive_type=archive_type,
            archive_size=100,
            max_evaluations=25000,
        )
        
        hv = compute_hypervolume(front, reference_point)
        
        results.append({
            "type": archive_type,
            "description": description,
            "front_size": len(front),
            "hypervolume": hv,
            "time": elapsed,
        })
        
        print(f"  Front size: {len(front)}")
        print(f"  Hypervolume: {hv:.6f}")
        print(f"  Time: {elapsed:.2f}s")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<40} {'Front':<8} {'HV':<12} {'Time':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['description']:<40} {r['front_size']:<8} {r['hypervolume']:<12.6f} {r['time']:<8.2f}s")
    
    print("\n" + "=" * 70)
    print("NOTES:")
    print("- CrowdingDistanceArchive: Good for 2-objective problems (fast)")
    print("- DistanceBasedArchive: Better for 3+ objective problems (more robust)")
    print("- Archive overhead is typically small compared to function evaluations")
    print("=" * 70)


if __name__ == "__main__":
    main()
