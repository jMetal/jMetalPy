"""Example: Parallel tuning of NSGA-II with multiple CPU cores.

This example demonstrates how to use parallel execution to speed up
the tuning process by distributing algorithm runs across CPU cores.

Usage:
    python tune_nsgaii_parallel.py

The parallelism is applied at the run level (within each trial), not
at the trial level. This is the recommended approach because:
1. It works safely with SQLite storage
2. It provides good speedup when N > 1 or when using many problems
3. It maintains reproducibility through deterministic seeding
"""

import os
import time
from pathlib import Path

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4

from jmetal.tuning import (
    TuningProtocol,
    TuningConfig,
    ParameterSpace,
)


def main():
    # Training set: ZDT problems
    problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4()]
    
    # Load reference fronts
    fronts_dir = Path(__file__).parent.parent.parent.parent / "resources" / "reference_fronts"
    reference_fronts = TuningProtocol.load_reference_fronts(problems, fronts_dir)
    
    # Load parameter space
    param_spaces_dir = Path(__file__).parent.parent.parent.parent / "src" / "jmetal" / "tuning" / "parameter_spaces"
    parameter_space = ParameterSpace.from_yaml(param_spaces_dir / "NSGAIIFloat.yaml")
    
    # Configuration with parallel runs enabled
    # Using 4 workers and 3 repetitions per problem
    config = TuningConfig(
        n_repeats=3,
        population_size=100,
        max_evaluations=10000,  # Reduced for faster example
        base_seed=42,
        indicator_names=["NHV", "IGD+"],
        parallel_runs=True,   # Enable parallel execution
        n_workers=4,          # Number of CPU cores (None = all available)
    )
    
    # Create tuning protocol
    protocol = TuningProtocol(
        algorithm_class=NSGAII,
        parameter_space=parameter_space,
        problems=problems,
        reference_fronts=reference_fronts,
        config=config,
        artifact_dir="artifacts_parallel",
        study_name="nsgaii_parallel_tuning",
    )
    
    print("=" * 60)
    print("Parallel NSGA-II Tuning Example")
    print("=" * 60)
    print(f"Problems: {[p.name() for p in problems]}")
    print(f"Repetitions per problem: {config.n_repeats}")
    print(f"Parallel workers: {config.n_workers}")
    print(f"Total runs per trial: {len(problems) * config.n_repeats}")
    print("=" * 60)
    
    # Create study with SQLite storage (safe with n_jobs=1)
    study = protocol.create_study(
        storage="sqlite:///parallel_tuning.db",
        load_if_exists=False,
    )
    
    # Run optimization with timing
    n_trials = 10
    print(f"\nRunning {n_trials} trials with parallel execution...\n")
    
    start_time = time.perf_counter()
    study.optimize(
        protocol.objective, 
        n_trials=n_trials,
        n_jobs=1,  # Keep at 1 for SQLite; parallelism is internal
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - start_time
    
    # Show results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Time per trial: {elapsed / n_trials:.2f} seconds")
    print(f"Best score: {study.best_value:.6f}")
    print("\nBest configuration:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save summary
    summary_path = protocol.save_summary(study, elapsed)
    if summary_path:
        print(f"\nSummary saved to: {summary_path}")
    
    # Comparison hint
    print("\n" + "-" * 60)
    print("To compare with sequential execution, run with:")
    print("  config = TuningConfig(..., parallel_runs=False)")
    print("-" * 60)


if __name__ == "__main__":
    main()
