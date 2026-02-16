"""Example: Tuning NSGA-II on ZDT1 using Optuna.

This example demonstrates how to use the jmetal.tuning module to perform
hyperparameter optimization for NSGA-II on the ZDT1 problem.

Requirements:
    - optuna
    - optuna-dashboard (optional, for visualization)

Run:
    python examples/multiobjective/tuning/tune_nsgaii_zdt1.py
"""

import time
from pathlib import Path

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.tuning import (
    TuningProtocol,
    TuningConfig,
    ParameterSpace,
)
from jmetal.util.solution import read_solutions


def load_reference_front(problem_name: str) -> np.ndarray:
    """Load reference front for a problem."""
    fronts_dir = Path(__file__).parent.parent.parent.parent / "resources/reference_fronts"
    csv_path = fronts_dir / f"{problem_name}.csv"
    pf_path = fronts_dir / f"{problem_name}.pf"
    
    # Try CSV first, then .pf format
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",")
    elif pf_path.exists():
        solutions = read_solutions(str(pf_path))
        return np.array([s.objectives for s in solutions])
    else:
        raise FileNotFoundError(f"No reference front found for {problem_name}")


def main():
    # 1. Define the problem(s)
    problems = [ZDT1()]
    
    # 2. Load reference fronts for all problems
    reference_fronts = {
        problem.name(): load_reference_front(problem.name())
        for problem in problems
    }
    
    # 3. Load the parameter space from YAML
    yaml_path = Path(__file__).parent.parent.parent.parent / "src/jmetal/tuning/parameter_spaces/NSGAIIFloat.yaml"
    param_space = ParameterSpace.from_yaml(yaml_path)
    
    # 4. Configure tuning parameters
    config = TuningConfig(
        n_repeats=3,                           # Number of runs per trial
        population_size=100,
        max_evaluations=25000,
        base_seed=42,
        indicator_names=["NHV", "IGD+"],       # Indicators to compute
        reference_point_offset=0.1,            # For hypervolume reference point
        aggregation="sum",                     # How to aggregate indicator values
    )
    
    # 5. Create the tuning protocol
    artifact_dir = Path("tuning_results/nsgaii_zdt1")
    protocol = TuningProtocol(
        algorithm_class=NSGAII,
        parameter_space=param_space,
        problems=problems,
        reference_fronts=reference_fronts,
        config=config,
        artifact_dir=artifact_dir,
        study_name="nsgaii_zdt1_tuning",
    )
    
    # 6. Print tuning info
    print("=" * 60)
    print("NSGA-II Hyperparameter Tuning on ZDT1")
    print("=" * 60)
    print(f"\nParameter space: {yaml_path.name}")
    print(f"Problems: {[p.name() for p in problems]}")
    print(f"Repeats per trial: {config.n_repeats}")
    print(f"Max evaluations: {config.max_evaluations}")
    print(f"Indicators: {list(config.indicator_names)}")
    print()
    
    # 7. Create and run the study
    n_trials = 30
    study = protocol.create_study(storage="sqlite:///tuning_results/tuning.db")
    
    print(f"Starting optimization with {n_trials} trials...")
    print("(This may take a while)")
    print()
    
    start_time = time.perf_counter()
    study.optimize(protocol.objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.perf_counter() - start_time
    
    # 8. Report results
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Total trials: {len(study.trials)}")
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.6f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 9. Save summary
    summary_path = protocol.save_summary(study, elapsed)
    if summary_path:
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
