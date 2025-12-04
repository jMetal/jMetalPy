#!/usr/bin/env python
"""
Validation script for tuned NSGA-II configuration.

This script:
1. Loads the best hyperparameters from a tuning JSON file
2. Runs NSGA-II on validation problems with the tuned configuration
3. Saves the Pareto fronts to files
4. Generates comparison plots (obtained front vs reference front)
5. Computes quality indicators for each problem

Usage:
    python examples/tuning/validate_tuning.py
    python examples/tuning/validate_tuning.py --config ./nsgaii_tuned_config.json
    python examples/tuning/validate_tuning.py --evaluations 25000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator
from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.operator.selection import RandomSelection, TournamentSelection
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.tuning.config import REFERENCE_FRONTS_DIR, VALIDATION_EVALUATIONS


# Default validation problems (same as training set)
DEFAULT_VALIDATION_PROBLEMS: List[Tuple[Problem, str]] = [
    (ZDT1(), "ZDT1"),
    (ZDT2(), "ZDT2"),
    (ZDT3(), "ZDT3"),
    (ZDT4(), "ZDT4"),
    (ZDT6(), "ZDT6"),
]


def load_config(config_path: str) -> dict:
    """Load the tuned configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_reference_front(problem_name: str) -> np.ndarray:
    """Load reference front for a problem."""
    ref_path = REFERENCE_FRONTS_DIR / f"{problem_name}.pf"
    solutions = read_solutions(str(ref_path))
    return np.array([s.objectives for s in solutions])


def create_algorithm(problem: Problem, config: dict, max_evaluations: int) -> NSGAII:
    """Create NSGA-II with tuned parameters."""
    params = config["best_params"]
    population_size = config["population_size"]
    
    # Build crossover operator
    crossover_type = params["crossover_type"]
    crossover_prob = params["crossover_probability"]
    
    if crossover_type == "sbx":
        crossover = SBXCrossover(
            probability=crossover_prob,
            distribution_index=params["crossover_eta"]
        )
    else:
        crossover = BLXAlphaCrossover(
            probability=crossover_prob,
            alpha=params.get("blx_alpha", 0.5)
        )
    
    # Build mutation operator
    mutation_type = params.get("mutation_type", "polynomial")
    
    # Calculate effective mutation probability: factor * (1/n)
    if "mutation_probability_factor" in params:
        n_variables = problem.number_of_variables()
        effective_mutation_prob = min(1.0, params["mutation_probability_factor"] / n_variables)
    else:
        # Backward compatibility with old configs
        effective_mutation_prob = params.get("mutation_probability", 1.0 / problem.number_of_variables())
    
    if mutation_type == "polynomial":
        mutation = PolynomialMutation(
            probability=effective_mutation_prob,
            distribution_index=params["mutation_eta"]
        )
    else:  # uniform
        mutation = UniformMutation(
            probability=effective_mutation_prob,
            perturbation=params["mutation_perturbation"]
        )
    
    # Build selection operator
    selection_type = params.get("selection_type", "tournament")
    
    if selection_type == "random":
        selection = RandomSelection()
    else:  # tournament
        tournament_size = params.get("tournament_size", 2)
        selection = TournamentSelection(tournament_size=tournament_size)
    
    # Create algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=params["offspring_population_size"],
        mutation=mutation,
        crossover=crossover,
        selection=selection,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    return algorithm


def save_front(front: np.ndarray, problem_name: str, output_dir: Path) -> None:
    """Save Pareto front to file."""
    fun_path = output_dir / f"FUN.NSGAII_TUNED.{problem_name}"
    np.savetxt(fun_path, front, fmt="%.10e", delimiter=" ")
    print(f"  Front saved to: {fun_path}")


def plot_fronts(obtained_front: np.ndarray, reference_front: np.ndarray, 
                problem_name: str, indicators: dict, output_dir: Path) -> None:
    """Generate comparison plot of obtained vs reference front."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot reference front
    ax.scatter(reference_front[:, 0], reference_front[:, 1], 
               c='blue', s=10, alpha=0.5, label='Reference Front')
    
    # Plot obtained front
    ax.scatter(obtained_front[:, 0], obtained_front[:, 1], 
               c='red', s=30, alpha=0.8, label='NSGA-II Tuned', marker='x')
    
    ax.set_xlabel('$f_1$', fontsize=12)
    ax.set_ylabel('$f_2$', fontsize=12)
    ax.set_title(f'NSGA-II Tuned on {problem_name}\n'
                 f'NHV: {indicators["nhv"]:.6f}, ε+: {indicators["epsilon"]:.6f}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"plot_{problem_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path}")


def plot_all_fronts(all_results: Dict, output_dir: Path) -> None:
    """Generate a combined plot with all problems."""
    n_problems = len(all_results)
    cols = min(3, n_problems)
    rows = (n_problems + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_problems == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (problem_name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        
        ref_front = data["reference_front"]
        obt_front = data["obtained_front"]
        indicators = data["indicators"]
        
        ax.scatter(ref_front[:, 0], ref_front[:, 1], 
                   c='blue', s=8, alpha=0.4, label='Reference')
        ax.scatter(obt_front[:, 0], obt_front[:, 1], 
                   c='red', s=20, alpha=0.8, label='Obtained', marker='x')
        
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title(f'{problem_name}\nNHV={indicators["nhv"]:.4f}, ε+={indicators["epsilon"]:.4f}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_problems, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('NSGA-II Tuned Configuration - Validation Results', fontsize=14)
    plt.tight_layout()
    
    combined_path = output_dir / "all_fronts_comparison.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCombined plot saved to: {combined_path}")


def validate(
    config_path: str,
    output_dir: str = "./validation_results",
    max_evaluations: int = VALIDATION_EVALUATIONS,
    problems: Optional[List[Tuple[Problem, str]]] = None,
) -> dict:
    """
    Validate a tuned configuration on a set of problems.
    
    Args:
        config_path: Path to the tuned configuration JSON file
        output_dir: Directory to save validation results
        max_evaluations: Maximum evaluations per problem
        problems: List of (Problem, name) tuples. If None, use default ZDT problems.
        
    Returns:
        Dictionary with validation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if problems is None:
        problems = DEFAULT_VALIDATION_PROBLEMS
    
    # Load configuration
    config = load_config(config_path)
    print("=" * 60)
    print("NSGA-II Tuned Configuration Validation")
    print("=" * 60)
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Validation evaluations: {max_evaluations}")
    print(f"\nTuned Parameters:")
    for key, value in config["best_params"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(f"  population_size: {config['population_size']}")
    print()
    
    all_results = {}
    summary_data = []
    ref_point_offset = config.get("ref_point_offset", 0.1)
    
    for problem, problem_name in problems:
        print(f"\n{'─' * 40}")
        print(f"Validating on {problem_name}")
        print(f"{'─' * 40}")
        
        # Load reference front
        reference_front = load_reference_front(problem_name)
        print(f"  Reference front: {len(reference_front)} solutions")
        
        # Create and run algorithm
        algorithm = create_algorithm(problem, config, max_evaluations)
        print(f"  Running NSGA-II ({max_evaluations} evaluations)...")
        algorithm.run()
        
        # Get non-dominated solutions
        solutions = get_non_dominated_solutions(algorithm.result())
        obtained_front = np.array([s.objectives for s in solutions])
        print(f"  Obtained front: {len(obtained_front)} solutions")
        
        # Compute indicators
        nhv_indicator = NormalizedHyperVolume(
            reference_front=reference_front,
            reference_point_offset=ref_point_offset
        )
        nhv_indicator.set_reference_front(reference_front)
        
        epsilon_indicator = AdditiveEpsilonIndicator(reference_front)
        
        nhv_value = float(nhv_indicator.compute(obtained_front))
        epsilon_value = float(epsilon_indicator.compute(obtained_front))
        
        indicators = {"nhv": nhv_value, "epsilon": epsilon_value}
        print(f"  Normalized HV: {nhv_value:.6f}")
        print(f"  Additive ε+: {epsilon_value:.6f}")
        
        # Save front
        save_front(obtained_front, problem_name, output_path)
        
        # Generate individual plot
        plot_fronts(obtained_front, reference_front, problem_name, indicators, output_path)
        
        # Store results
        all_results[problem_name] = {
            "obtained_front": obtained_front,
            "reference_front": reference_front,
            "indicators": indicators,
        }
        
        summary_data.append({
            "problem": problem_name,
            "solutions": len(obtained_front),
            "nhv": nhv_value,
            "epsilon": epsilon_value,
        })
    
    # Generate combined plot
    plot_all_fronts(all_results, output_path)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Problem':<10} {'Solutions':>10} {'NHV':>12} {'ε+':>12}")
    print("-" * 46)
    
    total_nhv = 0
    total_eps = 0
    for row in summary_data:
        print(f"{row['problem']:<10} {row['solutions']:>10} {row['nhv']:>12.6f} {row['epsilon']:>12.6f}")
        total_nhv += row['nhv']
        total_eps += row['epsilon']
    
    print("-" * 46)
    mean_nhv = total_nhv / len(summary_data)
    mean_eps = total_eps / len(summary_data)
    print(f"{'Mean':<10} {'':<10} {mean_nhv:>12.6f} {mean_eps:>12.6f}")
    print(f"\nComposite score (NHV + ε+): {mean_nhv + mean_eps:.6f}")
    
    if "best_value" in config:
        print(f"(Training best was: {config['best_value']:.6f})")
    
    # Save summary to JSON
    summary_path = output_path / "validation_summary.json"
    summary = {
        "config_path": str(config_path),
        "config": config,
        "validation_evaluations": max_evaluations,
        "results": summary_data,
        "mean_nhv": mean_nhv,
        "mean_epsilon": mean_eps,
        "composite_score": mean_nhv + mean_eps,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nAll results saved to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate tuned NSGA-II configuration"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./nsgaii_tuned_config.json",
        help="Path to tuned configuration JSON (default: ./nsgaii_tuned_config.json)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./validation_results",
        help="Output directory for validation results (default: ./validation_results)"
    )
    parser.add_argument(
        "--evaluations", "-e",
        type=int,
        default=VALIDATION_EVALUATIONS,
        help=f"Maximum evaluations per problem (default: {VALIDATION_EVALUATIONS})"
    )
    args = parser.parse_args()
    
    validate(
        config_path=args.config,
        output_dir=args.output_dir,
        max_evaluations=args.evaluations,
    )


if __name__ == "__main__":
    main()
