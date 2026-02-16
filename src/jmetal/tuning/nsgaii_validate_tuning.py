"""
Validation script for NSGA-II tuned configuration.

This script:
1. Loads the best hyperparameters from the tuning process
2. Runs NSGA-II on all training problems with the tuned configuration
3. Saves the Pareto fronts to files
4. Generates comparison plots (obtained front vs reference front)
5. Computes quality indicators for each problem
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator
from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "nsgaii_tuned_config.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "validation_results"

# Problems to validate (same as training set)
VALIDATION_PROBLEMS = [
    (ZDT1(), "ZDT1"),
    (ZDT2(), "ZDT2"),
    (ZDT3(), "ZDT3"),
    (ZDT4(), "ZDT4"),
    (ZDT6(), "ZDT6"),
]

# Use more evaluations for validation (as specified in config)
VALIDATION_EVALUATIONS = 20000


def load_config():
    """Load the tuned configuration."""
    with CONFIG_PATH.open("r") as f:
        return json.load(f)


def load_reference_front(problem_name: str) -> np.ndarray:
    """Load reference front for a problem."""
    ref_path = ROOT_DIR / f"resources/reference_fronts/{problem_name}.pf"
    solutions = read_solutions(str(ref_path))
    return np.array([s.objectives for s in solutions])


def create_algorithm(problem, config: dict):
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
    mutation_type = params.get("mutation_type", "polynomial")  # Default for backward compatibility
    
    # Calculate effective mutation probability: factor * (1/n)
    # Support both old format (mutation_probability) and new format (mutation_probability_factor)
    if "mutation_probability_factor" in params:
        n_variables = problem.number_of_variables()
        effective_mutation_prob = min(1.0, params["mutation_probability_factor"] / n_variables)
    else:
        # Backward compatibility with old configs
        effective_mutation_prob = params["mutation_probability"]
    
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
    
    # Create algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=params["offspring_population_size"],
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=VALIDATION_EVALUATIONS),
    )
    
    return algorithm


def save_front(front: np.ndarray, problem_name: str, output_dir: Path):
    """Save Pareto front to file."""
    fun_path = output_dir / f"FUN.NSGAII_TUNED.{problem_name}"
    np.savetxt(fun_path, front, fmt="%.10e", delimiter=" ")
    print(f"  Front saved to: {fun_path}")


def plot_fronts(obtained_front: np.ndarray, reference_front: np.ndarray, 
                problem_name: str, indicators: dict, output_dir: Path):
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


def plot_all_fronts(all_results: dict, output_dir: Path):
    """Generate a combined plot with all problems."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
    
    # Hide the 6th subplot (only 5 problems)
    axes[5].axis('off')
    
    plt.suptitle('NSGA-II Tuned Configuration - Validation Results', fontsize=14)
    plt.tight_layout()
    
    combined_path = output_dir / "all_fronts_comparison.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCombined plot saved to: {combined_path}")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config()
    print("=" * 60)
    print("NSGA-II Tuned Configuration Validation")
    print("=" * 60)
    print(f"\nConfiguration loaded from: {CONFIG_PATH}")
    print(f"Validation evaluations: {VALIDATION_EVALUATIONS}")
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
    
    for problem, problem_name in VALIDATION_PROBLEMS:
        print(f"\n{'─' * 40}")
        print(f"Validating on {problem_name}")
        print(f"{'─' * 40}")
        
        # Load reference front
        reference_front = load_reference_front(problem_name)
        print(f"  Reference front: {len(reference_front)} solutions")
        
        # Create and run algorithm
        algorithm = create_algorithm(problem, config)
        print(f"  Running NSGA-II ({VALIDATION_EVALUATIONS} evaluations)...")
        algorithm.run()
        
        # Get non-dominated solutions
        solutions = get_non_dominated_solutions(algorithm.result())
        obtained_front = np.array([s.objectives for s in solutions])
        print(f"  Obtained front: {len(obtained_front)} solutions")
        
        # Compute indicators
        nhv_indicator = NormalizedHyperVolume(
            reference_front=reference_front,
            reference_point_offset=config["ref_point_offset"]
        )
        nhv_indicator.set_reference_front(reference_front)
        
        epsilon_indicator = AdditiveEpsilonIndicator(reference_front)
        
        nhv_value = float(nhv_indicator.compute(obtained_front))
        epsilon_value = float(epsilon_indicator.compute(obtained_front))
        
        indicators = {"nhv": nhv_value, "epsilon": epsilon_value}
        print(f"  Normalized HV: {nhv_value:.6f}")
        print(f"  Additive ε+: {epsilon_value:.6f}")
        
        # Save front
        save_front(obtained_front, problem_name, OUTPUT_DIR)
        
        # Generate individual plot
        plot_fronts(obtained_front, reference_front, problem_name, indicators, OUTPUT_DIR)
        
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
    plot_all_fronts(all_results, OUTPUT_DIR)
    
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
    print(f"(Training best was: {config['best_value']:.6f})")
    
    # Save summary to JSON
    summary_path = OUTPUT_DIR / "validation_summary.json"
    summary = {
        "config": config,
        "validation_evaluations": VALIDATION_EVALUATIONS,
        "results": summary_data,
        "mean_nhv": mean_nhv,
        "mean_epsilon": mean_eps,
        "composite_score": mean_nhv + mean_eps,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
