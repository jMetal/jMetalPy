"""
Parameter space definition for NSGA-II hyperparameter tuning.

This module defines how hyperparameters are sampled during optimization.
Two modes are supported:
- "categorical": Uses categorical variables (compatible with TPE sampler)
- "continuous": Converts to continuous variables (compatible with CMA-ES sampler)
"""

from dataclasses import dataclass
from typing import Optional

import optuna


@dataclass
class NSGAIIParameters:
    """Container for NSGA-II hyperparameters."""
    offspring_population_size: int
    crossover_type: str  # "sbx" or "blxalpha"
    crossover_probability: float
    crossover_eta: Optional[float] = None  # For SBX
    blx_alpha: Optional[float] = None  # For BLX-alpha
    mutation_type: str = "polynomial"  # "polynomial" or "uniform"
    mutation_probability_factor: float = 1.0  # Actual prob = factor * (1/n)
    mutation_eta: Optional[float] = None  # For polynomial mutation
    mutation_perturbation: Optional[float] = None  # For uniform mutation


def sample_parameters_categorical(trial: optuna.Trial) -> NSGAIIParameters:
    """
    Sample hyperparameters using categorical variables.
    
    This mode is compatible with TPE sampler (default) and other samplers
    that handle categorical variables well.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        NSGAIIParameters with sampled values
    """
    # Offspring population size
    offspring_population_size = trial.suggest_categorical(
        "offspring_population_size", [1, 10, 50, 100, 150, 200]
    )
    
    # Crossover operator
    crossover_type = trial.suggest_categorical("crossover_type", ["sbx", "blxalpha"])
    crossover_probability = trial.suggest_float("crossover_probability", 0.7, 1.0)
    
    crossover_eta = None
    blx_alpha = None
    if crossover_type == "sbx":
        crossover_eta = trial.suggest_float("crossover_eta", 5.0, 400.0)
    else:
        blx_alpha = trial.suggest_float("blx_alpha", 0.0, 1.0)
    
    # Mutation operator
    mutation_type = trial.suggest_categorical("mutation_type", ["polynomial", "uniform"])
    mutation_probability_factor = trial.suggest_float("mutation_probability_factor", 0.5, 2.0)
    
    mutation_eta = None
    mutation_perturbation = None
    if mutation_type == "polynomial":
        mutation_eta = trial.suggest_float("mutation_eta", 5.0, 400.0)
    else:
        mutation_perturbation = trial.suggest_float("mutation_perturbation", 0.1, 2.0)
    
    return NSGAIIParameters(
        offspring_population_size=offspring_population_size,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        crossover_eta=crossover_eta,
        blx_alpha=blx_alpha,
        mutation_type=mutation_type,
        mutation_probability_factor=mutation_probability_factor,
        mutation_eta=mutation_eta,
        mutation_perturbation=mutation_perturbation,
    )


def sample_parameters_continuous(trial: optuna.Trial) -> NSGAIIParameters:
    """
    Sample hyperparameters using only continuous/integer variables.
    
    This mode converts categorical variables to numeric ones, making it
    compatible with samplers like CMA-ES that require continuous spaces.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        NSGAIIParameters with sampled values
    """
    # Offspring population size as integer (log scale favors smaller values)
    offspring_population_size = trial.suggest_int(
        "offspring_population_size", 1, 200, log=True
    )
    
    # Crossover type as float threshold
    crossover_type_idx = trial.suggest_float("crossover_type_idx", 0.0, 1.0)
    crossover_type = "sbx" if crossover_type_idx < 0.5 else "blxalpha"
    
    crossover_probability = trial.suggest_float("crossover_probability", 0.7, 1.0)
    
    # Sample both crossover parameters; only one will be used
    crossover_eta = trial.suggest_float("crossover_eta", 5.0, 400.0)
    blx_alpha = trial.suggest_float("blx_alpha", 0.0, 1.0)
    
    # Mutation type as float threshold
    mutation_type_idx = trial.suggest_float("mutation_type_idx", 0.0, 1.0)
    mutation_type = "polynomial" if mutation_type_idx < 0.5 else "uniform"
    
    mutation_probability_factor = trial.suggest_float("mutation_probability_factor", 0.5, 2.0)
    
    # Sample both mutation parameters; only one will be used
    mutation_eta = trial.suggest_float("mutation_eta", 5.0, 400.0)
    mutation_perturbation = trial.suggest_float("mutation_perturbation", 0.1, 2.0)
    
    return NSGAIIParameters(
        offspring_population_size=offspring_population_size,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        crossover_eta=crossover_eta if crossover_type == "sbx" else None,
        blx_alpha=blx_alpha if crossover_type == "blxalpha" else None,
        mutation_type=mutation_type,
        mutation_probability_factor=mutation_probability_factor,
        mutation_eta=mutation_eta if mutation_type == "polynomial" else None,
        mutation_perturbation=mutation_perturbation if mutation_type == "uniform" else None,
    )


def sample_parameters(trial: optuna.Trial, mode: str = "categorical") -> NSGAIIParameters:
    """
    Sample hyperparameters using the specified mode.
    
    Args:
        trial: Optuna trial object
        mode: "categorical" (for TPE) or "continuous" (for CMA-ES)
        
    Returns:
        NSGAIIParameters with sampled values
    """
    if mode == "categorical":
        return sample_parameters_categorical(trial)
    elif mode == "continuous":
        return sample_parameters_continuous(trial)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'categorical' or 'continuous'.")


def params_to_dict(params: NSGAIIParameters) -> dict:
    """Convert NSGAIIParameters to a dictionary for JSON serialization."""
    d = {
        "offspring_population_size": params.offspring_population_size,
        "crossover_type": params.crossover_type,
        "crossover_probability": params.crossover_probability,
        "mutation_type": params.mutation_type,
        "mutation_probability_factor": params.mutation_probability_factor,
    }
    
    if params.crossover_type == "sbx":
        d["crossover_eta"] = params.crossover_eta
    else:
        d["blx_alpha"] = params.blx_alpha
    
    if params.mutation_type == "polynomial":
        d["mutation_eta"] = params.mutation_eta
    else:
        d["mutation_perturbation"] = params.mutation_perturbation
    
    return d


def format_params_description(params: NSGAIIParameters) -> str:
    """Format parameters as a human-readable string for logging."""
    if params.crossover_type == "sbx":
        crossover_desc = f"SBX prob={params.crossover_probability:.4f}, eta={params.crossover_eta:.2f}"
    else:
        crossover_desc = f"BLX prob={params.crossover_probability:.4f}, alpha={params.blx_alpha:.2f}"
    
    if params.mutation_type == "polynomial":
        mutation_desc = f"Polynomial factor={params.mutation_probability_factor:.2f}, eta={params.mutation_eta:.2f}"
    else:
        mutation_desc = f"Uniform factor={params.mutation_probability_factor:.2f}, pert={params.mutation_perturbation:.2f}"
    
    return f"offspring={params.offspring_population_size}, {crossover_desc}, {mutation_desc}"
