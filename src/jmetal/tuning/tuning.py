"""
High-level API for algorithm hyperparameter tuning.

This module provides a simple interface for tuning algorithm hyperparameters
using Optuna. It is designed to be easy to use for non-expert users.

Example:
    from jmetal.tuning import tune
    from jmetal.problem import ZDT1, ZDT2
    
    # Simple usage
    result = tune("NSGAII", problems=[(ZDT1(), "ZDT1.pf"), (ZDT2(), "ZDT2.pf")], n_trials=100)
    print(result.best_params)
    
    # With observers for progress visualization
    from jmetal.tuning.observer import TuningProgressObserver, TuningPlotObserver
    result = tune("NSGAII", observers=[TuningProgressObserver(), TuningPlotObserver()])
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import optuna

from jmetal.core.problem import Problem

from .algorithms import TUNERS, AlgorithmTuner, TuningResult
from .config import (
    CONFIG_PATH,
    TRAINING_PROBLEMS as DEFAULT_TRAINING_PROBLEMS,
    TRAINING_EVALUATIONS,
    N_REPEATS,
    SEED,
    POPULATION_SIZE,
)

if TYPE_CHECKING:
    from .observer import TuningObserver
    from .tuning_config import ParameterSpaceConfig


def _configure_logging(quiet: bool = False) -> None:
    """Configure logging levels for tuning.
    
    Args:
        quiet: If True, suppress most logging output
    """
    if quiet:
        # Suppress jMetal algorithm debug logs
        logging.getLogger("jmetal").setLevel(logging.WARNING)
        # Suppress Optuna info logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        # Default: show info but not debug
        logging.getLogger("jmetal").setLevel(logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.INFO)


def tune(
    algorithm: str = "NSGAII",
    problems: Optional[List[Tuple[Problem, str]]] = None,
    n_trials: int = 100,
    n_evaluations: Optional[int] = None,
    n_repeats: Optional[int] = None,
    sampler: str = "tpe",
    mode: str = "categorical",
    seed: int = SEED,
    population_size: int = POPULATION_SIZE,
    output_path: Optional[str] = None,
    verbose: bool = True,
    observers: Optional[List["TuningObserver"]] = None,
    parameter_space: Optional["ParameterSpaceConfig"] = None,
) -> TuningResult:
    """
    Tune hyperparameters for a multi-objective optimization algorithm.
    
    This is the main entry point for hyperparameter tuning. It provides
    a simple interface that handles all the complexity internally.
    
    Args:
        algorithm: Algorithm name. Currently supported: "NSGAII"
        problems: List of (Problem, reference_front_file) tuples.
            The problem name is obtained from problem.name().
            None to use default ZDT problems.
        n_trials: Number of Optuna trials
        n_evaluations: Max evaluations per problem (default: 10000)
        n_repeats: Independent runs per trial (default: 1)
        sampler: Optuna sampler - "tpe", "cmaes", or "random"
        mode: Parameter space mode - "categorical" or "continuous"
        seed: Random seed for reproducibility
        population_size: Population size for the algorithm
        output_path: Path to save results JSON (optional)
        verbose: Print progress information (disabled if observers provided)
        observers: List of TuningObserver instances for progress visualization.
            When provided, verbose output is disabled in favor of observer output.
        parameter_space: Custom parameter space configuration from TuningConfig.
            When provided, limits the hyperparameter search to specified ranges.
        
    Returns:
        TuningResult with best parameters and metadata
        
    Example:
        # Basic usage with default problems
        result = tune("NSGAII", n_trials=50)
        
        # Custom problems with reference front files
        result = tune("NSGAII", problems=[
            (ZDT1(), "ZDT1.pf"),
            (ZDT2(), "ZDT2.pf"),
        ], n_trials=100)
        
        # Using CMA-ES sampler
        result = tune("NSGAII", sampler="cmaes", mode="continuous", n_trials=50)
        
        # With progress observers
        from jmetal.tuning.observer import TuningProgressObserver, TuningPlotObserver
        result = tune("NSGAII", observers=[
            TuningProgressObserver(),
            TuningPlotObserver(),
        ])
        
        # With custom parameter space from YAML config
        from jmetal.tuning import TuningConfig
        config = TuningConfig.from_yaml("my_config.yaml")
        result = tune("NSGAII", parameter_space=config.parameter_space)
    """
    # Get tuner class
    if algorithm not in TUNERS:
        available = ", ".join(TUNERS.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    tuner_class = TUNERS[algorithm]
    tuner = tuner_class(population_size=population_size, parameter_space=parameter_space)
    
    # Prepare problems
    if problems is None:
        training_problems = DEFAULT_TRAINING_PROBLEMS
    else:
        training_problems = problems
    
    # Set defaults
    if n_evaluations is None:
        n_evaluations = TRAINING_EVALUATIONS
    if n_repeats is None:
        n_repeats = N_REPEATS
    
    # Determine output mode: observers take precedence over verbose
    use_observers = observers is not None and len(observers) > 0
    show_verbose = verbose and not use_observers
    
    # Configure logging: quiet when using observers
    _configure_logging(quiet=use_observers)
    
    # Validate sampler/mode
    if sampler == "cmaes" and mode != "continuous":
        if show_verbose:
            print("Warning: CMA-ES requires continuous mode. Switching.")
        mode = "continuous"
    
    # Extract problem names using problem.name()
    problem_names = [problem.name() for problem, _ in training_problems]
    
    if show_verbose:
        print("=" * 60)
        print(f"Hyperparameter Tuning: {algorithm}")
        print("=" * 60)
        print(f"Trials: {n_trials}")
        print(f"Sampler: {sampler}")
        print(f"Mode: {mode}")
        print(f"Problems: {problem_names}")
        print(f"Evaluations per problem: {n_evaluations}")
        print(f"Repeats per trial: {n_repeats}")
        print("=" * 60)
    
    # Notify observers of tuning start
    if use_observers and observers is not None:
        for observer in observers:
            observer.on_tuning_start(n_trials, algorithm)
    
    # Create Optuna sampler
    optuna_sampler = _create_sampler(sampler, seed)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna_sampler,
        study_name=f"{algorithm.lower()}_tuning",
    )
    
    # Create objective function
    def objective(trial) -> float:
        params = tuner.sample_parameters(trial, mode=mode)
        
        if show_verbose:
            print(f"Trial {trial.number}: {tuner.format_params(params)}")
        
        score = tuner.evaluate_on_problems(
            problems=training_problems,
            params=params,
            max_evaluations=n_evaluations,
            n_repeats=n_repeats,
        )
        
        return score
    
    # Prepare callbacks for Optuna (observers act as callbacks)
    optuna_callbacks = observers if use_observers and observers is not None else None
    
    # Run optimization
    start_time = time.perf_counter()
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=show_verbose,
        callbacks=optuna_callbacks,
    )
    elapsed = time.perf_counter() - start_time
    
    # Notify observers of tuning end
    if use_observers and observers is not None:
        for observer in observers:
            observer.on_tuning_end(study)
    
    # Create result
    result = TuningResult(
        algorithm_name=algorithm,
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=n_trials,
        training_problems=problem_names,
        training_evaluations=n_evaluations,
        elapsed_seconds=elapsed,
        extra={
            "population_size": population_size,
            "n_repeats": n_repeats,
            "sampler": sampler,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        }
    )
    
    # Show results only if verbose and no observers (observers handle their own output)
    if show_verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Best score: {result.best_score:.6f}")
        print(f"Best parameters:")
        for key, value in result.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Save results
    if output_path is not None:
        _save_result(result, output_path)
        if show_verbose:
            print(f"\nResults saved to: {output_path}")
    
    return result


def _create_sampler(sampler_name: str, seed: int):
    """Create Optuna sampler."""
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def _save_result(result: TuningResult, path: str):
    """Save tuning result to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def describe_parameters(
    algorithm: str = "NSGAII",
    format: str = "txt",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Get a human-readable description of tunable parameters for an algorithm.
    
    This function is designed for users who want to understand what 
    hyperparameters can be tuned and what they mean, without reading code.
    
    Args:
        algorithm: Algorithm name (e.g., "NSGAII")
        format: Output format - "txt" (readable text), "json", or "yaml"
        output_path: If provided, save to file. Otherwise return as string.
        
    Returns:
        Parameter description as string (if output_path is None)
        
    Example:
        # Print to console
        print(describe_parameters("NSGAII"))
        
        # Save to file
        describe_parameters("NSGAII", format="json", output_path="nsgaii_params.json")
        describe_parameters("NSGAII", format="yaml", output_path="nsgaii_params.yaml")
    """
    if algorithm not in TUNERS:
        available = ", ".join(TUNERS.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    tuner = TUNERS[algorithm]()
    
    if output_path is not None:
        tuner.export_parameter_space(Path(output_path), format=format)
        return None
    
    return tuner.export_parameter_space(format=format)


def list_algorithms() -> List[str]:
    """
    Get list of supported algorithms for tuning.
    
    Returns:
        List of algorithm names
        
    Example:
        >>> list_algorithms()
        ['NSGAII']
    """
    return list(TUNERS.keys())
