"""
Objective function for NSGA-II hyperparameter tuning.

This module contains the logic to evaluate a hyperparameter configuration
by running NSGA-II on a set of training problems and computing quality indicators.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import copy

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.operator.crossover import SBXCrossover, BLXAlphaCrossover
from jmetal.operator.mutation import PolynomialMutation, UniformMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.core.quality_indicator import NormalizedHyperVolume, AdditiveEpsilonIndicator

from parameter_space import NSGAIIParameters
from config import (
    POPULATION_SIZE,
    TRAINING_EVALUATIONS,
    N_REPEATS,
    REFERENCE_POINT_OFFSET,
    get_reference_front_path,
)


def load_reference_front(filename: str) -> np.ndarray:
    """Load and return a reference front as numpy array.
    
    Args:
        filename: Full filename including extension (e.g., 'ZDT1.pf')
    """
    ref_path = get_reference_front_path(filename)
    solutions = read_solutions(str(ref_path))
    return np.array([s.objectives for s in solutions])


def build_crossover_operator(params: NSGAIIParameters):
    """Create the crossover operator from parameters."""
    if params.crossover_type == "sbx":
        return SBXCrossover(
            probability=params.crossover_probability,
            distribution_index=params.crossover_eta or 20.0,
        )
    else:
        return BLXAlphaCrossover(
            probability=params.crossover_probability,
            alpha=params.blx_alpha or 0.5,
        )


def build_mutation_operator(params: NSGAIIParameters, n_variables: int):
    """
    Create the mutation operator from parameters.
    
    The actual mutation probability is computed as:
        probability = mutation_probability_factor * (1 / n_variables)
    
    This adapts the mutation rate to the problem dimensionality.
    """
    effective_probability = params.mutation_probability_factor / n_variables
    # Clamp to [0, 1]
    effective_probability = max(0.0, min(1.0, effective_probability))
    
    if params.mutation_type == "polynomial":
        return PolynomialMutation(
            probability=effective_probability,
            distribution_index=params.mutation_eta or 20.0,
        )
    else:
        return UniformMutation(
            probability=effective_probability,
            perturbation=params.mutation_perturbation or 0.5,
        )


def run_nsgaii_single(
    problem: Problem,
    params: NSGAIIParameters,
    max_evaluations: int,
) -> List:
    """
    Run NSGA-II once on a problem and return the Pareto front.
    
    Args:
        problem: The optimization problem
        params: NSGA-II hyperparameters
        max_evaluations: Maximum number of function evaluations
        
    Returns:
        List of non-dominated solutions (Pareto front approximation)
    """
    # Create operators
    crossover = build_crossover_operator(params)
    mutation = build_mutation_operator(params, problem.number_of_variables())
    
    # Create and run algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=POPULATION_SIZE,
        offspring_population_size=params.offspring_population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    algorithm.run()
    
    # Return non-dominated solutions
    return get_non_dominated_solutions(algorithm.result())


def compute_indicators(
    front: List,
    reference_front: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute quality indicators for a Pareto front approximation.
    
    Args:
        front: List of non-dominated solutions
        reference_front: Reference Pareto front as numpy array
        
    Returns:
        Tuple of (normalized_hypervolume, additive_epsilon)
    """
    # Extract objective values as numpy array
    objectives = np.array([s.objectives for s in front])
    
    # Create indicators
    nhv_indicator = NormalizedHyperVolume(
        reference_front=reference_front,
        reference_point_offset=REFERENCE_POINT_OFFSET,
    )
    nhv_indicator.set_reference_front(reference_front)
    
    ae_indicator = AdditiveEpsilonIndicator(reference_front)
    
    # Compute indicators
    nhv_value = float(nhv_indicator.compute(objectives))
    ae_value = float(ae_indicator.compute(objectives))
    
    return nhv_value, ae_value


def evaluate_on_problem(
    problem: Problem,
    reference_front_file: str,
    params: NSGAIIParameters,
    max_evaluations: int,
    n_repeats: int = 1,
) -> Tuple[float, float]:
    """
    Evaluate configuration on a single problem with optional repetitions.
    
    Args:
        problem: The optimization problem
        reference_front_file: Reference front filename with extension (e.g., 'ZDT1.pf')
        params: NSGA-II hyperparameters
        max_evaluations: Maximum evaluations per run
        n_repeats: Number of independent runs
        
    Returns:
        Mean (normalized_hypervolume, additive_epsilon) across repeats
    """
    reference_front = load_reference_front(reference_front_file)
    
    nhv_values = []
    ae_values = []
    
    for _ in range(n_repeats):
        # Run NSGA-II
        front = run_nsgaii_single(copy.deepcopy(problem), params, max_evaluations)
        
        # Compute indicators
        nhv, ae = compute_indicators(front, reference_front)
        nhv_values.append(nhv)
        ae_values.append(ae)
    
    # Return mean values
    mean_nhv = float(np.mean(nhv_values))
    mean_ae = float(np.mean(ae_values))
    
    return mean_nhv, mean_ae


def evaluate_configuration(
    params: NSGAIIParameters,
    training_problems: List[Tuple[Problem, str]],
    max_evaluations: Optional[int] = None,
    n_repeats: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate a hyperparameter configuration on a set of training problems.
    
    The objective score is computed as:
        score = sum(NHV + AE) for each problem
    
    Note: NHV from NormalizedHyperVolume is already formulated so that
    lower is better. We sum both indicators (both lower = better).
    
    Args:
        params: NSGA-II hyperparameters
        training_problems: List of (problem, reference_front_file) tuples
        max_evaluations: Max evaluations per problem (default from config)
        n_repeats: Number of repetitions per problem (default from config)
        verbose: Print progress information
        
    Returns:
        Objective score to minimize
    """
    if max_evaluations is None:
        max_evaluations = TRAINING_EVALUATIONS
    if n_repeats is None:
        n_repeats = N_REPEATS
    
    problem_scores = []
    
    for problem, ref_front_file in training_problems:
        # Evaluate on this problem
        mean_nhv, mean_ae = evaluate_on_problem(
            problem=copy.deepcopy(problem),
            reference_front_file=ref_front_file,
            params=params,
            max_evaluations=max_evaluations,
            n_repeats=n_repeats,
        )
        
        # Both indicators: lower is better
        problem_score = mean_nhv + mean_ae
        problem_scores.append(problem_score)
        
        if verbose:
            print(f"  {problem.name()}: NHV={mean_nhv:.4f}, AE={mean_ae:.4f}, score={problem_score:.4f}")
    
    # Return mean across problems (consistent with FINAL_AGG="mean" in original)
    total_score = float(np.mean(problem_scores))
    
    if verbose:
        print(f"  Total score: {total_score:.4f}")
    
    return total_score


def create_objective_function(
    training_problems: List[Tuple[Problem, str]],
    sample_fn: Callable,
    max_evaluations: Optional[int] = None,
    n_repeats: Optional[int] = None,
):
    """
    Create an Optuna objective function.
    
    Args:
        training_problems: List of (problem, reference_front_file) tuples
        sample_fn: Function to sample parameters from a trial
        max_evaluations: Max evaluations per problem
        n_repeats: Number of repetitions per problem
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial) -> float:
        # Sample hyperparameters
        params = sample_fn(trial)
        
        # Evaluate configuration
        score = evaluate_configuration(
            params=params,
            training_problems=training_problems,
            max_evaluations=max_evaluations,
            n_repeats=n_repeats,
            verbose=False,
        )
        
        return score
    
    return objective
