import time
from typing import Generator, List, TypeVar

try:
    import dask
    from distributed import Client, as_completed
except ImportError:
    pass

from jmetal.util.normalization import normalize_solution_fronts, solutions_to_matrix
import numpy as np


from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import Algorithm, DynamicAlgorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.operator.selection import RandomSelection
from jmetal.core.problem import DynamicProblem, Problem
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, DominanceComparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.operator.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.density_estimator import HypervolumeContributionDensityEstimator


S = TypeVar("S")
R = TypeVar("R")

"""
.. module:: SMSEMOA
   :platform: Unix, Windows
   :synopsis: SMSEMOA (S-Metric Selection Evolutionary Multiobjective Algorithm) implementation.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""

class SMSEMOA(GeneticAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = None,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
    ):
        """
        SMSEMOA implementation (template based on NSGA-II).
        """
        if selection is None:
            selection = RandomSelection()
        super(SMSEMOA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
        )
        self.dominance_comparator = dominance_comparator

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """SMS-EMOA replacement: follows the jMetal logic, using HV contribution only for the last subfront."""
        merged_population = population + offspring_population

        # Compute non-dominated ranking and subfronts
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(merged_population)
        num_subfronts = ranking.get_number_of_subfronts()

        # Collect all subfronts except the last
        result_population: List[S] = []
        for i in range(num_subfronts - 1):
            result_population.extend(ranking.get_subfront(i))

        # Normalize merged_population and last subfront
        last_subfront = ranking.get_subfront(num_subfronts - 1)
        norm_merged, _ = normalize_solution_fronts(merged_population, merged_population, method="reference_only")
        norm_last, _ = normalize_solution_fronts(last_subfront, merged_population, method="reference_only")

        # Compute normalized reference point
        epsilon = 1e-6
        reference_point = np.max(norm_merged, axis=0) + epsilon

        # Compute HV contribution on normalized last subfront
        # Create temporary normalized solutions for HV calculation
        from copy import deepcopy
        norm_solutions = deepcopy(last_subfront)
        for sol, norm_obj in zip(norm_solutions, norm_last):
            sol.objectives = norm_obj.tolist()

        hv_estimator = HypervolumeContributionDensityEstimator(reference_point=reference_point)
        hv_estimator.compute_density_estimator(norm_solutions)
        # Transfer hv_contribution attributes to original solutions
        for orig, norm in zip(last_subfront, norm_solutions):
            orig.attributes["hv_contribution"] = norm.attributes["hv_contribution"]

        # Sort and truncate last subfront by HV contribution
        sorted_last_subfront = sorted(last_subfront, key=lambda s: s.attributes["hv_contribution"], reverse=True)
        result_population.extend(sorted_last_subfront[:len(last_subfront) - 1])

        # Compute non-dominated ranking
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(merged_population)
        num_subfronts = ranking.get_number_of_subfronts()

        # Collect all subfronts except the last
        result_population: List[S] = []
        for i in range(num_subfronts - 1):
            result_population.extend(ranking.get_subfront(i))

        # Truncate the last subfront using HV contribution
        last_subfront = ranking.get_subfront(num_subfronts - 1)
        hv_estimator = HypervolumeContributionDensityEstimator(reference_point=reference_point)
        hv_estimator.compute_density_estimator(last_subfront)
        # Sort by HV contribution (descending: keep largest contributions)
        sorted_last_subfront = sorted(last_subfront, key=lambda s: s.attributes["hv_contribution"], reverse=True)
        # Add all but one from the last subfront
        result_population.extend(sorted_last_subfront[:len(last_subfront) - 1])

        return result_population

    def result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "SMSEMOA"

