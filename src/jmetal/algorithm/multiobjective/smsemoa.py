from typing import Generator, List, TypeVar

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RandomSelection
from jmetal.util.comparator import Comparator
from jmetal.util.density_estimator import HypervolumeContributionDensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion

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

    from typing import List

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """
        SMS-EMOA replacement strategy.

        Implements replacement according to SMS-EMOA algorithm:
        1. Merge current population with offspring
        2. Compute non-dominated ranking
        3. Fill new population by fronts
        4. In the last front, remove solution with smallest HV contribution

        Args:
            population: Current population
            offspring_population: Offspring population (typically 1 solution)

        Returns:
            New population of size self.population_size
        """
        # Merge populations
        merged_population = population + offspring_population

        # Compute non-dominated ranking
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(merged_population)

        num_subfronts = ranking.get_number_of_subfronts()

        # Collect all subfronts except the last
        result_population: List[S] = []
        for i in range(num_subfronts - 1):
            result_population.extend(ranking.get_subfront(i))

        # Get the last subfront
        last_subfront = ranking.get_subfront(num_subfronts - 1)

        # If the entire last subfront fits, add it and return
        if len(result_population) + len(last_subfront) <= self.population_size:
            result_population.extend(last_subfront)
            return result_population

        # Otherwise, we need to truncate the last subfront using HV contribution
        # Calculate reference point in objective space (not normalized)
        # Use worst values from merged population
        objectives_array = np.array([s.objectives for s in merged_population])

        # Reference point should be worse than all solutions
        # For minimization: use maximum values + offset
        offset = 1.0  # Offset to ensure reference point is dominated by all solutions
        reference_point = np.max(objectives_array, axis=0) + offset

        # Compute HV contribution directly on last subfront (without normalization)
        hv_estimator = HypervolumeContributionDensityEstimator(
            reference_point=reference_point.tolist()
        )
        hv_estimator.compute_density_estimator(last_subfront)

        # Sort by HV contribution (descending: largest contributions first)
        sorted_last_subfront = sorted(
            last_subfront,
            key=lambda s: s.attributes["hv_contribution"],
            reverse=True
        )

        # Add all but one from the last subfront (remove the worst)
        result_population.extend(sorted_last_subfront[:len(last_subfront) - 1])

        return result_population

    def result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "SMSEMOA"

