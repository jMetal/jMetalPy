import time
from typing import Generator, List, TypeVar

try:
    import dask
    from distributed import Client, as_completed
except ImportError:
    pass

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

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistanceDensityEstimator()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "SMSEMOA"

