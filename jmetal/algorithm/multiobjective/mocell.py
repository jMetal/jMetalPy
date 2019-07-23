import time
from typing import TypeVar, List

import dask
from distributed import as_completed, Client

from jmetal.util.archive import BoundedArchive
from jmetal.util.neighborhood import Neighborhood
from jmetal.util.ranking import FastNonDominatedRanking

from jmetal.util.density_estimator import CrowdingDistance

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm, Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.operator import RankingAndCrowdingDistanceSelection, BinaryTournamentSelection
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.solutions.comparator import DominanceComparator, Comparator, RankingAndCrowdingDistanceComparator, \
    MultiComparator, SolutionAttributeComparator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: MOCell
   :platform: Unix, Windows
   :synopsis: MOCell (Multi-Objective Cellular evolutionary algorithm) implementation
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class MOCell(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 neighborhood: Neighborhood,
                 archive: BoundedArchive,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([SolutionAttributeComparator('dominance_ranking'),
                                      SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """
        MOCEll implementation as described in

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(MOCell, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            neighborhood = neighborhood,
            archive = archive,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator
        self.neighborhood = neighborhood
        self.archive = archive
        self.current_individual = 0
        self.current_neighbors = []

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = FastNonDominatedRanking()
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'MOCell'

