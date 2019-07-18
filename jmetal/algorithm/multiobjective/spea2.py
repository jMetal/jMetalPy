from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, RankingAndCrowdingDistanceComparator, StrengthAndKNNDistanceComparator, \
    SolutionAttributeComparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance, KNearestNeighborDensityEstimator
from jmetal.util.ranking import FastNonDominatedRanking, StrengthRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: SPEA2
   :platform: Unix, Windows
   :synopsis: SPEA2  implementation. Note that we do not follow the structure of the original SPEA2 code

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class SPEA2(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """
        NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """

        multi_comparator = MultiComparator([SolutionAttributeComparator('strength_ranking'),
                                            SolutionAttributeComparator("knn_density", lowest_is_best=False)])
        #multi_comparator = MultiComparator([SolutionAttributeComparator('strength_ranking'),
        #                                    SolutionAttributeComparator("knn_density", lowest_is_best=False)])
        #selection = BinaryTournamentSelection(comparator=StrengthAndKNNDistanceComparator())
        selection = BinaryTournamentSelection(comparator=StrengthAndKNNDistanceComparator())

        super(SPEA2, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = StrengthRanking() # StrengthRanking()
        density_estimator = KNearestNeighborDensityEstimator()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator).replace(population, offspring_population)
        return r

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'SPEA2'


"""
class NaryTournamentMatingPoolSelection():
    def __init__(self, tournament_size: int, mating_pool_size: int, comparator: Comparator):
        self.selection_operator = BinaryTournamentSelection(comparator)
        self.mating_pool_size = mating_pool_size

    def select(self, solution_list: List[S]):
        mating_pool = []

        while len(mating_pool) < self.mating_pool_size:
            mating_pool.append(self.selection_operator.execute(solution_list))

        return mating_pool
"""
