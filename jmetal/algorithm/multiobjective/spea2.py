from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.density_estimator import KNearestNeighborDensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.ranking import StrengthRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: SPEA2
   :platform: Unix, Windows
   :synopsis: SPEA2  implementation. Note that we do not follow the structure of the original SPEA2 code. We consider
   SPEA2 as a genetic algorithm with binary tournament selection, with a comparator based on the strength fitness and 
   the KNN distance, and a sequential replacement strategy based in iteratively (sequentially) 
   removing the worst solution of the population + offspring population. The worst solutions is selected again 
   considering the strength fitness and KNN distance. Note that the implementation is exactly the same of NSGA-II, 
   but using the fast nondominated sorting and the crowding distance density estimator, and the replacement follows a 
   one-shot scheme (once the solutions are ordered, the best ones are selected without recomputing the ranking and
   density estimator).

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
        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        """
        multi_comparator = MultiComparator([StrengthRanking.get_comparator(),
                                            KNearestNeighborDensityEstimator.get_comparator()])
        selection = BinaryTournamentSelection(comparator=multi_comparator)

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
        ranking = StrengthRanking(self.dominance_comparator)
        density_estimator = KNearestNeighborDensityEstimator()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.SEQUENTIAL)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'SPEA2'
