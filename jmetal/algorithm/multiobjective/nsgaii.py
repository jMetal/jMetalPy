from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic import GeneticAlgorithm
from jmetal.component.evaluator import Evaluator
from jmetal.component.generator import Generator
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator import RankingAndCrowdingDistanceSelection

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NSGAII(GeneticAlgorithm):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_size: int,
                 mating_pool_size: int,
                 max_evaluations: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 pop_generator: Generator = None,
                 pop_evaluator: Evaluator = None):
        """  NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1 and the mating pool size to 2.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param max_evaluations: Maximum number of evaluations/iterations.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(NSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            pop_generator=pop_generator,
            offspring_size=offspring_size,
            mating_pool_size=mating_pool_size,
            max_evaluations=max_evaluations,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            pop_evaluator=pop_evaluator
        )

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

    def get_result(self) -> R:
        return self.population

    def get_name(self) -> str:
        return 'Non-dominated Sorting Genetic Algorithm II (NSGA-II)'
