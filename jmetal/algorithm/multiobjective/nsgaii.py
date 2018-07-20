from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator, Evaluator
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RankingAndCrowdingDistanceSelection

S = TypeVar('S')
R = TypeVar(List[S])

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class NSGAII(GenerationalGeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 evaluator: Evaluator[S] = SequentialEvaluator[S]()):
        """  NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param max_evaluations: Maximum number of evaluations/iterations.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        :param evaluator: An evaluator object to evaluate the individuals of the population.
        """
        super(NSGAII, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            evaluator)

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
        return 'Non-dominated Sorting Genetic Algorithm II'
