from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator, Evaluator
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RankingAndCrowdingDistanceSelection
from jmetal.util.observable import Observable, DefaultObservable

S = TypeVar('S')
R = TypeVar(List[S])

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: Implementation of NSGA-II.

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
                 evaluator: Evaluator[S] = SequentialEvaluator[S](),
                 observable: Observable = DefaultObservable()):
        """ NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm templates section (:mod:`algorithm`) of the documentation.

        :param problem: The problem to solve.
        :param population_size:
        :param max_evaluations:
        :param mutation:
        :param crossover:
        :param selection:
        :param observable:
        :param evaluator: An evaluator object to evaluate the solutions in the population.
        """
        super(NSGAII, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            evaluator,
            observable)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[TypeVar('S')]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population:
        :param offspring_population:
        :return: New population.
        """
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

    def get_result(self) -> R:
        """:return: Population.
        """
        return self.population

    def get_name(self) -> str:
        return 'NSGA-II'
