from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RankingAndCrowdingDistanceSelection
from jmetal.util.observable import Observable, DefaultObservable

S = TypeVar('S')
R = TypeVar(List[S])


class NSGAII(GenerationalGeneticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 observable: Observable = DefaultObservable()):
        super(NSGAII, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            observable)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[TypeVar('S')]]:
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

    def get_name(self) -> str:
        return "NSGA-II"

    def get_result(self) -> R:
        return self.population


