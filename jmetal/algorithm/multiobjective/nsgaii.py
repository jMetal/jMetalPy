from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.crowding_distance import CrowdingDistance
from jmetal.util.observable import Observable, DefaultObservable
from jmetal.util.ranking import DominanceRanking

S = TypeVar('S')
R = TypeVar('R')


class NSGAII(GenerationalGeneticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 observable: Observable = DefaultObservable()):
        super(GenerationalGeneticAlgorithm, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            observable
        )

    def replacement(self, population: List[S], offspring_population: List[S]) \
            -> List[S]:
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)


class RankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):
    def __init__(self, max_population_size: int):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size

    def execute(self, solution_list: List[S]) -> List[S]:
        ranking = DominanceRanking()
        crowding_distance = CrowdingDistance()
        ranked_lists = ranking.compute_ranking(solution_list)

        ranking_index = 0
        new_solution_list = []
        while len(new_solution_list) < self.max_population_size:
            if len(ranking.get_subfront(ranking_index)) < self.max_population_size - len(new_solution_list):
                new_solution_list = new_solution_list + ranking.get_subfront(ranking_index)
                ranking_index += 1
            else:
                subfront = ranking.get_subfront(ranking_index)
                crowding_distance.compute_density_estimator(subfront)
                sorted_subfront = sorted(subfront, key=lambda x: x.attributes["distance"])
                for i in range((self.max_population_size - len(new_solution_list) - 1)):
                    new_solution_list.append(sorted_subfront[i])

        return new_solution_list