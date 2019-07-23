import time
from functools import cmp_to_key
from typing import TypeVar, List

import dask
from distributed import as_completed, Client

from jmetal.util.archive import BoundedArchive
from jmetal.util.neighborhood import Neighborhood
from jmetal.util.ranking import FastNonDominatedRanking, Ranking

from jmetal.util.density_estimator import CrowdingDistance, DensityEstimator

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
import copy

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
            offspring_population_size=1,
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
        self.comparator = MultiComparator([SolutionAttributeComparator('dominance_ranking'),
                                      SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])

    def init_progress(self) -> None:
        super().init_progress()
        for solution in self.solutions:
            self.archive.add(copy.copy(solution))

    def update_progress(self) -> None:
        super().update_progress()
        self.current_individual = (self.current_individual+1) % self.population_size

    def selection(self, population: List[S]):
        parents = []
        self.current_neighbors = self.neighborhood.get_neighbors(population, self.current_individual)
        self.current_neighbors.add(self.solutions[self.current_individual])

        parents.append(self.selection_operator.execute(self.current_neighbors))
        if len(self.archive.solution_list) > 0:
            parents.append(self.selection_operator.execute(self.archive.solution_list))
        else:
            parents.append(self.selection_operator.execute(self.current_neighbors))

        return parents

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = self.crossover_operator.execute(mating_population)
        self.mutation_operator.execute(offspring_population[0])

        return [offspring_population[0]]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        result = self.dominance_comparator.compare(self.solutions[self.current_individual], offspring_population[0])

        if result == 1: # the offspring individual dominates the current one
            location_of_current_individual = population.index(self.current_individual)
            population[location_of_current_individual] = offspring_population[0]
            self.archive.add(copy.copy(offspring_population[0]))
            #population = self.__insert_new_individual_when_it_dominates_the_current_one(population, offspring_population)
        elif result == 0: # the offspring and current individuals are non-dominated
            a = 5
            new_individual = offspring_population[0]
            self.current_neighbors.append(new_individual)
            new_individual.attributes["location"] = -1
            result_list = population[:]

            ranking: Ranking = FastNonDominatedRanking()
            ranking.compute_ranking(self.current_neighbors)

            density_estimator: DensityEstimator = CrowdingDistance()
            for i in range(ranking.get_number_of_subfronts()):
                density_estimator.compute_density_estimator(ranking.get_subfront(i))

        self.current_neighbors.sort(key=cmp_to_key(self.comparator.compare))
        worst_solution = self.current_neighbors[-1]

        if worst_solution.attributes["location"] == -1:
            self.archive.add(new_individual)
        else:
            pass
            

        return population

    def get_result(self) -> R:
        return self.archive.solution_list

    def get_name(self) -> str:
        return 'MOCell'


    """
    def __insert_new_individual_when_it_dominates_the_current_one(self, population, offspring_population):
        self.archive.add(offspring_population[0])
        result_list = population[:]
        location_of_current_individual = 
        result_list.
    """