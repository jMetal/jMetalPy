import copy
from functools import cmp_to_key
from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.archive import BoundedArchive
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance, DensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.neighborhood import Neighborhood
from jmetal.util.ranking import FastNonDominatedRanking, Ranking
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
                 neighborhood: Neighborhood,
                 archive: BoundedArchive,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """
        MOCEll implementation as described in:

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

        self.comparator = MultiComparator([FastNonDominatedRanking.get_comparator(),
                                           CrowdingDistance.get_comparator()])

    def init_progress(self) -> None:
        super().init_progress()
        for solution in self.solutions:
            self.archive.add(copy.copy(solution))

    def update_progress(self) -> None:
        super().update_progress()
        self.current_individual = (self.current_individual + 1) % self.population_size

    def selection(self, population: List[S]):
        parents = []

        self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
        self.current_neighbors.append(self.solutions[self.current_individual])

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
        result = self.dominance_comparator.compare(population[self.current_individual], offspring_population[0])

        if result == 1:  # the offspring individual dominates the current one
            population[self.current_individual] = offspring_population[0]
            self.archive.add(offspring_population[0])
        elif result == 0:  # the offspring and current individuals are non-dominated
            new_individual = offspring_population[0]

            self.current_neighbors.append(new_individual)

            ranking: Ranking = FastNonDominatedRanking()
            ranking.compute_ranking(self.current_neighbors)

            density_estimator: DensityEstimator = CrowdingDistance()
            for i in range(ranking.get_number_of_subfronts()):
                density_estimator.compute_density_estimator(ranking.get_subfront(i))

            self.current_neighbors.sort(key=cmp_to_key(self.comparator.compare))
            worst_solution = self.current_neighbors[-1]

            self.archive.add(new_individual)
            if worst_solution != new_individual:
                population[self.current_individual] = new_individual

        return population

    def get_result(self) -> R:
        return self.archive.solution_list

    def get_name(self) -> str:
        return 'MOCell'
