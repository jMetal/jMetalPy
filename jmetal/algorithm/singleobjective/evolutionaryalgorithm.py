from copy import copy
from typing import TypeVar, List

from jmetal.component.evaluator import Evaluator
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: evolutionary_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Evolutionary Algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ElitistEvolutionStrategy(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 mu: int,
                 lambd_a: int,
                 max_evaluations: int,
                 mutation: Mutation[S]):
        super(ElitistEvolutionStrategy, self).__init__()
        self.problem = problem
        self.mu = mu
        self.lambd_a = lambd_a
        self.max_evaluations = max_evaluations
        self.mutation = mutation

    def init_progress(self):
        self.evaluations = self.mu

    def update_progress(self):
        self.evaluations += self.lambd_a

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []
        for i in range(self.mu):
            population.append(self.problem.create_solution())
        return population

    def evaluate_population(self, population: List[S]) -> List[S]:
        for solution in population:
            self.problem.evaluate(solution)
        return population

    def selection(self, population: List[S]) -> List[S]:
        return population

    def reproduction(self, population: List[S]) -> List[S]:
        offspring_population = []
        for solution in population:
            for j in range(int(self.lambd_a / self.mu)):
                new_solution = copy(solution)
                offspring_population.append(self.mutation.execute(new_solution))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        for solution in offspring_population:
            self.population.append(solution)

        population.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(population[i])

        return new_population

    def get_result(self) -> R:
        return self.population[0]

    def get_name(self) -> str:
        return 'Elitist evolution Strategy'


class NonElitistEvolutionStrategy(ElitistEvolutionStrategy[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 mu: int,
                 lambd_a: int,
                 max_evaluations: int,
                 mutation: Mutation[S]):
        super(NonElitistEvolutionStrategy, self).__init__(problem, mu, lambd_a, max_evaluations, mutation)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        offspring_population.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(offspring_population[i])

        return new_population

    def get_name(self) -> str:
        return 'Non-Elitist evolution Strategy'


class GenerationalGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 evaluator: Evaluator[S]):
        super(GenerationalGeneticAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.evaluator = evaluator

    def init_progress(self):
        self.evaluations = self.population_size

    def update_progress(self):
        self.evaluations += self.population_size

        observable_data = {'evaluations': self.evaluations,
                           'computing time': self.get_current_computing_time(),
                           'population': self.population,
                           'reference_front': self.problem.reference_front}

        self.observable.notify_all(**observable_data)

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []

        for i in range(self.population_size):
            population.append(self.problem.create_solution())

        return population

    def evaluate_population(self, population: List[S]):
        return self.evaluator.evaluate(population, self.problem)

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.population_size):
            solution = self.selection_operator.execute(self.population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        self.__check_number_of_parents(number_of_parents_to_combine)

        offspring_population = []
        for i in range(0, self.population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.sort(key=lambda s: s.objectives[0])

        offspring_population.append(population[0])
        offspring_population.append(population[1])

        offspring_population.sort(key=lambda s: s.objectives[0])

        offspring_population.pop()
        offspring_population.pop()

        return offspring_population

    def get_result(self) -> R:
        """ :return: The best individual of the population.
        """
        return self.population[0]

    def __check_number_of_parents(self, number_of_parents_for_crossover: int):
        if self.population_size % number_of_parents_for_crossover != 0:
            raise Exception('Wrong number of parents')

    def get_name(self) -> str:
        return 'Generational Genetic Algorithm'
