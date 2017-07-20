from copy import copy
from typing import TypeVar, List

from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.observable import Observable, DefaultObservable

S = TypeVar('S')
R = TypeVar('R')


class ElitistEvolutionStrategy(EvolutionaryAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 mu: int,
                 lambdA: int,
                 max_evaluations: int,
                 mutation: Mutation[S]):
        super(ElitistEvolutionStrategy, self).__init__()
        self.problem = problem
        self.mu = mu
        self.lambdA = lambdA
        self.max_evaluations = max_evaluations
        self.mutation = mutation

    def init_progress(self):
        self.evaluations = self.mu

    def update_progress(self):
        self.evaluations += self.lambdA

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []
        for i in range(self.mu):
            population.append(self.problem.create_solution())
        return population

    def evaluate_population(self, population: List[S]):
        for solution in population:
            self.problem.evaluate(solution)
        return population

    def selection(self, population: List[S]):
        return population

    def reproduction(self, population: List[S]):
        offspring_population = []
        for solution in population:
            for j in range((int)(self.lambdA / self.mu)):
                new_solution = copy(solution)
                offspring_population.append(self.mutation.execute(new_solution))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) \
            -> List[S]:
        for solution in offspring_population:
            self.population.append(solution)

        population.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(population[i])

        return new_population

    def get_result(self) -> R:
        return self.population[0]

    def get_name(self):
        return "(" + str(self.mu) + "+" + str(self.lambdA) + ")ES"


class NonElitistEvolutionStrategy(ElitistEvolutionStrategy[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 mu: int, lambdA: int,
                 max_evaluations: int,
                 mutation: Mutation[S]):
        super(NonElitistEvolutionStrategy, self).__init__(problem, mu, lambdA,
                                                          max_evaluations, mutation)

    def replacement(self, population: List[S], offspring_population: List[S]) \
            -> List[S]:
        offspring_population.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(offspring_population[i])

        return new_population

    def get_name(self) -> str:
        return "(" + str(self.mu) + "," + str(self.lambdA) + ")ES"


class GenerationalGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 observable: Observable = DefaultObservable()):
        super(GenerationalGeneticAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.evaluations = 0
        self.observable = observable

    def init_progress(self):
        self.evaluations = self.population_size

    def update_progress(self):
        self.evaluations += self.population_size

        observable_data = {'evaluations': self.evaluations,
                           'population': self.population,
                           'computing time': self.get_current_computing_time()}
        self.observable.notify_all(**observable_data)

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []

        for i in range(self.population_size):
            population.append(self.problem.create_solution())

        return population

    def evaluate_population(self, population: List[S]):
        for solution in population:
            self.problem.evaluate(solution)
        return population

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.population_size):
            solution = self.selection_operator.execute(self.population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        self.__check_number_of_parents(population, number_of_parents_to_combine)

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

    def replacement(self, population: List[S], offspring_population: List[S]) \
            -> List[S]:
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

    def get_name(self) -> str:
        return "Generational Genetic Algorithm"

    def __check_number_of_parents(self, population: List[S], number_of_parents_for_crossover: int):
        if self.population_size % number_of_parents_for_crossover != 0:
            raise Exception("Wrong number of parents")

