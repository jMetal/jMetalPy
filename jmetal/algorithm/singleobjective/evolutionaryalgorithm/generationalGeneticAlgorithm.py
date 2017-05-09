from copy import deepcopy
from typing import TypeVar, List

from jmetal.core.algorithm.evolutionaryAlgorithm import EvolutionaryAlgorithm
from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.operator.selectionoperator import SelectionOperator
from jmetal.core.problem.problem import Problem

""" Class representing generational genetic algorithms """
__author__ = "Antonio J. Nebro"

S = TypeVar('S')
R = TypeVar('R')

class GenerationalGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation_operator: MutationOperator[S],
                 crossover_operator: CrossoverOperator[S, S],
                 selection_operator: SelectionOperator[List[S],S]):
        super(GenerationalGeneticAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.evaluations = 0

    def init_progress(self):
        self.evaluations = self.population_size

    def update_progress(self):
        self.evaluations += self.population_size

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []
        for i in range(self.population_size):
            population.append(self.problem.create_solution())

        p = (population.append(self.problem.create_solution()) for i in range (self.population_size))
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

    def reproduction(self, population: List[S]):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        self.__check_number_of_parents(population, number_of_parents_to_combine)

        offspring_population = []
        for i in range(0, self.population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(population[i+j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S])\
            -> List[S]:
        population.sort(key=lambda s: s.objectives[0])

        offspring_population.append(population[0])
        offspring_population.append(population[1])

        offspring_population.sort(key=lambda s: s.objectives[0])

        offspring_population.pop()
        offspring_population.pop()

        return offspring_population

    def get_result(self) -> R:
        return self.population[0]

    def get_name(self):
        return "Generational Genetic Algorithm"

    def __check_number_of_parents(self, population: List[S], number_of_parents_for_crossover: int) -> bool:
        if self.population_size % number_of_parents_for_crossover != 0:
            raise Exception("Wrong number of parents")