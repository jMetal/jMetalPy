from copy import deepcopy
from typing import TypeVar, List

from jmetal.core.algorithm.evolutionaryAlgorithm import EvolutionaryAlgorithm
from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.problem.problem import Problem

""" Class representing generational genetic algorithms """
__author__ = "Antonio J. Nebro"

S = TypeVar('S')
R = TypeVar('R')


class GenerationalGeneticAlgorithm(EvolutionaryAlgorithm[S, R]):
    def __init__(self, problem: Problem[S], population_size: int,
                 max_evaluations: int,
                 mutation_operator: MutationOperator[S],
                 crossover_operator: CrossoverOperator[S, S]):
        super(GenerationalGeneticAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
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
            for j in range((int)(self.lambdA/self.mu)):
                new_solution = deepcopy(solution)
                offspring_population.append(self.mutation_operator.execute(new_solution))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S])\
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
        return "("+str(self.mu)+ "+" + str(self.lambdA)+")ES"