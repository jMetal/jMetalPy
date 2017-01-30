from typing import TypeVar, Generic, List
from copy import deepcopy

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.algorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from jmetal.core.problem.problem import Problem
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.core.solution.solution import Solution
from jmetal.operator.mutation.bitflip import BitFlip
from jmetal.problem.singleobjective.onemax import OneMax

""" Class representing elitist evolution strategy algorithms """
__author__ = "Antonio J. Nebro"


S = TypeVar('S')
R = TypeVar('R')

class ElitistEvolutionStrategy(EvolutionaryAlgorithm(S, R)):
    def __init__(self, problem: Problem, mu: int, lambd: int, max_evaluations: int, mutation_operator: MutationOperator):
        self.problem = problem
        self.mu = mu
        self.lambd = lambd
        self.max_evaluations = max_evaluations
        self.mutation_operartor = mutation_operator
        self.evaluations = 0

    def init_progress(self):
        self.evaluations = self.mu

    def update_progress(self):
        self.evaluations += self.lambd

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[Solution]:
        population = List[Solution]
        for i in range(self.mu):
            population.append(self.problem.create_solution())
        return population

    def evaluate_population(self, population: List[Solution]):
        for solution in population:
            self.problem.evaluate(solution)

    def selection(self, population: List[Solution]):
        return population

    def reproduction(self, population: List[Solution]):
        offspring_population = List[Solution]
        for solution in population:
            for j in range(self.lambd/self.mu):
                new_solution = deepcopy(solution)
                offspring_population.append(new_solution)

        return offspring_population

    def replacement(self, population: List[Solution], offspring_population: List[Solution])\
            -> List[Solution]:
        for solution in offspring_population:
            population.append(solution)

        population.sort(key=solution.objectives[0])

        new_population = List[Solution]
        for i in range(self.mu):
            new_population.append(population[i])

        return new_population

    def get_result(self) -> List[Solution]:
        return self.population




algorithm = ElitistEvolutionStrategy(BinarySolution, [])(OneMax, mu=1, lambd=1,
                                     max_evaluations= 25000, mutation_operator=BitFlip)
algorithm.run()