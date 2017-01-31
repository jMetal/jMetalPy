from typing import TypeVar, Generic, List
from copy import deepcopy

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.algorithm.evolutionaryAlgorithm import EvolutionaryAlgorithm
from jmetal.core.problem.problem import Problem
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.operator.mutation.bitflip import BitFlip
from jmetal.operator.mutation.polynomial import Polynomial
from jmetal.problem.singleobjective.onemax import OneMax
from jmetal.problem.singleobjective.sphere import Sphere

""" Class representing elitist evolution strategy algorithms """
__author__ = "Antonio J. Nebro"

S = TypeVar('S')
R = TypeVar('R')


class ElitistEvolutionStrategy(EvolutionaryAlgorithm[S, R]):
    def __init__(self, problem: Problem[S], mu: int, lambdA: int,
                 max_evaluations: int, mutation_operator: MutationOperator[S]):
        super(ElitistEvolutionStrategy, self).__init__()
        self.problem = problem
        self.mu = mu
        self.lambdA = lambdA
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation_operator
        self.evaluations = 0

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