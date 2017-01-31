from typing import TypeVar, Generic, List
from copy import deepcopy

from jmetal.algorithm.singleobjective.evolutionaryalgorithm.elitistEvolutionStrategy import ElitistEvolutionStrategy
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


class NonElitistEvolutionStrategy(ElitistEvolutionStrategy[S, R]):
    def __init__(self, problem: Problem[S], mu: int, lambdA: int,
                 max_evaluations: int, mutation_operator: MutationOperator[S]):
        super(NonElitistEvolutionStrategy, self).__init__(problem, mu, lambdA,
                                                    max_evaluations, mutation_operator)

    def replacement(self, population: List[S], offspring_population: List[S])\
            -> List[S]:
        offspring_population.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(offspring_population[i])

        return new_population

    def get_name(self):
        return "("+str(self.mu)+ "," + str(self.lambdA)+")ES"