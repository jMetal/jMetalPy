from copy import copy
from typing import TypeVar, List

from jmetal.component.evaluator import Evaluator
from jmetal.component.generator import Generator
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: evolutionary_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Evolutionary Algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class EvolutionStrategy(EvolutionaryAlgorithm):

    def __init__(self,
                 problem: Problem,
                 max_evaluations: int,
                 mu: int,
                 lambda_: int,
                 mutation: Mutation,
                 elitist: bool = True,
                 pop_evaluator: Evaluator = None,
                 population_generator: Generator = None):
        super(EvolutionStrategy, self).__init__(
            problem=problem,
            population_size=mu,
            population_generator=population_generator,
            max_evaluations=max_evaluations,
            pop_evaluator=pop_evaluator
        )
        self.mu = mu
        self.lambda_ = lambda_
        self.elitist = elitist
        self.mutation = mutation

    def selection(self, population: List[S]) -> List[S]:
        return population

    def reproduction(self, population: List[S]) -> List[S]:
        offspring_population = []
        for solution in population:
            for j in range(int(self.lambda_ / self.mu)):
                new_solution = copy(solution)
                offspring_population.append(self.mutation.execute(new_solution))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population_pool = []

        if self.elitist:
            population_pool = population
            population_pool.extend(offspring_population)
        else:
            population_pool.extend(offspring_population)

        population_pool.sort(key=lambda s: s.objectives[0])

        new_population = []
        for i in range(self.mu):
            new_population.append(population_pool[i])

        return new_population

    def update_progress(self):
        self.evaluations += self.lambda_

        observable_data = {
            'problem': self.problem,
            'population': self.population,
            'evaluations': self.evaluations,
            'computing time': self.current_computing_time,
        }

        self.observable.notify_all(**observable_data)

    def get_result(self) -> R:
        return self.population[0]

    def get_name(self) -> str:
        return 'Elitist evolution Strategy'
