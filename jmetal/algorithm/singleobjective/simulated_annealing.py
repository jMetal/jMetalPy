import copy
import random
import threading
import time
from typing import TypeVar, List

import numpy

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: simulated_annealing
   :platform: Unix, Windows
   :synopsis: Implementation of Simulated Annealing.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class SimulatedAnnealing(Algorithm[S, R], threading.Thread):

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 termination_criterion: TerminationCriterion,
                 solution_generator: Generator = store.default_generator):
        super(SimulatedAnnealing, self).__init__()
        self.problem = problem
        self.mutation = mutation
        self.termination_criterion = termination_criterion
        self.solution_generator = solution_generator
        self.observable.register(termination_criterion)
        self.temperature = 1.0
        self.minimum_temperature = 0.000001
        self.alpha = 0.95
        self.counter = 0

    def create_initial_solutions(self) -> List[S]:
        return [self.solution_generator.new(self.problem)]

    def evaluate(self, solutions: List[S]) -> List[S]:
        return [self.problem.evaluate(solutions[0])]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = 0

    def step(self) -> None:
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution: Solution = self.mutation.execute(mutated_solution)
        mutated_solution = self.evaluate([mutated_solution])[0]

        acceptance_probability = self.compute_acceptance_probability(
            self.solutions[0].objectives[0],
            mutated_solution.objectives[0],
            self.temperature)

        if acceptance_probability > random.random():
            self.solutions[0] = mutated_solution

        self.temperature *= self.alpha

    def compute_acceptance_probability(self, current: float, new: float, temperature: float) -> float:
        if new < current:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            value = (new - current) / t
            return numpy.exp(-1.0 * value)

    def update_progress(self) -> None:
        self.evaluations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': ctime}

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Simulated Annealing'
