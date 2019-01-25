import threading
import time
from typing import Generic, TypeVar, List

from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class LocalSearch(Algorithm[S, R], threading.Thread):

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 termination_criterion: TerminationCriterion):
        super(LocalSearch, self).__init__()
        self.problem = problem
        self.mutation = mutation
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[S]:
        self.solutions.append(self.problem.create_solution())
        return self.solutions

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return [self.problem.evaluate(solution_list[0])]

    def init_progress(self) -> None:
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self) -> None:
        mutated_solution: Solution = self.mutation.execute(self.solutions[0])
        mutated_solution = self.problem.evaluate(mutated_solution)
        if mutated_solution.objectives[0] < self.solutions[0].objectives[0]:
            self.solutions[0] = mutated_solution

    def update_progress(self) -> None:
        self.evaluations += 1

        print("EVALs: " + str(self.evaluations) + ". Fitness: " + str(self.solutions[0].objectives[0]))

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        return self.observable

    def get_result(self) -> R:
        self.solutions[0]

    def get_name(self) -> str:
        return "LocalSearch"

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': self.get_result(), 'COMPUTING_TIME': ctime}


        
