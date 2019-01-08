from typing import TypeVar, List

from jmetal.util.archive import NonDominatedSolutionListArchive
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: RamdomSearch
   :platform: Unix, Windows
   :synopsis: Simple random search algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class RandomSearch(Algorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 termination_criterion: TerminationCriterion):
        super().__init__()
        self.problem = problem
        self.termination_criterion = termination_criterion
        self.archive = NonDominatedSolutionListArchive()

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution()]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return [self.problem.evaluate(solution_list[0])]

    def init_progress(self) -> None:
        self.evaluations = 1

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self) -> None:
        new_solution = self.problem.create_solution()
        self.problem.evaluate(new_solution)
        self.archive.add(new_solution)

    def update_progress(self) -> None:
        self.evaluations += 1

    def get_result(self) -> List[S]:
        return self.archive.solution_list

    @staticmethod
    def get_name() -> str:
        return 'Random Search Algorithm'
