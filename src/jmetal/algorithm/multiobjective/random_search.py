import time
from typing import TypeVar

from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.util.archive import NonDominatedSolutionsArchive
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

"""
.. module:: RamdomSearch
   :platform: Unix, Windows
   :synopsis: Simple random_search search algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class RandomSearch(Algorithm[S, R]):
    def __init__(
        self,
        problem: Problem[S],
        termination_criterion: TerminationCriterion | None = None,
    ):
        if termination_criterion is None:
            termination_criterion = StoppingByEvaluations(max_evaluations=25000)

        super().__init__()
        self.problem = problem
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.archive: NonDominatedSolutionsArchive[S] = NonDominatedSolutionsArchive()

    def observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": ctime,
        }

    def create_initial_solutions(self) -> list[S]:
        return [self.problem.create_solution()]

    def evaluate(self, solution_list: list[S]) -> list[S]:
        return [self.problem.evaluate(solution_list[0])]

    def init_progress(self) -> None:
        self.evaluations = 1

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self) -> None:
        new_solution = self.problem.create_solution()
        self.problem.evaluate(new_solution)
        self.archive.add(new_solution)

    def update_progress(self) -> None:
        self.evaluations += 1

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self) -> list[S]:
        return self.archive.solution_list

    def get_name(self) -> str:
        return "Random Search"

    @property
    def label(self) -> str:
        return f"{self.get_name()}.{self.problem.name()}"
