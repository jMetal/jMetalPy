from typing import TypeVar, List, Generic

from jmetal.component.archive import NonDominatedSolutionListArchive
from jmetal.core.problem import Problem

S = TypeVar('S')

"""
.. module:: RamdomSearch
   :platform: Unix, Windows
   :synopsis: Simple random search algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class RandomSearch(Generic[S]):

    def __init__(self,
                 problem: Problem[S],
                 max_evaluations: int = 25000):
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.archive = NonDominatedSolutionListArchive()

    def run(self) -> None:
        for i in range(self.max_evaluations):
            new_solution = self.problem.create_solution()
            self.problem.evaluate(new_solution)
            self.archive.add(new_solution)

    def get_result(self) -> List[S]:
        return self.archive.solution_list

    @staticmethod
    def get_name() -> str:
        return 'Random Search Algorithm'
