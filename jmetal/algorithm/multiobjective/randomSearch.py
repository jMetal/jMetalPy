from typing import TypeVar, List, Generic

from jmetal.core.problem.problem import Problem

from jmetal.component.nondominatedsolutionlistarchive import NonDominatedSolutionListArchive

""" Class representing simple random search algorithms """


S = TypeVar('S')

class RandomSearch(Generic[S]):
    def __init__(self, problem:Problem[S], max_evaluations:int):
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.archive = NonDominatedSolutionListArchive()

    def run(self):
        for i in range(self.max_evaluations):
            new_solution = self.problem.create_solution()
            self.problem.evaluate(new_solution)
            self.archive.add(new_solution)

    def get_name(self):
        return "Random Search Algorithm"

    def get_result(self) -> List[S]:
        return self.archive.get_solution_list()