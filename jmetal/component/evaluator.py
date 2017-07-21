from typing import TypeVar, List, Generic

from jmetal.core.problem import Problem

S = TypeVar('S')


class Evaluator(Generic[S]):
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass


class SequentialEvaluator(Evaluator[S]):
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        for solution in solution_list:
            problem.evaluate(solution)

        return solution_list