from abc import ABCMeta, abstractmethod
from typing import TypeVar, List, Generic

from jmetal.core.problem import Problem

S = TypeVar('S')


class Evaluator(Generic[S]):

    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem) -> None:
        problem.evaluate(solution)
        if problem.number_of_constraints > 0:
            problem.evaluate_constraints(solution)

    def get_name(self) -> str:
        return self.__class__.__name__