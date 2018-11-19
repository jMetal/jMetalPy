from multiprocessing.pool import ThreadPool
from typing import TypeVar, List

from jmetal.core.evaluator import Evaluator
from jmetal.core.problem import Problem

S = TypeVar('S')


class SequentialEvaluator(Evaluator[S]):

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class MapEvaluator(Evaluator[S]):

    def __init__(self, processes=None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        self.pool.map(lambda solution: Evaluator[S].evaluate_solution(solution, problem), solution_list)

        return solution_list
