from abc import ABCMeta, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import TypeVar, List, Generic

from dask.distributed import LocalCluster, Client, as_completed

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


class MultithreadedEvaluator(Evaluator[S]):

    def __init__(self, n_workers: int, processes: bool=True):
        """
        :param n_workers: Number of workers to start.
        :param processes: Whether to use processes (True) or threads (False).
        """
        cluster = LocalCluster(n_workers=n_workers, processes=processes)
        self.client = Client(cluster)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        futures = []
        for solution in solution_list:
            futures.append(self.client.submit(problem.evaluate, solution))

        evaluated_list = []
        for future in as_completed(futures):
            evaluated_list.append(future.result())

        return evaluated_list
