from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import TypeVar, List, Generic

from pyspark import SparkConf, SparkContext

from jmetal.core.problem import Problem

S = TypeVar('S')


class Evaluator(Generic[S], ABC):

    @abstractmethod
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem) -> None:
        problem.evaluate(solution)


class SequentialEvaluator(Evaluator[S]):

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class MapEvaluator(Evaluator[S]):

    def __init__(self, processes=None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)

        return solution_list


class SparkEvaluator(Evaluator[S]):
    def __init__(self):
        self.spark_conf = SparkConf()
        self.spark_context = SparkContext(conf=self.spark_conf)

        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate \
            .map(lambda s: problem.evaluate(s)) \
            .collect()
