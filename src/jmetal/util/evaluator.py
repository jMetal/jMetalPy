import functools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool, ThreadPool
from typing import Generic, List, TypeVar, Set

try:
    import dask
except ImportError:
    pass

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    pass

from jmetal.core.problem import Problem

S = TypeVar("S")


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
    def __init__(self, processes: int = None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)

        return solution_list


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)


class SparkEvaluator(Evaluator[S]):
    def __init__(self, processes: int = 8):
        self.spark_conf = SparkConf().setAppName("jmetalpy").setMaster(f"local[{processes}]")
        self.spark_context = SparkContext(conf=self.spark_conf)

        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate.map(lambda s: problem.evaluate(s)).collect()


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, scheduler="processes", number_of_cores=4):
        self.scheduler = scheduler
        self.number_of_cores = number_of_cores

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler, pool=ThreadPoolExecutor(self.number_of_cores)):
            return list(
                dask.compute(
                    *[dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list]
                )
            )


class EvaluatorWithCache(Evaluator[S]):
    """
    An evaluator wrapper that caches evaluated solutions to avoid redundant evaluations.

    This class extends an `Evaluator` by introducing caching. It ensures that solutions
    which have already been evaluated are not re-evaluated, improving efficiency when
    working with large sets of solutions.

    Attributes:
        cache (set): A set storing the variables of previously evaluated solutions.
        base_evaluator (Evaluator[S]): The base evaluator used for actual evaluation.
    """

    def __init__(self, base_evaluator: Evaluator[S]):
        """
        Initializes the evaluator with caching.

        Args:
            base_evaluator (Evaluator[S]): The underlying evaluator responsible for evaluating solutions.
        """
        self.cache: Set = set()
        self.base_evaluator = base_evaluator

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        """
        Evaluates a list of solutions while avoiding redundant evaluations.

        This method filters out solutions that have already been evaluated (i.e., whose variables
        are in the cache), evaluates only the new solutions, and updates the cache accordingly.

        Args:
            solution_list (List[S]): The list of solutions to evaluate.
            problem (Problem): The problem instance for which the solutions are being evaluated.

        Returns:
            List[S]: The list of newly evaluated solutions.
        """
        # Filter out already evaluated solutions
        filtered_solutions = [solution for solution in solution_list if solution.variables not in self.cache]

        # Evaluate only new solutions
        filtered_solutions = self.base_evaluator.evaluate(filtered_solutions, problem)

        # Add newly evaluated solutions to the cache
        for solution in filtered_solutions:
            self.cache.add(solution.variables)

        return filtered_solutions
