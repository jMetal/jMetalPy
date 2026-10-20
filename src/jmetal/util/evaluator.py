import copy
import functools
from abc import ABC, abstractmethod
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool, ThreadPool
from typing import Generic, TypeVar

try:
    import dask
except ImportError:
    pass

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    pass

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.archive import Archive

S = TypeVar("S")


class Evaluator(Generic[S], ABC):
    @abstractmethod
    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        pass

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem) -> None:
        problem.evaluate(solution)


class EvaluatorWithCache(Evaluator[S]):
    """Cache evaluations while delegating cache misses to another evaluator.

    The cache key is built from a solution's type and decision variables. Objective and
    constraint values are copied into the cache, so later mutations of an evaluated solution do
    not alter cached results. A cache is scoped to one problem instance and is cleared when the
    evaluator is used with a different problem.

    Args:
        base_evaluator: Evaluator used for solutions that are not already cached.
    """

    def __init__(self, base_evaluator: Evaluator[S]) -> None:
        self.base_evaluator = base_evaluator
        self.cache: dict[Hashable, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self._problem: Problem | None = None

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        if problem is not self._problem:
            self.cache.clear()
            self._problem = problem

        pending_solutions: dict[Hashable, list[S]] = {}
        for solution in solution_list:
            key = self._solution_key(solution)
            cached_values = self.cache.get(key)
            if cached_values is None:
                pending_solutions.setdefault(key, []).append(solution)
            else:
                self._restore_values(solution, cached_values)

        representatives = [solutions[0] for solutions in pending_solutions.values()]
        if not representatives:
            return solution_list

        evaluated_solutions = self.base_evaluator.evaluate(representatives, problem)
        if len(evaluated_solutions) != len(representatives):
            raise ValueError("The base evaluator must return one solution per input solution")

        for (key, equivalent_solutions), evaluated_solution in zip(
            pending_solutions.items(), evaluated_solutions, strict=True
        ):
            cached_values = (
                tuple(evaluated_solution.objectives),
                tuple(evaluated_solution.constraints),
            )
            self.cache[key] = cached_values
            for solution in equivalent_solutions:
                self._restore_values(solution, cached_values)

        return solution_list

    @classmethod
    def _solution_key(cls, solution: S) -> Hashable:
        return (type(solution), *(cls._to_hashable(value) for value in solution.variables))

    @classmethod
    def _to_hashable(cls, value: object) -> Hashable:
        if isinstance(value, Solution):
            return (
                type(value),
                *(cls._to_hashable(variable) for variable in value.variables),
            )
        if isinstance(value, dict):
            return frozenset(
                (cls._to_hashable(key), cls._to_hashable(item)) for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._to_hashable(item) for item in value)
        if isinstance(value, set):
            return frozenset(cls._to_hashable(item) for item in value)
        if isinstance(value, Hashable):
            return value

        raise TypeError(f"Solution variable of type {type(value).__name__} is not hashable")

    @staticmethod
    def _restore_values(
        solution: S, cached_values: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> None:
        objectives, constraints = cached_values
        solution.objectives = list(objectives)
        solution.constraints = list(constraints)


class SequentialEvaluator(Evaluator[S]):
    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class SequentialEvaluatorWithArchive(SequentialEvaluator[S]):
    """
    Sequential evaluator that maintains an archive of evaluated solutions.

    This evaluator extends SequentialEvaluator by automatically storing copies
    of evaluated solutions in an archive. This is useful for:
    - Maintaining a history of all evaluated solutions
    - Collecting best solutions found during optimization
    - Post-processing analysis of the optimization process

    Args:
        archive: Archive instance to store evaluated solutions

    Example:
        >>> from jmetal.util.archive import NonDominatedSolutionsArchive
        >>> from jmetal.util.evaluator import SequentialEvaluatorWithArchive
        >>>
        >>> archive = NonDominatedSolutionsArchive()
        >>> evaluator = SequentialEvaluatorWithArchive(archive)
        >>>
        >>> # Use with optimization algorithm
        >>> algorithm = NSGAII(
        ...     problem=problem,
        ...     population_size=100,
        ...     offspring_population_size=100,
        ...     mutation=mutation,
        ...     crossover=crossover,
        ...     selection=selection,
        ...     evaluator=evaluator
        ... )
        >>>
        >>> # After optimization, access collected solutions
        >>> best_solutions = evaluator.archive.solution_list
    """

    def __init__(self, archive: Archive[S]):
        """
        Initialize evaluator with archive.

        Args:
            archive: Archive instance to store evaluated solutions
        """
        self.archive = archive

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        """
        Evaluate solutions and store copies in the archive.

        Args:
            solution_list: List of solutions to evaluate
            problem: Problem instance used for evaluation

        Returns:
            List of evaluated solutions (same as input list)
        """
        # Evaluate solutions using parent implementation
        evaluated_solutions = super().evaluate(solution_list, problem)

        # Add copies of evaluated solutions to archive
        for solution in evaluated_solutions:
            # Use copy.copy to call solution.__copy__ implementations
            # which are more efficient and tailored for Solution types.
            self.archive.add(copy.copy(solution))

        return evaluated_solutions

    def get_archive(self) -> Archive[S]:
        """
        Get the archive containing evaluated solutions.

        Returns:
            Archive instance with stored solutions
        """
        return self.archive


class MapEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        self.pool.map(
            lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list
        )

        return solution_list


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)


class SparkEvaluator(Evaluator[S]):
    def __init__(self, processes: int = 8):
        self.spark_conf = SparkConf().setAppName("jmetalpy").setMaster(f"local[{processes}]")
        self.spark_context = SparkContext(conf=self.spark_conf)

        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate.map(lambda s: problem.evaluate(s)).collect()


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, scheduler="processes", number_of_cores=4):
        self.scheduler = scheduler
        self.number_of_cores = number_of_cores

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        with dask.config.set(
            scheduler=self.scheduler, pool=ThreadPoolExecutor(self.number_of_cores)
        ):
            return list(
                dask.compute(
                    *[
                        dask.delayed(evaluate_solution)(solution=solution, problem=problem)
                        for solution in solution_list
                    ]
                )
            )
