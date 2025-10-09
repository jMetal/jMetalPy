import copy
import functools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool, ThreadPool
from typing import Generic, List, TypeVar

try:
    import dask
except ImportError:
    pass

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    pass

from jmetal.core.problem import Problem
from jmetal.util.archive import Archive

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
    
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
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
            self.archive.add(copy.deepcopy(solution))
        
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
