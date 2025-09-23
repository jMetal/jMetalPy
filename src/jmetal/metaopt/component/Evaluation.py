from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, TypeVar, Generic, Optional, Any
from jmetal.core.problem import Problem
import logging
import os

S = TypeVar('S')

class Evaluation(ABC, Generic[S]):
    """
    Interface representing entities that evaluate a list of solutions.
    
    The type parameter S represents the type of solutions to be evaluated.
    """
    
    @abstractmethod
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """
        Evaluate a list of solutions.
        
        Args:
            solution_list: List of solutions to be evaluated.
            
        Returns:
            List of evaluated solutions.
        """
        pass
    
    @abstractmethod
    def computed_evaluations(self) -> int:
        """
        Get the number of evaluations computed so far.
        
        Returns:
            The number of evaluations.
        """
        pass
    
    @property
    @abstractmethod
    def problem(self) -> Problem[S]:
        """
        Get the problem being solved.
        
        Returns:
            The problem instance.
        """
        pass


class SequentialEvaluation(Evaluation[S]):
    """Evaluates a list of solutions sequentially.
    
    This implementation evaluates each solution in the input list one after another,
    updating the evaluation count after all solutions have been processed.
    
    Args:
        problem: The problem instance to use for evaluation. Must implement the Problem interface.
        
    Attributes:
        _problem: The problem instance being solved.
        _computed_evaluations: Counter for the total number of evaluations performed.
    """
    
    def __init__(self, problem: Problem[S]) -> None:
        """Initialize the sequential evaluator with a problem instance.
        
        Args:
            problem: The problem instance to evaluate solutions against.
            
        Raises:
            ValueError: If the provided problem is None.
        """
        if problem is None:
            raise ValueError("Problem instance cannot be None")
            
        self._problem: Problem[S] = problem
        self._computed_evaluations: int = 0
    
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """Evaluate all solutions in the provided list.
        
        Args:
            solution_list: A list of solutions to evaluate. Must not be None and should not contain None values.
            
        Returns:
            The input list with all solutions evaluated. The input list is modified in-place.
            
        Raises:
            ValueError: If solution_list is None or contains None values.
            
        Example:
            >>> problem = MyProblem()
            >>> evaluator = SequentialEvaluation(problem)
            >>> solutions = [Solution(...) for _ in range(10)]
            >>> evaluated = evaluator.evaluate(solutions)
            >>> evaluator.computed_evaluations()
            10
        """
        if not solution_list:  # Handles both None and empty list
            return solution_list  # No solutions to evaluate
                    
        # Evaluate all solutions
        for solution in solution_list:
            self._problem.evaluate(solution)
        
        self._computed_evaluations += len(solution_list)
        return solution_list
    
    def computed_evaluations(self) -> int:
        """Get the total number of evaluations performed.
        
        Returns:
            The total count of solutions evaluated by this instance.
        """
        return self._computed_evaluations
    
    @property
    def problem(self) -> Problem[S]:
        """Get the problem instance being used for evaluation.
        
        Returns:
            The problem instance that was provided during initialization.
            
        Note:
            The returned problem instance should be treated as read-only.
            Modifying it could lead to unexpected behavior.
        """
        return self._problem


class MultiThreadedEvaluation(Evaluation[S]):
    """Evaluates a list of solutions in parallel using multiple threads.
    
    This implementation distributes the evaluation of solutions across multiple
    worker threads, which can significantly improve performance for I/O-bound or
    CPU-bound evaluation tasks.
    
    Args:
        problem: The problem instance to use for evaluation.
        max_workers: Maximum number of worker threads to use. If None, it will
                   use the number of processors on the machine multiplied by 5.
                   This is a common heuristic for I/O-bound tasks.
    
    Attributes:
        _problem: The problem instance being solved.
        _computed_evaluations: Counter for the total number of evaluations performed.
        _max_workers: Maximum number of worker threads to use.
        _logger: Logger instance for debug and error messages.
    """
    
    def __init__(self, problem: Problem[S], max_workers: Optional[int] = None) -> None:
        """Initialize the multi-threaded evaluator.
        
        Args:
            problem: The problem instance to evaluate solutions against.
            max_workers: Maximum number of worker threads. If None, uses a default
                       based on the number of available CPUs.
        """
        if problem is None:
            raise ValueError("Problem instance cannot be None")
            
        self._problem: Problem[S] = problem
        self._computed_evaluations: int = 0
        self._max_workers: int = max_workers if max_workers is not None else (os.cpu_count() or 1) * 5
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)
    
    def _evaluate_solution(self, solution: S) -> S:
        """Helper method to evaluate a single solution.
        
        This method is called by worker threads.
        """
        try:
            self._problem.evaluate(solution)
            return solution
        except Exception as e:
            self._logger.error(f"Error evaluating solution: {e}", exc_info=True)
            raise
    
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """Evaluate all solutions in the provided list using multiple threads.
        
        Args:
            solution_list: A list of solutions to evaluate. Must not be None.
            
        Returns:
            The input list with all solutions evaluated. The input list is modified in-place.
            
        Raises:
            ValueError: If solution_list is None.
            
        Example:
            >>> problem = MyProblem()
            >>> evaluator = MultiThreadedEvaluation(problem, max_workers=4)
            >>> solutions = [Solution(...) for _ in range(10)]
            >>> evaluated = evaluator.evaluate(solutions)
            >>> evaluator.computed_evaluations()
            10
        """
        if solution_list is None:
            raise ValueError("Solution list cannot be None")
            
        if not solution_list:  # Empty list
            return solution_list
        
        # Use ThreadPoolExecutor to evaluate solutions in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all evaluation tasks
            future_to_solution = {
                executor.submit(self._evaluate_solution, solution): solution 
                for solution in solution_list
            }
            
            # Process results as they complete
            for future in as_completed(future_to_solution):
                try:
                    future.result()  # This will re-raise any exceptions from the worker
                except Exception as e:
                    self._logger.error(f"Error in worker thread: {e}")
                    raise
        
        self._computed_evaluations += len(solution_list)
        return solution_list
    
    def computed_evaluations(self) -> int:
        """Get the total number of evaluations performed.
        
        Returns:
            The total count of solutions evaluated by this instance.
        """
        return self._computed_evaluations
    
    @property
    def max_workers(self) -> int:
        """Get the maximum number of worker threads being used.
        
        Returns:
            The maximum number of worker threads.
        """
        return self._max_workers
    
    @property
    def problem(self) -> Problem[S]:
        """Get the problem instance being used for evaluation.
        
        Returns:
            The problem instance that was provided during initialization.
            
        Note:
            The returned problem instance should be treated as read-only.
            Modifying it could lead to unexpected behavior in multi-threaded context.
        """
        return self._problem