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
    
    Subclasses should call `super().__init__(problem)` in their `__init__` method
    to properly initialize the problem instance.
    
    Args:
        problem: The problem instance to be evaluated against.
    """
    
    def __init__(self, problem: Problem[S]) -> None:
        """Initialize the evaluator with a problem instance.
        
        Args:
            problem: The problem instance to evaluate solutions against.
            
        Raises:
            ValueError: If the problem is None.
        """
        if problem is None:
            raise ValueError("Problem instance cannot be None")
        self._problem = problem
    
    @property
    def problem(self) -> Problem[S]:
        """
        Get the problem being solved.
        
        Returns:
            The problem instance.
            
        Note:
            The returned problem instance should be treated as read-only.
            Modifying it could lead to unexpected behavior.
        """
        return self._problem
    
    @abstractmethod
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """
        Evaluate a list of solutions.
        
        Args:
            solution_list: List of solutions to be evaluated.
            
        Returns:
            List of evaluated solutions.
            
        Note:
            The input list is modified in-place and also returned for convenience.
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
            
        Note:
            The problem validation is handled by the parent class.
        """
        super().__init__(problem)
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
