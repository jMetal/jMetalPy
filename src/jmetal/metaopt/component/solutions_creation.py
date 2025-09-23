from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution

S = TypeVar('S', bound=Solution)

class SolutionsCreation(ABC, Generic[S]):
    """
    Abstract base class representing entities that create a list of solutions.
    
    The type parameter S represents the type of solutions to be created.
    Subclasses must implement the create() method.
    """
    
    @abstractmethod
    def create(self) -> List[S]:
        """
        Creates a list of solutions.
        
        Returns:
            A list of newly created solutions.
        """
        pass


class DefaultSolutionsCreation(SolutionsCreation[S]):
    """
    Default implementation of SolutionsCreation that creates solutions using the problem's create_solution() method.
    
    Args:
        problem: The problem instance used to create solutions.
        number_of_solutions_to_create: The number of solutions to create.
        
    Raises:
        ValueError: If problem is None or number_of_solutions_to_create is not a positive integer.
    """
    
    def __init__(self, problem: Problem[S], number_of_solutions_to_create: int):
        if problem is None:
            raise ValueError("Problem cannot be None")
        if not isinstance(number_of_solutions_to_create, int) or number_of_solutions_to_create <= 0:
            raise ValueError("Number of solutions to create must be a positive integer")
            
        self.problem = problem
        self.number_of_solutions_to_create = number_of_solutions_to_create
    
    def create(self) -> List[S]:
        """
        Creates a list of solutions by calling problem.create_solution() multiple times.
        
        Returns:
            A list of newly created solutions.
        """
        return [self.problem.create_solution() for _ in range(self. number_of_solutions_to_create)]