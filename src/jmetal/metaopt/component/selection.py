import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
from jmetal.core.solution import Solution

S = TypeVar('S', bound=Solution)


class Selection(ABC, Generic[S]):
    """
    Interface representing selection operations in evolutionary algorithms.
    
    This is a functional interface that defines a single method to select solutions
    from a given population.
    
    Type Parameters:
        S: Type of the solutions to be selected. Must extend Solution.
    
    Methods:
        select: Performs the selection operation on a list of solutions.
    """
    
    @abstractmethod
    def select(self, solution_list: List[S]) -> List[S]:
        """
        Selects solutions from the given list.
        
        Args:
            solution_list: The list of solutions to select from.
                          Should not be modified by this method.
                          
        Returns:
            List[S]: A new list containing the selected solutions.
            
        Raises:
            ValueError: If solution_list is None or empty.
        """
        pass


class RandomSelection(Selection[S]):
    """
    Random selection operator that selects solutions randomly from a given list.
    
    This implementation allows selection with or without replacement and can be configured
    to select a specific number of solutions.
    
    Args:
        number_of_solutions_to_select: The number of solutions to select. Must be positive.
        with_replacement: If True, solutions can be selected multiple times (with replacement).
                         If False, each solution can be selected only once.
                         
    Raises:
        ValueError: If number_of_solutions_to_select is not a positive integer.
    """
    
    def __init__(self, number_of_solutions_to_select: int, with_replacement: bool = False):
        if not isinstance(number_of_solutions_to_select, int) or number_of_solutions_to_select <= 0:
            raise ValueError("Number of solutions to select must be a positive integer")
            
        self.number_of_solutions_to_select = number_of_solutions_to_select
        self.with_replacement = with_replacement
    
    def select(self, solution_list: List[S]) -> List[S]:
        """
        Select solutions randomly from the given list.
        
        Args:
            solution_list: The list of solutions to select from. Should not be modified.
            
        Returns:
            List[S]: A new list containing the selected solutions.
            
        Raises:
            ValueError: If solution_list is empty or None.
            ValueError: If trying to select more solutions than available without replacement.
        """
        if not solution_list:
            raise ValueError("Cannot select from an empty or None solution list")
            
        if not self.with_replacement and self.number_of_solutions_to_select > len(solution_list):
            raise ValueError(
                f"Cannot select {self.number_of_solutions_to_select} solutions "
                f"without replacement from a list of {len(solution_list)} solutions"
            )
        
        if self.with_replacement:
            # Select with replacement (same solution can be selected multiple times)
            return random.choices(solution_list, k=self.number_of_solutions_to_select)
        else:
            # Select without replacement (no duplicates)
            return random.sample(solution_list, k=self.number_of_solutions_to_select)
    
    def __str__(self) -> str:
        return (
            f"RandomSelection("
            f"number_of_solutions_to_select={self.number_of_solutions_to_select}, "
            f"with_replacement={self.with_replacement}"
            f")"
        )
