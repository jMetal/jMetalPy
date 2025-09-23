"""
This module contains the Replacement interface for evolutionary algorithms.
"""
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
from jmetal.core.solution import Solution

S = TypeVar('S', bound=Solution)


class Replacement(Generic[S], ABC):
    """
    Interface representing replacement operations in evolutionary algorithms.
    
    This interface defines the contract for replacement strategies that determine how
    to create a new population from the current population and the offspring population.
    
    Type Parameters:
        S: Type of the solutions to be replaced. Must extend Solution.
    """

    @abstractmethod
    def replace(self, current_list: List[S], offspring_list: List[S]) -> List[S]:
        """Combine the current population and offspring into a new population.
        
        This method determines which solutions from the current population and offspring
        should be kept in the new population according to the replacement strategy.
        
        Args:
            current_list: The current population of solutions.
            offspring_list: The newly generated offspring solutions.
            
        Returns:
            List[S]: A new list containing the selected solutions for the next generation.
            
        Raises:
            ValueError: If either input list is None or if the replacement strategy
                      cannot be applied with the given inputs.
        """
        pass

    def get_attributes(self) -> dict:
        """Get a dictionary containing the operator's attributes.
        
        Returns:
            dict: A dictionary containing the operator's name and configuration parameters.
        """
        return {
            'name': self.__class__.__name__
        }

    def __str__(self) -> str:
        """Return a string representation of the replacement operator.
        
        Returns:
            str: A string describing the replacement operator.
        """
        return f"{self.get_attributes()['name']}()"
