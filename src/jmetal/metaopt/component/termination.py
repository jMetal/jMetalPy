from abc import ABC, abstractmethod
from typing import Dict, Any, final


class Termination(ABC):
    """
    Interface representing classes that check the termination condition of an algorithm.
    
    This is a functional interface that defines a single method to check if a termination
    condition has been met based on the algorithm's status data.
    
    Methods:
        is_met: Checks if the termination condition has been met.
    """
    
    @abstractmethod
    def is_met(self, algorithm_status_data: Dict[str, Any]) -> bool:
        """
        Checks if the termination condition has been met based on the algorithm's status data.
        
        Args:
            algorithm_status_data: A dictionary containing the current status of the algorithm.
                                 The specific keys and values depend on the algorithm implementation.
                                 Must contain an 'EVALUATIONS' key with an integer value.
                                 
        Returns:
            bool: True if the termination condition is met, False otherwise.
            
        Raises:
            KeyError: If the 'EVALUATIONS' key is missing from algorithm_status_data.
            TypeError: If the value for 'EVALUATIONS' is not an integer.
        """
        pass


class TerminationByEvaluations(Termination):
    """
    Termination condition based on a maximum number of evaluations.
    
    This class checks if the algorithm should terminate based on the number of evaluations
    performed so far, as reported in the algorithm status data.
    
    Args:
        maximum_number_of_evaluations: The maximum number of evaluations allowed before termination.
                                     Must be a positive integer.
    
    Raises:
        ValueError: If maximum_number_of_evaluations is not a positive integer.
    """
    
    def __init__(self, maximum_number_of_evaluations: int) -> None:
        if not isinstance(maximum_number_of_evaluations, int) or maximum_number_of_evaluations <= 0:
            raise ValueError("Maximum number of evaluations must be a positive integer")
            
        self._maximum_number_of_evaluations = maximum_number_of_evaluations
    
    def is_met(self, algorithm_status_data: Dict[str, Any]) -> bool:
        """
        Check if the maximum number of evaluations has been reached.
        
        Args:
            algorithm_status_data: A dictionary containing the algorithm's status.
                                 Must contain an 'EVALUATIONS' key with an integer value.
        
        Returns:
            bool: True if the current number of evaluations is greater than or equal to
                 the maximum allowed, False otherwise.
                
        Raises:
            KeyError: If the 'EVALUATIONS' key is missing from algorithm_status_data.
            TypeError: If the value for 'EVALUATIONS' is not an integer.
        """
        if 'EVALUATIONS' not in algorithm_status_data:
            raise KeyError("Algorithm status data must contain 'EVALUATIONS' key")
            
        current_evaluations = algorithm_status_data['EVALUATIONS']
        if not isinstance(current_evaluations, int):
            raise TypeError(f"'EVALUATIONS' must be an integer, got {type(current_evaluations).__name__}")
            
        return current_evaluations >= self._maximum_number_of_evaluations
    
    @property
    def maximum_number_of_evaluations(self) -> int:
        """Get the maximum number of evaluations before termination."""
        return self._maximum_number_of_evaluations