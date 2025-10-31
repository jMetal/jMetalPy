"""
This module defines the core operator interfaces for optimization in JMetalPy.

Operators are the building blocks of evolutionary algorithms, including mutation,
crossover, and selection operators. These operators are used to create variation
in the population and guide the search towards better solutions.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Generic, List, TypeVar, Callable, Any, TypeVar, Type

from jmetal.core.solution import Solution

# Type variables for operator implementation
S = TypeVar("S", bound=Solution)  # Type of the source solutions
R = TypeVar("R", bound=Solution)  # Type of the resulting solutions


class Operator(Generic[S, R], ABC):
    """Abstract base class for all operators in JMetalPy.
    
    An operator transforms one or more input solutions into one or more output solutions.
    This is the base class for all variation operators like mutation, crossover, and selection.
    
    Subclasses must implement the execute() method to define the operator's behavior
    and get_name() to provide a string identifier.
    """

    @abstractmethod
    def execute(self, source: S) -> R:
        """Execute the operator on the source solution(s).
        
        Args:
            source: The input solution or list of solutions to be transformed.
            
        Returns:
            The transformed solution or list of solutions.
            
        Note:
            The exact type and number of input and output solutions depend on the
            specific operator implementation.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            A string identifier for the operator (e.g., 'SBX', 'PolynomialMutation').
        """
        pass


def check_valid_probability_value(func: Callable) -> Callable:
    """Decorator to validate that a probability value is between 0 and 1.
    
    This decorator is used to ensure that probability values passed to operator
    constructors are within the valid range [0.0, 1.0].
    
    Args:
        func: The function to be decorated (typically __init__ of an operator).
        
    Returns:
        The wrapped function with probability validation.
        
    Raises:
        ValueError: If the probability is outside the [0.0, 1.0] range.
    """
    @wraps(func)
    def func_wrapper(self, probability: float) -> Any:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability must be between 0.0 and 1.0, but got {probability}"
            )
        return func(self, probability)
    return func_wrapper


class Mutation(Operator[S, S], ABC):
    """Abstract base class for mutation operators.
    
    Mutation operators introduce small random changes to a solution to maintain
    diversity in the population. Each solution has a probability of being mutated.
    
    Attributes:
        probability: The probability that a solution will be mutated (0.0 to 1.0).
    """

    @check_valid_probability_value
    def __init__(self, probability: float):
        """Initialize the mutation operator with a given probability.
        
        Args:
            probability: The probability of applying the mutation to a solution.
                        Must be between 0.0 and 1.0.
        """
        self.probability = probability


class Crossover(Operator[List[S], List[R]], ABC):
    """Abstract base class for crossover operators.
    
    Crossover operators combine genetic information from two or more parent solutions
    to produce new offspring solutions. This mimics biological recombination.
    
    Attributes:
        probability: The probability of applying the crossover to a set of parents.
    """

    @check_valid_probability_value
    def __init__(self, probability: float):
        """Initialize the crossover operator with a given probability.
        
        Args:
            probability: The probability of applying the crossover to a set of parents.
                        Must be between 0.0 and 1.0.
        """
        self.probability = probability

    @abstractmethod
    def get_number_of_parents(self) -> int:
        """Get the number of parent solutions required by this crossover.
        
        Returns:
            The number of parent solutions needed (typically 2 for most crossovers).
        """
        pass

    @abstractmethod
    def get_number_of_children(self) -> int:
        """Get the number of offspring solutions produced by this crossover.
        
        Returns:
            The number of offspring solutions generated (often equal to the number of parents).
        """
        pass


class Selection(Operator[List[S], R], ABC):
    """Abstract base class for selection operators.
    
    Selection operators are used to choose solutions from a population for reproduction.
    Different selection strategies can affect the exploration/exploitation balance.
    """

    def __init__(self):
        """Initialize the selection operator."""
        pass
