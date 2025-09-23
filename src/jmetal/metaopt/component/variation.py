"""
This module contains the Variation interface and its implementations for evolutionary algorithms.
"""
import math
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Dict, Any
from jmetal.core.solution import Solution
from jmetal.core.operator import Crossover, Mutation

S = TypeVar('S', bound=Solution)


class Variation(Generic[S], ABC):
    """
    Interface representing variation operations in evolutionary algorithms.
    
    This interface defines methods for creating new solutions by combining or modifying
    existing ones through operations like crossover and mutation.
    
    Type Parameters:
        S: Type of the solutions to be varied. Must extend Solution.
    """

    @abstractmethod
    def variate(self, solution_list: List[S], mating_pool: List[S]) -> List[S]:
        """Perform the variation operation on the given solutions.
        
        Args:
            solution_list: The current population of solutions. This is typically
                         used to maintain context but should not be modified.
            mating_pool: The solutions selected for variation. The number of solutions
                       should be at least equal to mating_pool_size().
                       
        Returns:
            List[S]: A new list containing the offspring solutions.
            
        Raises:
            ValueError: If the number of solutions in mating_pool is less than
                      mating_pool_size() or if any parameter is invalid.
        """
        pass

    @abstractmethod
    def mating_pool_size(self) -> int:
        """Get the number of solutions required for the variation operation.
        
        This method returns the minimum number of solutions needed from the mating pool
        to perform the variation operation. For example:
        - A standard two-parent crossover would return 2
        - A mutation operator would typically return 1
        - A three-parent crossover would return 3
        
        Returns:
            int: The number of solutions needed for variation.
        """
        pass

    @abstractmethod
    def offspring_population_size(self) -> int:
        """Get the number of solutions produced by the variation operation.
        
        This method returns the number of offspring solutions that will be generated
        from a single application of the variation operator. For example:
        - A standard crossover might produce 2 offspring from 2 parents
        - A mutation operator would typically return 1
        - Some operators might produce more offspring than parents
        
        Returns:
            int: The number of solutions produced by each variation operation.
        """
        pass

    def get_attributes(self) -> dict:
        """Get a dictionary containing the operator's attributes.
        
        This method provides metadata about the variation operator, which can be useful
        for logging, debugging, or adaptive algorithm control.
        
        Returns:
            dict: A dictionary containing the operator's name and configuration parameters.
            
        Example:
            >>> variation = SomeVariationOperator()
            >>> attrs = variation.get_attributes()
            >>> print(attrs['name'])
            'SomeVariationOperator'
        """
        return {
            'name': self.__class__.__name__,
            'mating_pool_size': self.mating_pool_size(),
            'offspring_population_size': self.offspring_population_size()
        }

    def __str__(self) -> str:
        """Return a string representation of the variation operator.
        
        The string representation includes the operator's name and its configuration
        parameters for easy identification and debugging.
        
        Returns:
            str: A string describing the variation operator.
            
        Example:
            >>> variation = SomeVariationOperator()
            >>> print(variation)
            'SomeVariationOperator(mating_pool=2, offspring=2)'
        """
        attrs = self.get_attributes()
        return (f"{attrs['name']} "
                f"(mating_pool={attrs['mating_pool_size']}, "
                f"offspring={attrs['offspring_population_size']})")


class CrossoverAndMutationVariation(Variation[S]):
    """
    A variation operator that combines crossover and mutation operations.
    
    This operator first applies crossover to create offspring solutions and then
    applies mutation to each of them. The number of parents required is determined
    by the crossover operator.
    
    Args:
        crossover: The crossover operator to use.
        mutation: The mutation operator to apply to each offspring.
        offspring_population_size: The desired number of offspring solutions.
        
    Raises:
        ValueError: If any argument is None or if the offspring population size is not positive.
    """

    def __init__(
            self,
            crossover: Crossover[S, S],
            mutation: Mutation[S],
            offspring_population_size: int
    ) -> None:
        """Initialize the variation operator with crossover, mutation, and population size.
        
        Args:
            crossover: The crossover operator to use for creating offspring.
            mutation: The mutation operator to apply to each offspring.
            offspring_population_size: The number of offspring to generate.
            
        Raises:
            ValueError: If any argument is invalid.
        """
        if not crossover:
            raise ValueError("Crossover operator cannot be None")
        if not mutation:
            raise ValueError("Mutation operator cannot be None")
        if offspring_population_size <= 0:
            raise ValueError("Offspring population size must be positive")

        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.offspring_population_size = offspring_population_size

        # Calculate the required mating pool size
        self._mating_pool_size = crossover.get_number_of_parents() * math.ceil(
            offspring_population_size / crossover.get_number_of_children()
        )

    def variate(self, solution_list: List[S], mating_pool: List[S]) -> List[S]:
        """Perform the variation operation by applying crossover and mutation.
        
        This method creates offspring by applying crossover and then mutation to the mating pool.
        It ensures the exact number of requested offspring is produced.
        
        Args:
            solution_list: The current population of solutions (unused in this implementation).
            mating_pool: The solutions selected for variation.
            
        Returns:
            List[S]: A new list containing the offspring solutions.
            
        Raises:
            ValueError: If the number of parents is not compatible with the crossover operator
                      or if the offspring population size cannot be achieved.
        """
        if not mating_pool:
            return []

        number_of_parents = self.crossover.get_number_of_parents()
        self._check_number_of_parents(mating_pool, number_of_parents)

        offspring_population = []
        num_parents = len(mating_pool)
        i = 0

        # Continue until we have enough offspring or run out of parents
        while (len(offspring_population) < self.offspring_population_size and
               i + number_of_parents <= num_parents):
            
            # Get the next set of parents
            parents = mating_pool[i:i + number_of_parents]
            i += number_of_parents
            
            # Apply crossover to get new offspring
            offspring = self.crossover.execute(parents)
            if not offspring:  # Handle case where no offspring are produced
                continue
                
            # Apply mutation to each offspring
            for solution in offspring:
                if len(offspring_population) >= self.offspring_population_size:
                    break
                self.mutation.execute(solution)
                offspring_population.append(solution)

        if len(offspring_population) != self.offspring_population_size:
            raise ValueError(
                f"Could not generate the required number of offspring. "
                f"Generated {len(offspring_population)} out of {self.offspring_population_size}"
            )

        return offspring_population

    def _check_number_of_parents(self, population: List[S], required_parents: int) -> None:
        """Check if the number of parents is compatible with the crossover operator.
        
        Args:
            population: The list of parent solutions.
            required_parents: The number of parents required by the crossover operator.
            
        Raises:
            ValueError: If the number of parents is not compatible.
        """
        if len(population) < required_parents or len(population) % required_parents != 0:
            raise ValueError(
                f"Wrong number of parents: the population size ({len(population)}) "
                f"must be a multiple of {required_parents}"
            )

    def mating_pool_size(self) -> int:
        """Get the number of solutions required for the variation operation.
        
        Returns:
            int: The number of solutions needed for variation.
        """
        return self._mating_pool_size

    def offspring_population_size(self) -> int:
        """Get the number of solutions produced by the variation operation.
        
        Returns:
            int: The number of solutions produced.
        """
        return self.offspring_population_size

    def get_attributes(self) -> dict:
        """Get a dictionary containing the operator's attributes.
        
        Returns:
            dict: A dictionary containing the operator's name and configuration parameters.
        """
        attrs = super().get_attributes()
        attrs.update({
            'crossover': str(self.crossover),
            'mutation': str(self.mutation)
        })
        return attrs
