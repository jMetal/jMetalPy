import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Callable, Optional
from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator, DominanceComparator

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


class NaryTournamentSelection(Selection[S]):
    """
    N-ary tournament selection operator that selects solutions through tournament competitions.
    
    This implementation selects solutions by running tournaments among 'tournament_size' 
    randomly chosen solutions from the population. The best solution from each tournament 
    (determined by the provided comparator) is selected.
    
    Args:
        number_of_solutions_to_select: The number of solutions to select. Must be positive.
        tournament_size: The number of solutions that participate in each tournament.
                        Must be between 2 and 10 (inclusive).
        comparator: The comparator used to determine the winner of each tournament.
                   Defaults to a comparator that prefers solutions with better fitness.
                   
    Raises:
        ValueError: If number_of_solutions_to_select is not a positive integer.
        ValueError: If tournament_size is not between 2 and 10 (inclusive).
    """
    
    def __init__(
        self, 
        number_of_solutions_to_select: int, 
        tournament_size: int = 2,
        comparator: Optional[Comparator] = None
    ):
        if not isinstance(number_of_solutions_to_select, int) or number_of_solutions_to_select <= 0:
            raise ValueError("Number of solutions to select must be a positive integer")
            
        if not (2 <= tournament_size <= 10):
            raise ValueError("Tournament size must be between 2 and 10 (inclusive)")
            
        self.number_of_solutions_to_select = number_of_solutions_to_select
        self.tournament_size = tournament_size
        self.comparator = comparator if comparator is not None else DominanceComparator()
    
    def select(self, solution_list: List[S]) -> List[S]:
        """
        Select solutions using tournament selection.
        
        Args:
            solution_list: The list of solutions to select from. Should not be modified.
            
        Returns:
            List[S]: A new list containing the selected solutions.
            
        Raises:
            ValueError: If solution_list is empty or None.
            ValueError: If the tournament size is larger than the solution list size.
        """
        if not solution_list:
            raise ValueError("Cannot select from an empty or None solution list")
            
        if self.tournament_size > len(solution_list):
            raise ValueError(
                f"Tournament size ({self.tournament_size}) cannot be larger than "
                f"the number of available solutions ({len(solution_list)})"
            )
        
        selected_solutions = []
        
        for _ in range(self.number_of_solutions_to_select):
            # Select tournament participants randomly without replacement
            participants = random.sample(solution_list, self.tournament_size)
            
            # Find the winner (the best solution according to the comparator)
            winner = participants[0]
            for participant in participants[1:]:
                if self.comparator.compare(participant, winner) < 0:
                    winner = participant
            
            selected_solutions.append(winner)
        
        return selected_solutions
    
    def __str__(self) -> str:
        return (
            f"NTournamentSelection("
            f"number_of_solutions_to_select={self.number_of_solutions_to_select}, "
            f"tournament_size={self.tournament_size}"
            f")"
        )
