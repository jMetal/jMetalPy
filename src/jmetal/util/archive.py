import copy
import random
from abc import ABC, abstractmethod
from threading import Lock
from typing import Generic, List, TypeVar

from jmetal.util.comparator import Comparator, DominanceComparator, SolutionAttributeComparator
from jmetal.util.density_estimator import DensityEstimator, CrowdingDistance

S = TypeVar('S')

"""
.. module:: archive
   :platform: Unix, Windows
   :synopsis: Archive implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Archive(Generic[S], ABC):
    def __init__(self):
        self.solution_list: List[S] = []

    @abstractmethod
    def add(self, solution: S) -> bool:
        pass

    def get(self, index: int) -> S:
        return self.solution_list[index]

    def size(self) -> int:
        return len(self.solution_list)

    def get_name(self) -> str:
        return self.__class__.__name__


class BoundedArchive(Archive[S]):
    def __init__(self, maximum_size: int, comparator: Comparator[S] = None, density_estimator: DensityEstimator = None,
                 dominance_comparator: Comparator[S] = DominanceComparator()):
        super(BoundedArchive, self).__init__()
        self.maximum_size = maximum_size
        self.comparator = comparator
        self.density_estimator = density_estimator
        self.non_dominated_solution_archive = NonDominatedSolutionsArchive(dominance_comparator=dominance_comparator)
        self.solution_list = self.non_dominated_solution_archive.solution_list

    def compute_density_estimator(self):
        self.density_estimator.compute_density_estimator(self.solution_list)

    def add(self, solution: S) -> bool:
        success = self.non_dominated_solution_archive.add(solution)

        if success:
            if self.size() > self.maximum_size:
                self.compute_density_estimator()
                worst_solution, index_to_remove = self.__find_worst_solution(self.solution_list)
                self.solution_list.pop(index_to_remove)

        return success

    def __find_worst_solution(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is None")
        elif len(solution_list) == 0:
            raise Exception("The solution list is empty")

        worst_solution = solution_list[0]
        index_to_remove = 0

        for solution_index, solution in enumerate(solution_list[1:]):
            if self.comparator.compare(worst_solution, solution) < 0:
                worst_solution = solution
                index_to_remove = solution_index + 1

        return worst_solution, index_to_remove


class NonDominatedSolutionsArchive(Archive[S]):
    """
    Archive that maintains only non-dominated solutions using Pareto dominance.
    
    This implementation efficiently manages a collection of solutions by:
    - Adding new non-dominated solutions
    - Removing solutions dominated by new ones
    - Preventing duplicate solutions based on objectives
    
    Time Complexity: O(n) per insertion, where n is archive size
    Space Complexity: O(n) for storing solutions
    """
    
    def __init__(self, dominance_comparator: Comparator = DominanceComparator(), 
                 objective_tolerance: float = 1e-10):
        """
        Initialize the non-dominated solutions archive.
        
        Args:
            dominance_comparator: Comparator to determine dominance relationships
            objective_tolerance: Tolerance for comparing floating-point objectives
        """
        super(NonDominatedSolutionsArchive, self).__init__()
        self.comparator = dominance_comparator
        self.objective_tolerance = objective_tolerance

    def _objectives_equal(self, solution1: S, solution2: S) -> bool:
        """
        Check if two solutions have equal objectives within tolerance.
        
        Args:
            solution1: First solution to compare
            solution2: Second solution to compare
            
        Returns:
            True if objectives are equal within tolerance, False otherwise
        """
        if len(solution1.objectives) != len(solution2.objectives):
            return False
            
        for obj1, obj2 in zip(solution1.objectives, solution2.objectives):
            if abs(obj1 - obj2) > self.objective_tolerance:
                return False
        return True

    def add(self, solution: S) -> bool:
        """
        Add a solution to the archive if it's non-dominated and not duplicate.
        
        This method efficiently handles the archive by:
        1. Checking if the new solution is dominated by existing ones
        2. Removing existing solutions dominated by the new one
        3. Preventing addition of duplicate solutions
        
        Args:
            solution: Solution to add to the archive
            
        Returns:
            True if solution was added, False if rejected (dominated or duplicate)
            
        Time Complexity: O(n) where n is the number of solutions in archive
        """
        # Handle empty archive case
        if not self.solution_list:
            self.solution_list.append(solution)
            return True

        # Check dominance against all existing solutions
        remaining_solutions = []
        
        for current_solution in self.solution_list:
            dominance_flag = self.comparator.compare(solution, current_solution)
            
            if dominance_flag == 1:
                # New solution is dominated by current -> reject immediately
                return False
            elif dominance_flag == 0:
                # No dominance relationship -> check for duplicates
                if self._objectives_equal(solution, current_solution):
                    # Duplicate found -> reject
                    return False
                # Keep the current solution as it's not dominated
                remaining_solutions.append(current_solution)
            # dominance_flag == -1: current solution is dominated -> don't add to remaining

        # Update archive with non-dominated solutions and add new one
        # IMPORTANT: Modify list in-place to maintain references from BoundedArchive
        self.solution_list.clear()
        self.solution_list.extend(remaining_solutions)
        self.solution_list.append(solution)
        return True


class CrowdingDistanceArchive(BoundedArchive[S]):
    def __init__(self, maximum_size: int, dominance_comparator=DominanceComparator()):
        super(CrowdingDistanceArchive, self).__init__(
            maximum_size=maximum_size,
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            dominance_comparator=dominance_comparator,
            density_estimator=CrowdingDistance(),
        )


class ArchiveWithReferencePoint(BoundedArchive[S]):
    def __init__(
            self,
            maximum_size: int,
            reference_point: List[float],
            comparator: Comparator[S],
            density_estimator: DensityEstimator,
    ):
        super(ArchiveWithReferencePoint, self).__init__(maximum_size, comparator, density_estimator)
        self.__reference_point = reference_point
        self.__comparator = comparator
        self.__density_estimator = density_estimator
        self.lock = Lock()

    def add(self, solution: S) -> bool:
        with self.lock:
            dominated_solution = None

            if self.__dominance_test(solution.objectives, self.__reference_point) == 0:
                if len(self.solution_list) == 0:
                    result = True
                else:
                    if random.uniform(0.0, 1.0) < 0.05:
                        result = True
                        dominated_solution = solution
                    else:
                        result = False
            else:
                result = True

            if result:
                result = super(ArchiveWithReferencePoint, self).add(solution)

            if result and dominated_solution is not None and len(self.solution_list) > 1:
                if dominated_solution in self.solution_list:
                    self.solution_list.remove(dominated_solution)

            if result and len(self.solution_list) > self.maximum_size:
                self.compute_density_estimator()

        return result

    def filter(self):
        # In case of having at least a solution which is non-dominated with the reference point, filter it
        if len(self.solution_list) > 1:
            self.solution_list[:] = [
                sol for sol in self.solution_list if self.__dominance_test(sol.objectives, self.__reference_point) != 0
            ]

    def update_reference_point(self, new_reference_point) -> None:
        with self.lock:
            self.__reference_point = new_reference_point

            first_solution = copy.deepcopy(self.solution_list[0])
            self.filter()

            if len(self.solution_list) == 0:
                self.solution_list.append(first_solution)

    def get_reference_point(self) -> List[float]:
        with self.lock:
            return self.__reference_point

    def __dominance_test(self, vector1: List[float], vector2: List[float]) -> int:
        best_is_one = 0
        best_is_two = 0

        for value1, value2 in zip(vector1, vector2):
            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value2 < value1:
                    best_is_two = 1

        if best_is_one > best_is_two:
            result = -1
        elif best_is_two > best_is_one:
            result = 1
        else:
            result = 0

        return result


class CrowdingDistanceArchiveWithReferencePoint(ArchiveWithReferencePoint[S]):
    def __init__(self, maximum_size: int, reference_point: List[float]):
        super(CrowdingDistanceArchiveWithReferencePoint, self).__init__(
            maximum_size=maximum_size,
            reference_point=reference_point,
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            density_estimator=CrowdingDistance(),
        )
