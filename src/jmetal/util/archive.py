import copy
import random
import numpy as np
from abc import ABC, abstractmethod
from threading import Lock
from typing import Generic, List, TypeVar

from jmetal.util.comparator import Comparator, DominanceComparator, SolutionAttributeComparator, ObjectiveComparator
from jmetal.util.density_estimator import DensityEstimator, CrowdingDistance
from jmetal.util.distance import EuclideanDistance
from jmetal.util.normalization import normalize_fronts

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


def distance_based_subset_selection(solution_list: List[S], subset_size: int, distance_measure=None) -> List[S]:
    """
    Selects a subset of solutions using distance-based selection.
    
    This function implements the algorithm from Java jMetal BestSolutionsArchive:
    1. For 2 objectives: Uses crowding distance selection
    2. For >2 objectives: Uses distance-based subset selection with normalization
    
    Args:
        solution_list: List of solutions to select from
        subset_size: Number of solutions to select
        distance_measure: Distance function to use (default: EuclideanDistance)
        
    Returns:
        List of selected solutions
        
    Raises:
        ValueError: If subset_size is larger than solution_list size or invalid parameters
    """
    if not solution_list:
        return []
        
    if subset_size <= 0:
        raise ValueError("Subset size must be positive")
        
    if subset_size >= len(solution_list):
        return solution_list[:]
        
    if distance_measure is None:
        distance_measure = EuclideanDistance()
    
    # Get number of objectives from first solution
    num_objectives = len(solution_list[0].objectives)
    
    # For 2 objectives, use crowding distance
    if num_objectives == 2:
        return _crowding_distance_selection(solution_list, subset_size)
    
    # For >2 objectives, use distance-based selection
    return _distance_based_selection(solution_list, subset_size, distance_measure)


def _crowding_distance_selection(solution_list: List[S], subset_size: int) -> List[S]:
    """
    Selects solutions using crowding distance for 2-objective problems.
    """
    # Create a temporary archive to calculate crowding distances
    archive = CrowdingDistanceArchive(len(solution_list))
    
    # Add all solutions to calculate crowding distances
    for solution in solution_list:
        archive.add(copy.deepcopy(solution))
    
    # Sort by crowding distance (descending)
    sorted_solutions = sorted(
        archive.solution_list,
        key=lambda sol: sol.attributes.get("crowding_distance", 0.0),
        reverse=True
    )
    
    return sorted_solutions[:subset_size]


def _distance_based_selection(solution_list: List[S], subset_size: int, distance_measure) -> List[S]:
    """
    Selects solutions using distance-based selection for >2 objective problems.
    
    Algorithm:
    1. Normalize objectives to [0,1] range
    2. Select random objective and sort solutions by that objective
    3. Select most extreme solutions first
    4. For remaining selections, choose solution with maximum minimum distance to selected ones
    """
    # Convert solutions to matrix for normalization
    objectives_matrix = np.array([sol.objectives for sol in solution_list])
    
    # Normalize objectives to [0,1] range using min-max normalization
    min_vals = np.min(objectives_matrix, axis=0)
    max_vals = np.max(objectives_matrix, axis=0)
    
    # Avoid division by zero for constant objectives
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    
    normalized_matrix = (objectives_matrix - min_vals) / ranges
    
    # Create list of (solution, normalized_objectives) pairs
    solution_data = [(solution_list[i], normalized_matrix[i]) for i in range(len(solution_list))]
    
    # Select random objective for initial sorting
    random_objective = random.randint(0, len(min_vals) - 1)
    
    # Sort by random objective using ObjectiveComparator
    comparator = ObjectiveComparator(random_objective)
    solution_data.sort(key=lambda x: x[0].objectives[random_objective])
    
    selected_solutions = []
    selected_normalized = []
    
    # Select first solution (best in random objective)
    selected_solutions.append(solution_data[0][0])
    selected_normalized.append(solution_data[0][1])
    
    # If we need more than one solution, select the last one too (worst in random objective)
    if subset_size > 1:
        selected_solutions.append(solution_data[-1][0])
        selected_normalized.append(solution_data[-1][1])
    
    # Select remaining solutions using maximum minimum distance criterion
    remaining_data = solution_data[1:-1]  # Exclude first and last already selected
    
    while len(selected_solutions) < subset_size and remaining_data:
        max_min_distance = -1
        best_candidate_idx = 0
        
        # Find candidate with maximum minimum distance to selected solutions
        for i, (candidate_sol, candidate_norm) in enumerate(remaining_data):
            min_distance = float('inf')
            
            # Calculate minimum distance to all selected solutions
            for selected_norm in selected_normalized:
                dist = distance_measure.get_distance(candidate_norm, selected_norm)
                min_distance = min(min_distance, dist)
            
            # Update best candidate if this one has larger minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate_idx = i
        
        # Add best candidate to selected solutions
        best_candidate = remaining_data[best_candidate_idx]
        selected_solutions.append(best_candidate[0])
        selected_normalized.append(best_candidate[1])
        
        # Remove selected candidate from remaining
        remaining_data.pop(best_candidate_idx)
    
    return selected_solutions


class BestSolutionsArchive(BoundedArchive[S]):
    """
    Archive that maintains the best solutions using distance-based subset selection.
    
    This archive extends BoundedArchive to use a sophisticated selection mechanism:
    - For 2 objectives: Uses crowding distance selection
    - For >2 objectives: Uses distance-based subset selection with normalization
    
    The implementation follows the Java jMetal BestSolutionsArchive algorithm.
    """
    
    def __init__(self, maximum_size: int, distance_measure=None, dominance_comparator=None):
        """
        Initialize the best solutions archive.
        
        Args:
            maximum_size: Maximum number of solutions to maintain
            distance_measure: Distance function for subset selection (default: EuclideanDistance)
            dominance_comparator: Comparator for dominance (default: DominanceComparator)
        """
        if distance_measure is None:
            distance_measure = EuclideanDistance()
        if dominance_comparator is None:
            dominance_comparator = DominanceComparator()
            
        # Initialize parent with dummy comparator and density estimator
        # We'll override the selection mechanism in our custom add method
        super(BestSolutionsArchive, self).__init__(
            maximum_size=maximum_size,
            comparator=SolutionAttributeComparator("dummy", lowest_is_best=True),
            density_estimator=CrowdingDistance(),
            dominance_comparator=dominance_comparator
        )
        
        self.distance_measure = distance_measure
        
    def add(self, solution: S) -> bool:
        """
        Add a solution to the archive using non-dominated sorting and distance-based selection.
        
        Args:
            solution: Solution to add
            
        Returns:
            True if solution was added or archive was modified, False otherwise
        """
        # First, add to non-dominated archive (this handles dominance)
        success = self.non_dominated_solution_archive.add(solution)
        
        if success and self.size() > self.maximum_size:
            # Apply distance-based subset selection
            selected_solutions = distance_based_subset_selection(
                self.solution_list, 
                self.maximum_size, 
                self.distance_measure
            )
            
            # Update solution list with selected solutions
            # IMPORTANT: Clear and extend to maintain reference from parent class
            self.solution_list.clear()
            self.solution_list.extend(selected_solutions)
        
        return success
    
    def compute_density_estimator(self):
        """
        Override parent method since we use distance-based selection instead.
        This method is called by parent class but we don't need density estimation.
        """
        # Do nothing - we use distance-based selection instead of density estimation
        pass
