import copy
import random
import threading
from abc import ABC, abstractmethod
from threading import Lock
from typing import Generic, List, TypeVar, Optional

import numpy as np

from jmetal.util.comparator import Comparator, DominanceComparator, SolutionAttributeComparator
from jmetal.util.density_estimator import DensityEstimator, CrowdingDistanceDensityEstimator
from jmetal.util.distance import DistanceMetric, DistanceCalculator

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
            density_estimator=CrowdingDistanceDensityEstimator(),
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
            density_estimator=CrowdingDistanceDensityEstimator(),
        )


def distance_based_subset_selection_robust(solution_list: List[S], subset_size: int, 
                                          metric: DistanceMetric = DistanceMetric.L2_SQUARED,
                                          weights: Optional[np.ndarray] = None,
                                          random_seed: Optional[int] = None,
                                          use_vectorized: bool = True) -> List[S]:
    """
    Robust distance-based subset selection with multiple metrics and improved normalization.
    
    This implementation follows the SafeBestSolutionsArchive approach:
    - For 2 objectives: Uses crowding distance selection (fast and standard)
    - For >2 objectives: Uses robust distance-based selection with smart normalization
    - Only normalizes dimensions with non-zero range to avoid division by zero
    - Supports multiple distance metrics for performance and flexibility
    
    Args:
        solution_list: List of solutions to select from
        subset_size: Number of solutions to select
        metric: Distance metric to use (default: L2_SQUARED)
        weights: Optional weights for TCHEBY_WEIGHTED metric
        random_seed: Optional seed for reproducible results
        use_vectorized: Whether to use vectorized implementation (default: True)
        
    Returns:
        List of selected solutions
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not solution_list:
        return []
        
    if subset_size <= 0:
        raise ValueError("Subset size must be positive")
        
    if subset_size >= len(solution_list):
        return solution_list[:]
    
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Get number of objectives from first solution
    num_objectives = len(solution_list[0].objectives)
    
    # For 2 objectives, use crowding distance (fast and standard in jMetal)
    if num_objectives == 2:
        return _crowding_distance_selection(solution_list, subset_size)
    
    # For >2 objectives, use robust distance-based selection
    return _robust_distance_based_selection(solution_list, subset_size, metric, weights, use_vectorized)


def _identify_valid_dimensions(objectives_matrix: np.ndarray) -> np.ndarray:
    """
    Identify dimensions with non-zero range to avoid normalization issues.
    
    Args:
        objectives_matrix: Matrix of objectives (n_solutions x n_objectives)
        
    Returns:
        Array of valid dimension indices (those with max > min)
    """
    min_vals = np.min(objectives_matrix, axis=0)
    max_vals = np.max(objectives_matrix, axis=0)
    
    # Find dimensions where max > min (non-constant objectives)
    valid_dims = np.where(max_vals > min_vals)[0]
    return valid_dims


def _robust_distance_based_selection(solution_list: List[S], subset_size: int,
                                    metric: DistanceMetric, weights: Optional[np.ndarray] = None,
                                    use_vectorized: bool = True) -> List[S]:
    """
    Robust distance-based selection for >2 objective problems.
    
    Algorithm improvements:
    1. Identify valid dimensions (non-zero range) for normalization
    2. Fallback to crowding distance if all dimensions are constant
    3. Use best solution in random objective as seed (not extremes)
    4. Apply efficient distance calculations with selectable metrics
    5. Choose between vectorized and original implementations
    
    Args:
        solution_list: List of solutions to select from
        subset_size: Number of solutions to select
        metric: Distance metric to use
        weights: Optional weights for TCHEBY_WEIGHTED metric
        use_vectorized: Whether to use vectorized implementation
    """
    # Convert solutions to matrix
    objectives_matrix = np.array([sol.objectives for sol in solution_list])
    n_solutions, n_objectives = objectives_matrix.shape
    
    # Identify valid dimensions (non-zero range)
    valid_dims = _identify_valid_dimensions(objectives_matrix)
    
    if len(valid_dims) == 0:
        # All objectives are constant -> fallback to crowding distance
        return _crowding_distance_selection(solution_list, subset_size)
    
    # Normalize only valid dimensions
    normalized_matrix = np.zeros((n_solutions, len(valid_dims)))
    min_vals = np.min(objectives_matrix[:, valid_dims], axis=0)
    max_vals = np.max(objectives_matrix[:, valid_dims], axis=0)
    ranges = max_vals - min_vals
    
    for i in range(n_solutions):
        normalized_matrix[i] = (objectives_matrix[i, valid_dims] - min_vals) / ranges
    
    # Project weights to valid dimensions if using weighted metric
    projected_weights = None
    if metric == DistanceMetric.TCHEBY_WEIGHTED:
        if weights is None:
            projected_weights = np.ones(len(valid_dims))
        else:
            if len(weights) != n_objectives:
                raise ValueError(f"Weights length ({len(weights)}) must match number of objectives ({n_objectives})")
            projected_weights = weights[valid_dims]
    
    # Seed selection: best solution in a random valid objective
    random_objective_idx = random.randint(0, len(valid_dims) - 1)
    random_objective = valid_dims[random_objective_idx]
    
    # Find best (minimum) value in the random objective
    seed_idx = np.argmin(objectives_matrix[:, random_objective])
    
    # Choose implementation based on parameter
    if use_vectorized:
        return _vectorized_subset_selection(solution_list, normalized_matrix, subset_size, 
                                           seed_idx, metric, projected_weights)
    else:
        return _original_subset_selection(solution_list, normalized_matrix, subset_size, 
                                         seed_idx, metric, projected_weights)


def _original_subset_selection(solution_list: List[S], normalized_matrix: np.ndarray,
                              subset_size: int, seed_idx: int, metric: DistanceMetric,
                              weights: Optional[np.ndarray] = None) -> List[S]:
    """
    Original (non-vectorized) implementation of subset selection for higher quality results.
    
    This function uses the original iterative approach which may be slower but can provide
    higher quality diversity selection in some cases.
    
    Args:
        solution_list: List of solutions to select from
        normalized_matrix: Normalized objective matrix
        subset_size: Number of solutions to select
        seed_idx: Index of seed solution
        metric: Distance metric to use
        weights: Optional weights for weighted metrics
        
    Returns:
        List[S]: Selected solutions
    """
    n_solutions = len(solution_list)
    
    # Initialize selection
    selected_indices = [seed_idx]
    selected_mask = np.zeros(n_solutions, dtype=bool)
    selected_mask[seed_idx] = True
    
    # Track minimum distances to selected solutions
    min_distances = np.full(n_solutions, np.inf)
    _update_min_distances_legacy(normalized_matrix, min_distances, selected_mask, seed_idx, metric, weights)
    
    # Iteratively select solutions with maximum minimum distance
    while len(selected_indices) < subset_size:
        # Find unselected solution with maximum minimum distance
        candidates = np.where(~selected_mask)[0]
        if len(candidates) == 0:
            break
            
        candidate_distances = min_distances[candidates]
        best_candidate_local_idx = np.argmax(candidate_distances)
        best_candidate_idx = candidates[best_candidate_local_idx]
        
        # Add to selection
        selected_indices.append(best_candidate_idx)
        selected_mask[best_candidate_idx] = True
        
        # Update minimum distances
        _update_min_distances_legacy(normalized_matrix, min_distances, selected_mask, best_candidate_idx, metric, weights)
    
    # Return selected solutions
    return [solution_list[i] for i in selected_indices]


def _vectorized_subset_selection(solution_list: List[S], normalized_matrix: np.ndarray,
                               subset_size: int, seed_idx: int, metric: DistanceMetric,
                               weights: Optional[np.ndarray] = None) -> List[S]:
    """
    Vectorized implementation of subset selection using optimized distance calculations.
    
    This function uses the new vectorized distance calculation methods to significantly
    improve performance when selecting subsets from large solution sets.
    
    Args:
        solution_list: List of solutions to select from
        normalized_matrix: Normalized objective matrix
        subset_size: Number of solutions to select
        seed_idx: Index of seed solution
        metric: Distance metric to use
        weights: Optional weights for weighted metrics
        
    Returns:
        List[S]: Selected solutions
    """
    n_solutions = len(solution_list)
    
    # Initialize selection with seed
    selected_indices = [seed_idx]
    
    # Iteratively select solutions with maximum minimum distance
    while len(selected_indices) < subset_size:
        # Calculate minimum distances using vectorized operations
        min_distances = DistanceCalculator.calculate_min_distances_vectorized(
            normalized_matrix, selected_indices, metric, weights
        )
        
        # Find unselected solution with maximum minimum distance
        # (selected solutions already have infinite distance)
        best_candidate_idx = np.argmax(min_distances[np.isfinite(min_distances)])
        
        # Convert to actual index (since argmax works on finite subset)
        finite_indices = np.where(np.isfinite(min_distances))[0]
        if len(finite_indices) == 0:
            # All remaining solutions are already selected
            break
            
        best_candidate_idx = finite_indices[np.argmax(min_distances[finite_indices])]
            
        # Add to selection
        selected_indices.append(best_candidate_idx)
    
    # Return selected solutions
    return [solution_list[i] for i in selected_indices]


def _update_min_distances_legacy(normalized_matrix: np.ndarray, min_distances: np.ndarray, 
                         selected_mask: np.ndarray, new_selected_idx: int,
                         metric: DistanceMetric, weights: Optional[np.ndarray] = None):
    """
    Legacy function for updating minimum distances (kept for compatibility).
    
    Note: This function is now deprecated in favor of vectorized operations.
    Use DistanceCalculator.calculate_min_distances_vectorized() instead.
    
    Args:
        normalized_matrix: Normalized objective matrix
        min_distances: Array of minimum distances to selected solutions
        selected_mask: Boolean mask of selected solutions
        new_selected_idx: Index of newly selected solution
        metric: Distance metric to use
        weights: Optional weights for weighted metrics
    """
    new_solution = normalized_matrix[new_selected_idx]
    
    for i in range(len(min_distances)):
        if selected_mask[i]:  # Skip already selected solutions
            continue
            
        # Calculate distance to new selected solution
        distance = DistanceCalculator.calculate_distance(
            normalized_matrix[i], new_solution, metric, weights
        )
        
        # Update minimum distance if this is closer
        if distance < min_distances[i]:
            min_distances[i] = distance


# Maintain backward compatibility
_update_min_distances = _update_min_distances_legacy


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


# Backward compatibility alias
def distance_based_subset_selection(solution_list: List[S], subset_size: int, distance_measure=None,
                                   metric: DistanceMetric = DistanceMetric.L2_SQUARED,
                                   weights: Optional[np.ndarray] = None,
                                   random_seed: Optional[int] = None) -> List[S]:
    """
    Backward compatibility wrapper for distance_based_subset_selection_robust.
    
    Args:
        solution_list: List of solutions to select from
        subset_size: Number of solutions to select  
        distance_measure: Deprecated parameter (ignored)
        metric: Distance metric to use
        weights: Optional weights for TCHEBY_WEIGHTED metric
        random_seed: Optional seed for reproducible results
        
    Returns:
        List of selected solutions
    """
    return distance_based_subset_selection_robust(solution_list, subset_size, metric, weights, random_seed)


class DistanceBasedArchive(BoundedArchive[S]):
    """
    Archive that maintains solutions using adaptive distance-based subset selection.
    
    This archive extends BoundedArchive to use a sophisticated selection mechanism:
    - For 2 objectives: Uses crowding distance selection
    - For >2 objectives: Uses robust distance-based subset selection with normalization
    
    The implementation follows the Java jMetal SafeBestSolutionsArchive algorithm.
    """
    
    def __init__(self, maximum_size: int, 
                 metric: DistanceMetric = DistanceMetric.L2_SQUARED,
                 weights: Optional[np.ndarray] = None,
                 random_seed: Optional[int] = None,
                 dominance_comparator=None,
                 use_vectorized: bool = True):
        """
        Initialize the distance-based archive.
        
        Args:
            maximum_size: Maximum number of solutions to maintain
            metric: Distance metric to use (default: L2_SQUARED)
            weights: Optional weights for TCHEBY_WEIGHTED metric
            random_seed: Optional seed for reproducible results
            dominance_comparator: Comparator for dominance (default: DominanceComparator)
            use_vectorized: Whether to use vectorized implementation (default: True)
        """
        if dominance_comparator is None:
            dominance_comparator = DominanceComparator()
            
        # Initialize parent with dummy comparator and density estimator
        # We'll override the selection mechanism in our custom add method
        super(DistanceBasedArchive, self).__init__(
            maximum_size=maximum_size,
            comparator=SolutionAttributeComparator("dummy", lowest_is_best=True),
            density_estimator=CrowdingDistanceDensityEstimator(),
            dominance_comparator=dominance_comparator
        )
        
        self.metric = metric
        self.weights = weights
        self.random_seed = random_seed
        self.use_vectorized = use_vectorized
        
        # Thread safety for concurrent access
        self._lock = threading.Lock()
        
    def add(self, solution: S) -> bool:
        """
        Add a solution to the archive using non-dominated sorting and distance-based selection.
        Thread-safe implementation for concurrent use.
        
        Args:
            solution: Solution to add
            
        Returns:
            True if solution was added or archive was modified, False otherwise
        """
        with self._lock:
            # First, add to non-dominated archive (this handles dominance)
            success = self.non_dominated_solution_archive.add(solution)
            
            if success and self.size() > self.maximum_size:
                # Apply distance-based subset selection
                selected_solutions = distance_based_subset_selection_robust(
                    self.solution_list, 
                    self.maximum_size,
                    self.metric,
                    self.weights,
                    self.random_seed,
                    self.use_vectorized
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
