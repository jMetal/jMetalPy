"""
This module provides quality indicators for evaluating multi-objective optimization results.

Quality indicators are essential for comparing and assessing the performance of
multi-objective optimization algorithms. This module includes various indicators
such as Generational Distance (GD), Inverted Generational Distance (IGD), and
Hypervolume (HV).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Dict, Any

import moocore
import numpy as np
from scipy import spatial


class QualityIndicator(ABC):
    """Abstract base class for all quality indicators.
    
    Quality indicators are used to assess the performance of multi-objective
    optimization algorithms by quantifying different aspects of the obtained
    solution sets, such as convergence, diversity, and spread.
    
    Args:
        is_minimization: If True, lower indicator values indicate better quality.
                        If False, higher values are better.
    """
    
    def __init__(self, is_minimization: bool):
        """Initialize the quality indicator with optimization direction."""
        self.is_minimization = is_minimization

    @abstractmethod
    def compute(self, solutions: np.ndarray) -> float:
        """Compute the quality indicator value for the given solutions.
        
        Args:
            solutions: A 2D numpy array of shape (m, n) where m is the number of
                     solutions and n is the number of objectives.
                     
        Returns:
            The computed quality indicator value.
            
        Raises:
            ValueError: If the input is invalid (e.g., empty array, wrong dimensions).
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the full name of the quality indicator.
        
        Returns:
            A string representing the full name of the indicator.
        """
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        """Get a short name or abbreviation for the quality indicator.
        
        Returns:
            A short string abbreviation for the indicator (e.g., 'GD', 'IGD', 'HV').
        """
        pass


class FitnessValue(QualityIndicator):
    """A simple fitness-based quality indicator.
    
    This indicator computes the average objective value of the solutions,
    which is useful for single-objective optimization or when a scalarization
    of multiple objectives is needed.
    
    Note:
        For multi-objective optimization, this indicator may not provide
        meaningful comparisons between solution sets.
    """
    
    def __init__(self, is_minimization: bool = True):
        """Initialize the fitness value indicator.
        
        Args:
            is_minimization: If True, lower fitness values are better.
        """
        super(FitnessValue, self).__init__(is_minimization=is_minimization)

    def compute(self, solutions: np.ndarray) -> float:
        """Compute the average fitness value of the solutions.
        
        Args:
            solutions: Array of solution objects with 'objectives' attribute.
            
        Returns:
            The mean of the objective values, with sign adjusted based on
            the optimization direction.
        """
        if self.is_minimization:
            mean = np.mean([s.objectives for s in solutions])
        else:
            mean = -np.mean([s.objectives for s in solutions])

        return mean

    def get_name(self) -> str:
        return "Fitness Value"

    def get_short_name(self) -> str:
        return "Fitness"


class GenerationalDistance(QualityIndicator):
    """Generational Distance (GD) quality indicator.
    
    GD measures the average distance from each solution in the obtained front to the
    nearest solution in the reference front. Lower values indicate better convergence
    to the reference front.
    
    Note:
        - GD = 0 indicates that all solutions are in the reference front.
        - Lower values indicate better convergence.
        
    Reference:
        Van Veldhuizen, D.A., Lamont, G.B. (1998): Multiobjective Evolutionary Algorithm 
        Research: A History and Analysis. Technical Report TR-98-03, Dept. Elec. Comput. Eng., 
        Air Force Inst. Technol.
    """
    
    def __init__(self, reference_front: np.ndarray = None):
        """Initialize the Generational Distance indicator.
        
        Args:
            reference_front: The reference front (Pareto front or approximation).
                           Each row represents a solution in the objective space.
        """
        super(GenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, solutions: np.ndarray) -> float:
        """Compute the Generational Distance value.
        
        Args:
            solutions: A 2D numpy array of shape (m, n) where m is the number of
                     solutions and n is the number of objectives.
                     
        Returns:
            The Generational Distance value (lower is better).
            
        Raises:
            ValueError: If the reference front is not set or if the input is invalid.
        """
        if self.reference_front is None:
            raise ValueError("Reference front must be set before computing GD")
        if solutions.size == 0:
            raise ValueError("Solutions array cannot be empty")
            
        # Compute pairwise distances between solutions and reference front
        distances = spatial.distance.cdist(solutions, self.reference_front)
        
        # For each solution, find the minimum distance to the reference front
        min_distances = np.min(distances, axis=1)
        
        # GD is the average of these minimum distances
        return float(np.mean(min_distances))

    def get_short_name(self) -> str:
        """Get the short name of the indicator.
        
        Returns:
            'GD' for Generational Distance.
        """
        return "GD"

    def get_name(self) -> str:
        """Get the full name of the indicator.
        
        Returns:
            'Generational Distance'.
        """
        return "Generational Distance"


class InvertedGenerationalDistance(QualityIndicator):
    """
    Inverted Generational Distance (IGD) quality indicator.
    
    IGD measures the average distance from each point in the reference front to the closest point 
    in the solution front. Lower values indicate better performance.
    
    Reference:
    Van Veldhuizen, D.A., Lamont, G.B. (1998): Multiobjective Evolutionary Algorithm Research: 
    A History and Analysis. Technical Report TR-98-03, Dept. Elec. Comput. Eng., 
    Air Force Inst. Technol.
    """
    
    def __init__(self, reference_front: np.array = None, pow: float = 2.0):
        """
        Initialize the IGD indicator.
        
        Args:
            reference_front: Reference front matrix (each row is a solution)
            pow: Power parameter for the Lp-norm (default: 2.0 for Euclidean distance)
        
        Raises:
            ValueError: If reference_front is None or empty
        """
        super(InvertedGenerationalDistance, self).__init__(is_minimization=True)
        if reference_front is None:
            raise ValueError("Reference front cannot be None")
        if len(reference_front) == 0:
            raise ValueError("Reference front cannot be empty")
        
        self.reference_front = reference_front
        self.pow = pow

    def compute(self, solutions: np.array) -> float:
        """
        Compute the IGD indicator value.
        
        Args:
            solutions: Solution front matrix (each row is a solution)
            
        Returns:
            The IGD indicator value
            
        Raises:
            ValueError: If solutions is empty or has different dimensionality than reference front
        """
        if solutions is None or len(solutions) == 0:
            raise ValueError("Solutions front cannot be None or empty")
        
        if solutions.shape[1] != self.reference_front.shape[1]:
            raise ValueError("Solutions and reference front must have the same number of objectives")

        # Compute distances from each reference point to closest solution point
        distances = spatial.distance.cdist(self.reference_front, solutions)
        min_distances = np.min(distances, axis=1)
        
        # Apply jMetal's IGD formula: IGD = (Σ(d^pow))^(1/pow) / N
        # where d is the minimum distance from each reference point to the solution front
        # This implementation matches exactly with jMetal's invertedGenerationalDistance method
        powered_distances = np.power(min_distances, self.pow)
        sum_root = np.power(np.sum(powered_distances), 1.0 / self.pow)
        return sum_root / len(self.reference_front)

    def get_short_name(self) -> str:
        return "IGD"

    def get_name(self) -> str:
        return "Inverted Generational Distance"


class InvertedGenerationalDistancePlus(QualityIndicator):
    """
    Inverted Generational Distance Plus (IGD+) quality indicator.
    
    IGD+ improves upon the standard IGD by using dominance-based distance calculation, making it more
    suitable for cases where the reference front may not be optimal.
    
    Reference:
    Ishibuchi et al. (2015): "A Study on Performance Evaluation Ability of a Modified
    Inverted Generational Distance Indicator", GECCO 2015
    """
    
    def __init__(self, reference_front: np.array = None):
        """
        Initialize the IGD+ indicator.
        
        Args:
            reference_front: Reference front matrix (each row is a solution)
        
        Raises:
            ValueError: If reference_front is None or empty
        """
        super(InvertedGenerationalDistancePlus, self).__init__(is_minimization=True)
        if reference_front is None:
            raise ValueError("Reference front cannot be None")
        if len(reference_front) == 0:
            raise ValueError("Reference front cannot be empty")
        
        self.reference_front = reference_front

    def _dominance_distance(self, reference_point: np.array, solution_point: np.array) -> float:
        """
        Compute the dominance distance between two vectors used in the IGD+ indicator.
        The dominance distance considers only the objectives where solution_point is worse 
        than reference_point (i.e., max(solution_point[i] - reference_point[i], 0)).
        
        Args:
            reference_point: First vector (reference point)
            solution_point: Second vector (solution point)
            
        Returns:
            The dominance distance between the two vectors
        """
        differences = np.maximum(solution_point - reference_point, 0.0)
        return np.linalg.norm(differences)

    def _distance_to_closest_vector_with_dominance_distance(self, reference_point: np.array, front: np.array) -> float:
        """
        Return the minimum dominance distance from a reference point to any point in the front.
        
        Args:
            reference_point: The reference vector
            front: Matrix where each row represents a point in the front
            
        Returns:
            The minimum dominance distance to the closest point in the front
        """
        min_distance = float('inf')
        for solution_point in front:
            distance = self._dominance_distance(reference_point, solution_point)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def compute(self, solutions: np.array) -> float:
        """
        Compute the IGD+ indicator value.
        
        Args:
            solutions: Solution front matrix (each row is a solution)
            
        Returns:
            The IGD+ indicator value
            
        Raises:
            ValueError: If solutions is empty or has different dimensionality than reference front
        """
        if solutions is None or len(solutions) == 0:
            raise ValueError("Solutions front cannot be None or empty")
        
        if solutions.shape[1] != self.reference_front.shape[1]:
            raise ValueError("Solutions and reference front must have the same number of objectives")

        # Compute dominance distances from each reference point to closest solution point
        sum_of_distances = 0.0
        for reference_point in self.reference_front:
            distance = self._distance_to_closest_vector_with_dominance_distance(reference_point, solutions)
            sum_of_distances += distance
        
        return sum_of_distances / len(self.reference_front)

    def get_short_name(self) -> str:
        return "IGD+"

    def get_name(self) -> str:
        return "Inverted Generational Distance Plus"


class AdditiveEpsilonIndicator(QualityIndicator):
    """
    Additive Epsilon (ε) quality indicator.
    
    Computes the additive epsilon indicator between two fronts, following the definition 
    of Zitzler et al. (2003). The returned value is the minimum value ε such that, 
    for each point in the reference front, there exists a point in the solution front 
    shifted by ε that weakly dominates the reference point (assuming minimization).
    
    Reference:
    E. Zitzler, L. Thiele, M. Laumanns, C.M. Fonseca, V.G. Da Fonseca (2003): 
    Performance Assessment of Multiobjective Optimizers: An Analysis and Review. 
    IEEE Transactions on Evolutionary Computation, 7(2), 117-132.
    """
    
    def __init__(self, reference_front: np.array = None):
        """
        Initialize the Additive Epsilon indicator.
        
        Args:
            reference_front: Reference front matrix (each row is a solution)
        
        Raises:
            ValueError: If reference_front is None or empty
        """
        super(AdditiveEpsilonIndicator, self).__init__(is_minimization=True)
        if reference_front is None:
            raise ValueError("Reference front cannot be None")
        if len(reference_front) == 0:
            raise ValueError("Reference front cannot be empty")
        
        self.reference_front = reference_front

    def compute(self, front: np.array) -> float:
        """
        Compute the additive epsilon indicator value.
        
        Args:
            front: Solution front matrix (each row is a solution)
            
        Returns:
            The additive epsilon indicator value
            
        Raises:
            ValueError: If front is empty or has different dimensionality than reference front
        """
        if front is None or len(front) == 0:
            raise ValueError("Solution front cannot be None or empty")
        
        if front.shape[1] != self.reference_front.shape[1]:
            raise ValueError("Solution and reference front must have the same number of objectives")

        maximum_epsilon = float('-inf')
        
        for reference_point in self.reference_front:
            # For each reference point, find the minimum over the front of the maximum objective-wise difference
            minimum_epsilon = min(
                max(solution_point - reference_point) for solution_point in front
            )
            maximum_epsilon = max(maximum_epsilon, minimum_epsilon)
        
        return maximum_epsilon

    def get_short_name(self) -> str:
        return "EP"

    def get_name(self) -> str:
        return "Additive Epsilon"


# Alias for backwards compatibility
EpsilonIndicator = AdditiveEpsilonIndicator
"""Legacy alias for AdditiveEpsilonIndicator.

This alias is maintained for backward compatibility. New code should use
AdditiveEpsilonIndicator directly.
"""


class HyperVolume(QualityIndicator):
    """Hypervolume (HV) quality indicator.
    
    The hypervolume indicator measures the volume of the objective space that is
    dominated by the solution set, bounded by a reference point. It is one of the
    most widely used indicators for multi-objective optimization as it captures
    both convergence and diversity in a single scalar value.
    
    This implementation uses the moocore library for efficient hypervolume computation.
    
    Note:
        - Higher hypervolume values indicate better quality (maximization).
        - The reference point must be worse than all solutions in all objectives.
    
    Reference:
        Zitzler, E., & Thiele, L. (1998). Multiobjective optimization using evolutionary
        algorithms - A comparative case study. In International Conference on Parallel
        Problem Solving from Nature (pp. 292-301).
    """

    def __init__(self, reference_point: List[float] = None):
        """Initialize the hypervolume indicator.
        
        Args:
            reference_point: The reference point that defines the upper bounds of
                           the hypervolume calculation. Must be worse than all
                           solutions in all objectives (for minimization problems).
        """
        super(HyperVolume, self).__init__(is_minimization=False)
        self.reference_point = reference_point
        self.hv = moocore.Hypervolume(ref=reference_point)

    def compute(self, solutions: np.ndarray) -> float:
        """Compute the hypervolume indicator value.
        
        Args:
            solutions: A 2D numpy array of shape (m, n) where m is the number of
                     solutions and n is the number of objectives.
                     
        Returns:
            The hypervolume value (higher is better).
            
        Raises:
            ValueError: If the reference point is not set or if any solution is
                       not dominated by the reference point.
        """
        if self.reference_point is None:
            raise ValueError("Reference point must be set before computing hypervolume")
            
        # The moocore.Hypervolume class uses __call__ for computation
        return float(self.hv(solutions))

    def get_name(self) -> str:
        return "Hypervolume"

    def get_short_name(self) -> str:
        return "HV"


class NormalizedHyperVolume(QualityIndicator):
    """Normalized Hypervolume (NHV) quality indicator.
    
    The normalized hypervolume is calculated as:
        NHV = 1 - (HV of the front / HV of the reference front)
    
    This indicator is useful for comparing solution sets when the absolute
    scale of the objectives is not known in advance. It assumes minimization
    of the indicator value (lower is better).
    
    The reference front should be a high-quality approximation of the true
    Pareto front for meaningful normalization.
    
    Note:
        - NHV = 0 when the front has the same hypervolume as the reference front.
        - NHV approaches 1 as the front quality decreases.
        - Negative values indicate the front is better than the reference front.
    """

    def __init__(self, reference_point: List[float]):
        """Initialize the normalized hypervolume indicator.
        
        Args:
            reference_point: The reference point for hypervolume computation.
                           Must be worse than all solutions in all objectives.
        """
        super().__init__(is_minimization=True)
        self.reference_point = reference_point
        self._hv = HyperVolume(reference_point=reference_point)
        self._reference_hypervolume = None  # Will be set by set_reference_front()

    def set_reference_front(self, reference_front: np.ndarray) -> None:
        """Set the reference front and compute its hypervolume.
        
        Args:
            reference_front: The reference front used for normalization.
            
        Raises:
            ValueError: If the reference front results in zero hypervolume.
        """
        self._reference_hypervolume = self._hv.compute(reference_front)
        if self._reference_hypervolume == 0:
            raise AssertionError("Hypervolume of reference front is zero")

    def compute(self, solutions: np.ndarray) -> float:
        """Compute the normalized hypervolume indicator value.
        
        Args:
            solutions: A 2D numpy array of shape (m, n) where m is the number of
                     solutions and n is the number of objectives.
                     
        Returns:
            The normalized hypervolume value (lower is better).
            
        Raises:
            RuntimeError: If the reference front has not been set.
        """
        if self._reference_hypervolume is None:
            raise RuntimeError("Reference front must be set before computing normalized hypervolume")
            
        hv = self._hv.compute(solutions=solutions)
        return 1.0 - (hv / self._reference_hypervolume)
        
    def get_short_name(self) -> str:
        """Get the short name of the indicator.
        
        Returns:
            'NHV' for Normalized Hypervolume.
        """
        return "NHV"
        
    def get_name(self) -> str:
        """Get the full name of the indicator.
        
        Returns:
            'Normalized Hypervolume'.
        """
        return "Normalized Hypervolume"

    def get_name(self) -> str:
        return "Normalized Hypervolume"
