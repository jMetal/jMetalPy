from abc import ABC, abstractmethod
from typing import Iterable

import moocore
import numpy as np
from scipy import spatial


class QualityIndicator(ABC):
    def __init__(self, is_minimization: bool):
        self.is_minimization = is_minimization

    @abstractmethod
    def compute(self, solutions: np.array):
        """
        :param solutions: [m, n] bi-dimensional numpy array, being m the number of solutions and n the dimension of
        each solution
        :return: the value of the quality indicator
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        pass


class FitnessValue(QualityIndicator):
    def __init__(self, is_minimization: bool = True):
        super(FitnessValue, self).__init__(is_minimization=is_minimization)

    def compute(self, solutions: np.array):
        if self.is_minimization:
            mean = np.mean([s.objectives for s in solutions])
        else:
            mean = -np.mean([s.objectives for s in solutions])

        return mean

    def get_name(self) -> str:
        return "Fitness"

    def get_short_name(self) -> str:
        return "Fitness"


class GenerationalDistance(QualityIndicator):
    def __init__(self, reference_front: np.array = None):
        """
        * Van Veldhuizen, D.A., Lamont, G.B.: Multiobjective Evolutionary Algorithm Research: A History and Analysis.
          Technical Report TR-98-03, Dept. Elec. Comput. Eng., Air Force. Inst. Technol. (1998)
        """
        super(GenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, solutions: np.array):
        if self.reference_front is None:
            raise Exception("Reference front is none")

        distances = spatial.distance.cdist(solutions, self.reference_front)

        return np.mean(np.min(distances, axis=1))

    def get_short_name(self) -> str:
        return "GD"

    def get_name(self) -> str:
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


class HyperVolume(QualityIndicator):
    """Hypervolume computation offered by the moocore project (https://multi-objective.github.io/moocore/python/)
    """

    def __init__(self, reference_point: list[float] = None):
        super(HyperVolume, self).__init__(is_minimization=False)
        self.referencePoint = reference_point
        self.hv = moocore.Hypervolume(ref=reference_point)

    def compute(self, solutions: np.array):
        """

        :return: The hypervolume that is dominated by a non-dominated front.
        """
        return self.hv(solutions)

    def get_name(self) -> str:
        return "HV"

    def get_short_name(self) -> str:
        return "Hypervolume quality indicator"

class NormalizedHyperVolume(QualityIndicator):
    """Implementation of the normalized hypervolume, which is calculated as follows:

    relative hypervolume = 1 - (HV of the front / HV of the reference front).

    Minimization is implicitly assumed here!
    """

    def __init__(self, reference_point: Iterable[float], reference_front: np.array):
        """Delegates the computation of the HyperVolume to `jMetal.core.quality_indicator.HyperVolume`.

        Fails if the HV of the reference front is zero."""
        self.reference_point = reference_point
        self._hv = HyperVolume(reference_point=reference_point)
        self._reference_hypervolume = self._hv.compute(reference_front)

        assert self._reference_hypervolume != 0, "Hypervolume of reference front is zero"

    def compute(self, solutions: np.array) -> float:
        hv = self._hv.compute(solutions=solutions)

        return 1 - (hv / self._reference_hypervolume)

    def get_short_name(self) -> str:
        return "NHV"

    def get_name(self) -> str:
        return "Normalized Hypervolume"
