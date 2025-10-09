from abc import ABC, abstractmethod
from functools import cmp_to_key
from typing import List, TypeVar

import numpy
from moocore import hv_contributions
from scipy.spatial.distance import euclidean

from jmetal.logger import get_logger
from jmetal.util.comparator import Comparator, SolutionAttributeComparator

logger = get_logger(__name__)

S = TypeVar("S")

"""
.. module:: density_estimator
   :platform: Unix, Windows
   :synopsis: Module including the implementation of density estimators.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


class DensityEstimator(List[S], ABC):
    """This is the interface of any density estimator algorithm."""

    @abstractmethod
    def compute_density_estimator(self, solutions: List[S]) -> float:
        pass

    @abstractmethod
    def sort(self, solutions: List[S]) -> List[S]:
        pass

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class CrowdingDistanceDensityEstimator(DensityEstimator[List[S]]):
    """This class implements a DensityEstimator based on the crowding distance of algorithm NSGA-II."""

    def compute_density_estimator(self, front: List[S]):
        """This function performs the computation of the crowding density estimation over the solution list.

        .. note::
           This method assign the distance in the inner elements of the solution list.

        :param front: The list of solutions.
        """
        size = len(front)

        if size == 0:
            return
        elif size == 1:
            front[0].attributes["crowding_distance"] = float("inf")
            return
        elif size == 2:
            front[0].attributes["crowding_distance"] = float("inf")
            front[1].attributes["crowding_distance"] = float("inf")
            return

        for i in range(len(front)):
            front[i].attributes["crowding_distance"] = 0.0

        number_of_objectives = len(front[0].objectives)

        for i in range(number_of_objectives):
            # Sort the population by Obj n
            front = sorted(front, key=lambda x: x.objectives[i])
            objective_minn = front[0].objectives[i]
            objective_maxn = front[len(front) - 1].objectives[i]

            # Set de crowding distance
            front[0].attributes["crowding_distance"] = float("inf")
            front[size - 1].attributes["crowding_distance"] = float("inf")

            for j in range(1, size - 1):
                distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass
                    # logger.warning('Minimum and maximum are the same!')
                else:
                    distance = distance / (objective_maxn - objective_minn)

                distance += front[j].attributes["crowding_distance"]
                front[j].attributes["crowding_distance"] = distance

    def sort(self, solutions: List[S]) -> List[S]:
        solutions.sort(key=cmp_to_key(self.get_comparator().compare))

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("crowding_distance", lowest_is_best=False)


class KNearestNeighborDensityEstimator(DensityEstimator[List[S]]):
    """This class implements a density estimator based on the distance to the k-th nearest solution."""

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.distance_matrix = []

    def compute_density_estimator(self, solutions: List[S]):
        solutions_size = len(solutions)
        if solutions_size <= self.k:
            return

        # Compute distance matrix
        self.distance_matrix = numpy.zeros(shape=(solutions_size, solutions_size))
        for i in range(solutions_size):
            for j in range(solutions_size):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = euclidean(
                    solutions[i].objectives, solutions[j].objectives
                )
        # Assign knn_density attribute
        for i in range(solutions_size):
            distances = list(self.distance_matrix[i])
            distances.sort()
            solutions[i].attributes["knn_density"] = distances[self.k]

    def sort(self, solutions: List[S]) -> List[S]:
        """
        Sort solutions by knn_density (highest first).
        """
        solutions.sort(key=lambda s: s.attributes.get("knn_density", float("inf")), reverse=True)

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("knn_density", lowest_is_best=False)

class HypervolumeContributionDensityEstimator(DensityEstimator[List[S]]):
    """Density estimator based on the hypervolume contribution of each solution."""

    def __init__(self, reference_point=None):
        super().__init__()
        if reference_point is None:
            raise ValueError("reference_point for hypervolume contribution cannot be None.")
        if isinstance(reference_point, (list, tuple, numpy.ndarray)) and len(reference_point) == 0:
            raise ValueError("reference_point for hypervolume contribution cannot be empty.")
        self.reference_point = reference_point

    def compute_density_estimator(self, solutions: List[S]):
        """
        Computes the hypervolume contribution for each solution in the list.
        Stores the value in solution.attributes["hv_contribution"].
        """
        if not solutions:
            return

        # Extract objective values from solutions
        objectives = [solution.objectives for solution in solutions]

        # Compute contributions
        contributions = hv_contributions(objectives, ref=self.reference_point)

        # Assign contribution to each solution
        for sol, hv in zip(solutions, contributions):
            sol.attributes["hv_contribution"] = hv

    def sort(self, solutions: List[S]) -> List[S]:
        """
        Sorts solutions by their hypervolume contribution (highest first).
        """
        solutions.sort(key=lambda s: s.attributes.get("hv_contribution", float('-inf')), reverse=True)

    @classmethod
    def get_comparator(cls) -> Comparator:
        """
        Returns a comparator for the "hv_contribution" attribute.
        """
        return SolutionAttributeComparator("hv_contribution", lowest_is_best=False)