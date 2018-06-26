import logging
from typing import TypeVar, List

logger = logging.getLogger(__name__)

S = TypeVar('S')

"""
.. module:: crowding_distance
   :platform: Unix, Windows
   :synopsis: Crowding distance implementation.

.. moduleauthor:: Álvaro Gómez Jáuregui <alvarogj@lcc.uma.es>
"""


class DensityEstimator(List[S]):
    """This is the interface of any density estimator algorithm.
    """

    def compute_density_estimator(self, solution_list: List[S]) -> float:
        pass


class CrowdingDistance(DensityEstimator[List[S]]):
    """This class implements a DensityEstimator based on the crowding distance.
    In consequence, the main method of this class is :func:`compute_density_estimator`.
    """

    def compute_density_estimator(self, solution_list: List[S]):
        """This function performs the computation of the crowding density estimation over the solution list.

        .. note::
           This method assign the distance in the inner elements of the solution list.

        :param solution_list: The list of solutions.
        """
        size = len(solution_list)

        if size is 0:
            return
        elif size is 1:
            solution_list[0].attributes["crowding_distance"] = float("inf")
            return
        elif size is 2:
            solution_list[0].attributes["crowding_distance"] = float("inf")
            solution_list[1].attributes["crowding_distance"] = float("inf")
            return

        for i in range(len(solution_list)):
            solution_list[i].attributes["crowding_distance"] = 0.0

        number_of_objectives = solution_list[0].number_of_objectives

        for i in range(number_of_objectives):
            # Sort the population by Obj n
            solution_list = sorted(solution_list, key=lambda x: x.objectives[i])
            objective_minn = solution_list[0].objectives[i]
            objective_maxn = solution_list[len(solution_list) - 1].objectives[i]

            # Set de crowding distance
            solution_list[0].attributes["crowding_distance"] = float("inf")
            solution_list[size - 1].attributes["crowding_distance"] = float("inf")

            for j in range(1, size - 1):
                distance = solution_list[j + 1].objectives[i] - solution_list[j - 1].objectives[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    logger.warning("Minimum and maximum are the same!")
                else:
                    distance = distance / (objective_maxn - objective_minn)

                distance += solution_list[j].attributes["crowding_distance"]
                solution_list[j].attributes["crowding_distance"] = distance
