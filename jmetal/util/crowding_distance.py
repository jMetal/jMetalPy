from typing import TypeVar, List
from copy import deepcopy
from operator import itemgetter

S = TypeVar('S')


class DensityEstimator(List[S]):
    def compute_density_estimator(self, solution_list: List[S]):
        pass


class CrowdingDistance(DensityEstimator[List[S]]):
    def compute_density_estimator(self, solution_list: List[S]):
        size = len(solution_list)

        if size is 0:
            return

        elif size is 1:
            solution_list[0].attributes["distance"] = float("inf")
            return

        elif size is 2:
            solution_list[0].attributes["distance"] = float("inf")
            solution_list[1].attributes["distance"] = float("inf")
            return

        front = deepcopy(solution_list)
        front = list(map(lambda x: 0.0, front))

        number_of_objectives = solution_list[0].number_of_objectives

        for i in range(len(number_of_objectives)):
            # Sort the population by Obj n
            front = sorted(front, key=itemgetter(i))
            objective_minn = front[0][i]
            objective_maxn = front[len(front) - 1][i]

            # Set de crowding distance
            front[0].attributes = float("inf")
            front[size - 1].attributes = float("inf")

            for j in range(1, size):
                distance = front[j + 1][i] - front[j - 1][i]
                distance = distance / (objective_maxn - objective_minn)
                distance += front[j].attributes["distance"]
                front[j].attributes["distance"] = distance
