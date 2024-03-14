from abc import ABC, abstractmethod
from math import sqrt

from jmetal.util.point import IdealPoint

"""
.. module:: aggregation_function
   :platform: Unix, Windows
   :synopsis: Implementation of aggregative (scalarizing) functions.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class AggregationFunction(ABC):
    @abstractmethod
    def compute(self, vector: [], weight_vector: []) -> float:
        pass

    @abstractmethod
    def update(self, vector: []) -> None:
        pass


class WeightedSum(AggregationFunction):
    def compute(self, vector: [], weight_vector: []) -> float:
        return sum(map(lambda x, y: x * y, vector, weight_vector))

    def update(self, vector: []) -> None:
        pass


class PenaltyBoundaryIntersection(AggregationFunction):
    def __init__(self, dimension: int, theta: float = 5.0):
        self.ideal_point = IdealPoint(dimension)
        self.theta = theta

    def compute(self, vector: [], weight_vector: []) -> float:
        d1 = d2 = nl = 0.0

        for i in range(len(vector)):
            d1 += (vector[i] - self.ideal_point.point[i]) * weight_vector[i];
            nl += pow(weight_vector[i], 2.0)

        nl = sqrt(nl)
        d1 = abs(d1) / nl

        for i in range(len(vector)):
            d2 += pow((vector[i] - self.ideal_point.point[i]) -
                           d1 * (weight_vector[i] / nl), 2.0)
        d2 = sqrt(d2)

        return d1 + self.theta * d2

    def update(self, vector: []) -> None:
        self.ideal_point.update(vector)


class Tschebycheff(AggregationFunction):
    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)

    def compute(self, vector: [], weight_vector: []) -> float:
        max_fun = -1.0e30

        for i in range(len(vector)):
            diff = abs(vector[i] - self.ideal_point.point[i])

            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = diff * weight_vector[i]

            if feval > max_fun:
                max_fun = feval

        return max_fun

    def update(self, vector: []) -> None:
        self.ideal_point.update(vector)
