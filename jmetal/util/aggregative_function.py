from abc import ABC, abstractmethod

from jmetal.util.point import IdealPoint

"""
.. module:: aggregative_function
   :platform: Unix, Windows
   :synopsis: Implementation of aggregative (scalarizing) functions.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class AggregativeFunction(ABC):

    @abstractmethod
    def compute(self, vector: [], weight_vector: []) -> float:
        pass

    @abstractmethod
    def update(self, vector: []) -> None:
        pass


class WeightedSum(AggregativeFunction):

    def compute(self, vector: [], weight_vector: []) -> float:
        return sum(map(lambda x, y: x * y, vector, weight_vector))

    def update(self, vector: []) -> None:
        pass


class Tschebycheff(AggregativeFunction):

    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)

    def compute(self, vector: [], weight_vector: []) -> float:
        max_fun = -1.0e+30

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
