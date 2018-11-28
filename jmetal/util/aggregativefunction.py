from abc import ABCMeta, abstractmethod

"""
.. module:: aggregativefunction
   :platform: Unix, Windows
   :synopsis: implementation of aggregative (scalarizing) functions.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class AggregativeFunction:
    __metaclass__ = ABCMeta

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
