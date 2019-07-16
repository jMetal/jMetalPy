from abc import ABC, abstractmethod

import numpy
from scipy.spatial import distance

"""
.. module:: distance
   :platform: Unix, Windows
   :synopsis: implementation of distances between entities

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Distance(ABC):

    @abstractmethod
    def get_distance(self, element1, element2) -> float:
        pass


class EuclideanDistance(Distance):

    def get_distance(self, list1: [], list2: []):
        return distance.euclidean(list1, list2)


class CosineDistance(Distance):

    def __init__(self, reference_point: []):
        self.reference_point = reference_point

    def get_distance(self, list1: [], list2: []):
        total = sum(numpy.multiply([(x - r) for x, r in zip(list1, self.reference_point)],
                                   [(y - r) for y, r in zip(list2, self.reference_point)]))

        a = distance.cosine([x - y for x, y in zip(list1, self.reference_point)],
                            [x - y for x, y in zip(list2, self.reference_point)])

        b = total / (self.__sum_of_distances_to_reference_point(list1) *
                     self.__sum_of_distances_to_reference_point(list2))

        return b

    def __sum_of_distances_to_reference_point(self, l: []):
        return sum([pow(x - y, 2.0) for x, y in zip(l, self.reference_point)])
