from abc import ABCMeta, abstractmethod
from typing import TypeVar,Generic, List
import numpy

from jmetal.core.solution import Solution

"""
.. module:: neighborhood
   :platform: Unix, Windows
   :synopsis: implementation of neighborhoods in the context of list of solutions. The working is,
   given the index of an element of the list, to find its neighbour solutions according to a criterion.
   NOTE: the current implementation only works with weight vectors of size 2

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

S = TypeVar('S')


class Neighborhood(Generic[S]):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        pass


class WeightVectorNeighborhood(Neighborhood[Solution]):
    def __init__(self, number_of_weight_vectors: int, neighborhood_size: int, weight_vector_size: int = 2):
        self.number_of_weight_vectors = number_of_weight_vectors
        self.neighborhood_size = neighborhood_size
        self.weight_vector_size = weight_vector_size

        self.neighborhood = numpy.zeros((number_of_weight_vectors, neighborhood_size), dtype = int)
        self.weight_vectors = numpy.zeros((number_of_weight_vectors, weight_vector_size))

        if weight_vector_size == 2:
            for i in range(0, number_of_weight_vectors):
                v = 1.0 * i / (number_of_weight_vectors - 1)
                self.weight_vectors[i, 0] = v
                self.weight_vectors[i, 1] = 1 - v

        self.__initialize_neighborhood(self.weight_vectors)

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        return [solution_list[i] for i in self.neighborhood[index]]

    def __initialize_neighborhood(self, weight_vectors) -> None:
        distance = numpy.zeros((len(weight_vectors), len(weight_vectors)))

        for i in range(len(weight_vectors)):
            for j in range(len(weight_vectors)):
                distance[i][j] = numpy.linalg.norm(weight_vectors[i] - weight_vectors[j])

            indexes = numpy.argsort(distance[i, :])
            self.neighborhood[i, :] = indexes[0:self.neighborhood_size]
