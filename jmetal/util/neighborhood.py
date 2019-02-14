from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, List

import numpy

from jmetal.core.solution import Solution

"""
.. module:: neighborhood
   :platform: Unix, Windows
   :synopsis: implementation of neighborhoods in the context of list of solutions. The goal is,
   given the index of an element of the list, to find its neighbour solutions according to a criterion.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

S = TypeVar('S')


class Neighborhood(Generic[S], ABC):

    @abstractmethod
    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        pass


class WeightNeighborhood(Neighborhood[Solution], ABC):

    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: str = None):
        self.number_of_weight_vectors = number_of_weight_vectors
        self.neighborhood_size = neighborhood_size
        self.weight_vector_size = weight_vector_size
        self.weights_path = weights_path

        self.neighborhood = numpy.zeros((number_of_weight_vectors, neighborhood_size), dtype=int)
        self.weight_vectors = numpy.zeros((number_of_weight_vectors, weight_vector_size))


class WeightVectorNeighborhood(WeightNeighborhood):

    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: str = None):
        super(WeightVectorNeighborhood, self).__init__(number_of_weight_vectors, neighborhood_size, weight_vector_size, weights_path)
        self.__initialize_uniform_weight(weight_vector_size, number_of_weight_vectors)
        self.__initialize_neighborhood()

    def __initialize_uniform_weight(self, weight_vector_size: int, number_of_weight_vectors: int) -> None:
        """ Precomputed weights from

        * Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II

        Downloaded from:

        * http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar
        """
        if weight_vector_size == 2:
            for i in range(0, number_of_weight_vectors):
                v = 1.0 * i / (number_of_weight_vectors - 1)
                self.weight_vectors[i, 0] = v
                self.weight_vectors[i, 1] = 1 - v
        else:
            file_name = 'W{}D_{}.dat'.format(weight_vector_size, number_of_weight_vectors)
            file_path = self.weights_path + '/' + file_name

            if Path(file_path).is_file():
                with open(file_path) as file:
                    for index, line in enumerate(file):
                        vector = [float(x) for x in line.split()]
                        self.weight_vectors[index][:] = vector
            else:
                raise FileNotFoundError('Failed to initialize weights: {} not found'.format(file_path))

    def __initialize_neighborhood(self) -> None:
        distance = numpy.zeros((len(self.weight_vectors), len(self.weight_vectors)))

        for i in range(len(self.weight_vectors)):
            for j in range(len(self.weight_vectors)):
                distance[i][j] = numpy.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])

            indexes = numpy.argsort(distance[i, :])
            self.neighborhood[i, :] = indexes[0:self.neighborhood_size]

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        neighbors_indexes = self.neighborhood[index]

        if any(i > len(solution_list) for i in neighbors_indexes):
            raise IndexError('Neighbor index out of range')

        return [solution_list[i] for i in neighbors_indexes]

    def get_neighborhood(self):
        return self.neighborhood
