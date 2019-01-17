from abc import ABC, abstractmethod

"""
.. module:: point
   :platform: Unix, Windows
   :synopsis: implementation of points of n-dimensions (e.g, ideal point, nadir point, etc.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Point(ABC):

    @abstractmethod
    def update(self, vector: []) -> None:
        pass


class IdealPoint(Point):

    def __init__(self, dimension: int):
        self.point = dimension * [float("inf")]

    def update(self, vector: []) -> None:
        self.point = [y if x > y else x for x, y in zip(self.point, vector)]


class ReferencePoint(list):
    """ A reference point exists in objective space an has a set of individuals
    associated to it. """

    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []
