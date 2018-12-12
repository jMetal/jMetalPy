from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Operator
   :platform: Unix, Windows
   :synopsis: Templates for operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Operator(Generic[S, R], ABC):
    """ Class representing operator """

    @abstractmethod
    def execute(self, source: S) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class Mutation(Operator[S, S], ABC):
    """ Class representing mutation operator. """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        self.probability = probability


class Crossover(Operator[List[S], List[R]], ABC):
    """ Class representing crossover operator. """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        self.probability = probability

    @abstractmethod
    def get_number_of_parents(self) -> int:
        pass

    @abstractmethod
    def get_number_of_children(self) -> int:
        pass


class Selection(Operator[S, R], ABC):
    """ Class representing selection operator. """

    def __init__(self):
        pass
