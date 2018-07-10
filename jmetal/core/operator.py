from typing import TypeVar, Generic, List

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Operator
   :platform: Unix, Windows
   :synopsis: Templates for operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Operator(Generic[S, R]):
    """ Class representing operator """

    def execute(self, source: S) -> R:
        pass

    def get_name(self):
        pass


class Mutation(Operator[S, S]):
    """ Class representing mutation operator. """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        self.probability = probability

    def execute(self, source: S) -> S:
        pass


class Crossover(Operator[List[S], List[R]]):
    """ Class representing crossover operator. """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        self.probability = probability

    def execute(self, source: S) -> R:
        pass

    def get_number_of_parents(self) -> int:
        pass


class Selection(Operator[S, R]):
    """ Class representing selection operator. """

    def __init__(self):
        pass

    def execute(self, source: S) -> R:
        pass
