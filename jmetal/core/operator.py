from typing import TypeVar, Generic, List

__author__ = "Antonio J. Nebro"

S = TypeVar('S')
R = TypeVar('R')


class Operator(Generic[S, R]):
    """ Class representing operators """

    def execute(self, source: S) -> R:
        pass

    def get_name(self):
        pass


class Mutation(Operator[S, S]):
    """ Class representing mutation operators """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: S) -> S:
        pass


class Crossover(Operator[List[S], List[R]]):
    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: S) -> R:
        pass

    def get_number_of_parents(self) -> int:
        pass


class Selection(Operator[S, R]):
    def __init__(self):
        pass

    def execute(self, source: S) -> R:
        pass
