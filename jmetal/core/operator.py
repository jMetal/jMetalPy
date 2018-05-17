from typing import TypeVar, Generic, List

__author__ = "Antonio J. Nebro"

Source = TypeVar('S')
Result = TypeVar('R')


class Operator(Generic[Source, Result]):
    """ Class representing operators """

    def execute(self, source: Source) -> Result:
        pass

    def get_name(self):
        pass


class Mutation(Operator[Source, Source]):
    """ Class representing mutation operators """

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: Source) -> Source:
        pass


class Crossover(Operator[List[Source], List[Result]]):
    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: Source) -> Result:
        pass

    def get_number_of_parents(self) -> int:
        pass


class Selection(Operator[Source, Result]):
    def execute(self, source: Source) -> Result:
        pass
