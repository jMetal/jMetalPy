from typing import TypeVar

from jmetal.core.operator.operator import Operator

Source = TypeVar('S')

""" Class representing mutation operators """
__author__ = "Antonio J. Nebro"


class MutationOperator(Operator[Source, Source]):
    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: Source) -> Source:
        pass
