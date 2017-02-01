from typing import TypeVar, List

from jmetal.core.operator.operator import Operator

Source = TypeVar('S')
Result = TypeVar('S')

""" Class representing crossover operators """
__author__ = "Antonio J. Nebro"


class CrossoverOperator(Operator[List[Source], List[Result]]):
    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception("The probability is greater than one: " + str(probability))
        elif probability < 0.0:
            raise Exception("The probability is lower than zero: " + str(probability))

        self.probability = probability

    def execute(self, source: Source) -> Source:
        pass
