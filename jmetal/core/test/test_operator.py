import unittest
from typing import TypeVar

from jmetal.core.operator import Mutation, Crossover
from jmetal.core.solution import Solution

S = TypeVar('S')
R = TypeVar('R')


class DummyMutation(Mutation[Solution]):

    def __init__(self, probability: float):
        super(DummyMutation, self).__init__(probability=probability)

    def execute(self, source: Solution) -> Solution:
        return None

    def get_name(self) -> str:
        return ""


class DummyCrossover(Crossover[Solution, Solution]):

    def __init__(self, probability: float):
        super(DummyCrossover, self).__init__(probability=probability)

    def execute(self, source: Solution) -> Solution:
        return None

    def get_name(self) -> str:
        return ""


class OperatorTestCase(unittest.TestCase):

    def test_should_mutation_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            DummyMutation(-1)

    def test_should_mutation_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            DummyMutation(1.1)

    def test_should_crossover_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            DummyCrossover(-1)

    def test_should_crossover_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            DummyMutation(1.1)


if __name__ == '__main__':
    unittest.main()
