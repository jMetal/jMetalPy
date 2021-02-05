import unittest

from jmetal.core.operator import Crossover, Mutation
from jmetal.core.solution import Solution


class DummyMutation(Mutation):
    def __init__(self, probability: float):
        super(DummyMutation, self).__init__(probability=probability)

    def execute(self, source: Solution) -> None:
        pass

    def get_name(self) -> str:
        pass


class DummyCrossover(Crossover):
    def __init__(self, probability: float):
        super(DummyCrossover, self).__init__(probability=probability)

    def execute(self, source: Solution) -> None:
        pass

    def get_name(self) -> str:
        pass


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


if __name__ == "__main__":
    unittest.main()
