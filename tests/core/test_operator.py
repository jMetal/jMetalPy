import unittest

from jmetal.core.operator import Crossover, Mutation


class FakeMutation(Mutation):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self, probability: float):
        super(FakeMutation, self).__init__(probability=probability)


class FakeCrossover(Crossover):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self, probability: float):
        super(FakeCrossover, self).__init__(probability=probability)


class OperatorTestCase(unittest.TestCase):
    def test_should_mutation_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            FakeMutation(-1)

    def test_should_mutation_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            FakeMutation(1.1)

    def test_should_crossover_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            FakeCrossover(-1)

    def test_should_crossover_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            FakeCrossover(1.1)


if __name__ == "__main__":
    unittest.main()
