import unittest

from jmetal.util.aggregative_function import WeightedSum


class WeightedSumTestCases(unittest.TestCase):

    def test_should_aggregative_sum_work_properly_with_2D_vectors(self) -> None:
        aggregative_function = WeightedSum()

        self.assertEqual(1.5, aggregative_function.compute([1.5, 2.9], [1.0, 0.0]))
        self.assertEqual(2.9, aggregative_function.compute([1.5, 2.9], [0.0, 1.0]))
        self.assertEqual(1.5 / 2.0 + 2.9 / 2.0, aggregative_function.compute([1.5, 2.9], [0.5, 0.5]))


if __name__ == '__main__':
    unittest.main()
