import unittest
from os.path import dirname, join

from jmetal.core.quality_indicator import HyperVolume, GenerationalDistance
from jmetal.core.solution import Solution
from jmetal.problem import ZDT1
from jmetal.util.solutions import read_solutions


class HyperVolumeTestCases(unittest.TestCase):

    def setUp(self):
        self.file_path = dirname(join(dirname(__file__)))

    def test_should_hypervolume_return_5_0(self):
        reference_point = [2, 2, 2]

        solution1 = Solution(1, 3)
        solution1.objectives = [1, 0, 1]

        solution2 = Solution(1, 3)
        solution2.objectives = [0, 1, 0]

        front = [solution1, solution2]

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertEqual(5.0, value)

    def test_should_hypervolume_return_the_correct_value_when_applied_to_the_ZDT1_reference_front(self):
        problem = ZDT1()
        problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

        reference_point = [1, 1]

        hv = HyperVolume(reference_point)
        value = hv.compute(problem.reference_front)

        self.assertAlmostEqual(0.666, value, delta=0.001)


class GenerationalDistanceTestCases(unittest.TestCase):

    def test_should_gd_return_the_closest_point_case_a(self):
        solution1 = Solution(1, 3)
        solution1.objectives = [1, 1, 1]

        solution2 = Solution(1, 3)
        solution2.objectives = [2, 2, 2]

        reference_front = [solution1, solution2]

        gd = GenerationalDistance(reference_front)
        value = gd.compute([solution1])

        self.assertEqual(0, value)

    def test_should_gd_return_0(self):
        solution1 = Solution(1, 3)
        solution1.objectives = [1, 0, 1]

        solution2 = Solution(1, 3)
        solution2.objectives = [0, 1, 0]

        reference_front = [solution1, solution2]

        gd = GenerationalDistance(reference_front)
        value = gd.compute(reference_front)

        self.assertEqual(0.0, value)


class InvertedGenerationalDistanceTestCases(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
