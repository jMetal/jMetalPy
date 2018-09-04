from os.path import dirname, join
import unittest

from jmetal.core.solution import Solution
from jmetal.problem import ZDT1
from jmetal.component.quality_indicator import HyperVolume


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
        problem = ZDT1(rf_path='resources/reference_front/ZDT1.pf')
        reference_point = [1, 1]

        hv = HyperVolume(reference_point)
        value = hv.compute(problem.reference_front)

        self.assertAlmostEqual(0.666, value, delta=0.001)


if __name__ == '__main__':
    unittest.main()
