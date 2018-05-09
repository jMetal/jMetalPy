import unittest

from jmetal.util.FrontUtils import read_front_from_file
from jmetal.util.quality_indicator import HyperVolume


class HyperVolumeTestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_hypervolume_return_5_0(self):
        reference_point = [2, 2, 2]
        front = [[1, 0, 1], [0, 1, 0]]

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertEqual(5.0, value)

    def test_should_hypervolume_return_the_correct_value_when_applied_to_the_ZDT1_reference_front(self):
        reference_point = [1, 1]
        front = read_front_from_file("../../../resources/data/ZDT1.pf")

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertAlmostEqual(0.666, value, delta=0.001)


if __name__ == '__main__':
    unittest.main()
