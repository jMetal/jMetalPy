import unittest

from jmetal.util.quality_indicator import HyperVolume


class HyperVolumeTestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_hyper_volume_return_5_0(self):
        reference_point = [2, 2, 2]
        front = [[1, 0, 1], [0, 1, 0]]

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertEqual(5.0, value)

if __name__ == '__main__':
    unittest.main()
