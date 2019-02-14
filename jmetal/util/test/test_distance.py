import unittest

from jmetal.util.distance import EuclideanDistance


class EuclideanDistanceTestCases(unittest.TestCase):

    def test_should_get_distance_work_properly_case_1(self):
        """ Case 1: [1], [1] -> distance == 0 """
        distance = EuclideanDistance()

        self.assertEqual(0, distance.get_distance([1], [1]))

    def test_should_get_distance_work_properly_case_2(self):
        """ Case 2: [1, 0, 0], [0, 1, 0] -> distance == 1.4142135623730951 """
        distance = EuclideanDistance()

        self.assertEqual(1.4142135623730951, distance.get_distance([1, 0, 0], [0, 1, 0]))

    def test_should_get_distance_work_properly_case_3(self):
        """ Case 3: [1, 1, 0], [0, 1, 0] -> distance == 1.0 """
        distance = EuclideanDistance()

        self.assertEqual(1.0, distance.get_distance([1, 1, 0], [0, 1, 0]))


"""
class CosineDistanceTestCases(unittest.TestCase):
    def test_should_identical_points_have_a_distance_of_zero(self):
        reference_point = [0.0, 0.0]
        distance = CosineDistance(reference_point)

        self.assertEqual(0.0, distance.get_distance([1.0, 1.0], [1.0, 1.0]))

    def test_should_points_in_the_same_direction_have_a_distance_of_zero(self):
        reference_point = [0.0, 0.0]
        distance = CosineDistance(reference_point)

        self.assertEqual(0.0, distance.get_distance([1.0, 1.0], [2.0, 2.0]))

    def test_should_two_perpendicular_points_have_a_distance_of_one(self):
        reference_point = [0.0, 0.0]
        distance = CosineDistance(reference_point)

        self.assertEqual(1.0, distance.get_distance([0.0, 1.0], [1.0, 0.0]))
"""

if __name__ == '__main__':
    unittest.main()
