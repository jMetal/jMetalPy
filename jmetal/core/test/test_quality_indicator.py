import os
import unittest
from pathlib import Path

import numpy as np

from jmetal.core.quality_indicator import (EpsilonIndicator,
                                           GenerationalDistance, HyperVolume,
                                           InvertedGenerationalDistance,
                                           NormalizedHyperVolume)

DIRNAME = os.path.dirname(os.path.abspath(__file__))


class GenerationalDistanceTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = GenerationalDistance([])
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Generational Distance", GenerationalDistance([]).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("GD", GenerationalDistance([]).get_short_name())

    def test_case1(self):
        """
        Case 1. Reference front: [[1.0, 1.0]], front: [[1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case2(self):
        """
        Case 2. Reference front: [[1.0, 1.0], [2.0, 2.0], front: [[1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case3(self):
        """
        Case 3. Reference front: [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], front: [[1.0, 1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0. Example with three objectives

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
        front = np.array([[1.0, 1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case4(self):
        """
        Case 4. reference front: [[1.0, 1.0], [2.0, 2.0]], front: [[1.5, 1.5]]
        Expected result: the distance to the nearest point of the reference front is the euclidean distance to any of the
        points of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.5, 1.5]])

        result = indicator.compute(front)

        self.assertEqual(np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2)), result)
        self.assertEqual(np.sqrt(pow(2.0 - 1.5, 2) + pow(2.0 - 1.5, 2)), result)

    def test_case5(self):
        """
        Case 5. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5]]
        Expected result: the distance to the nearest point of the reference front is the euclidean distance
        to the nearest point of the reference front ([1.0, 1.0])

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5]])

        result = indicator.compute(front)

        self.assertEqual(np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2)), result)
        self.assertEqual(np.sqrt(pow(2.0 - 1.5, 2) + pow(2.0 - 1.5, 2)), result)

    def test_case6(self):
        """
        Case 6. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2]]
        Expected result: the distance to the nearest point of the reference front is the average of the sum of each point
        of the front to the nearest point of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case7(self):
        """
        Case 7. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]]
        Expected result: the distance to the nearest point of the reference front is the sum of each point of the front to the
        nearest point of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))
        distance_of_third_point = np.sqrt(pow(2.1 - 1.9, 2) + pow(2.1 - 1.9, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point + distance_of_third_point) / 3.0, result)


class InvertedGenerationalDistanceTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = InvertedGenerationalDistance([])
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Inverted Generational Distance", InvertedGenerationalDistance([]).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("IGD", InvertedGenerationalDistance([]).get_short_name())

    def test_case1(self):
        """
        Case 1. Reference front: [[1.0, 1.0]], front: [[1.0, 1.0]]
        Expected result = 0.0
        Comment: simplest case

        :return:
        """
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case2(self):
        """
        Case 2. Reference front: [[1.0, 1.0], [2.0, 2.0], front: [[1.0, 1.0]]
        Expected result: average of the sum of the distances of the points of the reference front to the front

        :return:
        """
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        distance_of_first_point = np.sqrt(pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case3(self):
        """
        Case 3. Reference front: [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], front: [[1.0, 1.0, 1.0]]
        Expected result: average of the sum of the distances of the points of the reference front to the front.
        Example with three objectives

        :return:
        """
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
        front = np.array([[1.0, 1.0, 1.0]])

        result = indicator.compute(front)

        distance_of_first_point = np.sqrt(pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case4(self):
        """
        Case 4. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2]]
        Expected result: average of the sum of the distances of the points of the reference front to the front.
        Example with three objectives

        :return:
        """
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case5(self):
        """
        Case 5. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]]
        Expected result: average of the sum of the distances of the points of the reference front to the front.
        Example with three objectives

        :return:
        """
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.9, 2) + pow(2.0 - 1.9, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)


class EpsilonIndicatorTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = EpsilonIndicator(np.array([[1.0, 1.0], [2.0, 2.0]]))
        self.assertIsNotNone(indicator)


class HyperVolumeTestCases(unittest.TestCase):
    def test_should_hypervolume_return_5_0(self):
        reference_point = [2, 2, 2]

        front = np.array([[1, 0, 1], [0, 1, 0]])

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertEqual(5.0, value)

    def test_should_hypervolume_return_the_correct_value_when_applied_to_the_ZDT1_reference_front(self):
        filepath = Path(DIRNAME, "ZDT1.pf")
        front = []

        with open(filepath) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                front.append(vector)

        reference_point = [1, 1]

        hv = HyperVolume(reference_point)
        value = hv.compute(np.array(front))

        self.assertAlmostEqual(0.666, value, delta=0.001)


class NormalizedHyperVolumeTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = Path(DIRNAME, "ZDT1.pf")
        front = []

        with open(filepath) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                front.append(vector)

        cls._front = np.array(front)

    def test_should_hypervolume_return_zero_when_providing_reference_front(self):
        reference_point = [1, 1]
        reference_front = self._front

        hv = NormalizedHyperVolume(reference_point, reference_front=reference_front)
        value = hv.compute(reference_front)

        self.assertAlmostEqual(0, value, delta=0.001)

    def test_should_raise_AssertionError_when_reference_front_hv_is_zero(self):
        reference_point = [0, 0]
        reference_front = self._front

        with self.assertRaises(AssertionError):
            _ = NormalizedHyperVolume(reference_point, reference_front=reference_front)


if __name__ == "__main__":
    unittest.main()
