import unittest

from jmetal.util.point import IdealPoint


class IdealPointTestCases(unittest.TestCase):

    def test_should_constructor_create_a_correctly_initialized_point(self) -> None:
        point = IdealPoint(2)

        self.assertEqual(2, len(point.point))
        self.assertEqual(2 * [float("inf")], point.point)

    def test_should_update_with_one_point_work_properly(self) -> None:
        point = IdealPoint(3)

        vector = [2.2, -1.5, 3.5]
        point.update(vector)

        self.assertEqual(vector, point.point)

    def test_should_update_with_two_solutions_work_properly(self) -> None:
        point = IdealPoint(2)

        vector1 = [0.0, 1.0]
        vector2 = [1.0, 0.0]

        point.update(vector1)
        point.update(vector2)

        self.assertEqual([0.0, 0.0], point.point)

    def test_should_update_with_three_solutions_work_properly(self) -> None:
        point = IdealPoint(3)

        point.update([3.0, 1.0, 2.0])
        point.update([0.2, 4.0, 5.5])
        point.update([5.0, 6.0, 1.5])

        self.assertEqual([0.2, 1.0, 1.5], point.point)


if __name__ == '__main__':
    unittest.main()
