import unittest

from jmetal.component.density_estimator import CrowdingDistance
from jmetal.core.solution import Solution


class CrowdingDistanceTestCases(unittest.TestCase):

    def setUp(self):
        self.crowding = CrowdingDistance()

    def test_should_the_crowding_distance_of_an_empty_set_do_nothing(self):
        solution_list = []
        self.crowding.compute_density_estimator(solution_list)

    def test_should_the_crowding_distance_of_single_solution_be_infinity(self):
        solution = Solution(3, 3)
        solution_list = [solution]

        self.crowding.compute_density_estimator(solution_list)
        value = solution_list[0].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value)

    def test_should_the_crowding_distance_of_two_solutions_be_infinity(self):
        solution1 = Solution(3, 3)
        solution2 = Solution(3, 3)
        solution_list = [solution1, solution2]

        self.crowding.compute_density_estimator(solution_list)
        value_from_solution1 = solution_list[0].attributes["crowding_distance"]
        value_from_solution2 = solution_list[1].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value_from_solution1)
        self.assertEqual(float("inf"), value_from_solution2)

    def test_should_the_crowding_distance_of_three_solutions_correctly_assigned(self):
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)
        solution3 = Solution(2, 2)

        solution1.objectives[0] = 0.0
        solution1.objectives[1] = 1.0
        solution2.objectives[0] = 1.0
        solution2.objectives[1] = 0.0
        solution3.objectives[0] = 0.5
        solution3.objectives[1] = 0.5

        solution_list = [solution1, solution2, solution3]

        self.crowding.compute_density_estimator(solution_list)

        value_from_solution1 = solution_list[0].attributes["crowding_distance"]
        value_from_solution2 = solution_list[1].attributes["crowding_distance"]
        value_from_solution3 = solution_list[2].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value_from_solution1)
        self.assertEqual(float("inf"), value_from_solution2)
        self.assertEqual(2.0, value_from_solution3)

    def test_should_the_crowding_distance_of_four_solutions_correctly_assigned(self):
        solution1 = Solution(2, 2)
        solution2 = Solution(2, 2)
        solution3 = Solution(2, 2)
        solution4 = Solution(2, 2)

        solution1.objectives[0] = 0.0
        solution1.objectives[1] = 1.0
        solution2.objectives[0] = 1.0
        solution2.objectives[1] = 0.0
        solution3.objectives[0] = 0.5
        solution3.objectives[1] = 0.5
        solution4.objectives[0] = 0.75
        solution4.objectives[1] = 0.75

        solution_list = [solution1, solution2, solution3, solution4]

        self.crowding.compute_density_estimator(solution_list)

        value_from_solution1 = solution_list[0].attributes["crowding_distance"]
        value_from_solution2 = solution_list[1].attributes["crowding_distance"]
        value_from_solution3 = solution_list[2].attributes["crowding_distance"]
        value_from_solution4 = solution_list[3].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value_from_solution1)
        self.assertEqual(float("inf"), value_from_solution2)
        self.assertGreater(value_from_solution3, value_from_solution4)


if __name__ == "__main__":
    unittest.main()
