import unittest

from jmetal.core.solution import Solution
from jmetal.util.crowding_distance import CrowdingDistance


class CrowdingDistanceTestCases(unittest.TestCase):
    def setUp(self):
        self.crowding = CrowdingDistance()

    def should_the_crowding_distance_of_an_empty_set_do_nothing(self):
        solution_list = []
        self.crowding.compute_density_estimator(solution_list)

    def should_the_crowding_distance_of_single_solution_be_infinity(self):
        solution = Solution(3, 3)
        solution_list = [solution]

        self.crowding.compute_density_estimator(solution_list)
        value = solution_list[0].attributes["distance"]

        self.assertEqual(float("inf"), value)

    def should_the_crowding_distance_of_two_solutions_be_infinity(self):
        solution1 = Solution(3, 3)
        solution2 = Solution(3, 3)
        solution_list = [solution1, solution2]

        self.crowding.compute_density_estimator(solution_list)
        value_from_solution1 = solution_list[0].attributes["distance"]
        value_from_solution2 = solution_list[1].attributes["distance"]

        self.assertEqual(float("inf"), value_from_solution1)
        self.assertEqual(float("inf"), value_from_solution2)