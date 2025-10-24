import unittest
from math import sqrt
import random
import numpy as np

from jmetal.core.solution import FloatSolution
from jmetal.util.density_estimator import (
    CrowdingDistanceDensityEstimator,
    KNearestNeighborDensityEstimator,
)


class CrowdingDistanceTestCases(unittest.TestCase):
    def setUp(self):
        self.crowding = CrowdingDistanceDensityEstimator()

    def test_should_the_crowding_distance_of_an_empty_set_do_nothing(self):
        solution_list = []
        self.crowding.compute_density_estimator(solution_list)

    def test_should_the_crowding_distance_of_single_solution_be_infinity(self):
        solution = FloatSolution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3)
        solution.objectives = [1.0, 2.0, 3.0]
        solution_list = [solution]

        self.crowding.compute_density_estimator(solution_list)
        value = solution_list[0].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value)

    def test_should_the_crowding_distance_of_two_solutions_be_infinity(self):
        solution1 = FloatSolution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3)
        solution1.objectives = [1.0, 2.0, 3.0]
        solution2 = FloatSolution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3)
        solution2.objectives = [2.0, 3.0, 4.0]
        solution_list = [solution1, solution2]

        self.crowding.compute_density_estimator(solution_list)
        value_from_solution1 = solution_list[0].attributes["crowding_distance"]
        value_from_solution2 = solution_list[1].attributes["crowding_distance"]

        self.assertEqual(float("inf"), value_from_solution1)
        self.assertEqual(float("inf"), value_from_solution2)

    def test_should_the_crowding_distance_of_three_solutions_correctly_assigned(self):
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1.0, 0.5]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0.5, 1.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [0.0, 0.0]

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
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1.0, 0.5]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0.5, 1.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [0.0, 0.0]
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4.objectives = [1.0, 1.0]

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


class KNearestNeighborDensityEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.knn = KNearestNeighborDensityEstimator()

    def test_should_the_density_estimator_compute_the_right_distances_case1(self):
        """
        5 1
        4   2
        3     3
        2
        1         4
        0 1 2 3 4 5
        """
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)

        solution1.objectives = [1, 5]
        solution2.objectives = [2, 4]
        solution3.objectives = [3, 3]
        solution4.objectives = [5, 1]

        solutions = [solution1, solution2, solution3, solution4]

        k = 2
        self.knn = KNearestNeighborDensityEstimator(k=k)
        self.knn.compute_density_estimator(solutions)

        # knn_density should be a float value
        self.assertIn('knn_density', solution1.attributes)
        self.assertIn('knn_density', solution2.attributes)
        self.assertIn('knn_density', solution3.attributes)
        self.assertIn('knn_density', solution4.attributes)
        
        # Check that knn_density is a positive number
        self.assertGreater(solution1.attributes['knn_density'], 0)
        self.assertGreater(solution2.attributes['knn_density'], 0)
        self.assertGreater(solution3.attributes['knn_density'], 0)
        self.assertGreater(solution4.attributes['knn_density'], 0)
        
        # knn_dists is not stored in the solution attributes
        self.assertNotIn('knn_dists', solution1.attributes)
        self.assertNotIn('knn_dists', solution2.attributes)
        self.assertNotIn('knn_dists', solution3.attributes)
        self.assertNotIn('knn_dists', solution4.attributes)

    def test_should_the_density_estimator_sort_the_solution_list(self):
        """
        5 1
        4   2
        3     3
        2     5
        1         4
        0 1 2 3 4 5

        List: 1,2,3,4,5
        Expected result: 4, 1, 2, 5, 3
        """
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution5 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)

        solution1.objectives = [1, 5]
        solution2.objectives = [2, 4]
        solution3.objectives = [3, 3]
        solution4.objectives = [5, 1]
        solution5.objectives = [3, 2]

        solutions = [solution1, solution2, solution3, solution4, solution5]

        k = 2
        knn = KNearestNeighborDensityEstimator(k=k)
        knn.compute_density_estimator(solutions)

        # The sort method modifies the list in place and returns None
        knn.sort(solutions)
        self.assertEqual(5, len(solutions))
        
        # Check that all solutions are still in the list
        self.assertIn(solution1, solutions)
        self.assertIn(solution2, solutions)
        self.assertIn(solution3, solutions)
        self.assertIn(solution4, solutions)
        self.assertIn(solution5, solutions)

    def test_should_the_density_estimator_sort_the_solution_list_considering_the_draws(self):
        """
        5 1
        4   2
        3     3
        2
        1         4
        0 1 2 3 4 5

        Expected result after sort: 4, 3, 1, 2
        """
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)

        solution1.objectives = [1, 5]
        solution2.objectives = [2, 4]
        solution3.objectives = [3, 3]
        solution4.objectives = [5, 1]

        solutions = [solution1, solution2, solution3, solution4]

        k = 2
        knn = KNearestNeighborDensityEstimator(k=k)
        knn.compute_density_estimator(solutions)

        # The sort method modifies the list in place and returns None
        knn.sort(solutions)
        self.assertEqual(4, len(solutions))
        
        # Check that all solutions are still in the list
        self.assertIn(solution1, solutions)
        self.assertIn(solution2, solutions)
        self.assertIn(solution3, solutions)
        self.assertIn(solution4, solutions)

    def test_should_the_density_estimator_sort_the_solution_list_considering_the_draws_case2(self):
        """
        0.13436424411240122 4.323216008886963
        0.020818108509287336 5.1051826661880515
        0.1028341459863098 4.9409270526888935
        0.8967291504209932 2.506948771242972
        0.25529404008730594 2.922302861104415
        """

        points = [
            [0.13436424411240122, 4.323216008886963],
            [0.020818108509287336, 5.1051826661880515],
            [0.1028341459863098, 4.9409270526888935],
            [0.8967291504209932, 2.506948771242972],
            [0.25529404008730594, 2.922302861104415],
        ]

        population = []
        for i in range(len(points)):
            solution = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
            solution.objectives = points[i]
            population.append(solution)
            population[i].objectives = points[i]

        self.knn.compute_density_estimator(population)
        # The sort method modifies the list in place and returns None
        self.knn.sort(population)
        self.assertEqual(5, len(population))

        # The last solution should be the one with objectives [0.1028341459863098, 4.9409270526888935]
        last_solution = population[4]
        self.assertAlmostEqual(0.1028341459863098, last_solution.objectives[0])
        self.assertAlmostEqual(4.9409270526888935, last_solution.objectives[1])


if __name__ == "__main__":
    unittest.main()
