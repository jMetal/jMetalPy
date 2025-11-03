
import unittest
from functools import cmp_to_key

from jmetal.util.density_estimator import (
    CrowdingDistanceDensityEstimator,
    KNearestNeighborDensityEstimator,
    HypervolumeContributionDensityEstimator,
)


class DummySolution:
    def __init__(self, objectives):
        self.objectives = objectives
        self.attributes = {}
        self.number_of_objectives = len(objectives)
        self.variables = [0.0] * len(objectives)  # Dummy variables
        self.number_of_variables = len(objectives)
        self.number_of_constraints = 0
        self.constraints = []
        self.constraint_violation = 0.0

class TestCrowdingDistanceDensityEstimator(unittest.TestCase):
    def test_multiple_objectives(self):
        solutions = [
            DummySolution([1, 2, 3]),
            DummySolution([2, 3, 4]),
            DummySolution([3, 4, 5]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("crowding_distance", sol.attributes)
            self.assertIsInstance(sol.attributes["crowding_distance"], float)

    def test_identical_objective_values(self):
        solutions = [
            DummySolution([1, 1]),
            DummySolution([1, 1]),
            DummySolution([1, 1]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("crowding_distance", sol.attributes)
            self.assertIsInstance(sol.attributes["crowding_distance"], float)

    def test_comparator_consistency(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        sorted_solutions = sorted(solutions, key=cmp_to_key(self.estimator.get_comparator().compare))
        cd_values = [sol.attributes["crowding_distance"] for sol in sorted_solutions]
        self.assertEqual(cd_values, sorted(cd_values, reverse=True))
    def setUp(self):
        self.estimator = CrowdingDistanceDensityEstimator()

    def test_empty_list(self):
        solutions = []
        self.estimator.compute_density_estimator(solutions)
        self.assertEqual(solutions, [])

    def test_single_solution(self):
        solutions = [DummySolution([1, 2])]
        self.estimator.compute_density_estimator(solutions)
        self.assertEqual(solutions[0].attributes["crowding_distance"], float("inf"))

    def test_two_solutions(self):
        solutions = [DummySolution([1, 2]), DummySolution([2, 3])]
        self.estimator.compute_density_estimator(solutions)
        self.assertEqual(solutions[0].attributes["crowding_distance"], float("inf"))
        self.assertEqual(solutions[1].attributes["crowding_distance"], float("inf"))

    def test_multiple_solutions(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("crowding_distance", sol.attributes)
            self.assertIsInstance(sol.attributes["crowding_distance"], float)

    def test_sort(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        self.estimator.sort(solutions)
        cd_values = [sol.attributes["crowding_distance"] for sol in solutions]
        self.assertEqual(cd_values, sorted(cd_values, reverse=True))

class TestKNearestNeighborDensityEstimator(unittest.TestCase):
    def test_different_k_values(self):
        for k in [0, 1, 2]:
            estimator = KNearestNeighborDensityEstimator(k=k)
            solutions = [
                DummySolution([1, 2]),
                DummySolution([2, 3]),
                DummySolution([3, 4]),
            ]
            estimator.compute_density_estimator(solutions)
            for sol in solutions:
                if len(solutions) > k:
                    self.assertIn("knn_density", sol.attributes)
                else:
                    self.assertNotIn("knn_density", sol.attributes)

    def test_non_euclidean_points(self):
        solutions = [
            DummySolution([-1.5, 2.3]),
            DummySolution([0.0, 0.0]),
            DummySolution([1000, -999]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("knn_density", sol.attributes)
            self.assertIsInstance(sol.attributes["knn_density"], float)

    def test_tie_breaking(self):
        solutions = [
            DummySolution([1, 1]),
            DummySolution([1, 1]),
            DummySolution([2, 2]),
        ]
        self.estimator.compute_density_estimator(solutions)
        self.estimator.sort(solutions)
        knn_values = [sol.attributes["knn_density"] for sol in solutions]
        self.assertEqual(knn_values, sorted(knn_values, reverse=True))

    def test_comparator_consistency(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        sorted_solutions = sorted(solutions, key=cmp_to_key(self.estimator.get_comparator().compare))
        knn_values = [sol.attributes["knn_density"] for sol in sorted_solutions]
        self.assertEqual(knn_values, sorted(knn_values, reverse=True))
    def setUp(self):
        self.estimator = KNearestNeighborDensityEstimator(k=1)

    def test_empty_list(self):
        solutions = []
        self.estimator.compute_density_estimator(solutions)
        self.assertEqual(solutions, [])

    def test_single_solution(self):
        solutions = [DummySolution([1, 2])]
        self.estimator.compute_density_estimator(solutions)
        self.assertNotIn("knn_density", solutions[0].attributes)

    def test_multiple_solutions(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("knn_density", sol.attributes)
            self.assertIsInstance(sol.attributes["knn_density"], float)

    def test_sort(self):
        solutions = [
            DummySolution([1, 2]),
            DummySolution([2, 3]),
            DummySolution([3, 4]),
        ]
        self.estimator.compute_density_estimator(solutions)
        self.estimator.sort(solutions)
        knn_values = [sol.attributes["knn_density"] for sol in solutions]
        self.assertEqual(knn_values, sorted(knn_values, reverse=True))

    def test_k_greater_than_solutions(self):
        estimator = KNearestNeighborDensityEstimator(k=5)
        solutions = [DummySolution([1, 2]), DummySolution([2, 3])]
        estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertNotIn("knn_density", sol.attributes)

class TestHypervolumeContributionDensityEstimator(unittest.TestCase):
    def test_different_reference_points(self):
        for ref in [[10, 10], [0, 0], [100, 100]]:
            estimator = HypervolumeContributionDensityEstimator(reference_point=ref)
            solutions = [
                DummySolution([5, 1]),
                DummySolution([1, 5]),
                DummySolution([4, 2]),
            ]
            estimator.compute_density_estimator(solutions)
            for sol in solutions:
                self.assertIn("hv_contribution", sol.attributes)
                self.assertIsInstance(sol.attributes["hv_contribution"], float)

    def test_degenerate_fronts(self):
        solutions = [
            DummySolution([1, 1]),
            DummySolution([1, 1]),
            DummySolution([1, 1]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("hv_contribution", sol.attributes)
            self.assertIsInstance(sol.attributes["hv_contribution"], float)

    def test_high_dimensional_fronts(self):
        estimator = HypervolumeContributionDensityEstimator(reference_point=[10, 10, 10])
        solutions = [
            DummySolution([1, 2, 3]),
            DummySolution([2, 3, 4]),
            DummySolution([3, 4, 5]),
        ]
        estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("hv_contribution", sol.attributes)
            self.assertIsInstance(sol.attributes["hv_contribution"], float)

    def test_comparator_consistency(self):
        solutions = [
            DummySolution([5, 1]),
            DummySolution([1, 5]),
            DummySolution([4, 2]),
        ]
        self.estimator.compute_density_estimator(solutions)
        sorted_solutions = sorted(solutions, key=cmp_to_key(self.estimator.get_comparator().compare))
        hv_values = [sol.attributes["hv_contribution"] for sol in sorted_solutions]
        self.assertEqual(hv_values, sorted(hv_values, reverse=True))
    def setUp(self):
        self.reference_point = [6, 6]
        self.estimator = HypervolumeContributionDensityEstimator(reference_point=self.reference_point)

    def test_empty_list(self):
        solutions = []
        self.estimator.compute_density_estimator(solutions)
        self.assertEqual(solutions, [])

    def test_single_solution(self):
        solutions = [DummySolution([1, 2])]
        self.estimator.compute_density_estimator(solutions)
        self.assertIn("hv_contribution", solutions[0].attributes)

    def test_multiple_solutions(self):
        solutions = [
            DummySolution([5, 1]),
            DummySolution([1, 5]),
            DummySolution([4, 2]),
            DummySolution([4, 4]),
            DummySolution([5, 1]),
        ]
        self.estimator.compute_density_estimator(solutions)
        for sol in solutions:
            self.assertIn("hv_contribution", sol.attributes)
            self.assertIsInstance(sol.attributes["hv_contribution"], float)

    def test_sort(self):
        solutions = [
            DummySolution([5, 1]),
            DummySolution([1, 5]),
            DummySolution([4, 2]),
            DummySolution([4, 4]),
            DummySolution([5, 1]),
        ]
        self.estimator.compute_density_estimator(solutions)
        self.estimator.sort(solutions)
        hv_values = [sol.attributes["hv_contribution"] for sol in solutions]
        self.assertEqual(hv_values, sorted(hv_values, reverse=True))

    def test_invalid_reference_point_none(self):
        with self.assertRaises(ValueError):
            HypervolumeContributionDensityEstimator(reference_point=None)

    def test_invalid_reference_point_empty(self):
        with self.assertRaises(ValueError):
            HypervolumeContributionDensityEstimator(reference_point=[])

if __name__ == "__main__":
    unittest.main()
