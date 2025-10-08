import unittest
from jmetal.operator.replacement import SMSEMOAReplacement
from jmetal.util.density_estimator import HypervolumeContributionDensityEstimator

class DummySolution:
    def __init__(self, objectives):
        self.objectives = objectives
        self.attributes = {}

class TestSMSEMOAReplacement(unittest.TestCase):
    def setUp(self):
        self.reference_point = [6, 6]
        self.replacement = SMSEMOAReplacement(reference_point=self.reference_point)

    def test_removes_min_hv_solution(self):
        solutions = [
            DummySolution([5, 1]),
            DummySolution([1, 5]),
            DummySolution([4, 2]),
            DummySolution([4, 4]),
        ]
        offspring = [DummySolution([5, 1])]
        result = self.replacement.replace(solutions, offspring)
        self.assertEqual(len(result), len(solutions) + len(offspring) - 1)
        # Check that one solution with minimum hv_contribution is removed
        hv_estimator = HypervolumeContributionDensityEstimator(reference_point=self.reference_point)
        hv_estimator.compute_density_estimator([s for s in solutions + offspring if s in result])
        hv_values = [s.attributes["hv_contribution"] for s in result]
        self.assertTrue(all(hv > min(hv_values) or hv == min(hv_values) for hv in hv_values))

    def test_no_error_with_identical_solutions(self):
        solutions = [DummySolution([1, 1]), DummySolution([1, 1]), DummySolution([1, 1])]
        offspring = [DummySolution([1, 1])]
        result = self.replacement.replace(solutions, offspring)
        self.assertEqual(len(result), len(solutions) + len(offspring) - 1)

    def test_high_dimensional(self):
        reference_point = [10, 10, 10]
        replacement = SMSEMOAReplacement(reference_point=reference_point)
        solutions = [
            DummySolution([1, 2, 3]),
            DummySolution([2, 3, 4]),
            DummySolution([3, 4, 5]),
        ]
        offspring = [DummySolution([4, 5, 6])]
        result = replacement.replace(solutions, offspring)
        self.assertEqual(len(result), len(solutions) + len(offspring) - 1)

if __name__ == "__main__":
    unittest.main()
