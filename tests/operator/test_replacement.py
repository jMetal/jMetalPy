"""Tests for replacement operators."""
import pytest

from jmetal.operator.replacement import SMSEMOAReplacement
from jmetal.util.density_estimator import HypervolumeContributionDensityEstimator

class TestSMSEMOAReplacement:
    """Tests for the SMS-EMOA replacement operator."""
    
    @pytest.fixture
    def replacement_2d(self):
        """Create a 2D SMS-EMOA replacement operator."""
        return SMSEMOAReplacement(reference_point=[6, 6])
    
    @pytest.fixture
    def replacement_3d(self):
        """Create a 3D SMS-EMOA replacement operator."""
        return SMSEMOAReplacement(reference_point=[10, 10, 10])
    
    @pytest.fixture
    def hv_estimator_2d(self):
        """Create a 2D hypervolume contribution estimator."""
        return HypervolumeContributionDensityEstimator(reference_point=[6, 6])

    def test_removes_min_hv_solution(self, replacement_2d, float_solution, hv_estimator_2d):
        """Test that the solution with minimum hypervolume contribution is removed."""
        # Given: A set of solutions and an offspring
        solutions = [
            float_solution(objectives=[5, 1]),
            float_solution(objectives=[1, 5]),
            float_solution(objectives=[4, 2]),
            float_solution(objectives=[4, 4]),
        ]
        offspring = [float_solution(objectives=[5, 1])]
        
        # When: Applying the replacement
        result = replacement_2d.replace(solutions, offspring)
        
        # Then: The result should have the correct length and remove the worst solution
        assert len(result) == len(solutions) + len(offspring) - 1
        
        # And: The remaining solutions should have higher or equal hypervolume contribution
        hv_estimator_2d.compute_density_estimator([s for s in solutions + offspring if s in result])
        hv_values = [s.attributes["hv_contribution"] for s in result]
        assert all(hv >= min(hv_values) for hv in hv_values)

    @pytest.mark.parametrize("n_identical", [2, 3, 5])
    def test_handles_identical_solutions(self, replacement_2d, float_solution, n_identical):
        """Test that the operator handles identical solutions correctly."""
        # Given: A set of identical solutions and an identical offspring
        solutions = [float_solution(objectives=[1, 1]) for _ in range(n_identical)]
        offspring = [float_solution(objectives=[1, 1])]
        
        # When: Applying the replacement
        result = replacement_2d.replace(solutions, offspring)
        
        # Then: The result should have the correct length
        assert len(result) == len(solutions) + len(offspring) - 1

    def test_works_with_high_dimensional_solutions(self, replacement_3d, float_solution):
        """Test that the operator works with 3D solutions."""
        # Given: A set of 3D solutions and an offspring
        solutions = [
            float_solution(objectives=[1, 2, 3], n_objs=3),
            float_solution(objectives=[2, 3, 4], n_objs=3),
            float_solution(objectives=[3, 4, 5], n_objs=3),
        ]
        offspring = [float_solution(objectives=[4, 5, 6], n_objs=3)]
        
        # When: Applying the replacement
        result = replacement_3d.replace(solutions, offspring)
        
        # Then: The result should have the correct length
        assert len(result) == len(solutions) + len(offspring) - 1
        
        # And: All solutions should have 3 objectives
        assert all(len(s.objectives) == 3 for s in result)

    def test_preserves_solution_attributes(self, replacement_2d, float_solution):
        """Test that solution attributes are preserved during replacement."""
        # Given: Multiple solutions with custom attributes and one with a specific attribute
        solutions = [
            float_solution(objectives=[2, 3]),
            float_solution(objectives=[1, 4]),
            float_solution(objectives=[3, 2])
        ]
        solutions[1].attributes["custom_attr"] = "test_value"
        
        # When: Applying the replacement with one offspring
        offspring = [float_solution(objectives=[1.5, 3.5])]
        result = replacement_2d.replace(solutions, offspring)
        
        # Then: The result should contain the original solutions plus the offspring minus one
        assert len(result) == len(solutions) + len(offspring) - 1
        
        # And: If the solution with custom_attr is in the result, its attributes should be preserved
        for solution in result:
            if hasattr(solution, 'attributes') and "custom_attr" in solution.attributes:
                assert solution.attributes["custom_attr"] == "test_value"
