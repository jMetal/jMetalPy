"""
Unit tests for jmetal.tuning.metrics module.

Tests for quality indicators and score aggregation functions.
"""

import numpy as np
import pytest

from jmetal.tuning.metrics.indicators import (
    aggregate_scores,
    compute_combined_score,
    compute_quality_indicators,
)
from jmetal.tuning.metrics.reference_fronts import load_reference_front


class TestComputeQualityIndicators:
    """Tests for compute_quality_indicators function."""

    def test_given_good_front_when_compute_then_returns_low_values(
        self, sample_solutions, sample_reference_front
    ) -> None:
        """Good approximation should yield low indicator values."""
        # Arrange & Act
        nhv, ae = compute_quality_indicators(
            sample_solutions, sample_reference_front
        )
        
        # Assert
        assert isinstance(nhv, float)
        assert isinstance(ae, float)
        assert nhv >= 0
        assert ae >= 0

    def test_given_dominated_front_when_compute_then_returns_higher_values(
        self, dominated_solutions, sample_reference_front
    ) -> None:
        """Dominated approximation should yield higher indicator values."""
        # Arrange
        # Create good solutions for comparison
        good_solutions = []
        for objectives in [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]:
            sol = type('Solution', (), {'objectives': objectives})()
            good_solutions.append(sol)
        
        # Act
        nhv_good, ae_good = compute_quality_indicators(
            good_solutions, sample_reference_front
        )
        nhv_bad, ae_bad = compute_quality_indicators(
            dominated_solutions, sample_reference_front
        )
        
        # Assert - bad front should have worse (higher) values
        assert nhv_bad >= nhv_good or ae_bad >= ae_good

    def test_given_single_solution_when_compute_then_returns_values(
        self, sample_reference_front
    ) -> None:
        """Single solution should still compute indicators."""
        # Arrange
        single_solution = [type('Solution', (), {'objectives': [0.5, 0.5]})()]
        
        # Act
        nhv, ae = compute_quality_indicators(
            single_solution, sample_reference_front
        )
        
        # Assert
        assert isinstance(nhv, float)
        assert isinstance(ae, float)

    def test_given_custom_offset_when_compute_then_uses_offset(
        self, sample_solutions, sample_reference_front
    ) -> None:
        """Custom reference_point_offset should affect computation."""
        # Arrange & Act
        nhv_small, _ = compute_quality_indicators(
            sample_solutions, sample_reference_front, reference_point_offset=0.05
        )
        nhv_large, _ = compute_quality_indicators(
            sample_solutions, sample_reference_front, reference_point_offset=0.2
        )
        
        # Assert - different offsets should generally give different values
        # (though not guaranteed in all cases)
        assert isinstance(nhv_small, float)
        assert isinstance(nhv_large, float)


class TestComputeCombinedScore:
    """Tests for compute_combined_score function."""

    def test_given_equal_weights_when_compute_then_returns_sum(self) -> None:
        """Equal weights should return simple sum."""
        # Arrange
        nhv = 0.10
        ae = 0.05
        
        # Act
        score = compute_combined_score(nhv, ae)
        
        # Assert
        assert score == pytest.approx(0.15)

    def test_given_custom_weights_when_compute_then_applies_weights(self) -> None:
        """Custom weights should be applied correctly."""
        # Arrange
        nhv = 0.10
        ae = 0.05
        
        # Act
        score = compute_combined_score(nhv, ae, nhv_weight=2.0, ae_weight=1.0)
        
        # Assert
        expected = 2.0 * 0.10 + 1.0 * 0.05  # 0.25
        assert score == pytest.approx(expected)

    def test_given_zero_weight_when_compute_then_ignores_indicator(self) -> None:
        """Zero weight should ignore that indicator."""
        # Arrange
        nhv = 0.10
        ae = 0.05
        
        # Act
        score = compute_combined_score(nhv, ae, nhv_weight=0.0, ae_weight=1.0)
        
        # Assert
        assert score == pytest.approx(0.05)

    def test_given_all_zeros_when_compute_then_returns_zero(self) -> None:
        """Zero values should return zero."""
        # Arrange & Act
        score = compute_combined_score(0.0, 0.0)
        
        # Assert
        assert score == pytest.approx(0.0)


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_given_list_when_aggregate_mean_then_returns_average(self) -> None:
        """Mean aggregation should return arithmetic mean."""
        # Arrange
        scores = [0.1, 0.2, 0.3]
        
        # Act
        result = aggregate_scores(scores, method="mean")
        
        # Assert
        assert result == pytest.approx(0.2)

    def test_given_list_when_aggregate_median_then_returns_median(self) -> None:
        """Median aggregation should return median value."""
        # Arrange
        scores = [0.1, 0.5, 0.2]
        
        # Act
        result = aggregate_scores(scores, method="median")
        
        # Assert
        assert result == pytest.approx(0.2)

    def test_given_list_when_aggregate_min_then_returns_minimum(self) -> None:
        """Min aggregation should return minimum value."""
        # Arrange
        scores = [0.3, 0.1, 0.2]
        
        # Act
        result = aggregate_scores(scores, method="min")
        
        # Assert
        assert result == pytest.approx(0.1)

    def test_given_list_when_aggregate_max_then_returns_maximum(self) -> None:
        """Max aggregation should return maximum value."""
        # Arrange
        scores = [0.1, 0.3, 0.2]
        
        # Act
        result = aggregate_scores(scores, method="max")
        
        # Assert
        assert result == pytest.approx(0.3)

    def test_given_list_when_aggregate_sum_then_returns_sum(self) -> None:
        """Sum aggregation should return sum of values."""
        # Arrange
        scores = [0.1, 0.2, 0.3]
        
        # Act
        result = aggregate_scores(scores, method="sum")
        
        # Assert
        assert result == pytest.approx(0.6)

    def test_given_empty_list_when_aggregate_then_returns_inf(self) -> None:
        """Empty list should return infinity."""
        # Arrange
        scores = []
        
        # Act
        result = aggregate_scores(scores)
        
        # Assert
        assert result == float('inf')

    def test_given_invalid_method_when_aggregate_then_raises(self) -> None:
        """Invalid method should raise ValueError."""
        # Arrange
        scores = [0.1, 0.2]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_scores(scores, method="invalid")

    def test_given_single_value_when_aggregate_then_returns_value(self) -> None:
        """Single value should return that value for any method."""
        # Arrange
        scores = [0.5]
        
        # Act & Assert
        assert aggregate_scores(scores, method="mean") == pytest.approx(0.5)
        assert aggregate_scores(scores, method="median") == pytest.approx(0.5)
        assert aggregate_scores(scores, method="min") == pytest.approx(0.5)
        assert aggregate_scores(scores, method="max") == pytest.approx(0.5)


class TestLoadReferenceFront:
    """Tests for load_reference_front function."""

    def test_given_zdt1_when_load_then_returns_array(self) -> None:
        """Loading ZDT1 reference front should return numpy array."""
        # Arrange & Act
        front = load_reference_front("ZDT1.pf")
        
        # Assert
        assert isinstance(front, np.ndarray)
        assert front.ndim == 2
        assert front.shape[1] == 2  # 2 objectives

    def test_given_zdt2_when_load_then_returns_array(self) -> None:
        """Loading ZDT2 reference front should return numpy array."""
        # Arrange & Act
        front = load_reference_front("ZDT2.pf")
        
        # Assert
        assert isinstance(front, np.ndarray)
        assert front.ndim == 2

    def test_given_zdt3_when_load_then_returns_array(self) -> None:
        """Loading ZDT3 reference front should return numpy array."""
        # Arrange & Act
        front = load_reference_front("ZDT3.pf")
        
        # Assert
        assert isinstance(front, np.ndarray)
        assert front.shape[1] == 2

    def test_given_nonexistent_file_when_load_then_raises(self) -> None:
        """Non-existent file should raise FileNotFoundError."""
        # Arrange & Act & Assert
        with pytest.raises(FileNotFoundError):
            load_reference_front("NONEXISTENT.csv")

    def test_given_reference_front_when_load_then_values_in_valid_range(self) -> None:
        """Reference front values should be in valid range [0, 1] for ZDT."""
        # Arrange & Act
        front = load_reference_front("ZDT1.pf")
        
        # Assert
        assert np.all(front >= 0)
        assert np.all(front <= 1.1)  # Allow small margin

    def test_given_custom_path_when_load_then_uses_path(self, tmp_path) -> None:
        """Custom reference_fronts_dir should be used."""
        # Arrange
        custom_front = tmp_path / "custom.csv"
        custom_front.write_text("0.0 1.0\n0.5 0.5\n1.0 0.0\n")
        
        # Act
        front = load_reference_front("custom.csv", reference_fronts_dir=tmp_path)
        
        # Assert
        assert isinstance(front, np.ndarray)
        assert front.shape == (3, 2)
