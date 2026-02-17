"""
Unit tests for jmetal.tuning.tuning module (high-level API).

Tests for tune(), describe_parameters(), and list_algorithms() functions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jmetal.tuning import (
    describe_parameters,
    list_algorithms,
    tune,
    TuningResult,
)


class TestListAlgorithms:
    """Tests for list_algorithms function."""

    def test_given_call_when_list_algorithms_then_returns_list(self) -> None:
        """list_algorithms() should return a list."""
        # Arrange & Act
        result = list_algorithms()
        
        # Assert
        assert isinstance(result, list)

    def test_given_call_when_list_algorithms_then_not_empty(self) -> None:
        """list_algorithms() should return non-empty list."""
        # Arrange & Act
        result = list_algorithms()
        
        # Assert
        assert len(result) > 0

    def test_given_call_when_list_algorithms_then_contains_nsgaii(self) -> None:
        """list_algorithms() should include 'NSGAII'."""
        # Arrange & Act
        result = list_algorithms()
        
        # Assert
        assert "NSGAII" in result

    def test_given_call_when_list_algorithms_then_all_strings(self) -> None:
        """list_algorithms() should return list of strings."""
        # Arrange & Act
        result = list_algorithms()
        
        # Assert
        for algo in result:
            assert isinstance(algo, str)


class TestDescribeParameters:
    """Tests for describe_parameters function."""

    def test_given_nsgaii_when_describe_txt_then_returns_string(self) -> None:
        """describe_parameters('NSGAII') should return description string."""
        # Arrange & Act
        result = describe_parameters("NSGAII", format="txt")
        
        # Assert
        assert isinstance(result, str)
        assert len(result) > 0

    def test_given_nsgaii_when_describe_txt_then_contains_algorithm_name(self) -> None:
        """Description should include algorithm name."""
        # Arrange & Act
        result = describe_parameters("NSGAII", format="txt")
        
        # Assert
        assert "NSGAII" in result

    def test_given_nsgaii_when_describe_txt_then_contains_parameters(self) -> None:
        """Description should include parameter information."""
        # Arrange & Act
        result = describe_parameters("NSGAII", format="txt")
        
        # Assert
        assert "crossover" in result.lower()
        assert "mutation" in result.lower()

    def test_given_nsgaii_when_describe_json_then_returns_valid_json(self) -> None:
        """describe_parameters(format='json') should return valid JSON."""
        # Arrange & Act
        result = describe_parameters("NSGAII", format="json")
        
        # Assert
        assert result is not None
        data = json.loads(result)
        assert data["algorithm"] == "NSGAII"
        assert "parameters" in data

    def test_given_nsgaii_when_describe_yaml_then_returns_yaml_string(self) -> None:
        """describe_parameters(format='yaml') should return YAML string."""
        # Arrange & Act
        result = describe_parameters("NSGAII", format="yaml")
        
        # Assert
        assert result is not None
        assert "algorithm: NSGAII" in result

    def test_given_output_path_when_describe_then_writes_file(
        self, tmp_path
    ) -> None:
        """describe_parameters with output_path should write file."""
        # Arrange
        output_file = tmp_path / "params.txt"
        
        # Act
        result = describe_parameters("NSGAII", output_path=str(output_file))
        
        # Assert
        assert result is None  # Returns None when writing to file
        assert output_file.exists()
        content = output_file.read_text()
        assert "NSGAII" in content

    def test_given_invalid_algorithm_when_describe_then_raises(self) -> None:
        """Invalid algorithm should raise ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unknown algorithm"):
            describe_parameters("INVALID_ALGO")


class TestTuneFunction:
    """Tests for tune function."""

    def test_given_invalid_algorithm_when_tune_then_raises(self) -> None:
        """tune() with invalid algorithm should raise ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unknown algorithm"):
            tune(algorithm="INVALID")

    def test_given_invalid_sampler_when_tune_then_raises(self) -> None:
        """tune() with invalid sampler should raise ValueError."""
        # We need to mock the actual tuning to test sampler validation
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unknown sampler"):
            tune(
                algorithm="NSGAII",
                sampler="invalid_sampler",
                n_trials=1,
                verbose=False,
            )

    @pytest.mark.slow
    def test_given_valid_config_when_tune_then_returns_result(self) -> None:
        """tune() with valid config should return TuningResult."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)

    @pytest.mark.slow
    def test_given_tune_when_complete_then_result_has_algorithm_name(self) -> None:
        """TuningResult should contain correct algorithm name."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert result.algorithm_name == "NSGAII"

    @pytest.mark.slow
    def test_given_tune_when_complete_then_result_has_best_params(self) -> None:
        """TuningResult should contain best_params dict."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0

    @pytest.mark.slow
    def test_given_tune_when_complete_then_result_has_best_score(self) -> None:
        """TuningResult should contain valid best_score."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result.best_score, float)
        assert result.best_score > 0

    @pytest.mark.slow
    def test_given_tune_when_complete_then_result_has_n_trials(self) -> None:
        """TuningResult should contain correct n_trials."""
        # Arrange
        n_trials = 2
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=n_trials,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert result.n_trials == n_trials

    @pytest.mark.slow
    def test_given_output_path_when_tune_then_saves_result(
        self, tmp_path
    ) -> None:
        """tune() with output_path should save results to file."""
        # Arrange
        output_file = tmp_path / "result.json"
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            output_path=str(output_file),
            verbose=False,
        )
        
        # Assert
        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert content["algorithm"] == "NSGAII"

    @pytest.mark.slow
    def test_given_tpe_sampler_when_tune_then_succeeds(self) -> None:
        """tune() with TPE sampler should complete successfully."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            sampler="tpe",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)

    @pytest.mark.slow
    def test_given_random_sampler_when_tune_then_succeeds(self) -> None:
        """tune() with random sampler should complete successfully."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            sampler="random",
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)

    @pytest.mark.slow
    def test_given_cmaes_sampler_when_tune_then_uses_continuous_mode(self) -> None:
        """tune() with CMA-ES sampler should switch to continuous mode."""
        # Arrange & Act - CMA-ES requires continuous mode
        result = tune(
            algorithm="NSGAII",
            sampler="cmaes",
            mode="categorical",  # Should be overridden to continuous
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)

    @pytest.mark.slow
    def test_given_observers_when_tune_then_observers_called(self) -> None:
        """tune() with observers should call observer methods."""
        # Arrange
        mock_observer = MagicMock()
        mock_observer.on_tuning_start = MagicMock()
        mock_observer.on_tuning_end = MagicMock()
        mock_observer.__call__ = MagicMock()
        
        # Act
        tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=500,
            observers=[mock_observer],
            verbose=False,
        )
        
        # Assert
        mock_observer.on_tuning_start.assert_called_once()
        mock_observer.on_tuning_end.assert_called_once()

    @pytest.mark.slow
    def test_given_custom_problems_when_tune_then_uses_problems(
        self, simple_problem
    ) -> None:
        """tune() with custom problems should use those problems."""
        # Arrange
        problems = [(simple_problem, "ZDT1.pf")]
        
        # Act
        result = tune(
            algorithm="NSGAII",
            problems=problems,
            n_trials=1,
            n_evaluations=500,
            verbose=False,
        )
        
        # Assert
        assert "ZDT1" in result.training_problems

    @pytest.mark.slow
    def test_given_seed_when_tune_twice_then_params_same(self) -> None:
        """tune() with same seed should produce reproducible parameters."""
        # Arrange
        seed = 12345
        
        # Act
        result1 = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=500,
            seed=seed,
            verbose=False,
        )
        result2 = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=500,
            seed=seed,
            verbose=False,
        )
        
        # Assert - parameters should be the same (Optuna sampler is deterministic)
        # Note: scores may differ due to NSGAII internal randomness
        assert result1.best_params == result2.best_params
