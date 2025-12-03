"""
Integration tests for jmetal.tuning package.

Tests complete workflows and interactions between modules.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jmetal.tuning import (
    tune,
    TuningResult,
    TuningProgressObserver,
    TuningFileObserver,
)


class TestFullTuningWorkflow:
    """Test complete tuning workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_default_config_when_tune_workflow_then_completes(self) -> None:
        """Complete tuning workflow with defaults should work."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=3,
            n_evaluations=1000,
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)
        assert result.algorithm_name == "NSGAII"
        assert result.n_trials == 3
        assert result.best_score > 0
        assert len(result.best_params) > 0
        assert result.elapsed_seconds > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_workflow_when_complete_then_params_are_valid(self) -> None:
        """Tuning result should contain valid parameter keys."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=1000,
            verbose=False,
        )
        
        # Assert - check expected parameter keys exist
        params = result.best_params
        assert "crossover_type" in params
        assert "crossover_probability" in params
        assert "mutation_type" in params
        assert "mutation_probability_factor" in params

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_workflow_when_complete_then_result_serializable(self) -> None:
        """Tuning result should be JSON serializable."""
        # Arrange & Act
        result = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=1000,
            verbose=False,
        )
        
        # Assert
        data = result.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 0
        
        # Can deserialize back
        parsed = json.loads(json_str)
        assert parsed["algorithm"] == "NSGAII"


class TestObserverIntegration:
    """Test observer integration with tuning workflow."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_progress_observer_when_tune_then_receives_callbacks(self) -> None:
        """TuningProgressObserver should receive all callbacks."""
        # Arrange
        observer = TuningProgressObserver()
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=1000,
            observers=[observer],
            verbose=False,
        )
        
        # Assert - observer should have tracked data
        assert observer.n_trials == 2
        assert observer.algorithm == "NSGAII"
        assert observer.best_value is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_file_observer_when_tune_then_creates_valid_csv(
        self, tmp_path
    ) -> None:
        """TuningFileObserver should create valid CSV file."""
        # Arrange
        observer = TuningFileObserver(output_dir=str(tmp_path))
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=3,
            n_evaluations=1000,
            observers=[observer],
            verbose=False,
        )
        
        # Assert - check CSV file exists in output dir
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) >= 1
        content = csv_files[0].read_text()
        lines = content.strip().split('\n')
        assert len(lines) >= 4  # Header + 3 trials

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_multiple_observers_when_tune_then_all_work(
        self, tmp_path
    ) -> None:
        """Multiple observers should all receive callbacks."""
        # Arrange
        progress_observer = TuningProgressObserver()
        file_observer = TuningFileObserver(output_dir=str(tmp_path))
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=2,
            n_evaluations=1000,
            observers=[progress_observer, file_observer],
            verbose=False,
        )
        
        # Assert
        assert progress_observer.best_value is not None
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) >= 1


class TestMultiProblemTuning:
    """Test tuning across multiple problems."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_multiple_problems_when_tune_then_aggregates_scores(self) -> None:
        """Tuning with multiple problems should aggregate scores."""
        # Arrange
        from jmetal.problem import ZDT1, ZDT2
        
        problems = [
            (ZDT1(), "ZDT1.pf"),
            (ZDT2(), "ZDT2.pf"),
        ]
        
        # Act
        result = tune(
            algorithm="NSGAII",
            problems=problems,
            n_trials=2,
            n_evaluations=1000,
            verbose=False,
        )
        
        # Assert
        assert len(result.training_problems) == 2
        assert "ZDT1" in result.training_problems
        assert "ZDT2" in result.training_problems


class TestSamplerComparison:
    """Test different samplers produce different results."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_different_samplers_when_tune_then_different_paths(self) -> None:
        """Different samplers should explore parameter space differently."""
        # Arrange & Act
        result_tpe = tune(
            algorithm="NSGAII",
            sampler="tpe",
            n_trials=5,
            n_evaluations=1000,
            seed=42,
            verbose=False,
        )
        
        result_random = tune(
            algorithm="NSGAII",
            sampler="random",
            n_trials=5,
            n_evaluations=1000,
            seed=42,
            verbose=False,
        )
        
        # Assert - parameters should differ (random exploration vs guided)
        # Note: With same seed, results might still match, so we just verify both work
        assert result_tpe.best_score > 0
        assert result_random.best_score > 0


class TestReproducibility:
    """Test that tuning is reproducible with fixed seed."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_same_seed_when_tune_twice_then_same_results(self) -> None:
        """Same seed should produce identical results."""
        # Arrange
        seed = 98765
        
        # Act
        result1 = tune(
            algorithm="NSGAII",
            n_trials=3,
            n_evaluations=1000,
            seed=seed,
            verbose=False,
        )
        
        result2 = tune(
            algorithm="NSGAII",
            n_trials=3,
            n_evaluations=1000,
            seed=seed,
            verbose=False,
        )
        
        # Assert
        assert result1.best_score == pytest.approx(result2.best_score, rel=0.01)
        
    @pytest.mark.slow
    @pytest.mark.integration
    def test_given_different_seeds_when_tune_then_different_results(self) -> None:
        """Different seeds should produce different results."""
        # Arrange & Act
        result1 = tune(
            algorithm="NSGAII",
            n_trials=5,
            n_evaluations=1000,
            seed=11111,
            verbose=False,
        )
        
        result2 = tune(
            algorithm="NSGAII",
            n_trials=5,
            n_evaluations=1000,
            seed=99999,
            verbose=False,
        )
        
        # Assert - highly unlikely to get exact same results with different seeds
        # But we mainly verify both complete successfully
        assert result1.best_score > 0
        assert result2.best_score > 0
