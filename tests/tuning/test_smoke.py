"""
Smoke tests for jmetal.tuning package.

These tests verify basic functionality and should run quickly (~5 seconds).
Use these for quick CI checks to detect fundamental breakages.

Run with: pytest -m smoke tests/tuning/test_smoke.py
"""

import pytest


pytestmark = pytest.mark.smoke


class TestPackageImports:
    """Verify that all package imports work correctly."""

    def test_given_tuning_package_when_import_then_succeeds(self) -> None:
        """Importing the main tuning package should not raise errors."""
        # Arrange & Act
        import jmetal.tuning
        
        # Assert
        assert jmetal.tuning is not None

    def test_given_tuning_package_when_import_public_api_then_all_available(self) -> None:
        """All public API symbols should be importable from the main package."""
        # Arrange & Act
        from jmetal.tuning import (
            tune,
            describe_parameters,
            list_algorithms,
            AlgorithmTuner,
            TuningResult,
            ParameterInfo,
            NSGAIITuner,
            TUNERS,
            TuningObserver,
            TuningProgressObserver,
            TuningPlotObserver,
            TuningFileObserver,
            compute_quality_indicators,
            load_reference_front,
            aggregate_scores,
        )
        
        # Assert
        assert tune is not None
        assert describe_parameters is not None
        assert list_algorithms is not None
        assert AlgorithmTuner is not None
        assert TuningResult is not None
        assert ParameterInfo is not None
        assert NSGAIITuner is not None
        assert TUNERS is not None
        assert TuningObserver is not None
        assert TuningProgressObserver is not None
        assert TuningPlotObserver is not None
        assert TuningFileObserver is not None
        assert compute_quality_indicators is not None
        assert load_reference_front is not None
        assert aggregate_scores is not None

    def test_given_algorithms_submodule_when_import_then_succeeds(self) -> None:
        """Algorithms submodule should be importable."""
        # Arrange & Act
        from jmetal.tuning import algorithms
        from jmetal.tuning.algorithms import NSGAIITuner, AlgorithmTuner
        
        # Assert
        assert algorithms is not None
        assert NSGAIITuner is not None
        assert AlgorithmTuner is not None

    def test_given_observers_submodule_when_import_then_succeeds(self) -> None:
        """Observers submodule should be importable."""
        # Arrange & Act
        from jmetal.tuning import observers
        from jmetal.tuning.observers import (
            TuningObserver,
            TuningProgressObserver,
            TuningPlotObserver,
            TuningFileObserver,
        )
        
        # Assert
        assert observers is not None
        assert TuningObserver is not None
        assert TuningProgressObserver is not None
        assert TuningPlotObserver is not None
        assert TuningFileObserver is not None

    def test_given_metrics_submodule_when_import_then_succeeds(self) -> None:
        """Metrics submodule should be importable."""
        # Arrange & Act
        from jmetal.tuning import metrics
        from jmetal.tuning.metrics import (
            compute_quality_indicators,
            load_reference_front,
            aggregate_scores,
        )
        
        # Assert
        assert metrics is not None
        assert compute_quality_indicators is not None
        assert load_reference_front is not None
        assert aggregate_scores is not None

    def test_given_config_submodule_when_import_then_succeeds(self) -> None:
        """Config submodule should be importable."""
        # Arrange & Act
        from jmetal.tuning import config
        from jmetal.tuning.config import (
            POPULATION_SIZE,
            TRAINING_EVALUATIONS,
            TRAINING_PROBLEMS,
        )
        
        # Assert
        assert config is not None
        assert POPULATION_SIZE is not None
        assert TRAINING_EVALUATIONS is not None
        assert TRAINING_PROBLEMS is not None


class TestBasicInstantiation:
    """Verify that core classes can be instantiated."""

    def test_given_nsgaii_tuner_when_instantiate_then_succeeds(self) -> None:
        """NSGAIITuner should instantiate with default parameters."""
        # Arrange & Act
        from jmetal.tuning.algorithms import NSGAIITuner
        tuner = NSGAIITuner()
        
        # Assert
        assert tuner is not None
        assert tuner.name == "NSGAII"

    def test_given_nsgaii_tuner_when_instantiate_with_custom_population_then_stores_value(self) -> None:
        """NSGAIITuner should accept custom population size."""
        # Arrange
        from jmetal.tuning.algorithms import NSGAIITuner
        population_size = 200
        
        # Act
        tuner = NSGAIITuner(population_size=population_size)
        
        # Assert
        assert tuner.population_size == population_size

    def test_given_progress_observer_when_instantiate_then_succeeds(self) -> None:
        """TuningProgressObserver should instantiate with defaults."""
        # Arrange & Act
        from jmetal.tuning.observers import TuningProgressObserver
        observer = TuningProgressObserver()
        
        # Assert
        assert observer is not None

    def test_given_file_observer_when_instantiate_then_succeeds(self) -> None:
        """TuningFileObserver should instantiate with defaults."""
        # Arrange & Act
        from jmetal.tuning.observers import TuningFileObserver
        observer = TuningFileObserver()
        
        # Assert
        assert observer is not None

    def test_given_plot_observer_when_instantiate_then_succeeds(self) -> None:
        """TuningPlotObserver should instantiate with defaults."""
        # Arrange & Act
        from jmetal.tuning.observers import TuningPlotObserver
        observer = TuningPlotObserver()
        
        # Assert
        assert observer is not None

    def test_given_parameter_info_when_instantiate_then_succeeds(self) -> None:
        """ParameterInfo should instantiate correctly."""
        # Arrange & Act
        from jmetal.tuning.algorithms import ParameterInfo
        param = ParameterInfo(
            name="test_param",
            type="float",
            description="A test parameter",
            min_value=0.0,
            max_value=1.0,
        )
        
        # Assert
        assert param.name == "test_param"
        assert param.type == "float"

    def test_given_tuning_result_when_instantiate_then_succeeds(self) -> None:
        """TuningResult should instantiate correctly."""
        # Arrange & Act
        from jmetal.tuning.algorithms import TuningResult
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={"param1": 0.5},
            best_score=0.1,
            n_trials=10,
            training_problems=["ZDT1"],
            training_evaluations=10000,
        )
        
        # Assert
        assert result.algorithm_name == "NSGAII"
        assert result.best_score == 0.1


class TestMinimalAPI:
    """Verify that minimal API calls work."""

    def test_given_list_algorithms_when_call_then_returns_nonempty_list(self) -> None:
        """list_algorithms() should return a non-empty list."""
        # Arrange
        from jmetal.tuning import list_algorithms
        
        # Act
        algorithms = list_algorithms()
        
        # Assert
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "NSGAII" in algorithms

    def test_given_describe_parameters_when_call_for_nsgaii_then_returns_string(self) -> None:
        """describe_parameters() should return a description string."""
        # Arrange
        from jmetal.tuning import describe_parameters
        
        # Act
        description = describe_parameters("NSGAII")
        
        # Assert
        assert isinstance(description, str)
        assert len(description) > 0
        assert "NSGAII" in description

    def test_given_describe_parameters_when_invalid_algorithm_then_raises(self) -> None:
        """describe_parameters() should raise for unknown algorithm."""
        # Arrange
        from jmetal.tuning import describe_parameters
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown algorithm"):
            describe_parameters("INVALID_ALGO")

    def test_given_aggregate_scores_when_call_with_list_then_returns_float(self) -> None:
        """aggregate_scores() should return a float."""
        # Arrange
        from jmetal.tuning.metrics import aggregate_scores
        scores = [0.1, 0.2, 0.3]
        
        # Act
        result = aggregate_scores(scores)
        
        # Assert
        assert isinstance(result, float)
        assert result == pytest.approx(0.2)  # mean of [0.1, 0.2, 0.3]

    def test_given_tuners_registry_when_access_then_contains_nsgaii(self) -> None:
        """TUNERS registry should contain NSGAII."""
        # Arrange
        from jmetal.tuning import TUNERS
        
        # Act & Assert
        assert "NSGAII" in TUNERS
        assert TUNERS["NSGAII"] is not None


class TestTuneSingleTrial:
    """Test tune() with minimal configuration for smoke testing."""

    @pytest.mark.slow
    def test_given_tune_when_single_trial_then_returns_result(self) -> None:
        """tune() with 1 trial should complete and return a TuningResult."""
        # Arrange
        from jmetal.tuning import tune, TuningResult
        
        # Act
        result = tune(
            algorithm="NSGAII",
            n_trials=1,
            n_evaluations=1000,  # Minimal evaluations for speed
            verbose=False,
        )
        
        # Assert
        assert isinstance(result, TuningResult)
        assert result.algorithm_name == "NSGAII"
        assert result.n_trials == 1
        assert result.best_score > 0
        assert len(result.best_params) > 0
