"""
Unit tests for jmetal.tuning.tuning_config module.

Tests for TuningConfig dataclass and YAML loading functionality.
"""

from pathlib import Path
import tempfile

import pytest

from jmetal.tuning.tuning_config import (
    TuningConfig,
    ParameterRange,
    CategoricalParameter,
    CrossoverConfig,
    MutationConfig,
    ParameterSpaceConfig,
    ProblemConfig,
    OutputConfig,
)


class TestParameterRange:
    """Tests for ParameterRange dataclass."""

    def test_given_valid_range_when_create_then_stores_values(self) -> None:
        """ParameterRange should store min and max values."""
        # Arrange & Act
        param_range = ParameterRange(min=0.5, max=1.0)
        
        # Assert
        assert param_range.min == 0.5
        assert param_range.max == 1.0

    def test_given_invalid_range_when_create_then_raises_error(self) -> None:
        """ParameterRange should raise error if min > max."""
        # Act & Assert
        with pytest.raises(ValueError, match="min.*must be <= max"):
            ParameterRange(min=1.0, max=0.5)


class TestCategoricalParameter:
    """Tests for CategoricalParameter dataclass."""

    def test_given_valid_values_when_create_then_stores_values(self) -> None:
        """CategoricalParameter should store values list."""
        # Arrange & Act
        param = CategoricalParameter(values=[1, 10, 50, 100])
        
        # Assert
        assert param.values == [1, 10, 50, 100]

    def test_given_empty_values_when_create_then_raises_error(self) -> None:
        """CategoricalParameter should raise error if values is empty."""
        # Act & Assert
        with pytest.raises(ValueError, match="at least one value"):
            CategoricalParameter(values=[])


class TestCrossoverConfig:
    """Tests for CrossoverConfig dataclass."""

    def test_given_default_config_when_create_then_has_both_types(self) -> None:
        """Default CrossoverConfig should have both SBX and BLX-alpha."""
        # Arrange & Act
        config = CrossoverConfig()
        
        # Assert
        assert "sbx" in config.types
        assert "blxalpha" in config.types

    def test_given_single_type_when_check_is_fixed_then_true(self) -> None:
        """is_fixed_type should return True when only one type."""
        # Arrange
        config = CrossoverConfig(types=["sbx"])
        
        # Act & Assert
        assert config.is_fixed_type() is True
        assert config.get_fixed_type() == "sbx"

    def test_given_multiple_types_when_check_is_fixed_then_false(self) -> None:
        """is_fixed_type should return False when multiple types."""
        # Arrange
        config = CrossoverConfig(types=["sbx", "blxalpha"])
        
        # Act & Assert
        assert config.is_fixed_type() is False
        assert config.get_fixed_type() is None


class TestMutationConfig:
    """Tests for MutationConfig dataclass."""

    def test_given_default_config_when_create_then_has_both_types(self) -> None:
        """Default MutationConfig should have both polynomial and uniform."""
        # Arrange & Act
        config = MutationConfig()
        
        # Assert
        assert "polynomial" in config.types
        assert "uniform" in config.types


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_given_default_config_when_create_then_has_default_problems(self) -> None:
        """Default TuningConfig should have ZDT problems."""
        # Arrange & Act
        config = TuningConfig()
        
        # Assert
        assert len(config.problems) == 5
        problem_names = [p.name for p in config.problems]
        assert "ZDT1" in problem_names

    def test_given_config_when_to_dict_then_serializable(self) -> None:
        """TuningConfig.to_dict should return a serializable dictionary."""
        # Arrange
        config = TuningConfig(n_trials=50, n_evaluations=5000)
        
        # Act
        data = config.to_dict()
        
        # Assert
        assert data["n_trials"] == 50
        assert data["n_evaluations"] == 5000
        assert "parameter_space" in data
        assert "problems" in data

    def test_given_dict_when_from_dict_then_creates_config(self) -> None:
        """TuningConfig.from_dict should create config from dictionary."""
        # Arrange
        data = {
            "algorithm": "NSGAII",
            "n_trials": 75,
            "n_evaluations": 8000,
            "seed": 123,
            "problems": [
                {"name": "ZDT1", "reference_front": "ZDT1.pf"},
            ],
        }
        
        # Act
        config = TuningConfig.from_dict(data)
        
        # Assert
        assert config.n_trials == 75
        assert config.n_evaluations == 8000
        assert config.seed == 123
        assert len(config.problems) == 1
        assert config.problems[0].name == "ZDT1"


class TestTuningConfigYAML:
    """Tests for TuningConfig YAML loading."""

    def test_given_yaml_file_when_load_then_creates_config(self) -> None:
        """TuningConfig.from_yaml should load configuration from file."""
        # Arrange
        yaml_content = """
algorithm: NSGAII
n_trials: 30
n_evaluations: 6000
seed: 99

problems:
  - name: ZDT1
    reference_front: ZDT1.pf
  - name: ZDT2
    reference_front: ZDT2.pf

parameter_space:
  offspring_population_size:
    values: [50, 100]
  crossover:
    type: sbx
    probability: {min: 0.9, max: 1.0}
  mutation:
    type: polynomial
    probability_factor: {min: 0.8, max: 1.2}
"""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Act
            config = TuningConfig.from_yaml(temp_path)
            
            # Assert
            assert config.n_trials == 30
            assert config.n_evaluations == 6000
            assert config.seed == 99
            assert len(config.problems) == 2
            assert config.parameter_space.crossover.types == ["sbx"]
            assert config.parameter_space.mutation.types == ["polynomial"]
        finally:
            Path(temp_path).unlink()

    def test_given_nonexistent_file_when_load_then_raises_error(self) -> None:
        """TuningConfig.from_yaml should raise error for missing file."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            TuningConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_given_config_when_save_and_load_then_roundtrip(self) -> None:
        """TuningConfig should survive save/load roundtrip."""
        # Arrange
        original = TuningConfig(
            n_trials=42,
            n_evaluations=7500,
            seed=777,
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
        
        try:
            # Act
            original.to_yaml(temp_path)
            loaded = TuningConfig.from_yaml(temp_path)
            
            # Assert
            assert loaded.n_trials == 42
            assert loaded.n_evaluations == 7500
            assert loaded.seed == 777
        finally:
            Path(temp_path).unlink()


class TestTuningConfigGetProblems:
    """Tests for TuningConfig.get_problems_as_tuples."""

    def test_given_zdt_problems_when_get_tuples_then_returns_instances(self) -> None:
        """get_problems_as_tuples should return Problem instances."""
        # Arrange
        config = TuningConfig(problems=[
            ProblemConfig("ZDT1", "ZDT1.pf"),
            ProblemConfig("ZDT2", "ZDT2.pf"),
        ])
        
        # Act
        problems = config.get_problems_as_tuples()
        
        # Assert
        assert len(problems) == 2
        assert problems[0][0].name() == "ZDT1"
        assert problems[0][1] == "ZDT1.pf"
        assert problems[1][0].name() == "ZDT2"

    def test_given_unknown_problem_when_get_tuples_then_raises_error(self) -> None:
        """get_problems_as_tuples should raise error for unknown problems."""
        # Arrange
        config = TuningConfig(problems=[
            ProblemConfig("NonExistentProblem", "NonExistent.pf"),
        ])
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown problem"):
            config.get_problems_as_tuples()


class TestParameterSpaceIntegration:
    """Tests for ParameterSpaceConfig integration with NSGAIITuner."""

    def test_given_sbx_only_config_when_sample_then_only_sbx_crossover(self) -> None:
        """When crossover is fixed to SBX, sampled params should only contain SBX."""
        from jmetal.tuning.algorithms import NSGAIITuner
        from unittest.mock import MagicMock
        
        # Arrange - Create config with only SBX crossover
        param_space = ParameterSpaceConfig(
            crossover=CrossoverConfig(
                types=["sbx"],
                probability=ParameterRange(min=0.9, max=1.0),
                sbx_distribution_index=ParameterRange(min=5.0, max=20.0),
            ),
            mutation=MutationConfig(types=["polynomial"]),
        )
        tuner = NSGAIITuner(parameter_space=param_space)
        
        # Mock Optuna trial
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.return_value = 100
        mock_trial.suggest_float.side_effect = [0.95, 15.0, 1.0, 20.0]  # prob, eta, mut_factor, mut_eta
        
        # Act
        params = tuner.sample_parameters(mock_trial, mode="categorical")
        
        # Assert
        assert params["crossover_type"] == "sbx"
        assert "blx_alpha" not in params
        assert "crossover_eta" in params

    def test_given_custom_offspring_values_when_sample_then_uses_values(self) -> None:
        """When offspring_population_size has custom values, should use them."""
        from jmetal.tuning.algorithms import NSGAIITuner
        from unittest.mock import MagicMock
        
        # Arrange - Create config with custom offspring values
        param_space = ParameterSpaceConfig(
            offspring_population_size=CategoricalParameter(values=[25, 50, 75]),
        )
        tuner = NSGAIITuner(parameter_space=param_space)
        
        # Mock Optuna trial
        mock_trial = MagicMock()
        # Order: offspring, crossover_type, mutation_type, selection_type
        mock_trial.suggest_categorical.side_effect = [50, "sbx", "polynomial", "tournament"]
        mock_trial.suggest_float.side_effect = [0.9, 15.0, 1.0, 20.0]
        mock_trial.suggest_int.return_value = 2
        
        # Act
        params = tuner.sample_parameters(mock_trial, mode="categorical")
        
        # Assert
        # Verify the values passed to suggest_categorical include our custom values
        calls = mock_trial.suggest_categorical.call_args_list
        offspring_call = [c for c in calls if c[0][0] == "offspring_population_size"]
        assert len(offspring_call) == 1
        assert offspring_call[0][0][1] == [25, 50, 75]

    def test_given_no_parameter_space_when_sample_then_uses_defaults(self) -> None:
        """When no parameter_space, should use default parameter space."""
        from jmetal.tuning.algorithms import NSGAIITuner
        from unittest.mock import MagicMock
        
        # Arrange
        tuner = NSGAIITuner()  # No parameter_space
        
        # Mock Optuna trial
        mock_trial = MagicMock()
        # Order: offspring, crossover_type, mutation_type, selection_type
        mock_trial.suggest_categorical.side_effect = [100, "sbx", "polynomial", "tournament"]
        mock_trial.suggest_float.side_effect = [0.9, 15.0, 1.0, 20.0]
        mock_trial.suggest_int.return_value = 2
        
        # Act
        params = tuner.sample_parameters(mock_trial, mode="categorical")
        
        # Assert - default values should work
        assert "crossover_type" in params
        assert "mutation_type" in params
        assert "offspring_population_size" in params
