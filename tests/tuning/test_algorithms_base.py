"""
Unit tests for jmetal.tuning.algorithms.base module.

Tests for ParameterInfo, TuningResult, and AlgorithmTuner base class.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jmetal.tuning.algorithms.base import (
    AlgorithmTuner,
    ParameterInfo,
    TuningResult,
)


class TestParameterInfo:
    """Tests for ParameterInfo dataclass."""

    def test_given_basic_params_when_create_then_stores_values(self) -> None:
        """ParameterInfo should store basic parameters correctly."""
        # Arrange & Act
        param = ParameterInfo(
            name="test_param",
            type="float",
            description="A test parameter",
        )
        
        # Assert
        assert param.name == "test_param"
        assert param.type == "float"
        assert param.description == "A test parameter"

    def test_given_range_params_when_create_then_stores_bounds(self) -> None:
        """ParameterInfo should store min/max bounds."""
        # Arrange & Act
        param = ParameterInfo(
            name="probability",
            type="float",
            description="Probability value",
            min_value=0.0,
            max_value=1.0,
        )
        
        # Assert
        assert param.min_value == 0.0
        assert param.max_value == 1.0

    def test_given_categorical_param_when_create_then_stores_choices(self) -> None:
        """ParameterInfo should store categorical choices."""
        # Arrange & Act
        param = ParameterInfo(
            name="operator_type",
            type="categorical",
            description="Type of operator",
            choices=["SBX", "BLX_ALPHA"],
        )
        
        # Assert
        assert param.choices == ["SBX", "BLX_ALPHA"]

    def test_given_conditional_param_when_create_then_stores_condition(self) -> None:
        """ParameterInfo should store conditional dependency."""
        # Arrange & Act
        param = ParameterInfo(
            name="sbx_distribution",
            type="float",
            description="SBX distribution index",
            min_value=5.0,
            max_value=30.0,
            conditional_on="crossover_type",
            conditional_value="SBX",
        )
        
        # Assert
        assert param.conditional_on == "crossover_type"
        assert param.conditional_value == "SBX"

    def test_given_basic_param_when_to_dict_then_returns_required_fields(self) -> None:
        """to_dict() should return dict with required fields."""
        # Arrange
        param = ParameterInfo(
            name="test",
            type="int",
            description="Test parameter",
        )
        
        # Act
        result = param.to_dict()
        
        # Assert
        assert result["name"] == "test"
        assert result["type"] == "int"
        assert result["description"] == "Test parameter"

    def test_given_param_with_range_when_to_dict_then_includes_bounds(self) -> None:
        """to_dict() should include min/max when set."""
        # Arrange
        param = ParameterInfo(
            name="prob",
            type="float",
            description="Probability",
            min_value=0.0,
            max_value=1.0,
        )
        
        # Act
        result = param.to_dict()
        
        # Assert
        assert result["min"] == 0.0
        assert result["max"] == 1.0

    def test_given_param_with_choices_when_to_dict_then_includes_choices(self) -> None:
        """to_dict() should include choices when set."""
        # Arrange
        param = ParameterInfo(
            name="type",
            type="categorical",
            description="Type selection",
            choices=["A", "B", "C"],
        )
        
        # Act
        result = param.to_dict()
        
        # Assert
        assert result["choices"] == ["A", "B", "C"]

    def test_given_param_with_default_when_to_dict_then_includes_default(self) -> None:
        """to_dict() should include default value when set."""
        # Arrange
        param = ParameterInfo(
            name="size",
            type="int",
            description="Size parameter",
            default=100,
        )
        
        # Act
        result = param.to_dict()
        
        # Assert
        assert result["default"] == 100

    def test_given_conditional_param_when_to_dict_then_includes_condition(self) -> None:
        """to_dict() should include conditional info when set."""
        # Arrange
        param = ParameterInfo(
            name="eta",
            type="float",
            description="Distribution index",
            conditional_on="operator",
            conditional_value="SBX",
        )
        
        # Act
        result = param.to_dict()
        
        # Assert
        assert result["conditional_on"] == "operator"
        assert result["conditional_value"] == "SBX"


class TestTuningResult:
    """Tests for TuningResult dataclass."""

    def test_given_required_params_when_create_then_stores_values(self) -> None:
        """TuningResult should store all required parameters."""
        # Arrange & Act
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={"pop_size": 100},
            best_score=0.05,
            n_trials=50,
            training_problems=["ZDT1", "ZDT2"],
            training_evaluations=10000,
        )
        
        # Assert
        assert result.algorithm_name == "NSGAII"
        assert result.best_params == {"pop_size": 100}
        assert result.best_score == 0.05
        assert result.n_trials == 50
        assert result.training_problems == ["ZDT1", "ZDT2"]
        assert result.training_evaluations == 10000

    def test_given_elapsed_time_when_create_then_stores_value(self) -> None:
        """TuningResult should store elapsed time."""
        # Arrange & Act
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={},
            best_score=0.1,
            n_trials=10,
            training_problems=["ZDT1"],
            training_evaluations=5000,
            elapsed_seconds=120.5,
        )
        
        # Assert
        assert result.elapsed_seconds == 120.5

    def test_given_extra_fields_when_create_then_stores_dict(self) -> None:
        """TuningResult should store extra metadata."""
        # Arrange
        extra = {"sampler": "tpe", "seed": 42}
        
        # Act
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={},
            best_score=0.1,
            n_trials=10,
            training_problems=["ZDT1"],
            training_evaluations=5000,
            extra=extra,
        )
        
        # Assert
        assert result.extra["sampler"] == "tpe"
        assert result.extra["seed"] == 42

    def test_given_result_when_to_dict_then_returns_serializable(self) -> None:
        """to_dict() should return JSON-serializable dict."""
        # Arrange
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={"prob": 0.9},
            best_score=0.05,
            n_trials=100,
            training_problems=["ZDT1"],
            training_evaluations=10000,
            elapsed_seconds=60.0,
        )
        
        # Act
        data = result.to_dict()
        
        # Assert
        assert data["algorithm"] == "NSGAII"
        assert data["best_value"] == 0.05
        assert data["best_params"] == {"prob": 0.9}
        assert data["n_trials"] == 100
        # Verify it's JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_given_result_with_extra_when_to_dict_then_includes_extra(self) -> None:
        """to_dict() should merge extra fields into output."""
        # Arrange
        result = TuningResult(
            algorithm_name="NSGAII",
            best_params={},
            best_score=0.1,
            n_trials=10,
            training_problems=["ZDT1"],
            training_evaluations=5000,
            extra={"custom_field": "value"},
        )
        
        # Act
        data = result.to_dict()
        
        # Assert
        assert data["custom_field"] == "value"


class TestAlgorithmTunerBase:
    """Tests for AlgorithmTuner abstract base class."""

    def test_given_tuner_when_describe_parameters_then_returns_formatted_string(
        self, nsgaii_tuner
    ) -> None:
        """describe_parameters() should return formatted description."""
        # Arrange & Act
        description = nsgaii_tuner.describe_parameters()
        
        # Assert
        assert isinstance(description, str)
        assert "NSGAII" in description
        assert "Parameter Space" in description

    def test_given_tuner_when_export_json_then_returns_valid_json(
        self, nsgaii_tuner
    ) -> None:
        """export_parameter_space(format='json') should return valid JSON."""
        # Arrange & Act
        json_str = nsgaii_tuner.export_parameter_space(format="json")
        
        # Assert
        assert json_str is not None
        data = json.loads(json_str)
        assert data["algorithm"] == "NSGAII"
        assert "parameters" in data
        assert len(data["parameters"]) > 0

    def test_given_tuner_when_export_yaml_then_returns_yaml_string(
        self, nsgaii_tuner
    ) -> None:
        """export_parameter_space(format='yaml') should return YAML string."""
        # Arrange & Act
        yaml_str = nsgaii_tuner.export_parameter_space(format="yaml")
        
        # Assert
        assert yaml_str is not None
        assert "algorithm: NSGAII" in yaml_str
        assert "parameters:" in yaml_str

    def test_given_tuner_when_export_txt_then_returns_readable_text(
        self, nsgaii_tuner
    ) -> None:
        """export_parameter_space(format='txt') should return readable text."""
        # Arrange & Act
        txt_str = nsgaii_tuner.export_parameter_space(format="txt")
        
        # Assert
        assert txt_str is not None
        assert "NSGAII" in txt_str
        assert "Description:" in txt_str

    def test_given_tuner_when_export_to_file_then_creates_file(
        self, nsgaii_tuner, tmp_path
    ) -> None:
        """export_parameter_space() should write to file when path given."""
        # Arrange
        output_path = tmp_path / "params.json"
        
        # Act
        result = nsgaii_tuner.export_parameter_space(output_path, format="json")
        
        # Assert
        assert result is None  # Returns None when writing to file
        assert output_path.exists()
        content = output_path.read_text()
        assert "NSGAII" in content

    def test_given_tuner_when_format_params_with_floats_then_formats_precision(
        self, nsgaii_tuner
    ) -> None:
        """format_params() should format floats with limited precision."""
        # Arrange - use NSGA-II specific params
        params = {
            "offspring_population_size": 100,
            "crossover_type": "sbx",
            "crossover_probability": 0.9,
            "crossover_eta": 20.0,
            "mutation_type": "polynomial",
            "mutation_probability_factor": 1.0,
            "mutation_eta": 20.0,
        }
        
        # Act
        result = nsgaii_tuner.format_params(params)
        
        # Assert
        assert "100" in result
        assert "0.9" in result or "0.900" in result

    def test_given_tuner_when_format_params_with_strings_then_formats_correctly(
        self, nsgaii_tuner
    ) -> None:
        """format_params() should format correctly for different operator types."""
        # Arrange - use BLX_ALPHA crossover
        params = {
            "offspring_population_size": 100,
            "crossover_type": "blx_alpha",
            "crossover_probability": 0.9,
            "blx_alpha": 0.5,
            "mutation_type": "uniform",
            "mutation_probability_factor": 1.0,
            "mutation_perturbation": 0.5,
        }
        
        # Act
        result = nsgaii_tuner.format_params(params)
        
        # Assert
        assert "BLX" in result
        assert "offspring=100" in result

    def test_given_valid_reference_front_when_load_then_returns_array(
        self, nsgaii_tuner
    ) -> None:
        """load_reference_front() should return numpy array for valid file."""
        # Arrange & Act
        import numpy as np
        front = nsgaii_tuner.load_reference_front("ZDT1.pf")
        
        # Assert
        assert isinstance(front, np.ndarray)
        assert front.ndim == 2
        assert front.shape[1] == 2  # 2 objectives for ZDT1

    def test_given_invalid_reference_front_when_load_then_raises_error(
        self, nsgaii_tuner
    ) -> None:
        """load_reference_front() should raise FileNotFoundError for invalid file."""
        # Arrange & Act & Assert
        with pytest.raises(FileNotFoundError, match="Reference front not found"):
            nsgaii_tuner.load_reference_front("NONEXISTENT.csv")
