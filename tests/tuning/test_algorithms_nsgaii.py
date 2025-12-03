"""
Unit tests for jmetal.tuning.algorithms.nsgaii module.

Tests for NSGAIITuner class.
"""

from unittest.mock import MagicMock, patch

import pytest

from jmetal.tuning.algorithms.nsgaii import NSGAIITuner


class TestNSGAIITunerProperties:
    """Tests for NSGAIITuner basic properties."""

    def test_given_tuner_when_get_name_then_returns_nsgaii(self) -> None:
        """name property should return 'NSGAII'."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        name = tuner.name
        
        # Assert
        assert name == "NSGAII"

    def test_given_default_tuner_when_get_population_then_returns_default(self) -> None:
        """Default population_size should match config default."""
        # Arrange
        from jmetal.tuning.config import POPULATION_SIZE
        tuner = NSGAIITuner()
        
        # Act
        pop_size = tuner.population_size
        
        # Assert
        assert pop_size == POPULATION_SIZE

    def test_given_custom_population_when_create_then_stores_value(self) -> None:
        """Custom population_size should be stored."""
        # Arrange
        custom_size = 200
        
        # Act
        tuner = NSGAIITuner(population_size=custom_size)
        
        # Assert
        assert tuner.population_size == custom_size


class TestNSGAIITunerParameterSampling:
    """Tests for NSGAIITuner parameter sampling."""

    def test_given_categorical_mode_when_sample_then_uses_suggest_categorical(
        self, mock_optuna_trial
    ) -> None:
        """Categorical mode should use suggest_categorical for operator types."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.return_value = "SBX"
        mock_optuna_trial.suggest_float.return_value = 0.9
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        mock_optuna_trial.suggest_categorical.assert_called()
        assert "crossover_type" in params

    def test_given_continuous_mode_when_sample_then_uses_suggest_float(
        self, mock_optuna_trial
    ) -> None:
        """Continuous mode should use suggest_float for all parameters."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_float.return_value = 0.5
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="continuous")
        
        # Assert
        # In continuous mode, operator selection is based on float thresholds
        assert mock_optuna_trial.suggest_float.called

    def test_given_sbx_crossover_when_sample_then_includes_crossover_eta(
        self, mock_optuna_trial
    ) -> None:
        """SBX crossover should include crossover_eta parameter."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.side_effect = [1, "sbx", "polynomial"]
        mock_optuna_trial.suggest_float.return_value = 20.0
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        assert "crossover_eta" in params

    def test_given_blx_crossover_when_sample_then_includes_alpha(
        self, mock_optuna_trial
    ) -> None:
        """BLX_ALPHA crossover should include blx_alpha parameter."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.side_effect = [1, "blxalpha", "polynomial"]
        mock_optuna_trial.suggest_float.return_value = 0.5
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        assert "blx_alpha" in params

    def test_given_polynomial_mutation_when_sample_then_includes_mutation_eta(
        self, mock_optuna_trial
    ) -> None:
        """Polynomial mutation should include mutation_eta."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.side_effect = [1, "sbx", "polynomial"]
        mock_optuna_trial.suggest_float.return_value = 20.0
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        assert "mutation_eta" in params

    def test_given_sample_when_call_then_includes_crossover_probability(
        self, mock_optuna_trial
    ) -> None:
        """Sampled parameters should always include crossover_probability."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.side_effect = [1, "sbx", "polynomial"]
        mock_optuna_trial.suggest_float.return_value = 0.9
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        assert "crossover_probability" in params

    def test_given_sample_when_call_then_includes_mutation_probability_factor(
        self, mock_optuna_trial
    ) -> None:
        """Sampled parameters should always include mutation_probability_factor."""
        # Arrange
        tuner = NSGAIITuner()
        mock_optuna_trial.suggest_categorical.side_effect = [1, "sbx", "polynomial"]
        mock_optuna_trial.suggest_float.return_value = 1.0
        
        # Act
        params = tuner.sample_parameters(mock_optuna_trial, mode="categorical")
        
        # Assert
        assert "mutation_probability_factor" in params


class TestNSGAIITunerAlgorithmCreation:
    """Tests for NSGAIITuner.create_algorithm()."""

    def test_given_sbx_params_when_create_algorithm_then_returns_nsgaii(
        self, simple_problem
    ) -> None:
        """create_algorithm() with SBX params should return NSGAII instance."""
        # Arrange
        tuner = NSGAIITuner(population_size=50)
        params = {
            "offspring_population_size": 50,
            "crossover_type": "sbx",
            "crossover_probability": 0.9,
            "crossover_eta": 20.0,
            "mutation_type": "polynomial",
            "mutation_probability_factor": 1.0,
            "mutation_eta": 20.0,
        }
        
        # Act
        algorithm = tuner.create_algorithm(simple_problem, params, max_evaluations=1000)
        
        # Assert
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        assert isinstance(algorithm, NSGAII)

    def test_given_blx_params_when_create_algorithm_then_returns_nsgaii(
        self, simple_problem
    ) -> None:
        """create_algorithm() with BLX_ALPHA params should return NSGAII instance."""
        # Arrange
        tuner = NSGAIITuner(population_size=50)
        params = {
            "offspring_population_size": 50,
            "crossover_type": "blxalpha",
            "crossover_probability": 0.9,
            "blx_alpha": 0.5,
            "mutation_type": "uniform",
            "mutation_probability_factor": 1.0,
            "mutation_perturbation": 0.5,
        }
        
        # Act
        algorithm = tuner.create_algorithm(simple_problem, params, max_evaluations=1000)
        
        # Assert
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        assert isinstance(algorithm, NSGAII)

    def test_given_uniform_mutation_when_create_algorithm_then_succeeds(
        self, simple_problem
    ) -> None:
        """create_algorithm() with Uniform mutation should succeed."""
        # Arrange
        tuner = NSGAIITuner(population_size=50)
        params = {
            "offspring_population_size": 50,
            "crossover_type": "sbx",
            "crossover_probability": 0.9,
            "crossover_eta": 20.0,
            "mutation_type": "uniform",
            "mutation_probability_factor": 1.0,
            "mutation_perturbation": 0.5,
        }
        
        # Act
        algorithm = tuner.create_algorithm(simple_problem, params, max_evaluations=1000)
        
        # Assert
        assert algorithm is not None


class TestNSGAIITunerParameterSpace:
    """Tests for NSGAIITuner.get_parameter_space()."""

    def test_given_tuner_when_get_parameter_space_then_returns_list(self) -> None:
        """get_parameter_space() should return a list of ParameterInfo."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        params = tuner.get_parameter_space()
        
        # Assert
        assert isinstance(params, list)
        assert len(params) > 0

    def test_given_tuner_when_get_parameter_space_then_includes_crossover_type(
        self
    ) -> None:
        """Parameter space should include crossover_type."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        params = tuner.get_parameter_space()
        param_names = [p.name for p in params]
        
        # Assert
        assert "crossover_type" in param_names

    def test_given_tuner_when_get_parameter_space_then_includes_mutation_type(
        self
    ) -> None:
        """Parameter space should include mutation_type."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        params = tuner.get_parameter_space()
        param_names = [p.name for p in params]
        
        # Assert
        assert "mutation_type" in param_names

    def test_given_tuner_when_get_parameter_space_then_includes_probabilities(
        self
    ) -> None:
        """Parameter space should include probability parameters."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        params = tuner.get_parameter_space()
        param_names = [p.name for p in params]
        
        # Assert
        assert "crossover_probability" in param_names
        assert "mutation_probability_factor" in param_names

    def test_given_tuner_when_get_parameter_space_then_conditional_params_have_condition(
        self
    ) -> None:
        """Conditional parameters should have conditional_on set."""
        # Arrange
        tuner = NSGAIITuner()
        
        # Act
        params = tuner.get_parameter_space()
        crossover_eta_param = next((p for p in params if p.name == "crossover_eta"), None)
        
        # Assert
        assert crossover_eta_param is not None
        assert crossover_eta_param.conditional_on == "crossover_type"
        assert crossover_eta_param.conditional_value == "sbx"


class TestNSGAIITunerEvaluation:
    """Tests for NSGAIITuner evaluation methods."""

    @pytest.mark.slow
    def test_given_valid_config_when_evaluate_then_returns_tuple(
        self, simple_problem
    ) -> None:
        """evaluate() should return tuple of (nhv, epsilon)."""
        # Arrange
        tuner = NSGAIITuner(population_size=20)
        params = {
            "offspring_population_size": 20,
            "crossover_type": "sbx",
            "crossover_probability": 0.9,
            "crossover_eta": 20.0,
            "mutation_type": "polynomial",
            "mutation_probability_factor": 1.0,
            "mutation_eta": 20.0,
        }
        
        # Act
        nhv, epsilon = tuner.evaluate(
            problem=simple_problem,
            reference_front_file="ZDT1.pf",
            params=params,
            max_evaluations=1000,
            n_repeats=1,
        )
        
        # Assert
        assert isinstance(nhv, float)
        assert isinstance(epsilon, float)
        assert nhv >= 0
        assert epsilon >= 0

    @pytest.mark.slow
    def test_given_multiple_problems_when_evaluate_on_problems_then_returns_score(
        self, simple_problem
    ) -> None:
        """evaluate_on_problems() should return aggregated score."""
        # Arrange
        tuner = NSGAIITuner(population_size=20)
        problems = [(simple_problem, "ZDT1.pf")]
        params = {
            "offspring_population_size": 20,
            "crossover_type": "sbx",
            "crossover_probability": 0.9,
            "crossover_eta": 20.0,
            "mutation_type": "polynomial",
            "mutation_probability_factor": 1.0,
            "mutation_eta": 20.0,
        }
        
        # Act
        score = tuner.evaluate_on_problems(
            problems=problems,
            params=params,
            max_evaluations=1000,
            n_repeats=1,
        )
        
        # Assert
        assert isinstance(score, float)
        assert score >= 0
