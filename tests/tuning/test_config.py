"""
Unit tests for jmetal.tuning.config module.

Tests for configuration constants and default values.
"""

from pathlib import Path

import pytest

from jmetal.tuning.config import (
    POPULATION_SIZE,
    TRAINING_EVALUATIONS,
    VALIDATION_EVALUATIONS,
    NUMBER_OF_TRIALS,
    N_REPEATS,
    SEED,
)
from jmetal.tuning.config.paths import (
    CONFIG_PATH,
    REFERENCE_FRONTS_DIR,
    ROOT_DIR,
)
from jmetal.tuning.config.problems import TRAINING_PROBLEMS

# Aliases for backward compatibility in tests
REFERENCE_FRONTS_PATH = REFERENCE_FRONTS_DIR
RESOURCES_PATH = ROOT_DIR / "resources"


class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_given_population_size_when_check_then_positive(self) -> None:
        """POPULATION_SIZE should be a positive integer."""
        # Assert
        assert isinstance(POPULATION_SIZE, int)
        assert POPULATION_SIZE > 0

    def test_given_training_evaluations_when_check_then_positive(self) -> None:
        """TRAINING_EVALUATIONS should be a positive integer."""
        # Assert
        assert isinstance(TRAINING_EVALUATIONS, int)
        assert TRAINING_EVALUATIONS > 0

    def test_given_validation_evaluations_when_check_then_positive(self) -> None:
        """VALIDATION_EVALUATIONS should be a positive integer."""
        # Assert
        assert isinstance(VALIDATION_EVALUATIONS, int)
        assert VALIDATION_EVALUATIONS > 0

    def test_given_number_of_trials_when_check_then_positive(self) -> None:
        """NUMBER_OF_TRIALS should be a positive integer."""
        # Assert
        assert isinstance(NUMBER_OF_TRIALS, int)
        assert NUMBER_OF_TRIALS > 0

    def test_given_n_repeats_when_check_then_positive(self) -> None:
        """N_REPEATS should be a positive integer."""
        # Assert
        assert isinstance(N_REPEATS, int)
        assert N_REPEATS > 0

    def test_given_seed_when_check_then_is_integer(self) -> None:
        """SEED should be an integer."""
        # Assert
        assert isinstance(SEED, int)

    def test_given_training_evaluations_when_check_then_greater_than_population(
        self
    ) -> None:
        """TRAINING_EVALUATIONS should be greater than POPULATION_SIZE."""
        # Assert - need enough evaluations for at least a few generations
        assert TRAINING_EVALUATIONS > POPULATION_SIZE


class TestConfigPaths:
    """Tests for configuration paths."""

    def test_given_config_path_when_check_then_is_path(self) -> None:
        """CONFIG_PATH should be a Path object."""
        # Assert
        assert isinstance(CONFIG_PATH, Path)

    def test_given_reference_fronts_path_when_check_then_exists(self) -> None:
        """REFERENCE_FRONTS_PATH should point to existing directory."""
        # Assert
        assert isinstance(REFERENCE_FRONTS_PATH, Path)
        assert REFERENCE_FRONTS_PATH.exists()
        assert REFERENCE_FRONTS_PATH.is_dir()

    def test_given_resources_path_when_check_then_exists(self) -> None:
        """RESOURCES_PATH should point to existing directory."""
        # Assert
        assert isinstance(RESOURCES_PATH, Path)
        assert RESOURCES_PATH.exists()
        assert RESOURCES_PATH.is_dir()

    def test_given_reference_fronts_path_when_check_then_contains_zdt_fronts(
        self
    ) -> None:
        """REFERENCE_FRONTS_PATH should contain ZDT reference fronts."""
        # Arrange
        expected_files = ["ZDT1.pf", "ZDT2.pf", "ZDT3.pf"]
        
        # Assert
        for filename in expected_files:
            file_path = REFERENCE_FRONTS_PATH / filename
            assert file_path.exists(), f"Missing reference front: {filename}"


class TestTrainingProblems:
    """Tests for default training problems configuration."""

    def test_given_training_problems_when_check_then_not_empty(self) -> None:
        """TRAINING_PROBLEMS should be a non-empty list."""
        # Assert
        assert isinstance(TRAINING_PROBLEMS, list)
        assert len(TRAINING_PROBLEMS) > 0

    def test_given_training_problems_when_check_then_are_tuples(self) -> None:
        """Each training problem should be a (Problem, str) tuple."""
        # Assert
        for item in TRAINING_PROBLEMS:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_given_training_problems_when_check_then_problems_valid(self) -> None:
        """Each training problem should be a valid Problem instance."""
        # Arrange
        from jmetal.core.problem import Problem
        
        # Assert
        for problem, _ in TRAINING_PROBLEMS:
            assert isinstance(problem, Problem)

    def test_given_training_problems_when_check_then_reference_fronts_exist(
        self
    ) -> None:
        """Each training problem's reference front file should exist."""
        # Assert
        for _, ref_front_file in TRAINING_PROBLEMS:
            assert isinstance(ref_front_file, str)
            ref_path = REFERENCE_FRONTS_PATH / ref_front_file
            assert ref_path.exists(), f"Missing reference front: {ref_front_file}"

    def test_given_training_problems_when_check_then_includes_zdt_problems(
        self
    ) -> None:
        """TRAINING_PROBLEMS should include ZDT problems by default."""
        # Arrange
        problem_names = [p.name() for p, _ in TRAINING_PROBLEMS]
        
        # Assert - at least some ZDT problems should be included
        zdt_problems = [name for name in problem_names if name.startswith("ZDT")]
        assert len(zdt_problems) > 0
