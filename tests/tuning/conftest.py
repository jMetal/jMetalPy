"""
Shared fixtures for tuning package tests.

Provides reusable test fixtures for algorithms, observers, metrics,
and mock objects used across multiple test modules.
"""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from jmetal.problem import ZDT1


@pytest.fixture
def simple_problem() -> ZDT1:
    """Create a simple ZDT1 problem for quick tests.
    
    Returns:
        ZDT1 problem instance with default configuration.
    """
    return ZDT1()


@pytest.fixture
def mock_optuna_trial() -> MagicMock:
    """Create a mock Optuna trial for unit tests.
    
    Returns:
        MagicMock configured to behave like an Optuna trial.
    """
    trial = MagicMock()
    trial.suggest_int.return_value = 100
    trial.suggest_float.return_value = 0.9
    trial.suggest_categorical.return_value = "SBX"
    trial.number = 0
    trial.value = 0.5
    trial.state.name = "COMPLETE"
    return trial


@pytest.fixture
def mock_optuna_study() -> MagicMock:
    """Create a mock Optuna study for observer tests.
    
    Returns:
        MagicMock configured to behave like an Optuna study.
    """
    study = MagicMock()
    study.best_value = 0.05
    study.best_params = {
        "population_size": 100,
        "crossover_type": "SBX",
        "crossover_probability": 0.9,
        "mutation_probability": 0.01,
    }
    study.best_trial.number = 5
    study.best_trial.params = study.best_params
    return study


@pytest.fixture
def mock_frozen_trial() -> MagicMock:
    """Create a mock FrozenTrial for callback tests.
    
    Returns:
        MagicMock configured to behave like an Optuna FrozenTrial.
    """
    from optuna.trial import TrialState
    
    trial = MagicMock()
    trial.number = 3
    trial.value = 0.15
    trial.state = TrialState.COMPLETE
    trial.params = {
        "population_size": 100,
        "crossover_probability": 0.9,
    }
    return trial


@pytest.fixture
def sample_reference_front() -> np.ndarray:
    """Create a simple reference Pareto front for tests.
    
    Returns:
        Numpy array representing a simple 2D Pareto front.
    """
    return np.array([
        [0.0, 1.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [1.0, 0.0],
    ])


@pytest.fixture
def sample_solutions() -> List[MagicMock]:
    """Create mock solutions with objective values.
    
    Returns:
        List of mock solutions with .objectives attribute.
    """
    solutions = []
    for objectives in [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]:
        sol = MagicMock()
        sol.objectives = objectives
        solutions.append(sol)
    return solutions


@pytest.fixture
def dominated_solutions() -> List[MagicMock]:
    """Create mock solutions that are dominated (poor quality).
    
    Returns:
        List of mock solutions with dominated objective values.
    """
    solutions = []
    for objectives in [[0.8, 0.9], [0.9, 0.8], [0.85, 0.85]]:
        sol = MagicMock()
        sol.objectives = objectives
        solutions.append(sol)
    return solutions


@pytest.fixture
def nsgaii_tuner():
    """Create an NSGAIITuner instance.
    
    Returns:
        NSGAIITuner with default population size of 50.
    """
    from jmetal.tuning.algorithms import NSGAIITuner
    return NSGAIITuner(population_size=50)


@pytest.fixture
def temp_output_file(tmp_path):
    """Create a temporary file path for output tests.
    
    Args:
        tmp_path: pytest's temporary directory fixture.
        
    Returns:
        Path object for a temporary output file.
    """
    return tmp_path / "tuning_output.csv"


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file path for output tests.
    
    Args:
        tmp_path: pytest's temporary directory fixture.
        
    Returns:
        Path object for a temporary JSON file.
    """
    return tmp_path / "tuning_result.json"
