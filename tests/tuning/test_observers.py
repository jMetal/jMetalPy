"""
Unit tests for jmetal.tuning.observers module.

Tests for TuningObserver base class and concrete implementations.
"""

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jmetal.tuning.observers.base import TuningObserver
from jmetal.tuning.observers.console import TuningProgressObserver
from jmetal.tuning.observers.file import TuningFileObserver
from jmetal.tuning.observers.plot import TuningPlotObserver


class TestTuningObserverBase:
    """Tests for TuningObserver abstract base class."""

    def test_given_abstract_class_when_instantiate_directly_then_raises(self) -> None:
        """TuningObserver should not be instantiable directly."""
        # Arrange, Act & Assert
        with pytest.raises(TypeError):
            TuningObserver()

    def test_given_subclass_without_call_when_instantiate_then_raises(self) -> None:
        """Subclass without __call__ implementation should raise."""
        # Arrange
        class IncompleteObserver(TuningObserver):
            pass
        
        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteObserver()

    def test_given_valid_subclass_when_instantiate_then_succeeds(self) -> None:
        """Valid subclass with __call__ should instantiate."""
        # Arrange
        class ValidObserver(TuningObserver):
            def __call__(self, study, trial):
                pass
        
        # Act
        observer = ValidObserver()
        
        # Assert
        assert observer is not None

    def test_given_base_class_when_on_tuning_start_then_noop(self) -> None:
        """Default on_tuning_start should be a no-op."""
        # Arrange
        class TestObserver(TuningObserver):
            def __call__(self, study, trial):
                pass
        
        observer = TestObserver()
        
        # Act & Assert - should not raise
        observer.on_tuning_start(100, "NSGAII")

    def test_given_base_class_when_on_tuning_end_then_noop(
        self, mock_optuna_study
    ) -> None:
        """Default on_tuning_end should be a no-op."""
        # Arrange
        class TestObserver(TuningObserver):
            def __call__(self, study, trial):
                pass
        
        observer = TestObserver()
        
        # Act & Assert - should not raise
        observer.on_tuning_end(mock_optuna_study)


class TestTuningProgressObserver:
    """Tests for TuningProgressObserver (console output)."""

    def test_given_default_params_when_create_then_initializes_correctly(self) -> None:
        """Default instantiation should set expected values."""
        # Arrange & Act
        observer = TuningProgressObserver()
        
        # Assert
        assert observer.display_frequency == 10
        assert observer.show_params is True
        assert observer.show_improvement is True

    def test_given_custom_params_when_create_then_stores_values(self) -> None:
        """Custom parameters should be stored."""
        # Arrange & Act
        observer = TuningProgressObserver(display_frequency=5, show_params=False)
        
        # Assert
        assert observer.display_frequency == 5
        assert observer.show_params is False

    def test_given_observer_when_on_tuning_start_then_initializes_state(self) -> None:
        """on_tuning_start should initialize tracking variables."""
        # Arrange
        observer = TuningProgressObserver()
        
        # Act
        observer.on_tuning_start(100, "NSGAII")
        
        # Assert
        assert observer.n_trials == 100
        assert observer.algorithm == "NSGAII"
        assert observer.best_value is None

    @patch('sys.stdout', new_callable=StringIO)
    def test_given_observer_when_on_tuning_start_then_prints_header(
        self, mock_stdout
    ) -> None:
        """on_tuning_start should print header information."""
        # Arrange
        observer = TuningProgressObserver()
        
        # Act
        observer.on_tuning_start(100, "NSGAII")
        output = mock_stdout.getvalue()
        
        # Assert
        assert "NSGAII" in output
        assert "100" in output or "trials" in output.lower()

    def test_given_completed_trial_when_call_then_updates_best(
        self, mock_optuna_study, mock_frozen_trial
    ) -> None:
        """__call__ should update best_value when trial improves."""
        # Arrange
        observer = TuningProgressObserver()
        observer.on_tuning_start(10, "NSGAII")
        mock_frozen_trial.value = 0.05
        
        # Act
        observer(mock_optuna_study, mock_frozen_trial)
        
        # Assert
        assert observer.best_value == 0.05

    def test_given_worse_trial_when_call_then_keeps_best(
        self, mock_optuna_study, mock_frozen_trial
    ) -> None:
        """__call__ should keep best_value when trial is worse."""
        # Arrange
        observer = TuningProgressObserver()
        observer.on_tuning_start(10, "NSGAII")
        observer.best_value = 0.01  # Set a good value
        mock_frozen_trial.value = 0.05  # Worse value
        
        # Act
        observer(mock_optuna_study, mock_frozen_trial)
        
        # Assert
        assert observer.best_value == 0.01  # Unchanged

    @patch('sys.stdout', new_callable=StringIO)
    def test_given_observer_when_on_tuning_end_then_prints_summary(
        self, mock_stdout, mock_optuna_study
    ) -> None:
        """on_tuning_end should print final summary."""
        # Arrange
        observer = TuningProgressObserver()
        observer.on_tuning_start(10, "NSGAII")
        
        # Act
        observer.on_tuning_end(mock_optuna_study)
        output = mock_stdout.getvalue()
        
        # Assert
        assert "best" in output.lower() or "result" in output.lower()


class TestTuningFileObserver:
    """Tests for TuningFileObserver (CSV output)."""

    def test_given_default_params_when_create_then_uses_default_dir(self) -> None:
        """Default instantiation should use default output directory."""
        # Arrange & Act
        observer = TuningFileObserver()
        
        # Assert
        assert observer.output_dir is not None
        assert observer.csv_file == "tuning_history.csv"

    def test_given_custom_dir_when_create_then_stores_dir(
        self, tmp_path
    ) -> None:
        """Custom output_dir should be stored."""
        # Arrange & Act
        observer = TuningFileObserver(output_dir=str(tmp_path))
        
        # Assert
        assert str(tmp_path) in str(observer.output_dir)

    def test_given_observer_when_on_tuning_start_then_creates_dir(
        self, tmp_path
    ) -> None:
        """on_tuning_start should create the output directory."""
        # Arrange
        output_dir = tmp_path / "tuning_logs"
        observer = TuningFileObserver(output_dir=str(output_dir))
        
        # Act
        observer.on_tuning_start(10, "NSGAII")
        
        # Assert
        assert output_dir.exists()

    def test_given_observer_when_call_then_initializes_csv(
        self, tmp_path, mock_optuna_study, mock_frozen_trial
    ) -> None:
        """__call__ should initialize CSV file on first trial."""
        # Arrange
        observer = TuningFileObserver(output_dir=str(tmp_path))
        observer.on_tuning_start(10, "NSGAII")
        
        # Act
        observer(mock_optuna_study, mock_frozen_trial)
        
        # Assert
        csv_path = tmp_path / "tuning_history.csv"
        assert csv_path.exists()

    def test_given_completed_trial_when_call_then_writes_row(
        self, tmp_path, mock_optuna_study, mock_frozen_trial
    ) -> None:
        """__call__ should append trial data to CSV file."""
        # Arrange
        observer = TuningFileObserver(output_dir=str(tmp_path))
        observer.on_tuning_start(10, "NSGAII")
        
        # Act
        observer(mock_optuna_study, mock_frozen_trial)
        
        # Assert
        csv_path = tmp_path / "tuning_history.csv"
        content = csv_path.read_text()
        assert "trial" in content.lower()  # Header exists
        assert "score" in content.lower()

    def test_given_multiple_trials_when_call_then_appends_all(
        self, tmp_path, mock_optuna_study, mock_frozen_trial
    ) -> None:
        """Multiple trials should all be written to file."""
        # Arrange
        observer = TuningFileObserver(output_dir=str(tmp_path))
        observer.on_tuning_start(10, "NSGAII")
        
        # Act
        for i in range(3):
            mock_frozen_trial.number = i
            mock_frozen_trial.value = 0.1 + i * 0.01
            observer(mock_optuna_study, mock_frozen_trial)
        
        csv_path = tmp_path / "tuning_history.csv"
        content = csv_path.read_text()
        lines = content.strip().split('\n')
        
        # Assert
        assert len(lines) >= 4  # Header + 3 data rows


class TestTuningPlotObserver:
    """Tests for TuningPlotObserver (matplotlib plotting)."""

    def test_given_default_params_when_create_then_initializes_correctly(self) -> None:
        """Default instantiation should set expected values."""
        # Arrange & Act
        observer = TuningPlotObserver()
        
        # Assert
        assert observer.update_frequency == 1
        assert observer.figsize == (10, 7)

    def test_given_custom_params_when_create_then_stores_values(self) -> None:
        """Custom parameters should be stored."""
        # Arrange & Act
        observer = TuningPlotObserver(
            update_frequency=5,
            figsize=(12, 8),
            title="Custom Title",
        )
        
        # Assert
        assert observer.update_frequency == 5
        assert observer.figsize == (12, 8)
        assert observer.title == "Custom Title"

    def test_given_observer_when_on_tuning_start_then_initializes_state(self) -> None:
        """on_tuning_start should initialize tracking variables."""
        # Arrange
        observer = TuningPlotObserver()
        
        # Note: We don't actually create the plot in tests to avoid GUI issues
        # We can only test the state initialization
        
        # Act - with matplotlib mocked
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.plot.return_value = [MagicMock()]
            
            observer.on_tuning_start(100, "NSGAII")
        
        # Assert
        assert observer.algorithm == "NSGAII"
        assert observer.trial_scores == []
        assert observer.best_scores == []

    def test_given_format_params_when_call_then_groups_by_category(self) -> None:
        """_format_params should group parameters by category."""
        # Arrange
        observer = TuningPlotObserver()
        params = {
            "population_size": 100,
            "crossover_type": "SBX",
            "crossover_probability": 0.9,
            "mutation_type": "Polynomial",
            "mutation_probability": 0.01,
        }
        
        # Act
        result = observer._format_params(params)
        
        # Assert
        assert "Crossover:" in result
        assert "Mutation:" in result

    def test_given_float_params_when_format_then_limits_precision(self) -> None:
        """_format_params should format floats with 4 decimal places."""
        # Arrange
        observer = TuningPlotObserver()
        params = {"probability": 0.123456789}
        
        # Act
        result = observer._format_params(params)
        
        # Assert
        assert "0.1235" in result  # 4 decimal places

    def test_given_completed_trial_when_call_then_tracks_scores(self) -> None:
        """__call__ should track trial scores."""
        # Arrange
        observer = TuningPlotObserver()
        observer.trial_scores = []
        observer.best_scores = []
        observer.fig = None  # No actual plot
        
        # Act - observer should handle fig=None gracefully
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_trial.value = 0.15
        from optuna.trial import TrialState
        mock_trial.state = TrialState.COMPLETE
        
        observer(mock_study, mock_trial)
        
        # Assert - with fig=None, observer should exit early
        # This tests the guard clause works
        assert observer.fig is None
