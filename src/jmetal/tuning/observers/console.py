"""
Console-based progress observer for hyperparameter tuning.

Displays clean, formatted progress information without cluttering the console.
"""

import time
from typing import Any, Dict, Optional

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from .base import TuningObserver


class TuningProgressObserver(TuningObserver):
    """
    Console observer that displays clean, formatted progress information.
    
    Shows current trial, best configuration found, and timing information
    in a readable format without cluttering the console.
    
    Args:
        display_frequency: Show detailed info every N trials (default: 10)
        show_params: Whether to show parameter values (default: True)
        show_improvement: Highlight when a new best is found (default: True)
    
    Example:
        observer = TuningProgressObserver(display_frequency=5)
        result = tune("NSGAII", observers=[observer])
    """
    
    def __init__(
        self,
        display_frequency: int = 10,
        show_params: bool = True,
        show_improvement: bool = True,
    ):
        self.display_frequency = display_frequency
        self.show_params = show_params
        self.show_improvement = show_improvement
        self.start_time: Optional[float] = None
        self.n_trials: int = 0
        self.algorithm: str = ""
        self.best_value: Optional[float] = None
        self.last_improvement_trial: int = 0
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """Initialize progress tracking."""
        self.start_time = time.perf_counter()
        self.n_trials = n_trials
        self.algorithm = algorithm
        self.best_value = None
        self.last_improvement_trial = 0
        
        print("\n" + "=" * 65)
        print(f"  {algorithm} Hyperparameter Tuning")
        print("=" * 65)
        print(f"  Total trials: {n_trials}")
        print("=" * 65 + "\n")
    
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Display progress after each trial."""
        if trial.state != TrialState.COMPLETE:
            return
        
        current_trial = trial.number + 1
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        # Check for improvement
        is_new_best = False
        if self.best_value is None or trial.value < self.best_value:
            is_new_best = True
            self.best_value = trial.value
            self.last_improvement_trial = current_trial
        
        # Always show improvements
        if is_new_best and self.show_improvement:
            self._print_improvement(current_trial, trial, elapsed)
        
        # Show periodic summary
        elif current_trial % self.display_frequency == 0:
            self._print_summary(study, current_trial, elapsed)
    
    def _print_improvement(
        self, current_trial: int, trial: FrozenTrial, elapsed: float
    ) -> None:
        """Print new best configuration found."""
        print(f"★ Trial {current_trial}: NEW BEST = {trial.value:.6f}")
        if self.show_params:
            params_str = self._format_params(trial.params)
            print(f"  Parameters: {params_str}")
        print()
    
    def _print_summary(
        self, study: Study, current_trial: int, elapsed: float
    ) -> None:
        """Print periodic summary."""
        rate = current_trial / elapsed if elapsed > 0 else 0
        eta = (self.n_trials - current_trial) / rate if rate > 0 else 0
        
        print(f"─ Trial {current_trial}/{self.n_trials} │ "
              f"Best: {study.best_value:.6f} │ "
              f"Rate: {rate:.1f} trials/s │ "
              f"ETA: {self._format_time(eta)}")
    
    def on_tuning_end(self, study: Study) -> None:
        """Print final summary."""
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 65)
        print("  TUNING COMPLETED")
        print("=" * 65)
        print(f"  Best score: {study.best_value:.6f}")
        print(f"  Found at trial: {self.last_improvement_trial}")
        print(f"  Total time: {self._format_time(elapsed)}")
        print("\n  Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        print("=" * 65 + "\n")
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters as a compact string."""
        parts = []
        for key, value in params.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
