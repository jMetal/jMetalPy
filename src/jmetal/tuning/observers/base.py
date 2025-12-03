"""
Base class for tuning observers.

Tuning observers receive updates after each Optuna trial completes.
They follow the Optuna callback protocol: callback(study, trial).
"""

from abc import ABC, abstractmethod

from optuna.study import Study
from optuna.trial import FrozenTrial


class TuningObserver(ABC):
    """
    Abstract base class for tuning observers.
    
    Tuning observers receive updates after each Optuna trial completes.
    They follow the Optuna callback protocol: callback(study, trial).
    
    To create a custom observer, subclass this and implement __call__.
    Optionally override on_tuning_start and on_tuning_end for setup/cleanup.
    
    Example:
        class MyObserver(TuningObserver):
            def __call__(self, study, trial):
                print(f"Trial {trial.number}: {trial.value}")
    """
    
    @abstractmethod
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        Called after each trial completes.
        
        Args:
            study: The Optuna study object
            trial: The completed trial
        """
        pass
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """
        Called when tuning starts.
        
        Override this method for initialization logic.
        
        Args:
            n_trials: Total number of trials planned
            algorithm: Name of the algorithm being tuned
        """
        pass
    
    def on_tuning_end(self, study: Study) -> None:
        """
        Called when tuning ends.
        
        Override this method for cleanup logic.
        
        Args:
            study: The completed Optuna study
        """
        pass
