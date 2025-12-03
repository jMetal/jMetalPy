"""
Real-time matplotlib plotting observer for hyperparameter tuning.

Creates a live plot showing optimization progress.
"""

from typing import List, Optional

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from .base import TuningObserver


class TuningPlotObserver(TuningObserver):
    """
    Real-time plotting observer for hyperparameter tuning.
    
    Creates a live matplotlib plot showing:
    - Score vs trial number
    - Best score progression
    - Current best configuration
    
    Args:
        update_frequency: Update plot every N trials (default: 1)
        figsize: Figure size as (width, height) tuple
        title: Plot title (default: auto-generated)
    
    Example:
        observer = TuningPlotObserver(update_frequency=5)
        result = tune("NSGAII", observers=[observer])
    """
    
    def __init__(
        self,
        update_frequency: int = 1,
        figsize: tuple = (10, 6),
        title: Optional[str] = None,
    ):
        self.update_frequency = update_frequency
        self.figsize = figsize
        self.title = title
        self.fig = None
        self.ax = None
        self.trials_line = None
        self.best_line = None
        self.trial_scores: List[float] = []
        self.best_scores: List[float] = []
        self.algorithm = ""
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """Initialize the plot."""
        self.algorithm = algorithm
        self.trial_scores = []
        self.best_scores = []
        
        try:
            import matplotlib.pyplot as plt
            plt.ion()  # Enable interactive mode
            
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.ax.set_xlabel("Trial")
            self.ax.set_ylabel("Score (lower is better)")
            title = self.title or f"{algorithm} Hyperparameter Tuning"
            self.ax.set_title(title)
            self.ax.grid(True, alpha=0.3)
            
            # Initialize empty lines
            self.trials_line, = self.ax.plot([], [], 'b.', alpha=0.5, label='Trial score')
            self.best_line, = self.ax.plot([], [], 'r-', linewidth=2, label='Best score')
            self.ax.legend(loc='upper right')
            
            plt.show(block=False)
            plt.pause(0.1)
        except ImportError:
            print("Warning: matplotlib not available. Plot observer disabled.")
            self.fig = None
    
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Update the plot after each trial."""
        if self.fig is None or trial.state != TrialState.COMPLETE:
            return
        
        import matplotlib.pyplot as plt
        
        # Record data
        self.trial_scores.append(trial.value)
        current_best = min(self.trial_scores)
        self.best_scores.append(current_best)
        
        # Update plot every N trials
        if (trial.number + 1) % self.update_frequency == 0:
            trials = list(range(1, len(self.trial_scores) + 1))
            
            self.trials_line.set_data(trials, self.trial_scores)
            self.best_line.set_data(trials, self.best_scores)
            
            # Adjust axis limits
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Update title with current best
            title = self.title or f"{self.algorithm} Hyperparameter Tuning"
            self.ax.set_title(f"{title}\nBest: {current_best:.6f} (trial {trial.number + 1})")
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
    
    def on_tuning_end(self, study: Study) -> None:
        """Finalize the plot."""
        if self.fig is None:
            return
        
        import matplotlib.pyplot as plt
        
        # Final update
        title = self.title or f"{self.algorithm} Hyperparameter Tuning"
        self.ax.set_title(f"{title}\nFinal Best: {study.best_value:.6f}")
        
        # Mark the best trial
        best_trial_idx = study.best_trial.number
        if best_trial_idx < len(self.trial_scores):
            self.ax.axvline(x=best_trial_idx + 1, color='g', linestyle='--', 
                          alpha=0.7, label=f'Best trial ({best_trial_idx + 1})')
            self.ax.legend(loc='upper right')
        
        self.fig.canvas.draw()
        plt.ioff()  # Disable interactive mode
        plt.show(block=False)
