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
        figsize: tuple = (10, 7),
        title: Optional[str] = None,
    ):
        self.update_frequency = update_frequency
        self.figsize = figsize
        self.title = title
        self.fig = None
        self.ax = None
        self.trials_line = None
        self.best_line = None
        self.trial_numbers: List[int] = []  # Real trial numbers from Optuna
        self.trial_scores: List[float] = []
        self.best_scores: List[float] = []
        self.algorithm = ""
        self.config_text = None  # Text object for configuration display
        self.current_best_trial = None  # Track best trial for config updates
        self._local_trial_count = 0  # Count for update frequency
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """Initialize the plot."""
        self.algorithm = algorithm
        self.trial_numbers = []
        self.trial_scores = []
        self.best_scores = []
        self.current_best_trial = None
        self._local_trial_count = 0
        
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
            
            # Reserve space at bottom for configuration text
            self.fig.subplots_adjust(bottom=0.25)
            
            # Initialize configuration text (empty)
            self.config_text = self.fig.text(
                0.5, 0.01, "",
                ha='center', va='bottom',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='orange', alpha=0.9)
            )
            
            plt.show(block=False)
            plt.pause(0.1)
        except ImportError:
            print("Warning: matplotlib not available. Plot observer disabled.")
            self.fig = None
    
    def _format_params(self, params: dict) -> str:
        """Format parameters for display in footer."""
        crossover_params = []
        mutation_params = []
        other_params = []
        
        for name, value in params.items():
            if isinstance(value, float):
                formatted = f"{name}: {value:.4f}"
            else:
                formatted = f"{name}: {value}"
            
            if 'crossover' in name or 'blx' in name:
                crossover_params.append(formatted)
            elif 'mutation' in name:
                mutation_params.append(formatted)
            else:
                other_params.append(formatted)
        
        # Build footer text
        footer_parts = []
        if other_params:
            footer_parts.append("General: " + ", ".join(other_params))
        if crossover_params:
            footer_parts.append("Crossover: " + ", ".join(crossover_params))
        if mutation_params:
            footer_parts.append("Mutation: " + ", ".join(mutation_params))
        
        return "\n".join(footer_parts)
    
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Update the plot after each trial."""
        if self.fig is None or trial.state != TrialState.COMPLETE:
            return
        
        import matplotlib.pyplot as plt
        
        # Record data using real trial number from Optuna
        real_trial_number = trial.number + 1  # 1-indexed for display
        self.trial_numbers.append(real_trial_number)
        self.trial_scores.append(trial.value)
        current_best = min(self.trial_scores)
        self.best_scores.append(current_best)
        
        # Increment local counter for update frequency
        self._local_trial_count += 1
        
        # Check if this trial is the new best
        is_new_best = (self.current_best_trial is None or 
                       trial.value <= current_best)
        if is_new_best and trial.value == current_best:
            self.current_best_trial = trial
        
        # Update plot every N trials (based on local count)
        if self._local_trial_count % self.update_frequency == 0:
            # Use real trial numbers for x-axis
            self.trials_line.set_data(self.trial_numbers, self.trial_scores)
            self.best_line.set_data(self.trial_numbers, self.best_scores)
            
            # Adjust axis limits
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Update title with current best
            title = self.title or f"{self.algorithm} Hyperparameter Tuning"
            self.ax.set_title(f"{title}\nBest: {current_best:.6f} (trial {self.current_best_trial.number + 1})")
            
            # Update configuration text with current best trial parameters
            if self.current_best_trial is not None and self.config_text is not None:
                config_str = self._format_params(self.current_best_trial.params)
                footer_text = f"BEST CONFIG (Trial {self.current_best_trial.number + 1})\n{config_str}"
                self.config_text.set_text(footer_text)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
    
    def on_tuning_end(self, study: Study) -> None:
        """Finalize the plot with best configuration in footer."""
        if self.fig is None:
            return
        
        import matplotlib.pyplot as plt
        
        # Final update of title
        title = self.title or f"{self.algorithm} Hyperparameter Tuning"
        self.ax.set_title(f"{title}\nFinal Best Score: {study.best_value:.6f} (Trial {study.best_trial.number + 1})")
        
        # Mark the best trial with vertical line using real trial number
        best_trial_number = study.best_trial.number + 1  # 1-indexed
        self.ax.axvline(x=best_trial_number, color='g', linestyle='--', 
                      alpha=0.7, label=f'Best trial ({best_trial_number})')
        self.ax.legend(loc='upper right')
        
        # Final update of configuration text
        if self.config_text is not None:
            config_str = self._format_params(study.best_params)
            footer_text = f"BEST CONFIGURATION (Trial {study.best_trial.number + 1})\n{config_str}"
            self.config_text.set_text(footer_text)
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Disable interactive mode and keep window open
        plt.ioff()
        print("\n[Plot] Close the plot window to continue...")
        plt.show(block=True)  # Block to keep window open
