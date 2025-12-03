"""
Observers for hyperparameter tuning progress visualization.

This module provides callback-based observers that integrate with Optuna's
optimization process. They follow a similar pattern to jMetalPy's algorithm
observers but are adapted for hyperparameter tuning.

Example:
    from jmetal.tuning.observer import TuningProgressObserver, TuningPlotObserver
    
    # Console progress
    progress_observer = TuningProgressObserver()
    
    # Real-time plotting
    plot_observer = TuningPlotObserver()
    
    # Use with tune()
    result = tune("NSGAII", observers=[progress_observer, plot_observer])
"""

import csv
import json
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


class TuningObserver(ABC):
    """
    Abstract base class for tuning observers.
    
    Tuning observers receive updates after each Optuna trial completes.
    They follow the Optuna callback protocol: callback(study, trial).
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
        """Called when tuning starts. Override for initialization."""
        pass
    
    def on_tuning_end(self, study: Study) -> None:
        """Called when tuning ends. Override for cleanup."""
        pass


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


class TuningFileObserver(TuningObserver):
    """
    File-based observer that logs tuning progress to CSV and/or JSON.
    
    Useful for:
    - Post-hoc analysis of tuning runs
    - Comparing different tuning configurations
    - Creating custom visualizations
    
    Args:
        output_dir: Directory for output files
        csv_file: CSV filename (default: "tuning_history.csv")
        json_file: JSON filename for final results (default: "tuning_results.json")
        log_all_trials: Log every trial (True) or only improvements (False)
    
    Example:
        observer = TuningFileObserver(output_dir="./tuning_logs")
        result = tune("NSGAII", observers=[observer])
    """
    
    def __init__(
        self,
        output_dir: str = "./tuning_output",
        csv_file: str = "tuning_history.csv",
        json_file: str = "tuning_results.json",
        log_all_trials: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.csv_file = csv_file
        self.json_file = json_file
        self.log_all_trials = log_all_trials
        self.csv_path: Optional[Path] = None
        self.csv_writer = None
        self.csv_handle = None
        self.start_time: Optional[float] = None
        self.algorithm = ""
        self.n_trials = 0
        self.best_value: Optional[float] = None
        self.param_names: List[str] = []
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """Initialize output files."""
        self.start_time = time.perf_counter()
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.best_value = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / self.csv_file
    
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Log trial to CSV file."""
        if trial.state != TrialState.COMPLETE:
            return
        
        # Check if this is an improvement
        is_improvement = False
        if self.best_value is None or trial.value < self.best_value:
            is_improvement = True
            self.best_value = trial.value
        
        # Skip non-improvements if configured
        if not self.log_all_trials and not is_improvement:
            return
        
        # Initialize CSV on first trial (now we know the parameter names)
        if self.csv_writer is None:
            self.param_names = list(trial.params.keys())
            self._init_csv()
        
        # Write row
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        row = {
            'trial': trial.number + 1,
            'score': trial.value,
            'best_score': study.best_value,
            'is_best': is_improvement,
            'elapsed_seconds': elapsed,
            'datetime': datetime.now().isoformat(),
        }
        for param_name in self.param_names:
            row[param_name] = trial.params.get(param_name, '')
        
        self.csv_writer.writerow(row)
        self.csv_handle.flush()
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers."""
        fieldnames = ['trial', 'score', 'best_score', 'is_best', 
                     'elapsed_seconds', 'datetime'] + self.param_names
        
        self.csv_handle = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=fieldnames)
        self.csv_writer.writeheader()
    
    def on_tuning_end(self, study: Study) -> None:
        """Save final results and close files."""
        # Close CSV
        if self.csv_handle:
            self.csv_handle.close()
        
        # Save JSON summary
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        json_path = self.output_dir / self.json_file
        
        results = {
            'algorithm': self.algorithm,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number + 1,
            'total_trials': len(study.trials),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
            'csv_file': str(self.csv_path),
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTuning logs saved to: {self.output_dir}")


class TuningRichObserver(TuningObserver):
    """
    Rich-based console observer with beautiful formatted output.
    
    Uses the Rich library for enhanced console output including:
    - Live-updating progress table
    - Colored output for improvements
    - Formatted parameter display
    
    Falls back to TuningProgressObserver if Rich is not installed.
    
    Args:
        show_live_table: Show live-updating table (default: True)
        show_params: Show parameter values (default: True)
    
    Example:
        observer = TuningRichObserver()
        result = tune("NSGAII", observers=[observer])
    """
    
    def __init__(self, show_live_table: bool = True, show_params: bool = True):
        self.show_live_table = show_live_table
        self.show_params = show_params
        self.console = None
        self.live = None
        self.table = None
        self.start_time: Optional[float] = None
        self.n_trials = 0
        self.algorithm = ""
        self.best_value: Optional[float] = None
        self.best_params: Dict[str, Any] = {}
        self.best_trial: int = 0
        self._rich_available = False
    
    def on_tuning_start(self, n_trials: int, algorithm: str) -> None:
        """Initialize Rich display."""
        self.start_time = time.perf_counter()
        self.n_trials = n_trials
        self.algorithm = algorithm
        self.best_value = None
        self.best_params = {}
        self.best_trial = 0
        
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table
            from rich.panel import Panel
            
            self._rich_available = True
            self.console = Console()
            
            self.console.print(Panel(
                f"[bold blue]{algorithm}[/bold blue] Hyperparameter Tuning\n"
                f"Total trials: {n_trials}",
                title="🔧 Tuning Started",
                border_style="blue"
            ))
            
            if self.show_live_table:
                self.table = self._create_table()
                self.live = Live(self.table, console=self.console, refresh_per_second=4)
                self.live.start()
                
        except ImportError:
            print("Warning: Rich not installed. Using basic output.")
            print(f"\n{'='*60}")
            print(f"  {algorithm} Hyperparameter Tuning")
            print(f"  Total trials: {n_trials}")
            print(f"{'='*60}\n")
    
    def _create_table(self):
        """Create a Rich table for live display."""
        from rich.table import Table
        
        table = Table(title=f"{self.algorithm} Tuning Progress", 
                     show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        table.add_row("Current Trial", "0 / " + str(self.n_trials))
        table.add_row("Best Score", "-")
        table.add_row("Best Trial", "-")
        table.add_row("Elapsed", self._format_time(elapsed))
        
        if self.show_params and self.best_params:
            table.add_section()
            for key, value in self.best_params.items():
                if isinstance(value, float):
                    table.add_row(f"  {key}", f"{value:.6f}")
                else:
                    table.add_row(f"  {key}", str(value))
        
        return table
    
    def _update_table(self, current_trial: int) -> None:
        """Update the live table."""
        from rich.table import Table
        
        table = Table(title=f"{self.algorithm} Tuning Progress",
                     show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        rate = current_trial / elapsed if elapsed > 0 else 0
        eta = (self.n_trials - current_trial) / rate if rate > 0 else 0
        
        table.add_row("Current Trial", f"{current_trial} / {self.n_trials}")
        table.add_row("Best Score", f"{self.best_value:.6f}" if self.best_value else "-")
        table.add_row("Best Trial", str(self.best_trial) if self.best_trial else "-")
        table.add_row("Rate", f"{rate:.2f} trials/s")
        table.add_row("Elapsed", self._format_time(elapsed))
        table.add_row("ETA", self._format_time(eta))
        
        if self.show_params and self.best_params:
            table.add_section()
            table.add_row("[bold]Best Parameters[/bold]", "")
            for key, value in self.best_params.items():
                if isinstance(value, float):
                    table.add_row(f"  {key}", f"{value:.6f}")
                else:
                    table.add_row(f"  {key}", str(value))
        
        self.live.update(table)
    
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """Update display after each trial."""
        if trial.state != TrialState.COMPLETE:
            return
        
        current_trial = trial.number + 1
        
        # Check for improvement
        if self.best_value is None or trial.value < self.best_value:
            self.best_value = trial.value
            self.best_params = trial.params.copy()
            self.best_trial = current_trial
        
        if self._rich_available and self.live:
            self._update_table(current_trial)
        else:
            # Fallback: basic output every 10 trials
            if current_trial % 10 == 0:
                print(f"Trial {current_trial}/{self.n_trials} | "
                      f"Best: {self.best_value:.6f}")
    
    def on_tuning_end(self, study: Study) -> None:
        """Show final results."""
        if self._rich_available and self.live:
            self.live.stop()
        
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        if self._rich_available:
            from rich.panel import Panel
            from rich.table import Table
            
            # Final results table
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            
            table.add_row("Best Score", f"[green]{study.best_value:.6f}[/green]")
            table.add_row("Best Trial", str(study.best_trial.number + 1))
            table.add_row("Total Time", self._format_time(elapsed))
            table.add_row("", "")
            table.add_row("[bold]Best Parameters[/bold]", "")
            
            for key, value in study.best_params.items():
                if isinstance(value, float):
                    table.add_row(f"  {key}", f"{value:.6f}")
                else:
                    table.add_row(f"  {key}", str(value))
            
            self.console.print(Panel(table, title="✅ Tuning Completed", 
                                    border_style="green"))
        else:
            print(f"\n{'='*60}")
            print("  TUNING COMPLETED")
            print(f"{'='*60}")
            print(f"  Best score: {study.best_value:.6f}")
            print(f"  Total time: {self._format_time(elapsed)}")
            print(f"{'='*60}\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


# Convenience function to create default observers
def create_default_observers(
    console: bool = True,
    plot: bool = False,
    file: bool = False,
    rich: bool = False,
    output_dir: str = "./tuning_output",
) -> List[TuningObserver]:
    """
    Create a list of default observers based on options.
    
    Args:
        console: Include console progress observer
        plot: Include real-time plot observer
        file: Include file logging observer
        rich: Use Rich console observer instead of basic
        output_dir: Output directory for file observer
    
    Returns:
        List of configured observers
    
    Example:
        observers = create_default_observers(console=True, plot=True)
        result = tune("NSGAII", observers=observers)
    """
    observers = []
    
    if console:
        if rich:
            observers.append(TuningRichObserver())
        else:
            observers.append(TuningProgressObserver())
    
    if plot:
        observers.append(TuningPlotObserver())
    
    if file:
        observers.append(TuningFileObserver(output_dir=output_dir))
    
    return observers
