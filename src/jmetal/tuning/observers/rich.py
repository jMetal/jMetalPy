"""
Rich-based console observer with beautiful formatted output.

Uses the Rich library for enhanced console output. Falls back to basic
output if Rich is not installed.
"""

import time
from typing import Any, Dict, Optional

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from .base import TuningObserver


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
        self.console: Any = None
        self.live: Any = None
        self.table: Any = None
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
    
    def _create_table(self) -> Any:
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
        if self.best_value is None or (trial.value is not None and trial.value < self.best_value):
            self.best_value = trial.value
            self.best_params = trial.params.copy()
            self.best_trial = current_trial
        
        if self._rich_available and self.live:
            self._update_table(current_trial)
        else:
            # Fallback: basic output every 10 trials
            if current_trial % 10 == 0:
                print(f"Trial {current_trial}/{self.n_trials} | "
                      f"Best: {self.best_value:.6f}" if self.best_value else "-")
    
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
