"""
File-based observer that logs tuning progress to CSV and JSON.

Useful for post-hoc analysis and creating custom visualizations.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from .base import TuningObserver


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
        self.csv_writer: Optional[csv.DictWriter] = None
        self.csv_handle: Optional[Any] = None
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
        if self.best_value is None or (trial.value is not None and trial.value < self.best_value):
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
        row: Dict[str, Any] = {
            'trial': trial.number + 1,
            'score': trial.value,
            'best_score': study.best_value,
            'is_best': is_improvement,
            'elapsed_seconds': elapsed,
            'datetime': datetime.now().isoformat(),
        }
        for param_name in self.param_names:
            row[param_name] = trial.params.get(param_name, '')
        
        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
        if self.csv_handle is not None:
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
