"""
Command-line interfaces for hyperparameter tuning.

This module provides CLI scripts for running tuning experiments.

Available Scripts:
    - sequential: Run tuning in a single process
    - parallel: Run distributed tuning with multiple workers

Usage:
    # Sequential tuning
    python -m jmetal.tuning.cli.sequential --trials 100
    
    # Parallel tuning (requires PostgreSQL)
    python -m jmetal.tuning.cli.parallel --workers 4 --trials 500
"""

from .sequential import main as run_sequential
from .parallel import main as run_parallel

__all__ = [
    "run_sequential",
    "run_parallel",
]
