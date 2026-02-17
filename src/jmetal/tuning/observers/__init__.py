"""
Observers for hyperparameter tuning progress visualization.

This module provides callback-based observers that integrate with Optuna's
optimization process.

Available Observers:
    - TuningProgressObserver: Clean console output
    - TuningPlotObserver: Real-time matplotlib plots
    - TuningFileObserver: CSV and JSON logging
    - TuningRichObserver: Rich library enhanced output

Example:
    from jmetal.tuning.observers import TuningProgressObserver, TuningPlotObserver
    
    observers = [
        TuningProgressObserver(display_frequency=5),
        TuningPlotObserver(),
    ]
    result = tune("NSGAII", observers=observers)
"""

from typing import List

from .base import TuningObserver
from .console import TuningProgressObserver
from .plot import TuningPlotObserver
from .file import TuningFileObserver
from .rich import TuningRichObserver


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
    observers: List[TuningObserver] = []
    
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


__all__ = [
    # Base class
    "TuningObserver",
    # Implementations
    "TuningProgressObserver",
    "TuningPlotObserver",
    "TuningFileObserver",
    "TuningRichObserver",
    # Factory
    "create_default_observers",
]
