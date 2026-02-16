"""
Algorithm-specific tuning modules.

Each algorithm has its own tuner class that inherits from AlgorithmTuner.
"""

from .base import AlgorithmTuner, TuningResult, ParameterInfo
from .nsgaii import NSGAIITuner

# Registry of available tuners
TUNERS = {
    "NSGAII": NSGAIITuner,
}

__all__ = [
    "AlgorithmTuner",
    "TuningResult",
    "ParameterInfo",
    "NSGAIITuner",
    "TUNERS",
]
