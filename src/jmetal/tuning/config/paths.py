"""
Path configuration for hyperparameter tuning.

This module centralizes all path-related configuration constants.
"""

from pathlib import Path

# ROOT_DIR points to the jMetalPy project root (5 levels up from this file)
# Structure: config/paths.py -> config/ -> tuning/ -> jmetal/ -> src/ -> ROOT
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Output path for tuned configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "nsgaii_tuned_config.json"

# Directory containing reference Pareto fronts
REFERENCE_FRONTS_DIR = ROOT_DIR / "resources" / "reference_fronts"


def get_reference_front_path(reference_front_filename: str) -> Path:
    """
    Get the path to the reference front file.
    
    Args:
        reference_front_filename: Full name of the reference front file (with extension)
        
    Returns:
        Path to the reference front file
        
    Example:
        >>> path = get_reference_front_path("ZDT1.pf")
        >>> print(path)  # /path/to/jMetalPy/resources/reference_fronts/ZDT1.pf
    """
    return REFERENCE_FRONTS_DIR / reference_front_filename


def get_output_path(filename: str = "nsgaii_tuned_config.json") -> Path:
    """
    Get the path for saving tuning results.
    
    Args:
        filename: Output filename
        
    Returns:
        Path to the output file
    """
    return Path(__file__).resolve().parent.parent / filename
