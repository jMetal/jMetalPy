"""
Main entry point for quality indicator CLI when run as a module.

Usage:
    python -m jmetal.util.quality_indicator_cli <front.csv> <reference.csv> <indicator> [options]
"""

from .quality_indicator_cli import main

if __name__ == "__main__":
    main()