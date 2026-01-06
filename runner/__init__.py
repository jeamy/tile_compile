"""
Tile-Compile Runner Package

Modular implementation of the Methodik v3 pipeline runner.
Split from the monolithic tile_compile_runner.py for better maintainability.
"""

from .phases import run_phases

__all__ = ["run_phases"]
