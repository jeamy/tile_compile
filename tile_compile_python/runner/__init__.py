"""
Tile-Compile Runner Package - Methodik v4

Tile-centric reconstruction without global registration.
All registration is tile-local.
"""

from .tile_processor_v4 import (
    TileProcessor,
    TileProcessorConfig,
    overlap_add,
    build_initial_tile_grid,
    refine_tiles,
    global_coarse_normalize,
    compute_global_weights,
)
from .tile_local_registration_v4 import register_tile

__all__ = [
    "TileProcessor",
    "TileProcessorConfig",
    "overlap_add",
    "build_initial_tile_grid",
    "refine_tiles",
    "global_coarse_normalize",
    "compute_global_weights",
    "register_tile",
]
