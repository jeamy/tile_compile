# PATCH 03 – Adaptive Tile Grid (v4)
# Extension for tile_compile_backend/tile_grid.py

import math


def refine_tiles(tiles, warp_variances, threshold):
    """
    Split tiles with excessive warp variance.

    Args:
        tiles: list[(x0,y0,w,h)]
        warp_variances: list[float]
        threshold: variance threshold
    Returns:
        refined list of tiles
    """
    refined = []
    for (x0, y0, w, h), var in zip(tiles, warp_variances):
        if var < threshold or w <= 64 or h <= 64:
            refined.append((x0, y0, w, h))
            continue
        hw, hh = w // 2, h // 2
        refined.extend([
            (x0, y0, hw, hh),
            (x0 + hw, y0, w - hw, hh),
            (x0, y0 + hh, hw, h - hh),
            (x0 + hw, y0 + hh, w - hw, h - hh),
        ])
    return refined
```

Integration point:
- nach erstem TileProcessor-Durchlauf
- warp_variances aus lokalen Warps
