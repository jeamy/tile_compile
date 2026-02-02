# PATCH 07 – Adaptive Tile-Verfeinerung im Runner-Loop (v4)
# Integration der Warp-Varianz-basierten Tile-Splits

from runner.tile_processor_v4 import StreamingTileProcessor
from tile_compile_backend.tile_grid import build_initial_tile_grid
from tile_compile_backend.tile_grid_adaptive import refine_tiles
import numpy as np


def run_v4_with_adaptive_tiles(frame_paths, global_weights, cfg, image_shape):
    # Initiales Grid
    tiles = build_initial_tile_grid(image_shape, cfg)

    max_refine_passes = cfg.v4.max_refine_passes
    all_results = {}

    for refine_pass in range(max_refine_passes + 1):
        next_tiles = []
        warp_variances = []

        for tid, bbox in enumerate(tiles):
            tp = StreamingTileProcessor(
                tile_id=tid,
                bbox=bbox,
                frame_paths=frame_paths,
                global_weights=global_weights,
                cfg=cfg,
            )
            result, warps = tp.run()
            if result is None:
                all_results[bbox] = None
                warp_variances.append(float("inf"))
                continue

            all_results[bbox] = result

            # Warp-Varianz bestimmen
            dx = [w[0, 2] for w in warps]
            dy = [w[1, 2] for w in warps]
            var = float(np.var(dx) + np.var(dy))
            warp_variances.append(var)

        # Verfeinerung
        if refine_pass < max_refine_passes:
            tiles = refine_tiles(
                tiles,
                warp_variances,
                threshold=cfg.v4.tile_refine_variance_threshold,
            )
        else:
            break

    return all_results
