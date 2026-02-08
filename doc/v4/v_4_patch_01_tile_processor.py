# NEW FILE: runner/tile_processor_v4.py
# -----------------------------------------------------------------------------
# TileProcessor v4 – zentraler Rekonstruktionsoperator (Methodik v4)
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from runner.tile_local_registration_v4 import register_tile

EPS = 1e-6


def smooth_warps_translation(warps, window=5):
    """Temporal smoothing for translation-only warps."""
    if len(warps) < window:
        return warps
    smoothed = []
    half = window // 2
    for i in range(len(warps)):
        xs, ys = [], []
        for j in range(max(0, i-half), min(len(warps), i+half+1)):
            w = warps[j]
            xs.append(w[0, 2])
            ys.append(w[1, 2])
        w = warps[i].copy()
        w[0, 2] = float(np.median(xs))
        w[1, 2] = float(np.median(ys))
        smoothed.append(w)
    return smoothed


class TileProcessor:
    def __init__(self, tile_id, bbox, frames, global_weights, cfg):
        self.tile_id = tile_id
        self.bbox = bbox  # (x0, y0, w, h)
        self.frames = frames
        self.global_weights = global_weights
        self.cfg = cfg

        self.reference = None
        self.valid = True

    def _extract(self, frame):
        x0, y0, w, h = self.bbox
        return frame[y0:y0+h, x0:x0+w]

    def _apply_warp(self, tile, warp):
        h, w = tile.shape
        return cv2.warpAffine(tile, warp, (w, h), flags=cv2.INTER_LINEAR)

    def _initial_reference(self):
        stack = np.stack([self._extract(f) for f in self.frames], axis=0)
        return np.median(stack, axis=0)

    def run(self):
        ref = self._initial_reference()

        for it in range(self.cfg.v4.iterations):
            warped_tiles = []
            warps = []
            weights = []

            for f, Gf in zip(self.frames, self.global_weights):
                tile = self._extract(f)
                warp, cc = register_tile(
                    tile,
                    ref,
                    ecc_cc_min=self.cfg.registration.local_tiles.ecc_cc_min,
                )
                if warp is None:
                    continue

                warped = self._apply_warp(tile, warp)
                warped_tiles.append(warped)
                warps.append(warp)
                weights.append(Gf * np.exp(self.cfg.v4.beta * (cc - 1.0)))

            if len(warped_tiles) < self.cfg.registration.local_tiles.min_valid_frames:
                self.valid = False
                return None

            warps = smooth_warps_translation(warps)
            stack = np.stack(warped_tiles, axis=0)
            w = np.asarray(weights, dtype=np.float64)
            w /= max(EPS, np.sum(w))
            ref = np.sum(stack * w[:, None, None], axis=0)

        self.reference = ref
        return ref
