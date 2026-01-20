# PATCH 04 – Streaming TileProcessor v4 (OOM-safe, production-ready)
# Replaces in-memory frame lists with FITS tile streaming

import numpy as np
import cv2
from astropy.io import fits
from runner.tile_local_registration_v4 import register_tile

EPS = 1e-6


def smooth_warps_translation(warps, window=5):
    if len(warps) < window:
        return warps
    half = window // 2
    out = []
    for i, w in enumerate(warps):
        xs, ys = [], []
        for j in range(max(0, i-half), min(len(warps), i+half+1)):
            xs.append(warps[j][0, 2])
            ys.append(warps[j][1, 2])
        ws = w.copy()
        ws[0, 2] = float(np.median(xs))
        ws[1, 2] = float(np.median(ys))
        out.append(ws)
    return out


class StreamingTileProcessor:
    """
    Production-grade v4 TileProcessor.
    - No full frames in RAM
    - Reads only tile windows from disk
    - Deterministic memory usage
    """

    def __init__(self, tile_id, bbox, frame_paths, global_weights, cfg):
        self.tile_id = tile_id
        self.x0, self.y0, self.w, self.h = bbox
        self.frame_paths = frame_paths
        self.global_weights = global_weights
        self.cfg = cfg

        self.valid = True
        self.reference = None

    def _read_tile(self, path):
        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data
            return data[self.y0:self.y0+self.h, self.x0:self.x0+self.w].astype("float32")

    def run(self):
        # --- initial reference (median of streamed tiles)
        tiles = []
        for p in self.frame_paths:
            tiles.append(self._read_tile(p))
        ref = np.median(np.stack(tiles, axis=0), axis=0)
        del tiles

        # --- iterative refinement
        for _ in range(self.cfg.v4.iterations):
            warped_tiles = []
            warps = []
            weights = []

            for p, Gf in zip(self.frame_paths, self.global_weights):
                tile = self._read_tile(p)
                warp, cc = register_tile(
                    tile,
                    ref,
                    ecc_cc_min=self.cfg.registration.local_tiles.ecc_cc_min,
                )
                if warp is None:
                    continue
                warped = cv2.warpAffine(tile, warp, (self.w, self.h))
                warped_tiles.append(warped)
                warps.append(warp)
                weights.append(Gf * np.exp(self.cfg.v4.beta * (cc - 1.0)))

            if len(warped_tiles) < self.cfg.registration.local_tiles.min_valid_frames:
                self.valid = False
                return None, None

            warps = smooth_warps_translation(warps)
            stack = np.stack(warped_tiles, axis=0)
            w = np.asarray(weights, dtype=np.float64)
            w /= max(EPS, np.sum(w))
            ref = np.sum(stack * w[:, None, None], axis=0)

        self.reference = ref
        return ref, warps
