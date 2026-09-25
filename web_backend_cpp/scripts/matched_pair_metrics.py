#!/usr/bin/env python3
"""Matched-position quality metrics for a control/candidate pair of finished runs (read-only).

Implements the measurement plan of release_policy_v1.json for one session:
  star shape    same star coordinates and the same aperture/moment estimator on both outputs
  noise         same valid background pixels on both outputs (stars masked)
  signal        same unsaturated source apertures, same background subtraction
  coverage      per-arm valid fraction and their difference on the identical output raster

Stars are detected ONCE, on the control output, and measured at the same coordinates in both images, so the
ratios are paired per star. The star-level bootstrap interval is descriptive only: the policy's confidence
intervals are on SESSION level (one pair = one session) and need many sessions.

    matched_pair_metrics.py --control RUN --candidate RUN [--candidate-name drizzle_raw] [--channel G] --out FILE

Only reads run artifacts; refuses to write inside a run directory.
"""
import argparse
import json
import os
import sys

import numpy as np

CANDIDATE_FILES = {"drizzle_raw": "forward_drizzle_raw_%s.fit", "drizzle_uniform": "forward_drizzle_uniform_%s.fit",
                   "drizzle_multiband": "forward_drizzle_multiband_%s.fit"}
FWHM_PER_SIGMA = 2.354820045
WINDOW_R = 10          # cutout half size (21x21)
MOMENT_R = 6           # moment/shape aperture radius (pixels)
SIGNAL_R = 4           # flux aperture radius (pixels)
ISOLATION_R = 15       # no other >5 sigma peak within this distance
BORDER = 25
DETECT_SIGMA = 8.0
NEIGHBOR_SIGMA = 5.0
BOOTSTRAP = 2000
SEED = 20260925


def read_fits(path):
    from astropy.io import fits
    data = fits.getdata(path)
    return np.asarray(np.squeeze(data), dtype=np.float64)


def selected_candidate(run_dir):
    with open(os.path.join(run_dir, "artifacts", "forward_drizzle.json"), encoding="utf-8") as fh:
        return json.load(fh)["selected_candidate"]


def valid_mask(img):
    return np.isfinite(img) & (img != 0)


def robust_sigma(values):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def _blur(img, sigma):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _local_max(d, size=11):
    import cv2
    return d == cv2.dilate(d, np.ones((size, size), np.uint8))


def detect_stars(control, valid, saturation_fraction=0.9):
    """Returns (stars [(y, x)], all_peaks [(y, x)], sigma). Stars are isolated, unsaturated, away from the border."""
    filled = np.where(valid, control, np.median(control[valid]))
    d = filled - _blur(filled, 12.0)
    sigma = robust_sigma(d[valid])
    if not np.isfinite(sigma) or sigma <= 0:
        return [], np.empty((0, 2), int), float("nan")
    peaks_mask = _local_max(d) & valid & (d > NEIGHBOR_SIGMA * sigma)
    ys, xs = np.nonzero(peaks_mask)
    all_peaks = np.stack([ys, xs], axis=1)
    cell = ISOLATION_R
    grid = {}
    for i, (y, x) in enumerate(all_peaks):
        grid.setdefault((y // cell, x // cell), []).append(i)
    sat = saturation_fraction * float(np.max(control[valid]))
    h, w = control.shape
    stars = []
    for i, (y, x) in enumerate(all_peaks):
        if d[y, x] < DETECT_SIGMA * sigma or control[y, x] >= sat:
            continue
        if y < BORDER or x < BORDER or y >= h - BORDER or x >= w - BORDER:
            continue
        crowded = False
        for gy in (y // cell - 1, y // cell, y // cell + 1):
            for gx in (x // cell - 1, x // cell, x // cell + 1):
                for j in grid.get((gy, gx), []):
                    if j != i and (all_peaks[j][0] - y) ** 2 + (all_peaks[j][1] - x) ** 2 <= ISOLATION_R ** 2:
                        crowded = True
        if not crowded:
            stars.append((int(y), int(x)))
    return stars, all_peaks, sigma


_YY, _XX = np.mgrid[-WINDOW_R:WINDOW_R + 1, -WINDOW_R:WINDOW_R + 1]
_RR = np.hypot(_YY, _XX)


def measure_star(img, valid, y, x):
    """(fwhm, elongation, flux) at fixed integer coordinates, or None when the cutout is unusable."""
    sl = (slice(y - WINDOW_R, y + WINDOW_R + 1), slice(x - WINDOW_R, x + WINDOW_R + 1))
    if not valid[sl].all():
        return None
    patch = img[sl]
    ring = patch[(_RR >= WINDOW_R - 2) & (_RR <= WINDOW_R)]
    patch = patch - np.median(ring)
    w = np.clip(patch, 0, None) * (_RR <= MOMENT_R)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    cy, cx = (w * _YY).sum() / total, (w * _XX).sum() / total
    dy, dx = _YY - cy, _XX - cx
    m20, m02, m11 = (w * dx * dx).sum() / total, (w * dy * dy).sum() / total, (w * dx * dy).sum() / total
    tr, det = m20 + m02, m20 * m02 - m11 * m11
    disc = max(tr * tr / 4 - det, 0.0)
    lam_max, lam_min = tr / 2 + disc ** 0.5, tr / 2 - disc ** 0.5
    if lam_min <= 0:
        return None
    fwhm = FWHM_PER_SIGMA * (tr / 2) ** 0.5
    flux = float(patch[_RR <= SIGNAL_R].sum())
    return fwhm, (lam_max / lam_min) ** 0.5, flux


def star_mask(shape, peaks, radius=12):
    import cv2
    m = np.zeros(shape, np.uint8)
    for y, x in peaks:
        cv2.circle(m, (int(x), int(y)), radius, 1, -1)
    return m.astype(bool)


def bootstrap_ci(values, resamples=BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if v.size < 2:
        return None
    meds = np.median(v[rng.integers(0, v.size, (resamples, v.size))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def compare_images(control, candidate, saturation_fraction=0.9):
    if control.shape != candidate.shape:
        raise ValueError("control and candidate rasters differ: %s vs %s" % (control.shape, candidate.shape))
    vc, vd = valid_mask(control), valid_mask(candidate)
    common = vc & vd
    if not common.any():
        raise ValueError("no common valid pixels")
    stars, peaks, sigma = detect_stars(control, common, saturation_fraction)
    rows = []
    for y, x in stars:
        a, b = measure_star(control, common, y, x), measure_star(candidate, common, y, x)
        if a and b and a[0] > 0 and b[0] > 0 and a[2] > 0 and b[2] > 0:
            rows.append((a, b))
    fwhm_ratio = np.array([b[0] / a[0] for a, b in rows])
    elong_ratio = np.array([b[1] / a[1] for a, b in rows])
    signal_ratio = np.array([b[2] / a[2] for a, b in rows])
    # Noise: the same background pixels, stars and their surroundings masked, small-scale structure only.
    excluded = star_mask(control.shape, peaks) | ~common
    hp_c = control - _blur(np.where(common, control, np.median(control[common])), 6.0)
    hp_d = candidate - _blur(np.where(common, candidate, np.median(candidate[common])), 6.0)
    bg = ~excluded
    n_bg = int(bg.sum())
    noise_c, noise_d = robust_sigma(hp_c[bg]), robust_sigma(hp_d[bg])
    area = float(control.size)
    cov_c, cov_d, cov_i = vc.sum() / area, vd.sum() / area, common.sum() / area

    def summary(r):
        if r.size == 0:
            return {"median": None, "ci95_star_bootstrap": None}
        return {"median": float(np.median(r)), "ci95_star_bootstrap": bootstrap_ci(r)}

    return {
        "matched_stars": int(len(rows)),
        "detected_isolated_stars": int(len(stars)),
        "fwhm_ratio_candidate_over_control": summary(fwhm_ratio),
        "elongation_ratio_candidate_over_control": summary(elong_ratio),
        "signal_ratio_candidate_over_control": summary(signal_ratio),
        "signal_apertures": int(len(rows)),
        "noise": {"control": noise_c, "candidate": noise_d,
                  "ratio_candidate_over_control": (noise_d / noise_c) if noise_c else None, "valid_background_pixels": n_bg},
        "coverage": {"control": float(cov_c), "candidate": float(cov_d), "intersection": float(cov_i),
                     "coverage_loss": float(cov_c - cov_d),
                     "intersection_over_control": float(common.sum() / max(vc.sum(), 1))},
        "detection_sigma": float(sigma),
    }


def policy_flags(result, policy):
    """Which sample-size minimums of the frozen policy this single session satisfies (no verdict)."""
    c = policy["candidates"]["enable_adaptive_weights"]
    return {
        "matched_stars_ok": result["matched_stars"] >= c["minimum_matched_stars_per_session"],
        "background_pixels_ok": result["noise"]["valid_background_pixels"] >= c["minimum_valid_background_pixels_per_session"],
        "signal_apertures_ok": result["signal_apertures"] >= c["minimum_unsaturated_signal_apertures_per_session"],
        "common_area_ok": result["coverage"]["intersection_over_control"] >= c["minimum_coverage"],
    }


def load_pair(control_run, candidate_run, channel="G", candidate_name=None):
    sel_c, sel_d = selected_candidate(control_run), selected_candidate(candidate_run)
    name = candidate_name or sel_c
    if candidate_name is None and sel_c != sel_d:
        raise ValueError("selected candidates differ (%s vs %s); pass --candidate-name to compare one named output" % (sel_c, sel_d))
    if name not in CANDIDATE_FILES:
        raise ValueError("unknown candidate " + name)
    fname = CANDIDATE_FILES[name] % channel
    return (read_fits(os.path.join(control_run, "outputs", fname)), read_fits(os.path.join(candidate_run, "outputs", fname)),
            {"compared_output": fname, "selected_control": sel_c, "selected_candidate": sel_d})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-name", choices=sorted(CANDIDATE_FILES))
    ap.add_argument("--channel", default="G", choices=["R", "G", "B"])
    ap.add_argument("--policy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    for run in (args.control, args.candidate):
        if out.startswith(os.path.abspath(run) + os.sep):
            ap.error("refusing to write the report inside a run directory")
    control, candidate, meta = load_pair(args.control, args.candidate, args.channel, args.candidate_name)
    result = compare_images(control, candidate)
    result["source"] = dict(meta, channel=args.channel, control_run=os.path.basename(os.path.abspath(args.control)),
                            candidate_run=os.path.basename(os.path.abspath(args.candidate)))
    result["scope"] = ("paired per star on one session; the confidence interval is a star-level bootstrap and descriptive only, "
                       "the release policy requires session-level intervals")
    if args.policy:
        with open(args.policy, encoding="utf-8") as fh:
            result["policy_sample_size_flags"] = policy_flags(result, json.load(fh))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: result[k] for k in ("matched_stars", "fwhm_ratio_candidate_over_control", "noise", "coverage")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
