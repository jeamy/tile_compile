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
MATCH_R = 4            # the same star is searched in the candidate within this radius of the control position
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


def refine_center(img, y, x, radius=MATCH_R):
    """Position of the brightest smoothed pixel within `radius` of (y, x): the same star in the other image.

    Two runs of the same data are not registered identically (see docs), so a star may sit a pixel or two away
    from where the control has it. The offset is returned so it can be reported as a geometry diagnostic.
    """
    import cv2
    r = radius + 2
    patch = np.asarray(img[y - r: y + r + 1, x - r: x + r + 1], dtype=np.float64)
    sm = cv2.GaussianBlur(patch, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dist = np.hypot(yy, xx)
    sm[dist > radius] = -np.inf
    # The star is the NEAREST clear local maximum, not merely the brightest pixel: in a crowded field a brighter
    # neighbour inside the radius must not win over the star itself. "Clear" = well above the patch background/noise.
    finite = np.isfinite(sm)
    bg = float(np.median(sm[finite]))
    noise = max(1.4826 * float(np.median(np.abs(sm[finite] - bg))), 1e-12)
    is_max = sm == cv2.dilate(sm, np.ones((3, 3), np.uint8))
    cand = is_max & finite & (sm - bg > 5.0 * noise)
    if cand.any():
        idx = np.argwhere(cand)
        best = idx[np.argmin(dist[cand])]
        iy, ix = int(best[0]), int(best[1])
    else:
        iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return y + int(iy) - r, x + int(ix) - r


COARSE_R = 12          # first-pass search radius used to fit the geometry difference between the two runs
MIN_FIT_STARS = 30


def fit_geometry(pairs):
    """Robust affine map control -> candidate from coarse star pairs [(y, x, yc, xc)].

    Returns (matrix 2x3 in (x, y) order, inlier count) or None when there are too few pairs. Iteratively re-weighted
    least squares with a 1.5 px clip: the map describes the run-to-run registration difference (rotation, scale,
    translation) and is reported as a diagnostic.
    """
    p = np.asarray(pairs, float)
    if p.shape[0] < MIN_FIT_STARS:
        return None
    src = np.column_stack([p[:, 1], p[:, 0], np.ones(len(p))])
    dst = p[:, [3, 2]]
    keep = np.ones(len(p), bool)
    m = None
    for _ in range(6):
        if keep.sum() < MIN_FIT_STARS:
            return None
        m, *_ = np.linalg.lstsq(src[keep], dst[keep], rcond=None)
        res = np.hypot(*(src @ m - dst).T)
        new_keep = res < max(1.5, 3 * float(np.median(res[keep])))
        if (new_keep == keep).all():
            break
        keep = new_keep
    return m.T, int(keep.sum())


def geometry_summary(m, n_fit):
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    return {"affine_xy": [[float(v) for v in m[0]], [float(v) for v in m[1]]], "fit_stars": n_fit,
            "rotation_deg": float(np.degrees(np.arctan2(c - b, a + d))),
            "scale": float(np.sqrt(abs(a * d - b * c)))}


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


def estimate_shift(control, candidate, size=1024):
    """Integer (dy, dx) such that candidate[y + dy, x + dx] matches control[y, x] (translation only).

    Phase correlation on high-passed central crops. Used when two runs of the same data ended up on canvases of
    different size or origin (registration is not bit-reproducible between runs). Refuses a weak or large shift.
    """
    import cv2
    h = min(control.shape[0], candidate.shape[0], size)
    w = min(control.shape[1], candidate.shape[1], size)

    def crop(img):
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        c = np.array(img[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w], dtype=np.float64)
        c[~valid_mask(c)] = np.median(c[valid_mask(c)])
        return c - _blur(c, 12.0)
    a, b = crop(control), crop(candidate)
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    (sx, sy), response = cv2.phaseCorrelate(a, b, win)
    # phaseCorrelate reports how far b is displaced relative to a (b(x) ~ a(x - s)) in crop coordinates. Both crops are
    # centred on their own image, so the origin offset is that displacement plus the difference of the crop origins.
    dy = int(round(sy)) + (candidate.shape[0] // 2 - h // 2) - (control.shape[0] // 2 - h // 2)
    dx = int(round(sx)) + (candidate.shape[1] // 2 - w // 2) - (control.shape[1] // 2 - w // 2)
    if response < 0.05 or abs(dy) > 64 or abs(dx) > 64:
        raise ValueError("alignment refused: weak or large shift (dy=%d dx=%d response=%.3f)" % (dy, dx, response))
    return dy, dx, float(response)


def align_overlap(control, candidate, dy, dx):
    """Views of both images restricted to their common area, given candidate[y+dy, x+dx] ~ control[y, x]."""
    y0, x0 = max(0, -dy), max(0, -dx)
    y1 = min(control.shape[0], candidate.shape[0] - dy)
    x1 = min(control.shape[1], candidate.shape[1] - dx)
    if y1 - y0 < 2 * BORDER + 50 or x1 - x0 < 2 * BORDER + 50:
        raise ValueError("no usable overlap after alignment")
    return control[y0:y1, x0:x1], candidate[y0 + dy:y1 + dy, x0 + dx:x1 + dx]


def compare_images(control, candidate, saturation_fraction=0.9, align=False):
    raster = {"control": list(control.shape), "candidate": list(candidate.shape), "identical": control.shape == candidate.shape}
    alignment = None
    full_cov = (float(valid_mask(control).sum() / control.size), float(valid_mask(candidate).sum() / candidate.size))
    if align:
        dy, dx, resp = estimate_shift(control, candidate)
        alignment = {"dy": dy, "dx": dx, "phase_correlation_response": resp}
        if dy or dx or control.shape != candidate.shape:
            control, candidate = align_overlap(control, candidate, dy, dx)
    if control.shape != candidate.shape:
        raise ValueError("control and candidate rasters differ: %s vs %s (use --align)" % (control.shape, candidate.shape))
    vc, vd = valid_mask(control), valid_mask(candidate)
    common = vc & vd
    if not common.any():
        raise ValueError("no common valid pixels")
    stars, peaks, sigma = detect_stars(control, common, saturation_fraction)
    h, w = control.shape
    coarse = []
    for y, x in stars:
        yc, xc = refine_center(candidate, y, x, COARSE_R)
        coarse.append((y, x, yc, xc))
    fit = fit_geometry(coarse)
    geometry = None
    if fit:
        m, n_fit = fit
        geometry = geometry_summary(m, n_fit)
    rows, offsets, displacement, unmatched = [], [], [], 0
    for y, x in stars:
        if fit:
            xp, yp = m @ np.array([x, y, 1.0])
            yp, xp = int(round(yp)), int(round(xp))
        else:
            yp, xp = y, x
        if yp < BORDER or xp < BORDER or yp >= h - BORDER or xp >= w - BORDER:
            unmatched += 1
            continue
        yc, xc = refine_center(candidate, yp, xp)
        a, b = measure_star(control, common, y, x), measure_star(candidate, common, yc, xc)
        if a and b and a[0] > 0 and b[0] > 0 and a[2] > 0 and b[2] > 0:
            rows.append((a, b))
            offsets.append(float(np.hypot(yc - yp, xc - xp)))         # residual after the fitted geometry
            displacement.append(float(np.hypot(yc - y, xc - x)))      # total displacement between the two runs
        else:
            unmatched += 1
    off, disp = np.array(offsets), np.array(displacement)
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
        "unmatched_stars": int(unmatched),
        "star_offset_px": {"median": float(np.median(off)) if off.size else None,
                           "p95": float(np.percentile(off, 95)) if off.size else None, "search_radius": MATCH_R},
        "run_to_run_geometry": {"fit": geometry,
                                "total_displacement_px": {"median": float(np.median(disp)) if disp.size else None,
                                                          "p95": float(np.percentile(disp, 95)) if disp.size else None,
                                                          "max": float(disp.max()) if disp.size else None}},
        "noise": {"control": noise_c, "candidate": noise_d,
                  "ratio_candidate_over_control": (noise_d / noise_c) if noise_c else None, "valid_background_pixels": n_bg},
        "coverage": {"control": float(cov_c), "candidate": float(cov_d), "intersection": float(cov_i),
                     "coverage_loss": float(cov_c - cov_d),
                     "intersection_over_control": float(common.sum() / max(vc.sum(), 1))},
        "detection_sigma": float(sigma),
        "raster": raster,
        "alignment": alignment,
        "coverage_full_raster": {"control": full_cov[0], "candidate": full_cov[1], "loss": full_cov[0] - full_cov[1]},
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
    ap.add_argument("--align", action="store_true", help="translate-align the candidate onto the control (needed when the canvases differ)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    for run in (args.control, args.candidate):
        if out.startswith(os.path.abspath(run) + os.sep):
            ap.error("refusing to write the report inside a run directory")
    control, candidate, meta = load_pair(args.control, args.candidate, args.channel, args.candidate_name)
    result = compare_images(control, candidate, align=args.align)
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
    print(json.dumps({k: result[k] for k in ("matched_stars", "fwhm_ratio_candidate_over_control", "noise", "coverage", "raster", "alignment")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
