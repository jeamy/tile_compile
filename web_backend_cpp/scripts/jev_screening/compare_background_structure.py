#!/usr/bin/env python3
"""Compare the large-scale sky structure of two stacks of the same field (read-only): is a flat background real or processing?

    compare_background_structure.py --a STACK_A.fits --wcs-a A.wcs --b STACK_B.fits --wcs-b B.wcs [--out-map MAP.png] [--out FILE.json]

Both stacks are converted to luminance, the stars are removed (8x downsampling, median filter, 48 px smoothing) and B is resampled
onto A's pixel grid through the two WCS solutions (astropy), so the same sky is compared. Reported: overlap, and the correlation of the
large-scale background maps raw and after removing a plane / a quadratic surface (residual gradient). A high correlation means the
structure is in BOTH stacks (real sky signal); a low one means one of them carries artefacts or lost the structure.

The `.wcs` files written by the pipeline are FITS header cards (80 characters each) with or without line breaks; both are read.
Use LINEAR stacks (`stacked_rgb.fits`) for the structure question: a stretched image mixes in the stretch. The optional map draws both
maps side by side, each normalised to its own +-3 sigma, so shapes (not amplitudes) can be compared by eye.
"""
import argparse
import json
import sys
import warnings

import numpy as np


def read_wcs_header(path):
    from astropy.io import fits
    text = open(path, errors="replace").read().replace("\n", "").replace("\r", "")
    cards = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
    return fits.Header.fromstring("".join(c.ljust(80)[:80] for c in cards), sep="")


def load_luminance(path):
    from astropy.io import fits
    d = np.squeeze(np.asarray(fits.getdata(path), dtype=np.float32))
    if d.ndim == 3:
        d = d.mean(axis=0) if d.shape[0] == 3 else d.mean(axis=2)
    return d


def large_scale(lum, sigma=48.0, factor=8):
    """(smoothed star-free map at 1/factor resolution, mask of trustworthy pixels)."""
    import cv2
    good = np.isfinite(lum) & (lum > 0)
    fill = np.where(good, lum, np.median(lum[good])).astype(np.float32)
    small = cv2.resize(fill, (fill.shape[1] // factor, fill.shape[0] // factor), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, 5)                      # OpenCV's float median filter supports ksize <= 5
    big = cv2.GaussianBlur(med, (0, 0), sigma / factor)
    gs = cv2.resize(good.astype(np.uint8) * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 250
    gs = cv2.erode(gs.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return big, gs


def detrend(img, mask, degree):
    ys, xs = np.nonzero(mask)
    xn, yn = (xs - xs.mean()) / xs.std(), (ys - ys.mean()) / ys.std()
    cols = [np.ones(len(xs)), xn, yn] + ([xn * xn, yn * yn, xn * yn] if degree == 2 else [])
    a = np.column_stack(cols)
    v = img[mask]
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return v - a @ coef


def compare(lum_a, header_a, lum_b, header_b, factor=8):
    import cv2
    from astropy.wcs import WCS
    wa, wb = WCS(header_a, naxis=2), WCS(header_b, naxis=2)
    ba, ga = large_scale(lum_a, factor=factor)
    bb, gb = large_scale(lum_b, factor=factor)
    h, w = ba.shape
    yy, xx = np.mgrid[0:h, 0:w]
    world = wa.pixel_to_world_values(xx * factor + factor // 2, yy * factor + factor // 2)
    bx, by = wb.world_to_pixel_values(world[0], world[1])
    mapx, mapy = (np.asarray(bx) / factor).astype(np.float32), (np.asarray(by) / factor).astype(np.float32)
    bb_on_a = cv2.remap(bb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    gb_on_a = cv2.remap(gb.astype(np.float32), mapx, mapy, cv2.INTER_NEAREST, borderValue=0) > 0.5
    mask = ga & gb_on_a & np.isfinite(bb_on_a)
    if mask.sum() < 100:
        raise ValueError("the stacks barely overlap on the sky")
    a, b = ba[mask], bb_on_a[mask]
    out = {"overlap_fraction_of_a": float(mask.mean()), "overlap_pixels_at_1_over_%d" % factor: int(mask.sum()),
           "correlation_raw": float(np.corrcoef(a, b)[0, 1])}
    for degree, label in ((1, "plane"), (2, "quadratic")):
        ra, rb = detrend(ba, mask, degree), detrend(bb_on_a, mask, degree)
        out["correlation_after_%s_removed" % label] = float(np.corrcoef(ra, rb)[0, 1])
        out["std_after_%s_removed" % label] = {"a": float(ra.std()), "b": float(rb.std())}
    return out, np.where(mask, ba, np.nan), np.where(mask, bb_on_a, np.nan)


def side_by_side_map(map_a, map_b, path, labels=("A", "B")):
    import cv2
    mask = np.isfinite(map_a) & np.isfinite(map_b)

    def prep(x, label):
        ys, xs = np.nonzero(mask)
        xn, yn = (xs - xs.mean()) / xs.std(), (ys - ys.mean()) / ys.std()
        a = np.column_stack([np.ones(len(xs)), xn, yn])
        v = x[mask]
        coef, *_ = np.linalg.lstsq(a, v, rcond=None)
        res = np.full(x.shape, np.nan)
        res[mask] = v - a @ coef
        img = np.nan_to_num(np.clip(0.5 + (res / np.nanstd(res)) / 6, 0, 1), nan=0.0)
        col = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        col[~mask] = 0
        col = cv2.resize(col, (col.shape[1] * 3, col.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        cv2.putText(col, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return col
    cv2.imwrite(path, np.hstack([prep(map_a, labels[0]), prep(map_b, labels[1])]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--wcs-a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--wcs-b", required=True)
    ap.add_argument("--out")
    ap.add_argument("--out-map")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    result, ma, mb = compare(load_luminance(args.a), read_wcs_header(args.wcs_a), load_luminance(args.b), read_wcs_header(args.wcs_b))
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    if args.out_map:
        side_by_side_map(ma, mb, args.out_map, ("A", "B"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
