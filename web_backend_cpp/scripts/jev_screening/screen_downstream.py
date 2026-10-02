#!/usr/bin/env python3
"""Screen downstream parameters (denoise, BGE, PCC, stretch) on a COPY of a finished run, without a new reconstruction.

    screen_downstream.py --lab-run DIR --out-dir DIR [--from-phase BGE|PCC|HMS] [--only a,b,c] [--variants FILE.json]

1. Copy the finished run to a lab directory first (`cp -a RUN LAB`; never point --lab-run at an archived original).
2. For every variant the lab config is rewritten with exactly the listed changes, `tile_compile_runner resume-reconstruction
   --from-phase <phase>` is run (seconds to a few minutes), and the final `outputs/stacked_rgb_hms.fits` is measured against the
   unchanged baseline: pixel noise, sky span, colour span, mid-scale structure, and star width/signal/elongation on the same stars.
3. The lab config is restored at the end.

A resume may only change the sections its phase allows (BGE: bge, pcc, chroma_denoise, luma_denoise, hypermetric_stretch; PCC:
pcc, chroma_denoise, hypermetric_stretch; HMS: hypermetric_stretch). The runner compares against the ORIGINAL config recorded in
`artifacts/config_revisions/index.json` (entry `run_start`); a run started directly from the CLI has none, so the script writes
one from the lab config, exactly as the web backend does at run start. The baseline must reproduce the run's own
`stacked_rgb_hms.fits` bit-identically; the report says whether it did.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import matched_pair_metrics as mp  # noqa: E402

H, L, C, B = "hypermetric_stretch.", "luma_denoise.", "chroma_denoise.", "bge."
DEFAULT_VARIANTS = [
    ("base", {}),
    ("luma_off", {L + "enabled": False}), ("luma_blend_0.5", {L + "blend_amount": 0.5}), ("luma_blend_1.0", {L + "blend_amount": 1.0}),
    ("luma_wavelet_thr_1.0", {L + "wavelet.threshold_scale": 1.0}), ("luma_wavelet_thr_2.5", {L + "wavelet.threshold_scale": 2.5}),
    ("luma_bilateral_on", {L + "bilateral.enabled": True}), ("luma_no_star_prot", {L + "star_protection.enabled": False}),
    ("chroma_on", {C + "enabled": True}), ("chroma_on_blend_1.0", {C + "enabled": True, C + "blend.amount": 1.0}),
    ("chroma_on_large_scale_bias", {C + "enabled": True, C + "large_scale_bias.enabled": True}),
    ("bge_none", {B + "method": "none"}), ("bge_classic", {B + "method": "classic"}), ("bge_autobge", {B + "method": "autobge"}),
    ("hms_cast_off", {H + "color_cast_correction.enabled": False}), ("hms_cast_max_0.3", {H + "color_cast_correction.max_amount": 0.3}),
    ("hms_logd_2.5", {H + "fixed_log_d": 2.5}), ("hms_logd_5.0", {H + "fixed_log_d": 5.0}),
    ("hms_adaptive_anchor_off", {H + "adaptive_anchor": False}),
    ("hms_convergence_2", {H + "convergence_power": 2.0}), ("hms_convergence_6", {H + "convergence_power": 6.0}),
    ("hms_protect_b_2", {H + "protect_b": 2.0}), ("hms_color_grip_0.5", {H + "color_grip": 0.5}),
    ("hms_linear_expansion_0.3", {H + "linear_expansion": 0.3}), ("hms_shadow_conv_0.5", {H + "shadow_convergence": 0.5}),
    ("hms_sensor_rec709", {H + "sensor_profile": "rec709", H + "fallback_profile": "rec709"}),
    ("pcc_off", {"pcc.enabled": False}), ("pcc_chroma_strength_0.3", {"pcc.chroma_strength": 0.3}),
]


def set_path(config, dotted, value):
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node[key]
    if keys[-1] not in node:
        raise KeyError("no such config key (refusing to add it): " + dotted)
    node[keys[-1]] = value


def ensure_run_start_revision(lab_run):
    """Record the current lab config as the run_start revision unless one exists (see the module docstring)."""
    rev_dir = os.path.join(lab_run, "artifacts", "config_revisions")
    index = os.path.join(rev_dir, "index.json")
    if os.path.exists(index):
        return False
    os.makedirs(rev_dir, exist_ok=True)
    shutil.copy(os.path.join(lab_run, "config.yaml"), os.path.join(rev_dir, "run_cfg_lab_start.yaml"))
    with open(index, "w", encoding="utf-8") as fh:
        json.dump([{"revision_id": "run_cfg_lab_start", "file_name": "run_cfg_lab_start.yaml", "source": "run_start",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": os.path.basename(lab_run.rstrip("/"))}], fh, indent=2)
    return True


def load_rgb(path):
    from astropy.io import fits
    d = np.squeeze(np.asarray(fits.getdata(path), dtype=np.float64))
    return np.moveaxis(d, 0, -1) if d.ndim == 3 and d.shape[0] == 3 else d


def sky_metrics(img):
    """Kennzahlen of a final RGB image: noise, sky span (block medians p5-p95), colour span, mid-scale structure over noise."""
    import cv2
    lum = img.mean(axis=2)
    good = np.isfinite(lum) & (lum > 0)
    bs = 256
    h, w = lum.shape
    hh, ww = h // bs * bs, w // bs * bs

    def blocks(a):
        return a[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
    gb = blocks(good)
    vb = gb.mean(axis=1) > 0.9

    def bmed(a):
        return np.array([np.median(b[g]) for b, g in zip(blocks(a)[vb], gb[vb])])
    med = bmed(lum)
    sky = med <= np.percentile(med, 70)
    filled = np.where(good, lum, np.median(lum[good])).astype(np.float32)
    hp = filled - cv2.GaussianBlur(filled, (0, 0), 4)
    sig = 1.4826 * float(np.median(np.abs(hp[good] - np.median(hp[good]))))
    rg, bg = bmed(img[..., 0] - img[..., 1])[sky], bmed(img[..., 2] - img[..., 1])[sky]
    band = cv2.GaussianBlur(filled, (0, 0), 6) - cv2.GaussianBlur(filled, (0, 0), 24)
    return {"level": float(np.median(lum[good])), "pixel_noise": sig,
            "sky_span": float(np.percentile(med[sky], 95) - np.percentile(med[sky], 5)),
            "chroma_RG_span": float(np.percentile(rg, 95) - np.percentile(rg, 5)),
            "chroma_BG_span": float(np.percentile(bg, 95) - np.percentile(bg, 5)),
            "midscale_over_noise": float(np.std(band[good]) / sig)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lab-run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--from-phase", default="BGE", choices=["BGE", "PCC", "HMS"])
    ap.add_argument("--only")
    ap.add_argument("--variants", help="JSON list of {name, changes} instead of the default set")
    ap.add_argument("--runner")
    ap.add_argument("--project-root")
    args = ap.parse_args(argv)
    here_root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    runner = args.runner or os.path.join(here_root, "tile_compile_cpp", "build", "tile_compile_runner")
    project_root = args.project_root or here_root
    cfg_path = os.path.join(args.lab_run, "config.yaml")
    original = open(cfg_path, encoding="utf-8").read()
    variants = DEFAULT_VARIANTS
    if args.variants:
        variants = [(v["name"], v.get("changes", {})) for v in json.load(open(args.variants, encoding="utf-8"))]
    only = set(args.only.split(",")) if args.only else None
    os.makedirs(args.out_dir, exist_ok=True)
    created = ensure_run_start_revision(args.lab_run)
    baseline_sha = hashlib.sha256(open(os.path.join(args.lab_run, "outputs", "stacked_rgb_hms.fits"), "rb").read()).hexdigest()[:12]
    results, base_green, base_metrics = {}, None, None
    try:
        for name, changes in variants:
            if only and name not in only and name != "base":
                continue
            config = yaml.safe_load(original)
            for path, value in changes.items():
                set_path(config, path, value)
            with open(cfg_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, sort_keys=True)
            started = time.time()
            r = subprocess.run([runner, "resume-reconstruction", "--run-dir", args.lab_run, "--from-phase", args.from_phase,
                                "--project-root", project_root], capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
                results[name] = {"rc": r.returncode, "tail": tail}
                print(name, "FAILED", tail, flush=True)
                continue
            out = os.path.join(args.lab_run, "outputs", "stacked_rgb_hms.fits")
            shutil.copy(out, os.path.join(args.out_dir, name + ".fits"))
            img = load_rgb(out)
            m = sky_metrics(img)
            m["seconds"] = round(time.time() - started, 1)
            green = img[..., 1]
            if name == "base":
                base_green, base_metrics = green, m
                m["reproduces_original"] = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12] == baseline_sha
            elif base_green is not None:
                try:
                    s = mp.compare_images(base_green, green)
                    m.update(star_fwhm_ratio=s["fwhm_ratio_candidate_over_control"]["median"], star_signal_ratio=s["signal_ratio_candidate_over_control"]["median"],
                             star_elong_ratio=s["elongation_ratio_candidate_over_control"]["median"], stars=s["matched_stars"])
                except (ValueError, OSError) as error:
                    m["star_error"] = str(error)[:80]
                for k in ("pixel_noise", "sky_span", "chroma_RG_span", "chroma_BG_span", "midscale_over_noise"):
                    m["d_" + k] = m[k] / base_metrics[k] - 1
            results[name] = m
            print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not k.startswith("d_")}, flush=True)
    finally:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(original)
        with open(os.path.join(args.out_dir, "screen_downstream.json"), "w", encoding="utf-8") as fh:
            json.dump({"lab_run": args.lab_run, "from_phase": args.from_phase, "run_start_revision_created": created, "variants": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
