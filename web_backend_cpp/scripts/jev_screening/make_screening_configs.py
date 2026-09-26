#!/usr/bin/env python3
"""Generate the config files of a parameter-screening chain: one config per variant, each differing from the base in exactly
the listed keys.

    make_screening_configs.py --base BASE.yaml --out-dir DIR [--preset reconstruction|confirmation] [--variants FILE.json]
                              [--keep-downstream]

A variant file is a JSON list of {"name": "...", "changes": {"dotted.path": value, ...}}. Every changed key must already exist
in the base config (a typo is an error, never a silently added key). By default PCC, astrometry, BGE and the stretch are switched
off in EVERY arm (same for all): reconstruction screenings run on a frame subset, where PCC finds too few stars, and only the
reconstruction output is measured. The names are written to DIR/names.json in run order (control first).
The `confirmation` preset also normalises every arm (control included) to the control levels of release_policy_v2.json
(pixfrac 0.8, clipping 4.0/4.0), whatever the session's own config uses.
"""
import argparse
import json
import os
import sys

import yaml

R = "reconstruction."

PRESETS = {
    # the IC4605 screening of 2026-09-26 (docs/PI/pi_jev_effektgroessen_rekonstruktion_20260926.md)
    "reconstruction": [
        ("control", {}),
        ("pixfrac_0.6", {R + "drizzle.pixfrac": 0.6}), ("pixfrac_1.0", {R + "drizzle.pixfrac": 1.0}),
        ("clip_2_4", {R + "clipping.clip_sigma_low": 2.0, R + "clipping.clip_sigma_high": 4.0}),
        ("clip_3_3", {R + "clipping.clip_sigma_low": 3.0, R + "clipping.clip_sigma_high": 3.0}),
        ("clip_5_5", {R + "clipping.clip_sigma_low": 5.0, R + "clipping.clip_sigma_high": 5.0}),
        ("passes_2", {R + "drizzle.robust_passes": 2}), ("passes_6", {R + "drizzle.robust_passes": 6}),
        ("internal_scale_1", {R + "drizzle.internal_scale": 1}),
        ("multiband_levels_2", {R + "multiband.levels": 2}), ("multiband_levels_4", {R + "multiband.levels": 4}),
        ("pyramid_sharp_0.8", {R + "quality.pyramid.sharpness_weight": 0.8, R + "quality.pyramid.snr_weight": 0.2}),
        ("pyramid_snr_0.7", {R + "quality.pyramid.sharpness_weight": 0.3, R + "quality.pyramid.snr_weight": 0.7}),
        ("pyramid_score_scale_2.5", {R + "quality.pyramid.score_scale": 2.5}),
        ("exponent_0.8", {"global_metrics.weight_exponent_scale": 0.8}), ("exponent_1.8", {"global_metrics.weight_exponent_scale": 1.8}),
        ("prewarp_lanczos4", {"registration.prewarp_interpolation": "lanczos4"}),
        ("prewarp_linear", {"registration.prewarp_interpolation": "linear"}),
    ],
    # the pre-registered confirmation arms of release_policy_v2.json
    "confirmation": [
        ("control", {}),
        ("pixfrac_1.0", {R + "drizzle.pixfrac": 1.0}),
        ("clip_5_5", {R + "clipping.clip_sigma_low": 5.0, R + "clipping.clip_sigma_high": 5.0}),
    ],
}

# Levels every arm of a preset is normalised to BEFORE its own change (the control levels of release_policy_v2.json): a session whose own
# config happens to use another level (e.g. clipping 2/4) would otherwise not have the pre-registered control.
PRESET_BASELINES = {
    "confirmation": {R + "drizzle.pixfrac": 0.8, R + "clipping.clip_sigma_low": 4.0, R + "clipping.clip_sigma_high": 4.0},
}

DOWNSTREAM_OFF = {"pcc.enabled": False, "astrometry.enabled": False, "hypermetric_stretch.enabled": False,
                  "bge.method": "none", "registration.use_astrometry": False}


def set_path(config, dotted, value):
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError("no such config section: " + dotted)
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError("no such config key (refusing to add it): " + dotted)
    node[keys[-1]] = value


def build(base, variants, downstream_off=True, baseline=None):
    """Returns {name: config}; the control (empty changes) must come first and no name may repeat.

    `baseline` (dotted path -> value) is applied to EVERY arm first, so all arms share the same control levels."""
    names = [n for n, _ in variants]
    if len(set(names)) != len(names):
        raise ValueError("duplicate variant names")
    if not variants or variants[0][0] != "control" or variants[0][1]:
        raise ValueError("the first variant must be an unchanged 'control'")
    out = {}
    for name, changes in variants:
        config = yaml.safe_load(yaml.safe_dump(base))
        for path, value in (baseline or {}).items():
            set_path(config, path, value)
        for path, value in changes.items():
            set_path(config, path, value)
        if downstream_off:
            for path, value in DOWNSTREAM_OFF.items():
                set_path(config, path, value)
        out[name] = config
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--preset", choices=sorted(PRESETS))
    grp.add_argument("--variants")
    ap.add_argument("--keep-downstream", action="store_true", help="do not switch off PCC/astrometry/BGE/stretch")
    args = ap.parse_args(argv)
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        ap.error("output directory exists and is not empty")
    with open(args.base, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    if args.preset:
        variants = PRESETS[args.preset]
    else:
        with open(args.variants, encoding="utf-8") as fh:
            variants = [(v["name"], v.get("changes", {})) for v in json.load(fh)]
    configs = build(base, variants, not args.keep_downstream, PRESET_BASELINES.get(args.preset))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, config in configs.items():
        with open(os.path.join(args.out_dir, name + ".yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=True)
    with open(os.path.join(args.out_dir, "names.json"), "w", encoding="utf-8") as fh:
        json.dump(list(configs), fh)
    print("%d configs written to %s" % (len(configs), args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
