#!/usr/bin/env python3
"""Evaluate a reconstruction screening: every variant run against its control on matched stars (read-only).

    evaluate_reconstruction_screening.py --runs-dir RUNS --prefix scr_ic4605_120 --out FILE.json [--variants a,b,c]

Compares run <prefix>_<variant> with <prefix>_control (green channel of the control's selected output, translation-aligned,
stars re-matched; see matched_pair_metrics.py) and reports per variant: matched stars, star width/elongation/signal ratio,
noise ratio, the coverage-gate figure n_eff p10, the pairing check (reference frame, raster) and the selected output.
A variant whose run has no result (e.g. it failed a protected gate) is listed with the reason instead of numbers.
Ratios are candidate/control: below 1.0 is lower noise, narrower stars.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import matched_pair_metrics as mp  # noqa: E402


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_summary(run):
    d = read_json(os.path.join(run, "artifacts", "forward_drizzle.json"))
    return {"selected": d["selected_candidate"], "ref_frame": read_json(os.path.join(run, "artifacts", "global_registration.json"))["ref_frame"],
            "n_eff_p10": read_json(os.path.join(run, "artifacts", "sampling_geometry.json"))["coverage_gate"].get("min_channel_n_eff_p10")}


def failure_reason(run):
    events = os.path.join(run, "logs", "run_events.jsonl")
    if not os.path.exists(events):
        return "no_run"
    reason = None
    with open(events, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("error"):
                reason = e["error"]
    return reason or "no_result"


def evaluate(runs_dir, prefix, variants=None):
    control = os.path.join(runs_dir, prefix + "_control")
    ctl = run_summary(control)
    if variants is None:
        variants = sorted(n[len(prefix) + 1:] for n in os.listdir(runs_dir) if n.startswith(prefix + "_") and n != prefix + "_control")
    out = {"control": ctl, "variants": {}}
    for name in variants:
        run = os.path.join(runs_dir, prefix + "_" + name)
        if not os.path.exists(os.path.join(run, "artifacts", "forward_drizzle.json")):
            out["variants"][name] = {"no_result": failure_reason(run)}
            continue
        row = run_summary(run)
        row["pairing"] = {"same_reference_frame": row["ref_frame"] == ctl["ref_frame"]}
        try:
            c, k, meta = mp.load_pair(control, run, "G", ctl["selected"])
            r = mp.compare_images(c, k, align=True)
            row["pairing"]["identical_raster"] = r["raster"]["identical"]
            row.update({
                "matched_stars": r["matched_stars"],
                "fwhm_ratio": r["fwhm_ratio_candidate_over_control"]["median"],
                "elongation_ratio": r["elongation_ratio_candidate_over_control"]["median"],
                "signal_ratio": r["signal_ratio_candidate_over_control"]["median"],
                "noise_ratio": r["noise"]["ratio_candidate_over_control"],
                "valid_background_pixels": r["noise"]["valid_background_pixels"],
                "n_eff_ratio": (row["n_eff_p10"] / ctl["n_eff_p10"]) if ctl["n_eff_p10"] else None,
                "selected_unchanged": row["selected"] == ctl["selected"],
            })
        except (ValueError, OSError) as error:
            row["error"] = str(error)[:160]
        out["variants"][name] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", help="comma separated variant names (default: all runs with the prefix)")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if out.startswith(os.path.abspath(args.runs_dir) + os.sep):
        ap.error("refusing to write the report inside the runs directory")
    result = evaluate(args.runs_dir, args.prefix, args.variants.split(",") if args.variants else None)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("%-26s %6s %8s %8s %8s %8s %8s" % ("variant", "stars", "fwhm", "elong", "signal", "noise", "n_eff"))
    for name, r in result["variants"].items():
        if "no_result" in r:
            print("%-26s no result: %s" % (name, r["no_result"]))
        elif "error" in r:
            print("%-26s error: %s" % (name, r["error"]))
        else:
            print("%-26s %6d %8.4f %8.4f %8.4f %8.4f %8.2f" % (name, r["matched_stars"], r["fwhm_ratio"], r["elongation_ratio"],
                                                              r["signal_ratio"], r["noise_ratio"], r["n_eff_p10"] or 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
