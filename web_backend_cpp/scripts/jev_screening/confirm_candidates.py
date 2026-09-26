#!/usr/bin/env python3
"""Apply release_policy_v2.json to the session reports of a confirmation screening and write the verdict with its intervals.

    confirm_candidates.py --policy release_policy_v2.json --out verdict.json  m31=galaxy:m31_confirmation.json  m42=nebula:m42_confirmation.json ...

Each session is NAME=GROUP:REPORT.json, REPORT being the output of evaluate_reconstruction_screening.py. GROUP is the object group
(the policy does not define its meaning; the confirmation of 2026-09-26 used galaxy / emission-or-reflection nebula). Candidates and
arms: set_pixfrac <- pixfrac_1.0, set_clip_sigmas <- clip_5_5. A missing or failed arm (no noise_ratio) counts as a failed coverage
gate for that session. The verdict says whether the pre-registered endpoints hold on these sessions; it is not a release.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_v2  # noqa: E402

ARMS = {"set_pixfrac": "pixfrac_1.0", "set_clip_sigmas": "clip_5_5"}
FIELDS = ("noise_ratio", "fwhm_ratio", "elongation_ratio", "signal_ratio", "selected_unchanged", "n_eff_ratio", "matched_stars")


def session_row(group, variant):
    if not variant or "noise_ratio" not in variant:
        return None, "no_result"
    pairing = variant.get("pairing", {})
    ok = bool(pairing.get("same_reference_frame")) and bool(pairing.get("identical_raster"))
    if not ok:
        return None, "pairing_invalid"
    row = {k: variant[k] for k in FIELDS}
    row.update(group=group, gate_passed=True, background_pixels=variant["valid_background_pixels"])
    return row, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("sessions", nargs="+", help="NAME=GROUP:REPORT.json")
    args = ap.parse_args(argv)
    with open(args.policy, encoding="utf-8") as fh:
        policy = json.load(fh)
    policy_v2.validate(policy)
    reports = {}
    for spec in args.sessions:
        name, rest = spec.split("=", 1)
        group, path = rest.split(":", 1)
        with open(path, encoding="utf-8") as fh:
            reports[name] = (group, json.load(fh))
    out = {"policy": os.path.basename(args.policy), "sessions": sorted(reports), "candidates": {}}
    for cid, arm in ARMS.items():
        rows, excluded, failed_gate = [], {}, 0
        for name, (group, rep) in sorted(reports.items()):
            row, why = session_row(group, rep["variants"].get(arm))
            if row:
                rows.append(row)
            else:
                excluded[name] = why
                if why == "no_result":
                    failed_gate += 1
        verdict = policy_v2.candidate_verdict(policy, cid, rows)
        if failed_gate:
            verdict["holds"] = False
            verdict["reasons"].append("coverage_gate_failed")
        verdict["excluded_sessions"] = excluded
        out["candidates"][cid] = verdict
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for cid, v in out["candidates"].items():
        print(cid, "holds" if v["holds"] else "does NOT hold", v["reasons"], {k: round(x["upper"], 4) for k, x in v["intervals_95"].items() if k == "noise_ratio"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
