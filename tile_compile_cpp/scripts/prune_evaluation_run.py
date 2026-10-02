#!/usr/bin/env python3
"""Prune the intermediate data of a FINISHED evaluation run, keeping what comparisons/references need.

Retention rule (docs/PI/pi_jev_decisions_plan_de.md section 1.1, item 5): keep result artifacts, metrics,
configs and logs; remove regenerable intermediates (`cache/`, `outputs/calibrated/`). A pruned run can no longer
be resumed. Dry run by default; nothing is removed without --apply.

    prune_evaluation_run.py RUN_DIR [--apply]

Refuses a run that is not a complete run directory, that has not ended successfully (its cache may be the only
thing that lets it be resumed), or whose target is a symlink.
"""
import argparse
import json
import os
import shutil
import sys

PRUNABLE = ("cache", os.path.join("outputs", "calibrated"))
REQUIRED = ("config.yaml", "logs", "artifacts")


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def finished_successfully(run_dir):
    events = os.path.join(run_dir, "logs", "run_events.jsonl")
    if not os.path.isfile(events):
        return False
    last = None
    with open(events, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "run_end":
                last = e
    return bool(last and last.get("success") is True)


def plan(run_dir):
    """Returns (targets, refusal); targets are (path, bytes) that exist and may be removed."""
    if not os.path.isdir(run_dir) or os.path.islink(run_dir):
        return [], "not a plain run directory"
    missing = [r for r in REQUIRED if not os.path.exists(os.path.join(run_dir, r))]
    if missing:
        return [], "not a run directory (missing %s)" % ", ".join(missing)
    if not finished_successfully(run_dir):
        return [], "run has no successful run_end event; its cache may be needed to resume it"
    targets = []
    for rel in PRUNABLE:
        path = os.path.join(run_dir, rel)
        if os.path.islink(path):
            return [], "%s is a symlink" % rel
        if os.path.isdir(path):
            targets.append((path, dir_size(path)))
    return targets, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--apply", action="store_true", help="really remove (default: dry run)")
    args = ap.parse_args(argv)
    targets, refusal = plan(os.path.abspath(args.run_dir))
    if refusal:
        print("refused: " + refusal, file=sys.stderr)
        return 2
    total = sum(b for _p, b in targets)
    for path, size in targets:
        print("%s %.2f GB  %s" % ("remove" if args.apply else "would remove", size / 1e9, path))
        if args.apply:
            shutil.rmtree(path)
    print("%s %.2f GB" % ("freed" if args.apply else "would free", total / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
