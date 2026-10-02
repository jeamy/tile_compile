#!/usr/bin/env python3
"""Generate an atomic adaptive-weighting A/B config pair from a trusted base."""

import argparse
import hashlib
import json
from pathlib import Path
import yaml


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--dark-master", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.base_config.is_file() or not args.dark_master.is_file():
        parser.error("base config and dark master must exist")
    with args.base_config.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    if not isinstance(base, dict) or not isinstance(base.get("global_metrics"), dict):
        parser.error("base config lacks global_metrics")
    if not isinstance(base.get("calibration"), dict) or not base["calibration"].get("use_dark"):
        parser.error("base config must enable dark calibration")
    base["calibration"]["dark_master"] = str(args.dark_master.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {}
    for enabled, arm in ((False, "off"), (True, "on")):
        variant = yaml.safe_load(yaml.safe_dump(base, sort_keys=True))
        variant["global_metrics"]["adaptive_weights"] = enabled
        path = args.output_dir / f"config_adaptive_{arm}.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=True), encoding="utf-8")
        paths[arm] = path
    off = yaml.safe_load(paths["off"].read_text(encoding="utf-8"))
    on = yaml.safe_load(paths["on"].read_text(encoding="utf-8"))
    on["global_metrics"]["adaptive_weights"] = False
    if on != off:
        raise ValueError("generated arms differ outside adaptive_weights")
    manifest = {
        "schema_version": "pi.jev-test-pair.v1",
        "base_config_sha256": sha256(args.base_config),
        "dark_master_sha256": sha256(args.dark_master),
        "arms": {arm: {"path": str(path), "sha256": sha256(path)} for arm, path in paths.items()},
        "only_difference": "global_metrics.adaptive_weights",
    }
    (args.output_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
