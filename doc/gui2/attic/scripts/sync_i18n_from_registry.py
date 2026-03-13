#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "gui2_control_registry.yaml"
DEFAULT_I18N_DIR = ROOT / "i18n"
DEFAULT_EXTRA_KEYS = DEFAULT_I18N_DIR / "additional_keys.yaml"
LOCALES = ("de", "en")


def _strip_yaml_scalar(raw: str) -> str:
    value = raw.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def extract_registry_keys(registry_path: Path) -> set[str]:
    text = registry_path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^\s*(label_key|tooltip_key):\s*(.+?)\s*$", line)
        if not m:
            continue
        key = _strip_yaml_scalar(m.group(2))
        if key:
            keys.add(key)
    return keys


def extract_additional_keys(extra_keys_path: Path) -> set[str]:
    if not extra_keys_path.exists():
        return set()
    keys: set[str] = set()
    for raw_line in extra_keys_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^-\s*([a-z0-9_]+(?:\.[a-z0-9_]+)+)\s*$", line)
        if m:
            keys.add(m.group(1))
            continue
        m = re.match(r"^[a-z_]+\s*:\s*([a-z0-9_]+(?:\.[a-z0-9_]+)+)\s*$", line)
        if m:
            keys.add(m.group(1))
    return keys


def load_locale_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    out: dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = "" if v is None else str(v)
    return out


def write_locale_map(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync/check i18n locale files against control-registry keys plus optional additional keys."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--i18n-dir", type=Path, default=DEFAULT_I18N_DIR)
    parser.add_argument("--extra-keys", type=Path, default=DEFAULT_EXTRA_KEYS)
    parser.add_argument("--check", action="store_true", help="Only check for missing keys; do not modify files.")
    args = parser.parse_args()

    keys = extract_registry_keys(args.registry) | extract_additional_keys(args.extra_keys)
    if not keys:
        print(f"No i18n keys found in {args.registry} and {args.extra_keys}")
        return 1

    failed = False
    for locale in LOCALES:
        path = args.i18n_dir / f"{locale}.json"
        current = load_locale_map(path)
        missing = sorted(keys - set(current))
        if args.check:
            print(f"{locale}: missing={len(missing)} file={path}")
            for key in missing[:20]:
                print(f"  - {key}")
            if missing:
                failed = True
            continue

        for key in missing:
            current[key] = f"TODO::{key}"
        write_locale_map(path, current)
        print(f"{locale}: total={len(current)} added={len(missing)} file={path}")

    if args.check and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
