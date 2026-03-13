#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_mockups.py"


def load_generate_mockups_module():
    spec = importlib.util.spec_from_file_location("generate_mockups", GEN)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load generate_mockups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_generate_mockups_module()
    spacing = mod.SPACING_1920

    expected = {
        ("global", "wrapper_inner_padding"): 24,
        ("global", "field_label_to_input_gap"): 12,
        ("dashboard", "readiness_first_row_y"): 470,
        ("dashboard", "readiness_row_h"): 46,
        ("dashboard", "readiness_row_step"): 56,
        ("parameter_studio", "registration_row1_y"): 424,
        ("parameter_studio", "registration_row2_y"): 516,
        ("parameter_studio", "registration_row2_hgap"): 20,
        ("parameter_studio", "section_title_gap"): 34,
        ("run_monitor", "artifact_list_first_y"): 480,
        ("run_monitor", "artifact_button_row_y"): 804,
        ("run_monitor", "artifact_secondary_button_y"): 868,
        ("history_tools", "astrometry_first_input_y"): 356,
        ("history_tools", "astrometry_row_step"): 86,
        ("history_tools", "astrometry_plate_solve_y"): 528,
    }

    failed = False
    print("Spacing token check:")
    for path, exp in expected.items():
        section, key = path
        got = spacing.get(section, {}).get(key)
        ok = got == exp
        print(f"- {section}.{key}: expected={exp} actual={got} {'OK' if ok else 'FAIL'}")
        if not ok:
            failed = True

    # Derived safety checks.
    readiness_bottom = (
        spacing["dashboard"]["readiness_first_row_y"]
        + 4 * spacing["dashboard"]["readiness_row_step"]
        + spacing["dashboard"]["readiness_row_h"]
    )
    # Guardrails card is y=402..784 in the mockup.
    readiness_ok = readiness_bottom <= 784
    print(f"- dashboard.readiness_bottom<=784: actual={readiness_bottom} {'OK' if readiness_ok else 'FAIL'}")
    if not readiness_ok:
        failed = True

    # Run monitor artifact cards: 5 rows, h=52, step=62.
    artifact_last_bottom = spacing["run_monitor"]["artifact_list_first_y"] + 4 * 62 + 52
    button_top = spacing["run_monitor"]["artifact_button_row_y"]
    artifact_gap_ok = button_top - artifact_last_bottom >= 20
    print(
        f"- run_monitor.artifact_button_gap>=20: actual={button_top - artifact_last_bottom} "
        f"{'OK' if artifact_gap_ok else 'FAIL'}"
    )
    if not artifact_gap_ok:
        failed = True

    if failed:
        print("Result: FAIL")
        return 1
    print("Result: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
