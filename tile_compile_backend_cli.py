"""
Tile-Compile Backend CLI

Command-line interface for backend operations:
- Config validation and schema checking
- Input scanning and frame discovery
- Run listing and status queries
- Log retrieval and artifact listing
- Config text manipulation

This CLI provides utility commands used by the GUI and for
manual pipeline management. All commands output JSON for
easy parsing by the GUI.

Usage:
    python tile_compile_backend_cli.py <command> [args]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
import hashlib
import re

from tile_compile_backend.config_io import load_config_text, save_config_text
from tile_compile_backend.schema import load_schema_json
from tile_compile_backend.validate import validate_config_yaml_text
from tile_compile_backend.scan import scan_input
from tile_compile_backend.runs import list_runs
from tile_compile_backend.logs import get_run_logs
from tile_compile_backend.artifacts import list_artifacts
from tile_compile_backend.status import get_run_status
import subprocess


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_get_schema(_: argparse.Namespace) -> int:
    schema = load_schema_json()
    _print_json(schema)
    return 0


def _default_gui_state_path() -> Path:
    return (Path(__file__).resolve().parent / "tile_compile_gui_state.json").resolve()


def cmd_load_gui_state(args: argparse.Namespace) -> int:
    p = Path(args.path).expanduser() if args.path is not None else _default_gui_state_path()
    if not p.is_absolute():
        p = p.resolve()

    state: dict[str, Any] = {}
    if p.exists() and p.is_file():
        try:
            raw = p.read_text(encoding="utf-8")
            obj = json.loads(raw)
            if isinstance(obj, dict):
                state = obj
        except Exception:
            state = {}

    _print_json({"ok": True, "path": str(p), "state": state})
    return 0


def cmd_save_gui_state(args: argparse.Namespace) -> int:
    p = Path(args.path).expanduser() if args.path is not None else _default_gui_state_path()
    if not p.is_absolute():
        p = p.resolve()

    if args.stdin:
        raw = sys.stdin.read()
    else:
        raw = args.json

    if raw is None:
        sys.stderr.write("save-gui-state requires JSON text either as argument or via --stdin\n")
        return 2

    try:
        obj = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"save-gui-state: failed to parse JSON: {e}\n")
        return 2

    if not isinstance(obj, dict):
        sys.stderr.write("save-gui-state: state must be a JSON object\n")
        return 2

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_json({"ok": True, "path": str(p), "saved": True})
    return 0


def cmd_load_config(args: argparse.Namespace) -> int:
    yaml_text = load_config_text(args.path)
    _print_json({"path": args.path, "yaml": yaml_text})
    return 0


def cmd_save_config(args: argparse.Namespace) -> int:
    if args.stdin:
        yaml_text = sys.stdin.read()
    else:
        yaml_text = args.yaml

    if yaml_text is None:
        sys.stderr.write("save-config requires YAML text either as argument or via --stdin\n")
        return 2

    save_config_text(args.path, yaml_text)
    _print_json({"path": args.path, "saved": True})
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    if args.path is not None:
        yaml_text = load_config_text(args.path)
    else:
        if args.stdin:
            yaml_text = sys.stdin.read()
        else:
            yaml_text = args.yaml

    result = validate_config_yaml_text(yaml_text=yaml_text, schema_path=args.schema)
    if args.path is not None:
        result["path"] = args.path
    _print_json(result)
    if args.strict_exit_codes:
        return 0 if result.get("valid") else 1
    return 0


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _validate_siril_script(path: Path) -> tuple[bool, list[str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return False, [f"failed to read script: {e}"]

    lines: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    text = "\n".join(lines).lower()

    violations: list[str] = []

    if "-weight" in text:
        violations.append("-weight")

    if "-drizzle" in text and "-drizzle=0" not in text and "-drizzle 0" not in text:
        violations.append("-drizzle")

    if "-norm" in text and "-norm=none" not in text and "-norm none" not in text and "-nonorm" not in text:
        violations.append("-norm")

    if "-rej" in text and "-rej=none" not in text and "-rej none" not in text:
        violations.append("-rej")

    for tok in [
        "autostretch",
        "asinh",
        "linstretch",
        "logstretch",
        "modasinh",
        "mtf",
        "histo",
        "hist",
        "wavelet",
        "stretch",
    ]:
        if tok in text:
            violations.append(tok)

    for cmd in ["select", "unselect"]:
        if re.search(rf"(^|\n){re.escape(cmd)}\b", text):
            violations.append(cmd)

    return len(violations) == 0, sorted(set(violations))


def _extract_siril_save_targets(script_path: Path) -> list[str]:
    try:
        raw = script_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    targets: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if not low.startswith("save "):
            continue
        rest = s.split(None, 1)[1].strip() if len(s.split(None, 1)) == 2 else ""
        if not rest:
            continue
        target = Path(rest.strip().strip('"').strip("'")).name
        if target:
            targets.append(target)
    return targets


def _get_path(obj: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def cmd_validate_siril_scripts(args: argparse.Namespace) -> int:
    if args.path is not None:
        yaml_text = load_config_text(args.path)
    else:
        if args.stdin:
            yaml_text = sys.stdin.read()
        else:
            yaml_text = args.yaml

    errors: list[str] = []
    warnings: list[str] = []

    try:
        cfg = yaml.safe_load(yaml_text)
    except Exception as e:  # noqa: BLE001
        _print_json({"ok": False, "errors": [f"failed to parse config as YAML: {e}"], "warnings": []})
        return 1 if args.strict_exit_codes else 0

    if not isinstance(cfg, dict):
        _print_json({"ok": False, "errors": ["configuration root must be a mapping/object"], "warnings": []})
        return 1 if args.strict_exit_codes else 0

    project_root = Path(__file__).resolve().parent

    registration_cfg = _get_path(cfg, ["registration"]) if isinstance(cfg.get("registration"), dict) else {}
    stacking_cfg = _get_path(cfg, ["stacking"]) if isinstance(cfg.get("stacking"), dict) else {}
    if not isinstance(registration_cfg, dict):
        registration_cfg = {}
    if not isinstance(stacking_cfg, dict):
        stacking_cfg = {}

    reg_script_cfg = registration_cfg.get("siril_script")
    reg_is_default = not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip())
    reg_script_path = (
        Path(str(reg_script_cfg)).expanduser().resolve()
        if isinstance(reg_script_cfg, str) and reg_script_cfg.strip()
        else (project_root / "siril_register_osc.ssf").resolve()
    )

    stack_script_cfg = stacking_cfg.get("siril_script")
    stack_is_default = not (isinstance(stack_script_cfg, str) and stack_script_cfg.strip())
    stack_script_path = (
        Path(str(stack_script_cfg)).expanduser().resolve()
        if isinstance(stack_script_cfg, str) and stack_script_cfg.strip()
        else (project_root / "siril_stack_average.ssf").resolve()
    )

    color_mode = _get_path(cfg, ["data", "color_mode"])
    bayer_pattern = _get_path(cfg, ["data", "bayer_pattern"])
    stack_method = _get_path(cfg, ["stacking", "method"])

    if reg_is_default:
        if str(color_mode).upper() != "OSC":
            errors.append(
                "default registration script requires data.color_mode=OSC; set registration.siril_script for non-OSC inputs"
            )
        if bayer_pattern is None:
            warnings.append("data.bayer_pattern is not set; default OSC script assumes CFA inputs")

    if stack_is_default and str(stack_method).lower() != "average":
        errors.append(
            "default stacking script is only defined for stacking.method=average; set stacking.siril_script for other methods"
        )

    reg_exists = reg_script_path.exists() and reg_script_path.is_file()
    stack_exists = stack_script_path.exists() and stack_script_path.is_file()

    reg_policy_ok = False
    reg_policy_violations: list[str] = []
    if reg_exists:
        reg_policy_ok, reg_policy_violations = _validate_siril_script(reg_script_path)
        if not reg_policy_ok:
            errors.append(
                "registration script violates policy: " + ", ".join(reg_policy_violations)
            )
    else:
        errors.append(f"registration script not found: {reg_script_path}")

    stack_policy_ok = False
    stack_policy_violations: list[str] = []
    save_targets: list[str] = []
    if stack_exists:
        stack_policy_ok, stack_policy_violations = _validate_siril_script(stack_script_path)
        if not stack_policy_ok:
            errors.append("stacking script violates policy: " + ", ".join(stack_policy_violations))
        save_targets = _extract_siril_save_targets(stack_script_path)
        if not save_targets:
            warnings.append("stacking script contains no 'save' commands (output target cannot be inferred)")
    else:
        errors.append(f"stacking script not found: {stack_script_path}")

    result = {
        "ok": len(errors) == 0,
        "project_root": str(project_root),
        "path": args.path,
        "registration": {
            "script_path": str(reg_script_path),
            "script_is_default": reg_is_default,
            "exists": reg_exists,
            "sha256": _sha256_file(reg_script_path) if reg_exists else None,
            "policy_ok": reg_policy_ok,
            "policy_violations": reg_policy_violations,
        },
        "stacking": {
            "script_path": str(stack_script_path),
            "script_is_default": stack_is_default,
            "exists": stack_exists,
            "sha256": _sha256_file(stack_script_path) if stack_exists else None,
            "policy_ok": stack_policy_ok,
            "policy_violations": stack_policy_violations,
            "save_targets": save_targets,
        },
        "errors": errors,
        "warnings": warnings,
    }

    _print_json(result)
    if args.strict_exit_codes:
        return 0 if result.get("ok") else 1
    return 0


def cmd_validate_ssf(args: argparse.Namespace) -> int:
    p = Path(args.path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    exists = p.exists() and p.is_file()
    if not exists:
        errors.append(f"script not found: {p}")

    policy_ok = False
    policy_violations: list[str] = []
    save_targets: list[str] = []
    sha256 = _sha256_file(p) if exists else None

    if exists:
        policy_ok, policy_violations = _validate_siril_script(p)
        if not policy_ok:
            errors.append("script violates policy: " + ", ".join(policy_violations))
        save_targets = _extract_siril_save_targets(p)
        if args.expect_save and not save_targets:
            errors.append("expected at least one 'save' command but none found")

    result = {
        "ok": len(errors) == 0,
        "script_path": str(p),
        "exists": exists,
        "sha256": sha256,
        "policy_ok": policy_ok,
        "policy_violations": policy_violations,
        "save_targets": save_targets,
        "errors": errors,
        "warnings": warnings,
    }
    _print_json(result)
    if args.strict_exit_codes:
        return 0 if result.get("ok") else 1
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    result = scan_input(
        input_path=args.input_path,
        frames_min=args.frames_min,
        project_cache_dir=args.project_cache_dir,
        with_checksums=args.with_checksums,
    )
    _print_json(result)
    if args.strict_exit_codes:
        return 0 if result.get("ok") else 1
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    res = list_runs(args.runs_dir)
    _print_json(res)
    return 0


def cmd_get_run_logs(args: argparse.Namespace) -> int:
    res = get_run_logs(args.run_dir, tail=args.tail)
    _print_json(res)
    return 0


def cmd_list_artifacts(args: argparse.Namespace) -> int:
    res = list_artifacts(args.run_dir)
    _print_json(res)
    return 0


def cmd_get_run_status(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    result = get_run_status(run_dir)
    _print_json(result)
    return 0


def cmd_resume_run(args: argparse.Namespace) -> int:
    """Resume a run from a specific phase by calling tile_compile_runner.py resume."""
    run_dir = Path(args.run_dir).resolve()
    
    if not run_dir.exists() or not run_dir.is_dir():
        _print_json({"ok": False, "error": f"Run directory not found: {run_dir}"})
        return 1
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        # Assume project root is two levels up from run directory
        project_root = run_dir.parent.parent
    
    # Build command to run tile_compile_runner.py resume
    runner_script = project_root / "tile_compile_runner.py"
    if not runner_script.exists():
        _print_json({"ok": False, "error": f"Runner script not found: {runner_script}"})
        return 1
    
    cmd = [
        sys.executable,
        str(runner_script),
        "resume",
        "--run-dir", str(run_dir),
        "--from-phase", str(args.from_phase),
    ]
    
    if args.project_root:
        cmd.extend(["--project-root", str(project_root)])
    
    # Execute the resume command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        _print_json({
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "run_dir": str(run_dir),
            "from_phase": args.from_phase,
        })
        return result.returncode
    except Exception as e:
        _print_json({"ok": False, "error": str(e)})
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tile_compile_backend_cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_schema = sub.add_parser("get-schema")
    p_schema.set_defaults(func=cmd_get_schema)

    p_load = sub.add_parser("load-config")
    p_load.add_argument("path")
    p_load.set_defaults(func=cmd_load_config)

    p_save = sub.add_parser("save-config")
    p_save.add_argument("path")
    p_save.add_argument("yaml", nargs="?")
    p_save.add_argument("--stdin", action="store_true")
    p_save.set_defaults(func=cmd_save_config)

    p_gui_load = sub.add_parser("load-gui-state")
    p_gui_load.add_argument("--path", default=None)
    p_gui_load.set_defaults(func=cmd_load_gui_state)

    p_gui_save = sub.add_parser("save-gui-state")
    p_gui_save.add_argument("--path", default=None)
    p_gui_save.add_argument("json", nargs="?")
    p_gui_save.add_argument("--stdin", action="store_true")
    p_gui_save.set_defaults(func=cmd_save_gui_state)

    p_validate = sub.add_parser("validate-config")
    src = p_validate.add_mutually_exclusive_group(required=True)
    src.add_argument("--path")
    src.add_argument("--yaml")
    p_validate.add_argument("--stdin", action="store_true")
    p_validate.add_argument(
        "--schema",
        default=None,
        help="Optional path to tile_compile.schema.json (defaults to repo root tile_compile.schema.json)",
    )
    p_validate.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help="Return exit code 1 when validation fails (CLI mode). Default: always 0 and rely on JSON result.",
    )
    p_validate.set_defaults(func=cmd_validate_config)

    p_siril = sub.add_parser("validate-siril-scripts")
    src_siril = p_siril.add_mutually_exclusive_group(required=True)
    src_siril.add_argument("--path")
    src_siril.add_argument("--yaml")
    p_siril.add_argument("--stdin", action="store_true")
    p_siril.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help="Return exit code 1 when ok=false (CLI mode). Default: always 0 and rely on JSON result.",
    )
    p_siril.set_defaults(func=cmd_validate_siril_scripts)

    p_ssf = sub.add_parser("validate-ssf")
    p_ssf.add_argument("path")
    p_ssf.add_argument(
        "--expect-save",
        action="store_true",
        help="Fail validation if the script contains no 'save' command (useful for stacking scripts).",
    )
    p_ssf.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help="Return exit code 1 when ok=false (CLI mode). Default: always 0 and rely on JSON result.",
    )
    p_ssf.set_defaults(func=cmd_validate_ssf)

    p_scan = sub.add_parser("scan")
    p_scan.add_argument("input_path")
    p_scan.add_argument(
        "--frames-min",
        type=int,
        default=1,
        help="Hard error if frames_detected < frames_min",
    )
    p_scan.add_argument(
        "--project-cache-dir",
        default=None,
        help="Project cache root directory (defaults to <input_path>/../.tile_compile)",
    )
    p_scan.add_argument(
        "--with-checksums",
        action="store_true",
        help="Compute per-frame sha256 checksums (can be slow)",
    )
    p_scan.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help="Return exit code 1 when scan ok=false (CLI mode). Default: always 0 and rely on JSON result.",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_list_runs = sub.add_parser("list-runs")
    p_list_runs.add_argument("runs_dir")
    p_list_runs.set_defaults(func=cmd_list_runs)

    p_logs = sub.add_parser("get-run-logs")
    p_logs.add_argument("run_dir")
    p_logs.add_argument("--tail", type=int, default=None)
    p_logs.set_defaults(func=cmd_get_run_logs)

    p_art = sub.add_parser("list-artifacts")
    p_art.add_argument("run_dir")
    p_art.set_defaults(func=cmd_list_artifacts)

    p_status = sub.add_parser("get-run-status")
    p_status.add_argument("run_dir")
    p_status.set_defaults(func=cmd_get_run_status)

    p_resume = sub.add_parser("resume-run")
    p_resume.add_argument("run_dir", help="Path to existing run directory")
    p_resume.add_argument("--from-phase", type=int, required=True, help="Phase number to resume from (0-11)")
    p_resume.add_argument("--project-root", default=None, help="Project root directory (optional)")
    p_resume.set_defaults(func=cmd_resume_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
