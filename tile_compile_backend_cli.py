import argparse
import json
import sys

from tile_compile_backend.config_io import load_config_text, save_config_text
from tile_compile_backend.schema import load_schema_json
from tile_compile_backend.validate import validate_config_yaml_text
from tile_compile_backend.scan import scan_input
from tile_compile_backend.runs import list_runs
from tile_compile_backend.logs import get_run_logs
from tile_compile_backend.artifacts import list_artifacts
from tile_compile_backend.status import get_run_status


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_get_schema(_: argparse.Namespace) -> int:
    schema = load_schema_json()
    _print_json(schema)
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
    return 0 if result.get("valid") else 1


def cmd_scan(args: argparse.Namespace) -> int:
    result = scan_input(
        input_path=args.input_path,
        frames_min=args.frames_min,
        project_cache_dir=args.project_cache_dir,
        with_checksums=args.with_checksums,
    )
    _print_json(result)
    return 0 if result.get("ok") else 1


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
    res = get_run_status(args.run_dir)
    _print_json(res)
    return 0


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
    p_validate.set_defaults(func=cmd_validate_config)

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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
