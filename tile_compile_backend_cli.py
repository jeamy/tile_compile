import argparse
import json
import sys

from tile_compile_backend.config_io import load_config_text, save_config_text
from tile_compile_backend.schema import load_schema_json


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
    save_config_text(args.path, args.yaml)
    _print_json({"path": args.path, "saved": True})
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
    p_save.add_argument("yaml")
    p_save.set_defaults(func=cmd_save_config)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
