from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.process_manager import InMemoryJobStore


@dataclass
class BackendRuntime:
    project_root: Path
    cli_path: Path
    runner_path: Path
    runs_dir: Path
    default_config_path: Path
    stats_script: Path

    @staticmethod
    def autodetect() -> "BackendRuntime":
        project_root = Path(__file__).resolve().parents[3]

        cli_env = os.getenv("TILE_COMPILE_CLI")
        runner_env = os.getenv("TILE_COMPILE_RUNNER")
        runs_env = os.getenv("TILE_COMPILE_RUNS_DIR")
        config_env = os.getenv("TILE_COMPILE_CONFIG_PATH")
        stats_env = os.getenv("TILE_COMPILE_STATS_SCRIPT")

        cli_path = _resolve_binary(
            cli_env,
            [
                project_root / "tile_compile_cpp/build/tile_compile_cli",
                project_root / "tile_compile_cpp/build/apps/tile_compile_cli",
            ],
        )
        runner_path = _resolve_binary(
            runner_env,
            [
                project_root / "tile_compile_cpp/build/tile_compile_runner",
                project_root / "tile_compile_cpp/build/apps/tile_compile_runner",
            ],
        )
        runs_dir = Path(runs_env).expanduser() if runs_env else project_root / "tile_compile_cpp/build/runs"
        default_config_path = (
            Path(config_env).expanduser() if config_env else project_root / "tile_compile_cpp/tile_compile.yaml"
        )
        stats_script = (
            Path(stats_env).expanduser()
            if stats_env
            else project_root / "tile_compile_cpp/scripts/generate_report.py"
        )
        return BackendRuntime(
            project_root=project_root,
            cli_path=cli_path,
            runner_path=runner_path,
            runs_dir=runs_dir,
            default_config_path=default_config_path,
            stats_script=stats_script,
        )

    def resolve_run_dir(self, run_id_or_path: str, runs_dir_override: str | None = None) -> Path:
        candidate = Path(run_id_or_path).expanduser()
        if candidate.is_absolute():
            return candidate
        base = Path(runs_dir_override).expanduser() if runs_dir_override else self.runs_dir
        return base / run_id_or_path

    def resolve_runs_dir(self, runs_dir_override: str | None = None) -> Path:
        return Path(runs_dir_override).expanduser() if runs_dir_override else self.runs_dir


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    parsed_json: dict[str, Any] | list[Any] | None


class CommandExecutionError(RuntimeError):
    def __init__(self, message: str, *, command: list[str], exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_sec: int = 120,
    stdin_text: str | None = None,
) -> CommandResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            input=stdin_text,
            timeout=timeout_sec,
            check=False,
        )
        parsed = _try_parse_json(proc.stdout)
        return CommandResult(
            command=command,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            parsed_json=parsed,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=command,
            exit_code=127,
            stdout="",
            stderr=str(exc),
            parsed_json=None,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"timeout after {timeout_sec}s",
            parsed_json=None,
        )


def run_json_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_sec: int = 120,
    stdin_text: str | None = None,
) -> dict[str, Any] | list[Any]:
    result = run_command(command, cwd=cwd, timeout_sec=timeout_sec, stdin_text=stdin_text)
    if result.exit_code != 0:
        raise CommandExecutionError(
            "command failed",
            command=result.command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if result.parsed_json is None:
        raise CommandExecutionError(
            "command returned non-JSON output",
            command=result.command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result.parsed_json


def launch_background_command(
    *,
    job_store: InMemoryJobStore,
    job_id: str,
    command: list[str],
    cwd: Path | None = None,
    stdin_text: str | None = None,
) -> None:
    def _worker() -> None:
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            job_store.set_process(job_id, proc)
            stdout, stderr = proc.communicate(stdin_text)
            exit_code = proc.returncode
            job_store.set_exit_code(job_id, exit_code)
            data_patch = {
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
            }
            parsed = _try_parse_json(stdout)
            if parsed is not None:
                data_patch["result"] = parsed
            job_store.merge_data(job_id, data_patch)
            if job_store.get(job_id) and job_store.get(job_id).state == "cancelled":
                return
            job_store.set_state(job_id, "ok" if exit_code == 0 else "error")
        except Exception as exc:
            job_store.merge_data(job_id, {"command": command, "error": str(exc)})
            if job_store.get(job_id) and job_store.get(job_id).state != "cancelled":
                job_store.set_state(job_id, "error")
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
            job_store.clear_process(job_id)

    thread = threading.Thread(target=_worker, name=f"job-{job_id}", daemon=True)
    thread.start()


def resolve_python() -> str:
    return shutil.which("python3") or shutil.which("python") or "python3"


def _resolve_binary(env_path: str | None, fallback_candidates: list[Path]) -> Path:
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    if fallback_candidates:
        return fallback_candidates[0]
    raise RuntimeError("no binary candidates provided")


def _try_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, (dict, list)):
            return parsed
        return None
    except json.JSONDecodeError:
        return None
