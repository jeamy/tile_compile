from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
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
    allowed_roots: list[Path]
    input_search_roots: list[Path]

    @staticmethod
    def autodetect() -> "BackendRuntime":
        project_root = Path(__file__).resolve().parents[3]

        cli_env = os.getenv("TILE_COMPILE_CLI")
        runner_env = os.getenv("TILE_COMPILE_RUNNER")
        runs_env = os.getenv("TILE_COMPILE_RUNS_DIR")
        config_env = os.getenv("TILE_COMPILE_CONFIG_PATH")
        stats_env = os.getenv("TILE_COMPILE_STATS_SCRIPT")
        allowed_roots_env = os.getenv("TILE_COMPILE_ALLOWED_ROOTS")
        input_roots_env = os.getenv("TILE_COMPILE_INPUT_SEARCH_ROOTS")

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
        allowed_roots = _resolve_allowed_roots(
            allowed_roots_env,
            defaults=[
                project_root,
                runs_dir,
                Path.home(),
                Path("/tmp"),
                Path("/media"),
            ],
        )
        input_search_roots = _resolve_allowed_roots(input_roots_env, defaults=[])
        return BackendRuntime(
            project_root=project_root,
            cli_path=cli_path,
            runner_path=runner_path,
            runs_dir=runs_dir,
            default_config_path=default_config_path,
            stats_script=stats_script,
            allowed_roots=allowed_roots,
            input_search_roots=input_search_roots,
        )

    def resolve_run_dir(self, run_id_or_path: str, runs_dir_override: str | None = None) -> Path:
        candidate = Path(run_id_or_path).expanduser()
        if candidate.is_absolute():
            return self.ensure_path_allowed(candidate, label="run_dir")
        base = self.resolve_runs_dir(runs_dir_override)
        return self.ensure_path_allowed(base / run_id_or_path, label="run_dir")

    def resolve_runs_dir(self, runs_dir_override: str | None = None) -> Path:
        candidate = Path(runs_dir_override).expanduser() if runs_dir_override else self.runs_dir
        return self.ensure_path_allowed(candidate, label="runs_dir")

    def ensure_path_allowed(
        self,
        path: Path | str,
        *,
        must_exist: bool = False,
        label: str = "path",
    ) -> Path:
        candidate = Path(path).expanduser()
        resolved = candidate.resolve(strict=False)
        for root in self.allowed_roots:
            root_resolved = root.expanduser().resolve(strict=False)
            if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                if must_exist and not candidate.exists():
                    raise SecurityPolicyError(
                        code="PATH_NOT_FOUND",
                        message=f"{label} does not exist",
                        details={"path": str(candidate)},
                    )
                return candidate
        raise SecurityPolicyError(
            code="PATH_NOT_ALLOWED",
            message=f"{label} is outside allowed roots",
            details={
                "path": str(candidate),
                "allowed_roots": [str(x) for x in self.allowed_roots],
            },
        )

    def resolve_input_path(
        self,
        path: Path | str,
        *,
        must_exist: bool = False,
        label: str = "input_path",
    ) -> Path:
        raw = Path(path).expanduser()
        if raw.is_absolute():
            checked = self.ensure_path_allowed(raw, must_exist=False, label=label)
            if must_exist and not checked.exists():
                raise SecurityPolicyError(
                    code="PATH_NOT_FOUND",
                    message=f"{label} does not exist",
                    details={"path": str(raw)},
                )
            return checked

        tried: list[str] = []
        for base in self.input_search_roots:
            try:
                base_checked = self.ensure_path_allowed(base, must_exist=False, label=f"{label}_base")
            except SecurityPolicyError:
                continue
            probe = base_checked / raw
            try:
                probe_checked = self.ensure_path_allowed(probe, must_exist=False, label=label)
            except SecurityPolicyError:
                continue
            tried.append(str(probe_checked))
            if probe_checked.exists():
                return probe_checked

        if must_exist:
            raise SecurityPolicyError(
                code="PATH_NOT_FOUND",
                message=f"{label} does not exist or relative lookup is not configured",
                details={
                    "path": str(path),
                    "tried": tried,
                    "hint": "Use an absolute path, or configure TILE_COMPILE_INPUT_SEARCH_ROOTS for relative inputs",
                },
            )
        return raw


@dataclass
class SecurityPolicyError(RuntimeError):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


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


class CommandPolicy:
    """Allow only explicit trusted executables and safe invocation patterns."""

    def __init__(self, runtime: BackendRuntime) -> None:
        self.runtime = runtime
        self._allowed_exact = {
            runtime.cli_path.expanduser().resolve(strict=False),
            runtime.runner_path.expanduser().resolve(strict=False),
        }
        self._allowed_named = {"astap", "astap_cli", "dpkg-deb", "unzip"}
        self._python_execs = {
            Path(sys.executable).expanduser().resolve(strict=False),
        }
        for candidate in _preferred_python_candidates(runtime.project_root):
            if candidate.exists():
                self._python_execs.add(candidate.expanduser().resolve(strict=False))
        for name in ("python3", "python"):
            resolved = shutil.which(name)
            if resolved:
                self._python_execs.add(Path(resolved).expanduser().resolve(strict=False))

    def validate(self, command: list[str]) -> None:
        if not command:
            raise SecurityPolicyError(code="COMMAND_INVALID", message="empty command is not allowed")
        executable = self._resolve_executable(command[0])
        executable_name = executable.name

        if executable in self._allowed_exact:
            return
        if executable in self._python_execs:
            self._validate_python_command(command)
            return
        if executable_name in self._allowed_named:
            return

        raise SecurityPolicyError(
            code="COMMAND_NOT_ALLOWED",
            message="command executable is not whitelisted",
            details={"executable": str(executable)},
        )

    def _validate_python_command(self, command: list[str]) -> None:
        if len(command) < 2:
            raise SecurityPolicyError(
                code="COMMAND_NOT_ALLOWED",
                message="python invocation without script is not allowed",
            )
        script = Path(command[1]).expanduser()
        script_resolved = script.resolve(strict=False)
        stats_script_resolved = self.runtime.stats_script.expanduser().resolve(strict=False)
        if script_resolved != stats_script_resolved:
            raise SecurityPolicyError(
                code="COMMAND_NOT_ALLOWED",
                message="python invocation is restricted to stats script",
                details={"script": str(script), "allowed_script": str(self.runtime.stats_script)},
            )
        self.runtime.ensure_path_allowed(script, must_exist=True, label="stats_script")
        if len(command) >= 3:
            self.runtime.ensure_path_allowed(Path(command[2]), label="stats_run_dir")

    def _resolve_executable(self, executable: str) -> Path:
        candidate = Path(executable).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        found = shutil.which(executable)
        if found:
            return Path(found).expanduser().resolve(strict=False)
        return candidate.resolve(strict=False)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_sec: int = 120,
    stdin_text: str | None = None,
    command_policy: CommandPolicy | None = None,
) -> CommandResult:
    if command_policy is not None:
        command_policy.validate(command)
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
    command_policy: CommandPolicy | None = None,
) -> dict[str, Any] | list[Any]:
    result = run_command(
        command,
        cwd=cwd,
        timeout_sec=timeout_sec,
        stdin_text=stdin_text,
        command_policy=command_policy,
    )
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
    command_policy: CommandPolicy | None = None,
) -> None:
    if command_policy is not None:
        command_policy.validate(command)

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
                start_new_session=True,
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


def resolve_python(runtime: BackendRuntime | None = None) -> str:
    project_root = runtime.project_root if runtime is not None else Path(__file__).resolve().parents[3]
    for candidate in _preferred_python_candidates(project_root):
        if candidate.exists():
            return str(candidate)
    return shutil.which("python3") or shutil.which("python") or "python3"


def _preferred_python_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / ".venv/bin/python3",
        project_root / ".venv/bin/python",
        project_root / ".venv/Scripts/python.exe",
    ]


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


def _resolve_allowed_roots(raw: str | None, *, defaults: list[Path]) -> list[Path]:
    if not raw:
        return defaults
    items: list[Path] = []
    for part in raw.split(os.pathsep):
        token = part.strip()
        if not token:
            continue
        items.append(Path(token).expanduser())
    return items or defaults
