from __future__ import annotations

from pathlib import Path

import pytest

from app.services.command_runner import BackendRuntime, CommandPolicy, SecurityPolicyError


def _runtime(tmp_path: Path) -> BackendRuntime:
    cli = tmp_path / "tile_compile_cli"
    runner = tmp_path / "tile_compile_runner"
    stats = tmp_path / "generate_report.py"
    cfg = tmp_path / "tile_compile.yaml"
    runs = tmp_path / "runs"

    for p in [cli, runner, stats, cfg]:
        p.write_text("#!/bin/sh\n", encoding="utf-8")
    runs.mkdir(parents=True, exist_ok=True)

    return BackendRuntime(
        project_root=tmp_path,
        cli_path=cli,
        runner_path=runner,
        runs_dir=runs,
        default_config_path=cfg,
        stats_script=stats,
        allowed_roots=[tmp_path, Path("/tmp")],
        input_search_roots=[tmp_path],
    )


def test_path_policy_blocks_outside_allowed_roots(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(SecurityPolicyError) as exc:
        runtime.ensure_path_allowed(Path("/etc/passwd"), label="test_path")
    assert exc.value.code == "PATH_NOT_ALLOWED"


def test_command_policy_allows_cli_and_runner(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    policy = CommandPolicy(runtime)
    policy.validate([str(runtime.cli_path), "get-schema"])
    policy.validate([str(runtime.runner_path), "resume", "--run-dir", str(runtime.runs_dir), "--from-phase", "PCC"])


def test_command_policy_rejects_unknown_executable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    policy = CommandPolicy(runtime)
    with pytest.raises(SecurityPolicyError) as exc:
        policy.validate(["/bin/sh", "-c", "echo test"])
    assert exc.value.code == "COMMAND_NOT_ALLOWED"


def test_command_policy_restricts_python_to_stats_script(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    policy = CommandPolicy(runtime)

    # allowed
    policy.validate(["python3", str(runtime.stats_script), str(runtime.runs_dir)])

    # blocked: different script
    other_script = tmp_path / "other.py"
    other_script.write_text("print('x')\n", encoding="utf-8")
    with pytest.raises(SecurityPolicyError) as exc:
        policy.validate(["python3", str(other_script), str(runtime.runs_dir)])
    assert exc.value.code == "COMMAND_NOT_ALLOWED"


def test_resolve_input_path_uses_search_roots(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    runtime.input_search_roots = [cache_root]
    target = cache_root / "IC434_ligths_all"
    target.mkdir(parents=True, exist_ok=True)

    resolved = runtime.resolve_input_path("IC434_ligths_all", must_exist=True, label="input_path")
    assert resolved == target
