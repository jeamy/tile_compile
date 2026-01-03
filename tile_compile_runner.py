import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astropy.io import fits
import yaml


_STOP = False


def _handle_signal(_signum, _frame):
    global _STOP
    _STOP = True


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _json_dumps_canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _emit(event: dict, log_fp=None):
    line = _json_dumps_canonical(event).decode("utf-8")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def _copy_config(config_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_path)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _discover_frames(input_dir: Path, pattern: str):
    paths = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return paths


def _resolve_project_root(start: Path) -> Path:
    p = start
    if p.is_file():
        p = p.parent
    p = p.resolve()
    while True:
        if (p / "tile_compile_runner.py").exists() or (p / "tile_compile_backend_cli.py").exists():
            return p
        if p.parent == p:
            return start.resolve()
        p = p.parent


def _load_gui_state(project_root: Path) -> dict:
    path = project_root / "tile_compile_gui_state.json"
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_siril_exe(project_root: Path) -> Tuple[Optional[str], str]:
    exe = shutil.which("siril")
    if exe:
        return exe, "path"

    st = _load_gui_state(project_root)
    candidate = st.get("sirilExe") or st.get("siril_exe")
    if isinstance(candidate, str) and candidate.strip():
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate, "gui_state"
        exe2 = shutil.which(candidate)
        if exe2:
            return exe2, "gui_state"

    return None, "none"


def _phase_start(run_id: str, log_fp, phase_id: int, phase_name: str, extra: Optional[Dict[str, Any]] = None):
    ev: Dict[str, Any] = {
        "type": "phase_start",
        "run_id": run_id,
        "phase": phase_id,
        "phase_name": phase_name,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        ev.update(extra)
    _emit(ev, log_fp)


def _phase_end(
    run_id: str,
    log_fp,
    phase_id: int,
    phase_name: str,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
):
    ev: Dict[str, Any] = {
        "type": "phase_end",
        "run_id": run_id,
        "phase": phase_id,
        "phase_name": phase_name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if extra:
        ev.update(extra)
    _emit(ev, log_fp)


def _stop_requested(run_id: str, log_fp, phase_id: int, phase_name: str) -> bool:
    if not _STOP:
        return False
    _emit(
        {
            "type": "run_stop_requested",
            "run_id": run_id,
            "phase": phase_id,
            "phase_name": phase_name,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
        log_fp,
    )
    return True


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        shutil.copy2(src, dst)


def _siril_setext_from_suffix(suffix: str) -> str:
    s = (suffix or "").lower().lstrip(".")
    if s in {"fit", "fits", "fts"}:
        return "fit"
    return "fit"


def _is_fits_image_path(p: Path) -> bool:
    suf = p.suffix.lower()
    return suf in {".fit", ".fits", ".fts"}


def _fits_is_cfa(path: Path) -> Optional[bool]:
    try:
        hdr = fits.getheader(str(path), ext=0)
    except Exception:
        return None
    naxis3 = hdr.get("NAXIS3")
    try:
        naxis3_i = int(naxis3) if naxis3 is not None else None
    except Exception:
        naxis3_i = None
    if naxis3_i is not None and naxis3_i >= 3:
        return False
    return hdr.get("BAYERPAT") is not None


def _fits_get_bayerpat(path: Path) -> Optional[str]:
    try:
        hdr = fits.getheader(str(path), ext=0)
    except Exception:
        return None
    v = hdr.get("BAYERPAT")
    if not isinstance(v, str):
        return None
    vv = v.strip().upper()
    if vv in {"RGGB", "BGGR", "GBRG", "GRBG"}:
        return vv
    return None


def _derive_prefix_from_pattern(pattern: str, default_prefix: str) -> str:
    s = str(pattern or "").strip()
    if not s:
        return default_prefix
    i = s.find("{")
    if i > 0:
        return s[:i]
    j = s.find("*")
    if j > 0:
        return s[:j]
    stem = Path(s).stem
    if stem:
        return stem + "_"
    return default_prefix


def _validate_siril_script(path: Path) -> Tuple[bool, List[str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, [f"failed to read script: {e}"]

    lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    text = "\n".join(lines).lower()

    violations: List[str] = []

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


def _run_siril_script(
    siril_exe: str,
    work_dir: Path,
    script_path: Path,
    artifacts_dir: Path,
    log_name: str,
    timeout_s: Optional[int] = None,
    quiet: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / log_name

    script_sha256 = None
    try:
        script_sha256 = _sha256_bytes(script_path.read_bytes())
    except Exception:
        script_sha256 = None

    base_cmd = [siril_exe, "-d", str(work_dir), "-s", str(script_path)]
    cmd = base_cmd + (["-q"] if quiet else [])

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if quiet and proc.returncode != 0:
            combined = (stdout + "\n" + stderr).lower()
            if "unknown option -q" in combined or "unbekannte option -q" in combined:
                cmd2 = base_cmd
                proc2 = subprocess.run(
                    cmd2,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                stdout2 = proc2.stdout or ""
                stderr2 = proc2.stderr or ""
                log_path.write_text(
                    "# attempt_with_q\n" + stdout + "\n" + stderr + "\n" + "# attempt_without_q\n" + stdout2 + "\n" + stderr2,
                    encoding="utf-8",
                )
                return proc2.returncode == 0, {
                    "cmd": cmd2,
                    "returncode": proc2.returncode,
                    "seconds": max(0.0, time.time() - started),
                    "log_path": str(log_path),
                    "script_path": str(script_path),
                    "script_sha256": script_sha256,
                    "retry": {"cmd": cmd, "returncode": proc.returncode},
                }

        log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")
        return proc.returncode == 0, {
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "script_sha256": script_sha256,
        }
    except subprocess.TimeoutExpired as e:
        log_path.write_text((e.stdout or "") + "\n" + (e.stderr or ""), encoding="utf-8")
        return False, {
            "cmd": cmd,
            "returncode": None,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "script_sha256": script_sha256,
            "error": "timeout",
        }
    except Exception as e:
        log_path.write_text(str(e), encoding="utf-8")
        return False, {
            "cmd": cmd,
            "returncode": None,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "script_sha256": script_sha256,
            "error": str(e),
        }


def _extract_siril_save_targets(script_path: Path) -> List[str]:
    try:
        raw = script_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    targets: List[str] = []
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


def _run_siril(
    siril_exe: str,
    work_dir: Path,
    script_text: str,
    artifacts_dir: Path,
    script_name: str,
    log_name: str,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    script_path = artifacts_dir / script_name
    log_path = artifacts_dir / log_name
    script_path.write_text(script_text, encoding="utf-8")

    cmd = [siril_exe, "-s", str(script_path), "-d", str(work_dir)]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        return proc.returncode == 0, {
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
        }
    except subprocess.TimeoutExpired as e:
        log_path.write_text((e.stdout or "") + "\n" + (e.stderr or ""), encoding="utf-8")
        return False, {
            "cmd": cmd,
            "returncode": None,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "error": "timeout",
        }
    except Exception as e:
        log_path.write_text(str(e), encoding="utf-8")
        return False, {
            "cmd": cmd,
            "returncode": None,
            "seconds": max(0.0, time.time() - started),
            "log_path": str(log_path),
            "script_path": str(script_path),
            "error": str(e),
        }


def _pick_output_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _run_phases(
    run_id: str,
    log_fp,
    dry_run: bool,
    run_dir: Path,
    project_root: Path,
    cfg: Dict[str, Any],
    frames: List[Path],
    siril_exe: Optional[str],
) -> bool:
    outputs_dir = run_dir / "outputs"
    artifacts_dir = run_dir / "artifacts"
    work_dir = run_dir / "work"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        phases = [
            (1, "registration"),
            (2, "global_normalization"),
            (3, "global_metrics"),
            (4, "tile_geometry"),
            (5, "local_tile_metrics"),
            (6, "tile_reconstruction"),
            (7, "state_clustering"),
            (8, "synthetic_frames"),
            (9, "final_stacking"),
            (10, "validation"),
        ]
        for phase_id, phase_name in phases:
            _phase_start(run_id, log_fp, phase_id, phase_name)
            if _stop_requested(run_id, log_fp, phase_id, phase_name):
                return False
            _phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "dry_run"})
        return True

    d = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    frames_min = d.get("frames_min")
    color_mode = str(d.get("color_mode") or "").strip().upper()
    bayer_pattern = str(d.get("bayer_pattern") or "GBRG").strip().upper()
    try:
        frames_min_i = int(frames_min) if frames_min is not None else None
    except Exception:
        frames_min_i = None
    if frames_min_i is not None and frames_min_i > 0 and len(frames) < frames_min_i:
        _phase_start(run_id, log_fp, 1, "registration")
        _phase_end(
            run_id,
            log_fp,
            1,
            "registration",
            "error",
            {"error": f"frames.count={len(frames)} < data.frames_min={frames_min_i}"},
        )
        return False

    registration_cfg = cfg.get("registration") if isinstance(cfg.get("registration"), dict) else {}
    stacking_cfg = cfg.get("stacking") if isinstance(cfg.get("stacking"), dict) else {}
    synthetic_cfg = cfg.get("synthetic") if isinstance(cfg.get("synthetic"), dict) else {}

    reg_engine = str(registration_cfg.get("engine") or "")
    stack_engine = str(stacking_cfg.get("engine") or "")

    reg_script_cfg = registration_cfg.get("siril_script")
    reg_script_path = (
        Path(str(reg_script_cfg)).expanduser().resolve()
        if isinstance(reg_script_cfg, str) and reg_script_cfg.strip()
        else (project_root / "siril_register_osc.ssf").resolve()
    )
    stack_script_cfg = stacking_cfg.get("siril_script")
    stack_script_path = (
        Path(str(stack_script_cfg)).expanduser().resolve()
        if isinstance(stack_script_cfg, str) and stack_script_cfg.strip()
        else (project_root / "siril_stack_average.ssf").resolve()
    )

    reg_out_name = str(registration_cfg.get("output_dir") or "registered")
    reg_pattern = str(registration_cfg.get("registered_filename_pattern") or "reg_{index:05d}.fit")

    syn_dir_name = str(stacking_cfg.get("input_dir") or "synthetic")
    syn_pattern = str(stacking_cfg.get("input_pattern") or "syn_*.fits")
    syn_prefix = _derive_prefix_from_pattern(syn_pattern, "syn_")
    syn_ext = _siril_setext_from_suffix(Path(syn_pattern).suffix or ".fits")

    stack_output_file = str(stacking_cfg.get("output_file") or "stacked.fit")
    stack_method_cfg = str(stacking_cfg.get("method") or "")
    stack_method = stack_method_cfg.strip().lower()

    phase_id = 1
    phase_name = "registration"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    if reg_engine != "siril":
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"registration.engine not supported: {reg_engine!r}"})
        return False
    if not siril_exe:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "siril executable not found"})
        return False

    cfa_flag = _fits_is_cfa(frames[0]) if frames else None
    header_bayerpat = _fits_get_bayerpat(frames[0]) if frames else None
    warn = None
    if header_bayerpat is not None and header_bayerpat != bayer_pattern:
        warn = "bayer_pattern mismatch (config vs header)"

    using_default_reg_script = not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip())
    if using_default_reg_script and (color_mode != "OSC" or cfa_flag is not True):
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "error": "default registration script requires OSC/CFA; set registration.siril_script for non-OSC inputs",
                "color_mode": color_mode,
                "cfa": cfa_flag,
            },
        )
        return False

    if not reg_script_path.exists() or not reg_script_path.is_file():
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"registration script not found: {reg_script_path}"})
        return False

    ok_script, violations = _validate_siril_script(reg_script_path)
    if not ok_script:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "registration script violates policy", "script": str(reg_script_path), "violations": violations},
        )
        return False

    reg_work = work_dir / "registration"
    reg_work.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames, start=1):
        dst = reg_work / f"seq{i:05d}.fit"
        _safe_symlink_or_copy(src, dst)

    ok, meta = _run_siril_script(
        siril_exe=siril_exe,
        work_dir=reg_work,
        script_path=reg_script_path,
        artifacts_dir=artifacts_dir,
        log_name="siril_registration.log",
        quiet=True,
    )
    if not ok:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "siril": meta,
                "bayer_pattern": bayer_pattern,
                "bayer_pattern_header": header_bayerpat,
                "cfa": cfa_flag,
                "warning": warn,
            },
        )
        return False

    reg_out = outputs_dir / reg_out_name
    reg_out.mkdir(parents=True, exist_ok=True)
    registered = sorted([p for p in reg_work.iterdir() if p.is_file() and p.name.lower().startswith("r_") and _is_fits_image_path(p)])
    if not registered:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "no registered frames produced by Siril (expected r_*.fit*)", "siril": meta},
        )
        return False

    moved = 0
    for idx, src in enumerate(registered, start=1):
        try:
            dst_name = reg_pattern.format(index=idx)
        except Exception:
            dst_name = src.name
        shutil.copy2(src, reg_out / dst_name)
        moved += 1
    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "siril": meta,
            "output_dir": str(reg_out),
            "registered_count": moved,
            "bayer_pattern": bayer_pattern,
            "bayer_pattern_header": header_bayerpat,
            "cfa": cfa_flag,
            "warning": warn,
        },
    )

    for phase_id, phase_name in [
        (2, "global_normalization"),
        (3, "global_metrics"),
        (4, "tile_geometry"),
        (5, "local_tile_metrics"),
        (6, "tile_reconstruction"),
        (7, "state_clustering"),
    ]:
        _phase_start(run_id, log_fp, phase_id, phase_name)
        if _stop_requested(run_id, log_fp, phase_id, phase_name):
            return False
        _phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "not_implemented"})

    phase_id = 8
    phase_name = "synthetic_frames"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    reg_out_dir = outputs_dir / reg_out_name
    registered_files = (
        sorted([p for p in reg_out_dir.iterdir() if p.is_file() and _is_fits_image_path(p)])
        if reg_out_dir.exists()
        else []
    )
    if not registered_files:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "no registered frames found"})
        return False

    syn_min = synthetic_cfg.get("frames_min")
    syn_max = synthetic_cfg.get("frames_max")
    try:
        syn_min_i = int(syn_min) if syn_min is not None else 15
    except Exception:
        syn_min_i = 15
    try:
        syn_max_i = int(syn_max) if syn_max is not None else 30
    except Exception:
        syn_max_i = 30
    use_all_registered = bool(stack_method == "average")
    n = len(registered_files) if use_all_registered else min(len(registered_files), syn_max_i)
    if n < syn_min_i:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": f"registered_count={len(registered_files)} < synthetic.frames_min={syn_min_i}"},
        )
        return False
    syn_out = outputs_dir / syn_dir_name
    syn_out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(registered_files[:n], start=1):
        dst = syn_out / f"{syn_prefix}{i:05d}.{syn_ext}"
        shutil.copy2(src, dst)
    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {"synthetic_dir": str(syn_out), "synthetic_count": n, "use_all_registered": use_all_registered},
    )

    phase_id = 9
    phase_name = "final_stacking"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    if stack_engine != "siril":
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"stacking.engine not supported: {stack_engine!r}"})
        return False
    if not siril_exe:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "siril executable not found"})
        return False

    stack_work = work_dir / "stacking"
    stack_work.mkdir(parents=True, exist_ok=True)
    syn_files = sorted([p for p in syn_out.iterdir() if p.is_file() and _is_fits_image_path(p)])

    using_default_stack_script = not (isinstance(stack_script_cfg, str) and stack_script_cfg.strip())
    if using_default_stack_script and stack_method != "average":
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "error": "default stacking script is only defined for stacking.method=average; set stacking.siril_script for other methods",
                "stacking_method": stack_method,
            },
        )
        return False
    if not stack_script_path.exists() or not stack_script_path.is_file():
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"stacking script not found: {stack_script_path}"})
        return False

    ok_script, violations = _validate_siril_script(stack_script_path)
    if not ok_script:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "stacking script violates policy", "script": str(stack_script_path), "violations": violations},
        )
        return False

    for i, src in enumerate(syn_files, start=1):
        dst = stack_work / f"seq{i:05d}.fit"
        _safe_symlink_or_copy(src, dst)

    before = sorted([p.name for p in stack_work.iterdir() if p.is_file()])
    ok, meta = _run_siril_script(
        siril_exe=siril_exe,
        work_dir=stack_work,
        script_path=stack_script_path,
        artifacts_dir=artifacts_dir,
        log_name="siril_stacking.log",
        quiet=True,
    )
    if not ok:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"siril": meta, "method": stack_method})
        return False

    after_files = sorted([p for p in stack_work.iterdir() if p.is_file()])
    after_names = {p.name for p in after_files}
    new_names = sorted(list(after_names.difference(set(before))))

    save_targets = _extract_siril_save_targets(stack_script_path)
    candidates: List[Path] = []
    for name in save_targets:
        candidates.extend(
            [
                stack_work / name,
                stack_work / (name + ".fit"),
                stack_work / (name + ".fits"),
                stack_work / (name + ".fts"),
                stack_work / (Path(name).stem + ".fit"),
                stack_work / (Path(name).stem + ".fits"),
                stack_work / (Path(name).stem + ".fts"),
            ]
        )
    out_basename = Path(stack_output_file).name
    candidates.extend(
        [
            stack_work / out_basename,
            stack_work / (out_basename + ".fit"),
            stack_work / (out_basename + ".fits"),
            stack_work / (out_basename + ".fts"),
            stack_work / (Path(out_basename).stem + ".fit"),
            stack_work / (Path(out_basename).stem + ".fits"),
            stack_work / (Path(out_basename).stem + ".fts"),
        ]
    )
    produced = _pick_output_file(candidates)
    if produced is None and len(new_names) == 1:
        lone = stack_work / new_names[0]
        if lone.is_file() and _is_fits_image_path(lone):
            produced = lone
    if produced is None:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "error": "expected output not found",
                "siril": meta,
                "method": stack_method,
                "save_targets": save_targets,
                "new_files": new_names,
            },
        )
        return False

    if using_default_stack_script and stack_method == "average":
        n_stack = len(syn_files)
        if n_stack > 0:
            try:
                hdr = fits.getheader(str(produced), ext=0)
                data = fits.getdata(str(produced), ext=0)
                if data is not None:
                    data_f = data.astype("float32", copy=False)
                    data_f = data_f / float(n_stack)
                    fits.writeto(str(produced), data_f, header=hdr, overwrite=True)
            except Exception:
                pass

    final_out = outputs_dir / Path(stack_output_file)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(final_out))
    extra: Dict[str, Any] = {"siril": meta, "method": stack_method, "output": str(final_out)}
    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

    phase_id = 10
    phase_name = "validation"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    _phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "not_implemented"})

    return True


def cmd_run(args) -> int:
    config_path = Path(args.config).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()

    if not config_path.exists() or not config_path.is_file():
        sys.stderr.write(f"config not found: {config_path}\n")
        return 2

    if not input_dir.exists() or not input_dir.is_dir():
        sys.stderr.write(f"input_dir not found: {input_dir}\n")
        return 2

    run_id = str(uuid.uuid4())
    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{ts_compact}_{run_id}"

    project_root = _resolve_project_root(Path.cwd())
    siril_exe, siril_source = _resolve_siril_exe(project_root)

    paths = {
        "run_dir": str(run_dir),
        "runs_dir": str(runs_dir),
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "pattern": args.pattern,
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "color_mode_confirmed": args.color_mode_confirmed,
        "tools": {
            "siril_exe": siril_exe,
            "siril_source": siril_source,
        },
    }
    (run_dir / "run_metadata.json").write_bytes(_json_dumps_canonical(run_metadata))

    log_path = run_dir / "logs" / "run_events.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fp:
        config_bytes = _read_bytes(config_path)
        config_hash = _sha256_bytes(config_bytes)
        _copy_config(config_path, run_dir / "config.yaml")
        (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")

        try:
            cfg = yaml.safe_load(config_bytes.decode("utf-8"))
        except Exception as e:
            cfg = None
            _emit(
                {
                    "type": "runner_error",
                    "run_id": run_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "error": f"failed to parse config as YAML: {e}",
                },
                log_fp,
            )

        frames = _discover_frames(input_dir, args.pattern)
        frames_manifest = {
            "input_dir": str(input_dir),
            "pattern": args.pattern,
            "frames": [str(p) for p in frames],
        }
        frames_manifest_bytes = _json_dumps_canonical(frames_manifest)
        frames_manifest_id = _sha256_bytes(frames_manifest_bytes)
        (run_dir / "frames_manifest.json").write_bytes(frames_manifest_bytes)
        (run_dir / "frames_manifest_id.txt").write_text(frames_manifest_id + "\n", encoding="utf-8")

        _emit(
            {
                "type": "run_start",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "paths": paths,
                "config_hash": config_hash,
                "frames_manifest_id": frames_manifest_id,
                "frame_count": len(frames),
                "dry_run": bool(args.dry_run),
                "color_mode_confirmed": args.color_mode_confirmed,
                "tools": {
                    "siril_exe": siril_exe,
                    "siril_source": siril_source,
                },
            },
            log_fp,
        )

        if not isinstance(cfg, dict):
            _emit(
                {
                    "type": "run_end",
                    "run_id": run_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                },
                log_fp,
            )
            return 0

        ok = _run_phases(
            run_id,
            log_fp,
            dry_run=bool(args.dry_run),
            run_dir=run_dir,
            project_root=project_root,
            cfg=cfg,
            frames=frames,
            siril_exe=siril_exe,
        )

        _emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": "ok" if ok else "stopped",
            },
            log_fp,
        )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tile_compile_runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--input-dir", required=True)
    p_run.add_argument("--pattern", default="*.fit*")
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--color-mode-confirmed", default=None)
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv=None) -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
