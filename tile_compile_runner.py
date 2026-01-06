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
import numpy as np
import yaml

# New imports for Methodik v3 integration
try:
    from tile_compile_backend.policy import PhaseManager, PolicyValidator
    from tile_compile_backend.metrics import MetricsCalculator, TileMetricsCalculator, compute_channel_metrics
    from tile_compile_backend.reconstruction import reconstruct_channels
    from tile_compile_backend.synthetic import generate_channel_synthetic_frames
    from tile_compile_backend.clustering import cluster_channels
    from tile_compile_backend.configuration import validate_and_prepare_configuration
    from tile_compile_backend.registration import CFARegistration, BayerPattern
    from tile_compile_backend.linearity import validate_frames_linearity
    from tile_compile_backend.tile_grid import generate_multi_channel_grid
except Exception:
    PhaseManager = None
    PolicyValidator = None
    MetricsCalculator = None
    TileMetricsCalculator = None
    compute_channel_metrics = None
    reconstruct_channels = None
    generate_channel_synthetic_frames = None
    cluster_channels = None
    validate_and_prepare_configuration = None
    CFARegistration = None
    BayerPattern = None
    validate_frames_linearity = None
    generate_multi_channel_grid = None

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None


_STOP = False

# Reduced mode thresholds (Methodik v3 §1.4)
_FRAMES_MIN_DEFAULT = 50
_FRAMES_OPTIMAL_DEFAULT = 800
_FRAMES_REDUCED_THRESHOLD_DEFAULT = 200


def _get_assumptions_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Get assumptions configuration with defaults."""
    assumptions = cfg.get("assumptions") if isinstance(cfg.get("assumptions"), dict) else {}
    return {
        "frames_min": int(assumptions.get("frames_min", _FRAMES_MIN_DEFAULT)),
        "frames_optimal": int(assumptions.get("frames_optimal", _FRAMES_OPTIMAL_DEFAULT)),
        "frames_reduced_threshold": int(assumptions.get("frames_reduced_threshold", _FRAMES_REDUCED_THRESHOLD_DEFAULT)),
        "reduced_mode_skip_clustering": bool(assumptions.get("reduced_mode_skip_clustering", True)),
        "reduced_mode_cluster_range": assumptions.get("reduced_mode_cluster_range", [5, 10]),
        "exposure_time_tolerance_percent": float(assumptions.get("exposure_time_tolerance_percent", 5.0)),
        "registration_residual_warn_px": float(assumptions.get("registration_residual_warn_px", 0.5)),
        "registration_residual_max_px": float(assumptions.get("registration_residual_max_px", 1.0)),
        "elongation_warn": float(assumptions.get("elongation_warn", 0.3)),
        "elongation_max": float(assumptions.get("elongation_max", 0.4)),
    }


def _is_reduced_mode(frame_count: int, assumptions: Dict[str, Any]) -> bool:
    """Check if reduced mode should be activated (Methodik v3 §1.4)."""
    return frame_count < assumptions["frames_reduced_threshold"]


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


def _cfa_downsample_sum2x2(mosaic: np.ndarray) -> np.ndarray:
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
    a = mosaic[0::2, 0::2].astype("float32", copy=False)
    b = mosaic[0::2, 1::2].astype("float32", copy=False)
    c = mosaic[1::2, 0::2].astype("float32", copy=False)
    d = mosaic[1::2, 1::2].astype("float32", copy=False)
    return a + b + c + d


def _opencv_prepare_ecc_image(img: np.ndarray) -> np.ndarray:
    f = img.astype("float32", copy=False)
    med = float(np.median(f))
    f = f - med
    sd = float(np.std(f))
    if sd > 0:
        f = f / sd
    bg = cv2.GaussianBlur(f, (0, 0), 12.0)
    f = f - bg
    f = cv2.GaussianBlur(f, (0, 0), 1.0)
    f = cv2.normalize(f, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return f


def _opencv_count_stars(img01: np.ndarray) -> int:
    corners = cv2.goodFeaturesToTrack(
        img01,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7,
        useHarrisDetector=False,
    )
    return int(0 if corners is None else len(corners))


def _opencv_ecc_warp(
    moving01: np.ndarray,
    ref01: np.ndarray,
    allow_rotation: bool,
    init_warp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    motion = cv2.MOTION_EUCLIDEAN if allow_rotation else cv2.MOTION_TRANSLATION
    warp = (init_warp.copy() if init_warp is not None else np.eye(2, 3, dtype=np.float32)).astype(np.float32, copy=False)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 1e-6)
    cc, warp = cv2.findTransformECC(ref01, moving01, warp, motion, criteria, None, 5)
    return warp, float(cc)


def _opencv_phasecorr_translation(moving01: np.ndarray, ref01: np.ndarray) -> Tuple[float, float]:
    win = cv2.createHanningWindow((ref01.shape[1], ref01.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref01.astype("float32", copy=False), moving01.astype("float32", copy=False), win)
    return float(dx), float(dy)


def _opencv_alignment_score(moving01: np.ndarray, ref01: np.ndarray) -> float:
    a = moving01.astype("float32", copy=False)
    b = ref01.astype("float32", copy=False)
    am = float(np.mean(a))
    bm = float(np.mean(b))
    da = a - am
    db = b - bm
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if denom <= 0:
        return -1.0
    return float(np.sum(da * db) / denom)


def _opencv_best_translation_init(moving01: np.ndarray, ref01: np.ndarray) -> np.ndarray:
    dx, dy = _opencv_phasecorr_translation(moving01, ref01)
    candidates = [
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32),
    ]
    best = candidates[0]
    best_score = -1e9
    for M in candidates:
        warped = cv2.warpAffine(
            moving01,
            M,
            (ref01.shape[1], ref01.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        s = _opencv_alignment_score(warped, ref01)
        if s > best_score:
            best_score = s
            best = M
    return best


def _warp_cfa_mosaic_via_subplanes(mosaic: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
        h, w = mosaic.shape[:2]
    hh = h // 2
    ww = w // 2
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    border = cv2.BORDER_CONSTANT
    border_val = 0

    p00 = mosaic[0::2, 0::2].astype("float32", copy=False)
    p01 = mosaic[0::2, 1::2].astype("float32", copy=False)
    p10 = mosaic[1::2, 0::2].astype("float32", copy=False)
    p11 = mosaic[1::2, 1::2].astype("float32", copy=False)

    w00 = cv2.warpAffine(p00, warp, (ww, hh), flags=flags, borderMode=border, borderValue=border_val)
    w01 = cv2.warpAffine(p01, warp, (ww, hh), flags=flags, borderMode=border, borderValue=border_val)
    w10 = cv2.warpAffine(p10, warp, (ww, hh), flags=flags, borderMode=border, borderValue=border_val)
    w11 = cv2.warpAffine(p11, warp, (ww, hh), flags=flags, borderMode=border, borderValue=border_val)

    out = np.zeros((h, w), dtype=np.float32)
    out[0::2, 0::2] = w00
    out[0::2, 1::2] = w01
    out[1::2, 0::2] = w10
    out[1::2, 1::2] = w11
    return out


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
        if low.startswith("save "):
            rest = s.split(None, 1)[1].strip() if len(s.split(None, 1)) == 2 else ""
            if not rest:
                continue
            target = Path(rest.strip().strip('"').strip("'")).name
            if target:
                targets.append(target)
            continue

        if low.startswith("stack ") and "-out=" in low:
            m = re.search(r"-out=([^\s]+)", s)
            if m:
                out_arg = m.group(1).strip().strip('"').strip("'")
                target = Path(out_arg).name
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


def _read_fits_float(path: Path) -> Tuple[np.ndarray, Any]:
    hdr = fits.getheader(str(path), ext=0)
    data = fits.getdata(str(path), ext=0)
    if data is None:
        raise RuntimeError("no FITS data")
    return np.asarray(data).astype("float32", copy=False), hdr


def _split_rgb_frame(data: np.ndarray) -> Dict[str, np.ndarray]:
    if data.ndim == 2:
        return {"R": data, "G": data, "B": data}
    if data.ndim != 3:
        raise RuntimeError(f"unsupported FITS data ndim={data.ndim}")
    if data.shape[0] == 3:
        return {"R": data[0, :, :], "G": data[1, :, :], "B": data[2, :, :]}
    if data.shape[-1] == 3:
        return {"R": data[:, :, 0], "G": data[:, :, 1], "B": data[:, :, 2]}
    raise RuntimeError(f"unsupported RGB FITS layout: shape={data.shape}")


def _split_cfa_channels(mosaic: np.ndarray, bayer_pattern: str) -> Dict[str, np.ndarray]:
    bp = str(bayer_pattern or "GBRG").strip().upper()
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]

    if bp == "RGGB":
        r = mosaic[0::2, 0::2]
        g1 = mosaic[0::2, 1::2]
        g2 = mosaic[1::2, 0::2]
        b = mosaic[1::2, 1::2]
    elif bp == "BGGR":
        b = mosaic[0::2, 0::2]
        g1 = mosaic[0::2, 1::2]
        g2 = mosaic[1::2, 0::2]
        r = mosaic[1::2, 1::2]
    elif bp == "GBRG":
        g1 = mosaic[0::2, 0::2]
        b = mosaic[0::2, 1::2]
        r = mosaic[1::2, 0::2]
        g2 = mosaic[1::2, 1::2]
    elif bp == "GRBG":
        g1 = mosaic[0::2, 0::2]
        r = mosaic[0::2, 1::2]
        b = mosaic[1::2, 0::2]
        g2 = mosaic[1::2, 1::2]
    else:
        raise RuntimeError(f"unsupported bayer_pattern: {bp}")

    g = (g1.astype("float32", copy=False) + g2.astype("float32", copy=False)) * 0.5
    return {
        "R": r.astype("float32", copy=False),
        "G": g.astype("float32", copy=False),
        "B": b.astype("float32", copy=False),
    }


def _normalize_frames(frames: List[np.ndarray], mode: str) -> Tuple[List[np.ndarray], float]:
    if not frames:
        return [], 0.0
    meds = [float(np.median(f)) for f in frames]
    target = float(np.median(np.asarray(meds, dtype=np.float32)))
    out: List[np.ndarray] = []
    m = str(mode or "background").strip().lower()
    for f, med in zip(frames, meds):
        if m == "median":
            scale = (target / med) if med not in (0.0, -0.0) else 1.0
            out.append((f * float(scale)).astype("float32", copy=False))
        else:
            out.append((f - (med - target)).astype("float32", copy=False))
    return out, target


def _to_uint8(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img).astype("float32", copy=False)
    mn = float(np.min(f))
    mx = float(np.max(f))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(f.shape, dtype=np.uint8)
    x = (f - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


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
            (0, "SCAN_INPUT"),
            (1, "REGISTRATION"),
            (2, "CHANNEL_SPLIT"),
            (3, "NORMALIZATION"),
            (4, "GLOBAL_METRICS"),
            (5, "TILE_GRID"),
            (6, "LOCAL_METRICS"),
            (7, "TILE_RECONSTRUCTION"),
            (8, "STATE_CLUSTERING"),
            (9, "SYNTHETIC_FRAMES"),
            (10, "STACKING"),
            (11, "DONE"),
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
        _phase_start(run_id, log_fp, 0, "SCAN_INPUT")
        _phase_end(
            run_id,
            log_fp,
            0,
            "SCAN_INPUT",
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

    if not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip()):
        if not reg_script_path.exists():
            alt = (project_root / "siril_scripts" / "siril_register_osc.ssf").resolve()
            if alt.exists():
                reg_script_path = alt
    if not (isinstance(stack_script_cfg, str) and stack_script_cfg.strip()):
        if not stack_script_path.exists():
            alt = (project_root / "siril_scripts" / "siril_stack_average.ssf").resolve()
            if alt.exists():
                stack_script_path = alt

    reg_out_name = str(registration_cfg.get("output_dir") or "registered")
    reg_pattern = str(registration_cfg.get("registered_filename_pattern") or "reg_{index:05d}.fit")

    stack_input_dir_name = str(stacking_cfg.get("input_dir") or reg_out_name)
    stack_input_pattern = str(stacking_cfg.get("input_pattern") or "reg_*.fit")

    stack_output_file = str(stacking_cfg.get("output_file") or "stacked.fit")
    stack_method_cfg = str(stacking_cfg.get("method") or "")
    stack_method = stack_method_cfg.strip().lower()

    phase_id = 0
    phase_name = "SCAN_INPUT"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    cfa_flag0 = _fits_is_cfa(frames[0]) if frames else None
    header_bayerpat0 = _fits_get_bayerpat(frames[0]) if frames else None
    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "frame_count": len(frames),
            "color_mode": color_mode,
            "bayer_pattern": bayer_pattern,
            "bayer_pattern_header": header_bayerpat0,
            "cfa": cfa_flag0,
        },
    )

    phase_id = 1
    phase_name = "REGISTRATION"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    if reg_engine == "opencv_cfa":
        if cv2 is None:
            _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "opencv (cv2) not available; install python opencv to use registration.engine=opencv_cfa"})
            return False

        allow_rotation = bool(registration_cfg.get("allow_rotation"))
        min_star_matches = registration_cfg.get("min_star_matches")
        try:
            min_star_matches_i = int(min_star_matches) if min_star_matches is not None else 1
        except Exception:
            min_star_matches_i = 1

        reg_out = outputs_dir / reg_out_name
        reg_out.mkdir(parents=True, exist_ok=True)

        ref_idx = 0
        ref_stars = -1
        ref_lum01: Optional[np.ndarray] = None
        ref_hdr = None
        ref_path = None
        for i, p in enumerate(frames):
            try:
                data = fits.getdata(str(p), ext=0)
                if data is None:
                    continue
                lum = _cfa_downsample_sum2x2(np.asarray(data))
                lum01 = _opencv_prepare_ecc_image(lum)
                stars = _opencv_count_stars(lum01)
                if stars > ref_stars:
                    ref_stars = stars
                    ref_idx = i
                    ref_lum01 = lum01
                    ref_hdr = fits.getheader(str(p), ext=0)
                    ref_path = p
            except Exception:
                continue

        if ref_lum01 is None or ref_stars < max(1, min_star_matches_i):
            _phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {"error": "failed to select registration reference (insufficient stars)", "min_star_matches": min_star_matches_i, "best_star_count": ref_stars},
            )
            return False

        corrs: List[float] = []
        registered_count = 0
        for idx, p in enumerate(frames):
            try:
                src_data = fits.getdata(str(p), ext=0)
                if src_data is None:
                    raise RuntimeError("no FITS data")
                src_hdr = fits.getheader(str(p), ext=0)
                mosaic = np.asarray(src_data)
                lum = _cfa_downsample_sum2x2(mosaic)
                lum01 = _opencv_prepare_ecc_image(lum)
                stars = _opencv_count_stars(lum01)
                if stars < max(1, min_star_matches_i):
                    raise RuntimeError(f"insufficient stars: {stars} < {min_star_matches_i}")

                if idx == ref_idx:
                    warped = mosaic.astype("float32", copy=False)
                    cc = 1.0
                else:
                    init = _opencv_best_translation_init(lum01, ref_lum01)
                    try:
                        warp, cc = _opencv_ecc_warp(lum01, ref_lum01, allow_rotation=allow_rotation, init_warp=init)
                    except Exception:
                        warp, cc = init, 0.0
                    if not np.isfinite(cc) or cc < 0.15:
                        warp, cc = init, float(cc if np.isfinite(cc) else 0.0)
                    warped = _warp_cfa_mosaic_via_subplanes(mosaic, warp)

                try:
                    dst_name = reg_pattern.format(index=registered_count + 1)
                except Exception:
                    dst_name = f"reg_{registered_count + 1:05d}.fit"
                dst_path = reg_out / dst_name
                fits.writeto(str(dst_path), warped.astype("float32", copy=False), header=src_hdr, overwrite=True)
                registered_count += 1
                corrs.append(float(cc))
            except Exception as e:
                _phase_end(
                    run_id,
                    log_fp,
                    phase_id,
                    phase_name,
                    "error",
                    {
                        "error": "opencv_cfa registration failed",
                        "frame": str(p),
                        "frame_index": idx,
                        "reference_index": ref_idx,
                        "reference_frame": str(ref_path) if ref_path is not None else None,
                        "details": str(e),
                    },
                )
                return False

        extra: Dict[str, Any] = {
            "engine": "opencv_cfa",
            "output_dir": str(reg_out),
            "registered_count": registered_count,
            "reference_index": ref_idx,
            "min_star_matches": min_star_matches_i,
            "allow_rotation": allow_rotation,
        }
        if corrs:
            extra["ecc_corr_min"] = float(min(corrs))
            extra["ecc_corr_mean"] = float(sum(corrs) / len(corrs))
        _phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

    else:
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

    reg_out_dir = outputs_dir / reg_out_name
    registered_files = sorted([p for p in reg_out_dir.iterdir() if p.is_file() and _is_fits_image_path(p)]) if reg_out_dir.exists() else []
    if not registered_files:
        _phase_end(run_id, log_fp, 1, "REGISTRATION", "error", {"error": "no registered frames found"})
        return False

    phase_id = 2
    phase_name = "CHANNEL_SPLIT"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    frames_target = d.get("frames_target")
    try:
        frames_target_i = int(frames_target) if frames_target is not None else 0
    except Exception:
        frames_target_i = 0
    analysis_count = len(registered_files) if frames_target_i <= 0 else min(len(registered_files), frames_target_i)

    channels: Dict[str, List[np.ndarray]] = {"R": [], "G": [], "B": []}
    cfa_registered = None
    for p in registered_files[:analysis_count]:
        data, _hdr = _read_fits_float(p)
        is_cfa = (_fits_is_cfa(p) is True)
        if cfa_registered is None:
            cfa_registered = is_cfa
        if is_cfa:
            split = _split_cfa_channels(data, bayer_pattern)
        else:
            try:
                split = _split_rgb_frame(data)
            except Exception:
                if data.ndim != 2:
                    _phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": "unsupported registered frame layout for channel split", "frame": str(p), "shape": list(data.shape)},
                    )
                    return False
                split = {"R": data, "G": data, "B": data}
        channels["R"].append(split["R"])
        channels["G"].append(split["G"])
        channels["B"].append(split["B"])

    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "registered_dir": str(reg_out_dir),
            "registered_count": len(registered_files),
            "analysis_count": analysis_count,
            "cfa": bool(cfa_registered),
        },
    )

    phase_id = 3
    phase_name = "NORMALIZATION"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    norm_cfg = cfg.get("normalization") if isinstance(cfg.get("normalization"), dict) else {}
    norm_mode = str(norm_cfg.get("mode") or "background")
    per_channel = bool(norm_cfg.get("per_channel", True))
    norm_target: Optional[float] = None
    if per_channel:
        channels["R"], _ = _normalize_frames(channels["R"], norm_mode)
        channels["G"], _ = _normalize_frames(channels["G"], norm_mode)
        channels["B"], _ = _normalize_frames(channels["B"], norm_mode)
    else:
        meds = [float(np.median(f)) for ch in ("R", "G", "B") for f in channels[ch]]
        norm_target = float(np.median(np.asarray(meds, dtype=np.float32))) if meds else None
        out: Dict[str, List[np.ndarray]] = {"R": [], "G": [], "B": []}
        for ch in ("R", "G", "B"):
            for f in channels[ch]:
                med = float(np.median(f))
                if str(norm_mode).strip().lower() == "median":
                    scale = (norm_target / med) if (norm_target is not None and med not in (0.0, -0.0)) else 1.0
                    out[ch].append((f * float(scale)).astype("float32", copy=False))
                else:
                    out[ch].append((f - (med - float(norm_target or med))).astype("float32", copy=False))
        channels = out

    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {"mode": norm_mode, "per_channel": per_channel, "target_median": norm_target},
    )

    phase_id = 4
    phase_name = "GLOBAL_METRICS"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    weights_cfg = cfg.get("global_metrics") if isinstance(cfg.get("global_metrics"), dict) else {}
    w = weights_cfg.get("weights") if isinstance(weights_cfg.get("weights"), dict) else {}
    try:
        w_bg = float(w.get("background", 1.0 / 3.0))
    except Exception:
        w_bg = 1.0 / 3.0
    try:
        w_noise = float(w.get("noise", 1.0 / 3.0))
    except Exception:
        w_noise = 1.0 / 3.0
    try:
        w_grad = float(w.get("gradient", 1.0 / 3.0))
    except Exception:
        w_grad = 1.0 / 3.0

    channel_metrics: Dict[str, Dict[str, Any]] = {}
    for ch in ("R", "G", "B"):
        frs = channels[ch]
        bgs = [float(np.median(f)) for f in frs]
        noises = [float(np.std(f)) for f in frs]
        grads = [float(np.mean(np.hypot(*np.gradient(f.astype("float32", copy=False))))) for f in frs]

        def _norm01(vals: List[float]) -> List[float]:
            if not vals:
                return []
            a = np.asarray(vals, dtype=np.float32)
            mn = float(np.min(a))
            mx = float(np.max(a))
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                return [0.0 for _ in vals]
            return [float(x) for x in ((a - mn) / (mx - mn)).tolist()]

        bg_n = _norm01(bgs)
        noise_n = _norm01(noises)
        grad_n = _norm01(grads)
        gfc = [float(w_bg * (1.0 - b) + w_noise * (1.0 - n) + w_grad * g) for b, n, g in zip(bg_n, noise_n, grad_n)]

        channel_metrics[ch] = {
            "global": {
                "background_level": bgs,
                "noise_level": noises,
                "gradient_energy": grads,
                "G_f_c": gfc,
            }
        }

    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"analysis_count": analysis_count})

    phase_id = 5
    phase_name = "TILE_GRID"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    tile_cfg = cfg.get("tile") if isinstance(cfg.get("tile"), dict) else {}
    try:
        min_tile_size = int(tile_cfg.get("min_size") or 32)
    except Exception:
        min_tile_size = 32
    try:
        max_divisor = int(tile_cfg.get("max_divisor") or 8)
    except Exception:
        max_divisor = 8
    try:
        overlap = float(tile_cfg.get("overlap_fraction") or 0.25)
    except Exception:
        overlap = 0.25

    rep = {ch: channels[ch][0] for ch in ("R", "G", "B") if channels[ch]}
    if not rep:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "no frames for tile grid"})
        return False
    h0, w0 = next(iter(rep.values())).shape[:2]
    max_tile_size = max(min_tile_size, int(min(h0, w0) // max(1, max_divisor)))
    grid_cfg = {"min_tile_size": min_tile_size, "max_tile_size": max_tile_size, "overlap": overlap}

    if generate_multi_channel_grid is None:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "tile_grid backend not available"})
        return False
    tile_grids = generate_multi_channel_grid({k: _to_uint8(v) for k, v in rep.items()}, grid_cfg)
    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"grid_cfg": grid_cfg})

    phase_id = 6
    phase_name = "LOCAL_METRICS"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    if TileMetricsCalculator is None:
        _phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "metrics backend not available"})
        return False

    try:
        tile_size_i = int(tile_grids.get("G", {}).get("tile_size") or tile_grids.get("R", {}).get("tile_size") or min_tile_size)
    except Exception:
        tile_size_i = min_tile_size
    try:
        overlap_i = float(tile_grids.get("G", {}).get("overlap") or overlap)
    except Exception:
        overlap_i = overlap
    tile_calc = TileMetricsCalculator(tile_size=tile_size_i, overlap=overlap_i)

    lm_cfg = cfg.get("local_metrics") if isinstance(cfg.get("local_metrics"), dict) else {}
    star_mode = lm_cfg.get("star_mode") if isinstance(lm_cfg.get("star_mode"), dict) else {}
    star_w = star_mode.get("weights") if isinstance(star_mode.get("weights"), dict) else {}
    try:
        w_fwhm = float(star_w.get("fwhm", 1.0 / 3.0))
    except Exception:
        w_fwhm = 1.0 / 3.0
    try:
        w_round = float(star_w.get("roundness", 1.0 / 3.0))
    except Exception:
        w_round = 1.0 / 3.0
    try:
        w_con = float(star_w.get("contrast", 1.0 / 3.0))
    except Exception:
        w_con = 1.0 / 3.0

    for ch in ("R", "G", "B"):
        q_local: List[List[float]] = []
        q_mean: List[float] = []
        q_var: List[float] = []
        for f in channels[ch]:
            tm = tile_calc.calculate_tile_metrics(f)
            fwhm = np.asarray(tm.get("fwhm") or [], dtype=np.float32)
            rnd = np.asarray(tm.get("roundness") or [], dtype=np.float32)
            con = np.asarray(tm.get("contrast") or [], dtype=np.float32)
            if fwhm.size == 0:
                q = np.zeros((0,), dtype=np.float32)
            else:
                mn = float(np.min(fwhm))
                mx = float(np.max(fwhm))
                if mx > mn:
                    inv_fwhm = 1.0 - (fwhm - mn) / (mx - mn)
                else:
                    inv_fwhm = np.ones_like(fwhm)
                q = (w_fwhm * inv_fwhm + w_round * rnd + w_con * con).astype(np.float32, copy=False)
            q_local.append([float(x) for x in q.tolist()])
            q_mean.append(float(np.mean(q)) if q.size else 0.0)
            q_var.append(float(np.var(q)) if q.size else 0.0)

        channel_metrics[ch]["tiles"] = {"Q_local": q_local, "tile_quality_mean": q_mean, "tile_quality_variance": q_var}

    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"tile_size": tile_size_i, "overlap": overlap_i})

    phase_id = 7
    phase_name = "TILE_RECONSTRUCTION"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    reconstructed: Dict[str, np.ndarray] = {}
    hdr0 = None
    try:
        hdr0 = fits.getheader(str(registered_files[0]), ext=0)
    except Exception:
        hdr0 = None
    for ch in ("R", "G", "B"):
        frs = channels[ch]
        gfc = np.asarray(channel_metrics[ch]["global"].get("G_f_c") or [], dtype=np.float32)
        if frs and gfc.size == len(frs) and float(np.sum(gfc)) > 0:
            wsum = float(np.sum(gfc))
            w_norm = (gfc / wsum).astype(np.float32, copy=False)
            out = np.zeros_like(frs[0], dtype=np.float32)
            for f, ww in zip(frs, w_norm):
                out += f.astype(np.float32, copy=False) * float(ww)
            reconstructed[ch] = out
        elif frs:
            reconstructed[ch] = np.mean(np.asarray(frs, dtype=np.float32), axis=0).astype(np.float32, copy=False)
        else:
            reconstructed[ch] = np.zeros((1, 1), dtype=np.float32)

        out_path = outputs_dir / f"reconstructed_{ch}.fits"
        fits.writeto(str(out_path), reconstructed[ch].astype("float32", copy=False), header=hdr0, overwrite=True)

    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"outputs": [f"reconstructed_{c}.fits" for c in ("R", "G", "B")]})

    phase_id = 8
    phase_name = "STATE_CLUSTERING"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    # Reduced mode check (Methodik v3 §1.4)
    assumptions_cfg = _get_assumptions_config(cfg)
    frame_count = len(registered_files)
    reduced_mode = _is_reduced_mode(frame_count, assumptions_cfg)
    
    clustering_cfg = synthetic_cfg.get("clustering") if isinstance(synthetic_cfg.get("clustering"), dict) else {}
    clustering_results = None
    clustering_skipped = False
    
    if reduced_mode and assumptions_cfg["reduced_mode_skip_clustering"]:
        # Skip clustering in reduced mode
        clustering_skipped = True
    elif cluster_channels is not None:
        try:
            # Adjust cluster range for reduced mode
            if reduced_mode:
                reduced_range = assumptions_cfg["reduced_mode_cluster_range"]
                clustering_cfg = dict(clustering_cfg)
                clustering_cfg["cluster_count_range"] = reduced_range
            clustering_results = cluster_channels(channels, channel_metrics, clustering_cfg)
        except Exception:
            clustering_results = None

    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "enabled": bool(clustering_results is not None),
            "reduced_mode": reduced_mode,
            "skipped": clustering_skipped,
        },
    )

    phase_id = 9
    phase_name = "SYNTHETIC_FRAMES"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False

    syn_out = outputs_dir / "synthetic"
    syn_out.mkdir(parents=True, exist_ok=True)
    synthetic_channels: Optional[Dict[str, List[np.ndarray]]] = None
    synthetic_count = 0
    synthetic_skipped = False
    
    # Skip synthetic frames in reduced mode if clustering was skipped
    if reduced_mode and clustering_skipped:
        synthetic_skipped = True
    else:
        try:
            metrics_for_syn: Dict[str, Dict[str, Any]] = {}
            for ch in ("R", "G", "B"):
                g = channel_metrics.get(ch, {}).get("global", {}) if isinstance(channel_metrics.get(ch), dict) else {}
                t = channel_metrics.get(ch, {}).get("tiles", {}) if isinstance(channel_metrics.get(ch), dict) else {}
                g2 = dict(g)
                if "noise_level" in g2:
                    g2["noise_level"] = np.asarray(g2["noise_level"], dtype=np.float32)
                if "background_level" in g2:
                    g2["background_level"] = np.asarray(g2["background_level"], dtype=np.float32)
                metrics_for_syn[ch] = {"global": g2, "tiles": t}

            if generate_channel_synthetic_frames is not None:
                synthetic_channels = generate_channel_synthetic_frames(channels, metrics_for_syn, synthetic_cfg)
        except Exception:
            synthetic_channels = None

        if synthetic_channels:
            hdr_syn = None
            try:
                hdr_syn = fits.getheader(str(registered_files[0]), ext=0)
            except Exception:
                hdr_syn = None
            for ch in ("R", "G", "B"):
                frs = synthetic_channels.get(ch) or []
                for i, f in enumerate(frs, start=1):
                    outp = syn_out / f"syn_{ch}_{i:05d}.fits"
                    fits.writeto(str(outp), np.asarray(f).astype("float32", copy=False), header=hdr_syn, overwrite=True)
                synthetic_count = max(synthetic_count, len(frs))

    _phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "synthetic_dir": str(syn_out),
            "synthetic_count": synthetic_count,
            "enabled": bool(synthetic_channels),
            "reduced_mode": reduced_mode,
            "skipped": synthetic_skipped,
        },
    )

    phase_id = 10
    phase_name = "STACKING"
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
    stack_src_dir = outputs_dir / Path(stack_input_dir_name)
    stack_files = (
        sorted([p for p in stack_src_dir.glob(stack_input_pattern) if p.is_file() and _is_fits_image_path(p)])
        if stack_src_dir.exists()
        else []
    )
    if not stack_files:
        _phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "no stacking input frames found", "input_dir": str(stack_src_dir), "input_pattern": stack_input_pattern},
        )
        return False

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

    for i, src in enumerate(stack_files, start=1):
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
    if produced is None:
        new_fits = [stack_work / n for n in new_names if (stack_work / n).is_file() and _is_fits_image_path(stack_work / n)]
        if len(new_fits) == 1:
            produced = new_fits[0]
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
        n_stack = len(stack_files)
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

    phase_id = 11
    phase_name = "DONE"
    _phase_start(run_id, log_fp, phase_id, phase_name)
    if _stop_requested(run_id, log_fp, phase_id, phase_name):
        return False
    _phase_end(run_id, log_fp, phase_id, phase_name, "ok", {})

    return True


def process_pipeline(config, input_frames):
    """
    Process entire pipeline using new modules with comprehensive validation
    """
    # Validate configuration
    config = validate_and_prepare_configuration(config)
    
    # Create phase manager
    phase_manager = PhaseManager()
    
    # Input scan and validation
    phase_manager.advance_phase('SCAN_INPUT', {'frames': input_frames})
    
    # Linearity Validation
    linearity_result = validate_frames_linearity(
        np.array(input_frames), 
        config.get('linearity', {})
    )
    
    if linearity_result['overall_linearity'] < 0.9:
        raise ValueError("Input frames fail linearity validation")
    
    # Registration with CFA-aware method
    registration_result = CFARegistration.register_cfa_frames(
        linearity_result['valid_frames'], 
        bayer_pattern=BayerPattern[config.get('bayer_pattern', 'GBRG')]
    )
    
    registered_frames = registration_result['registered_frames']
    phase_manager.advance_phase('REGISTRATION', {'registered_frames': registered_frames})
    
    # Channel Split
    channels = {
        'R': [frame for frame in registered_frames],
        'G': [frame for frame in registered_frames],
        'B': [frame for frame in registered_frames]
    }
    phase_manager.advance_phase('CHANNEL_SPLIT', {'channels': channels})
    
    # Compute Metrics
    channel_metrics = compute_channel_metrics(channels)
    phase_manager.advance_phase('GLOBAL_METRICS', {'metrics': channel_metrics})
    
    # Adaptive Tile Grid Generation
    tile_grids = generate_multi_channel_grid(
        channels, 
        config.get('tile_grid', {})
    )
    phase_manager.advance_phase('TILE_GRID', {'grids': tile_grids})
    
    # Synthetic Frame Generation
    synthetic_frames = generate_channel_synthetic_frames(
        channels, 
        channel_metrics, 
        config.get('synthetic', {})
    )
    phase_manager.advance_phase('SYNTHETIC_FRAMES', {'synthetic_frames': synthetic_frames})
    
    # State Clustering
    clustering_results = cluster_channels(
        channels, 
        channel_metrics, 
        config.get('clustering', {})
    )
    phase_manager.advance_phase('STATE_CLUSTERING', {'clustering': clustering_results})
    
    # Tile Reconstruction
    reconstructed_channels = reconstruct_channels(
        channels, 
        channel_metrics
    )
    phase_manager.advance_phase('TILE_RECONSTRUCTION', {'reconstructed_channels': reconstructed_channels})
    
    # Final Stacking
    # TODO: Implement final stacking logic
    phase_manager.advance_phase('STACKING')
    
    phase_manager.advance_phase('DONE')
    
    return reconstructed_channels

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

        status = "ok" if ok else ("stopped" if _STOP else "error")
        _emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": status,
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
