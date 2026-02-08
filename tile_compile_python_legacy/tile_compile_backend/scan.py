from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits


@dataclass
class ScanIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _list_frame_files(input_path: Path) -> list[Path]:
    exts = {".fit", ".fits"}
    files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_project_cache_dir(input_path: Path) -> Path:
    # Cache lives alongside the input directory, not inside runs.
    # This keeps scan results reusable across runs and stable for a given project.
    return input_path.resolve().parent / ".tile_compile"


def scan_input(
    input_path: str,
    frames_min: int = 1,
    project_cache_dir: str | None = None,
    with_checksums: bool = False,
) -> dict:
    issues: list[ScanIssue] = []

    in_dir = Path(input_path)
    if not in_dir.exists() or not in_dir.is_dir():
        return {
            "ok": False,
            "errors": [
                {
                    "severity": "error",
                    "code": "input_path_not_found",
                    "message": f"input_path does not exist or is not a directory: {input_path}",
                }
            ],
            "warnings": [],
        }

    files = _list_frame_files(in_dir)
    frames_detected = len(files)

    if frames_detected < int(frames_min):
        issues.append(
            ScanIssue(
                severity="error",
                code="too_few_frames",
                message=f"frames_detected ({frames_detected}) < frames_min ({frames_min})",
            )
        )

    image_width: int | None = None
    image_height: int | None = None

    has_bayerpat = False
    bayer_pattern: str | None = None
    bayer_pattern_inconsistent = False

    frames: list[dict[str, Any]] = []

    for p in files:
        try:
            hdr = fits.getheader(str(p), ext=0)
        except Exception as e:  # noqa: BLE001
            issues.append(
                ScanIssue(
                    severity="error",
                    code="fits_read_error",
                    message=f"failed to read FITS header for {p.name}: {e}",
                )
            )
            continue

        naxis1 = hdr.get("NAXIS1")
        naxis2 = hdr.get("NAXIS2")
        if naxis1 is None or naxis2 is None:
            issues.append(
                ScanIssue(
                    severity="error",
                    code="fits_missing_axis",
                    message=f"missing NAXIS1/NAXIS2 in FITS header for {p.name}",
                )
            )
            continue

        try:
            w = int(naxis1)
            h = int(naxis2)
        except Exception as e:  # noqa: BLE001
            issues.append(
                ScanIssue(
                    severity="error",
                    code="fits_axis_not_int",
                    message=f"invalid NAXIS1/NAXIS2 in FITS header for {p.name}: {e}",
                )
            )
            continue

        if image_width is None:
            image_width = w
            image_height = h
        else:
            if w != image_width or h != image_height:
                issues.append(
                    ScanIssue(
                        severity="error",
                        code="inconsistent_image_dimensions",
                        message=(
                            f"inconsistent image size: expected {image_width}x{image_height}, got {w}x{h} in {p.name}"
                        ),
                    )
                )

        bpat = hdr.get("BAYERPAT")
        if isinstance(bpat, str) and bpat.strip():
            has_bayerpat = True
            bp = bpat.strip().upper()
            if bp in {"RGGB", "BGGR", "GBRG", "GRBG"}:
                if bayer_pattern is None:
                    bayer_pattern = bp
                elif bayer_pattern != bp:
                    bayer_pattern_inconsistent = True

        entry: dict[str, Any] = {
            "file_name": p.name,
            "abs_path": str(p.resolve()),
        }
        if with_checksums:
            entry["sha256"] = _compute_file_sha256(p)
        frames.append(entry)

    if image_width is None or image_height is None:
        # No readable frames -> hard error
        issues.append(
            ScanIssue(
                severity="error",
                code="no_readable_frames",
                message="no readable FITS frames found",
            )
        )
        image_width = 0
        image_height = 0

    requires_user_confirmation = False
    color_mode_candidates: list[str] = ["OSC"]
    color_mode = "OSC"
    if not has_bayerpat:
        # Do not implicitly decide. GUI must confirm / finalize.
        requires_user_confirmation = True
        color_mode = "UNKNOWN"
        issues.append(
            ScanIssue(
                severity="warning",
                code="color_mode_ambiguous",
                message="BAYERPAT not found in FITS headers; color_mode requires user confirmation",
            )
        )
    if bayer_pattern_inconsistent:
        requires_user_confirmation = True
        issues.append(
            ScanIssue(
                severity="warning",
                code="bayer_pattern_inconsistent",
                message="BAYERPAT differs across frames; bayer_pattern requires user confirmation",
            )
        )

    manifest_obj: dict[str, Any] = {
        "version": 1,
        "input_path": str(in_dir.resolve()),
        "frames": frames,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "frames_detected": int(frames_detected),
        "color_mode": color_mode,
        "bayer_pattern": bayer_pattern,
        "color_mode_candidates": color_mode_candidates,
        "requires_user_confirmation": bool(requires_user_confirmation),
        "with_checksums": bool(with_checksums),
    }

    manifest_bytes = _canonical_json_bytes(manifest_obj)
    frames_manifest_id = _sha256_bytes(manifest_bytes)

    cache_root = Path(project_cache_dir) if project_cache_dir is not None else _default_project_cache_dir(in_dir)
    cache_dir = cache_root / "cache" / "frames_manifests"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_manifest_path = cache_dir / f"{frames_manifest_id}.json"
    if not cache_manifest_path.exists():
        cache_manifest_path.write_bytes(manifest_bytes)

    errors = [i.__dict__ for i in issues if i.severity == "error"]
    warnings = [i.__dict__ for i in issues if i.severity == "warning"]

    ok = len(errors) == 0
    result: dict[str, Any] = {
        "ok": ok,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "frames_detected": int(frames_detected),
        "color_mode": color_mode,
        "bayer_pattern": bayer_pattern,
        "color_mode_candidates": color_mode_candidates,
        "requires_user_confirmation": bool(requires_user_confirmation),
        "frames_manifest_id": frames_manifest_id,
        "project_cache_dir": str(cache_root.resolve()),
        "frames_manifest_path": str(cache_manifest_path.resolve()),
        "errors": errors,
        "warnings": warnings,
    }
    return result
