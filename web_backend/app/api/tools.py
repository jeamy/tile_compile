from __future__ import annotations

import bz2
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request

from app.schemas import JobAccepted
from app.services.command_runner import launch_background_command

router = APIRouter(prefix="/tools", tags=["tools"])

ASTAP_CLI_URL = (
    "https://sourceforge.net/projects/astap-program/files/"
    "linux_installer/astap_command-line_version_Linux_amd64.zip/download"
)
ASTAP_CATALOGS: dict[str, dict[str, str]] = {
    "d05": {"filename": "d05_star_database.zip", "description": "smallest"},
    "d20": {"filename": "d20_star_database.zip", "description": "medium"},
    "d50": {"filename": "d50_star_database.zip", "description": "recommended"},
    "d80": {"filename": "d80_star_database.deb", "description": "largest"},
}
ASTAP_SF_BASE = "https://sourceforge.net/projects/astap-program/files/star_databases/"
SIRIL_NUM_CHUNKS = 48
SIRIL_URL_TEMPLATE = "https://zenodo.org/records/14738271/files/siril_cat1_healpix8_xpsamp_{chunk}.dat.bz2?download=1"


@router.post("/astrometry/detect")
def astrometry_detect(payload: dict[str, Any]) -> dict[str, Any]:
    astap_data_dir = _resolve_astap_data_dir(payload)
    catalog_dir = Path(str(payload.get("catalog_dir", ""))).expanduser() if payload.get("catalog_dir") else astap_data_dir
    astap_bin = _resolve_astap_bin(payload, astap_data_dir)

    catalog_states: dict[str, bool] = {}
    for cat_id in ASTAP_CATALOGS:
        catalog_states[cat_id] = _is_astap_catalog_installed(catalog_dir, cat_id)

    return {
        "installed": astap_bin is not None and astap_bin.exists(),
        "binary": str(astap_bin) if astap_bin else "",
        "data_dir": str(astap_data_dir),
        "catalog_dir": str(catalog_dir),
        "catalogs": catalog_states,
    }


@router.post("/astrometry/install-cli", response_model=JobAccepted)
def astrometry_install(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    data_dir = _resolve_astap_data_dir(body)
    url = str(body.get("url") or ASTAP_CLI_URL)

    def _worker(job_id: str) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "astap_cli.zip"
        request.app.state.job_store.merge_data(job_id, {"stage": "download", "url": url, "data_dir": str(data_dir)})
        _download_file(
            url,
            archive_path,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            timeout_s=1800,
        )

        request.app.state.job_store.merge_data(job_id, {"stage": "extract", "archive": str(archive_path)})
        _safe_extract_zip(archive_path, data_dir)
        archive_path.unlink(missing_ok=True)

        target = data_dir / "astap_cli"
        if not target.exists():
            candidate = _find_astap_candidate(data_dir)
            if candidate is not None:
                shutil.copy2(candidate, target)
        if not target.exists():
            raise RuntimeError("astap_cli executable not found after extraction")
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        request.app.state.job_store.merge_data(job_id, {"stage": "done", "binary": str(target)})

    return _start_custom_job(request, "astrometry_install_cli", {"payload": body}, _worker)


@router.post("/astrometry/catalog/download", response_model=JobAccepted)
def astrometry_catalog_download(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    catalog_id = str(body.get("catalog_id", "d50")).lower()
    if catalog_id not in ASTAP_CATALOGS:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": f"unknown catalog_id '{catalog_id}'"}},
        )

    data_dir = _resolve_astap_data_dir(body)
    filename = ASTAP_CATALOGS[catalog_id]["filename"]
    url = str(body.get("url") or f"{ASTAP_SF_BASE}{filename}/download")

    def _worker(job_id: str) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / filename
        request.app.state.job_store.merge_data(
            job_id,
            {"stage": "download", "catalog_id": catalog_id, "url": url, "archive": str(archive_path)},
        )
        _download_file(
            url,
            archive_path,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            timeout_s=1800,
        )

        request.app.state.job_store.merge_data(job_id, {"stage": "extract"})
        if archive_path.suffix.lower() == ".zip":
            _safe_extract_zip(archive_path, data_dir)
        elif archive_path.suffix.lower() == ".deb":
            _extract_deb_catalog(archive_path, data_dir, catalog_id)
        else:
            raise RuntimeError(f"unsupported archive format: {archive_path.name}")
        archive_path.unlink(missing_ok=True)
        request.app.state.job_store.merge_data(job_id, {"stage": "done", "installed": _is_astap_catalog_installed(data_dir, catalog_id)})

    return _start_custom_job(
        request,
        "astrometry_catalog_download",
        {"payload": body, "catalog_id": catalog_id},
        _worker,
    )


@router.post("/astrometry/catalog/cancel")
def astrometry_catalog_cancel(request: Request) -> dict[str, Any]:
    cancelled = False
    for job in request.app.state.job_store.list():
        if job.job_type == "astrometry_catalog_download" and job.state == "running":
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
    return {"ok": cancelled}


@router.post("/astrometry/solve", response_model=JobAccepted)
def astrometry_solve(request: Request, payload: dict[str, Any]) -> JobAccepted:
    solve_file = str(payload.get("solve_file", "")).strip()
    if not solve_file:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "solve_file is required"}},
        )
    fits_path = Path(solve_file).expanduser()
    if not fits_path.exists():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": f"solve_file not found: {fits_path}"}},
        )

    astap_data_dir = _resolve_astap_data_dir(payload)
    astap_bin = _resolve_astap_bin(payload, astap_data_dir)
    if astap_bin is None or not astap_bin.exists():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "ASTAP CLI not found; install or provide astap_cli path"}},
        )

    search_radius = int(payload.get("search_radius_deg", 180))
    wcs_path = _guess_wcs_path(fits_path)
    cmd = [
        str(astap_bin),
        "-f",
        str(fits_path),
        "-d",
        str(astap_data_dir),
        "-r",
        str(search_radius),
    ]
    job = request.app.state.job_store.create(
        "astrometry_solve",
        {"payload": payload, "command": cmd, "wcs_path": str(wcs_path)},
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=request.app.state.runtime.project_root,
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.post("/astrometry/save-solved")
def astrometry_save_solved(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = payload.get("input_path")
    output_path = payload.get("output_path")
    if not input_path or not output_path:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_path and output_path are required"}},
        )
    src = Path(str(input_path)).expanduser()
    dst = Path(str(output_path)).expanduser()
    if not src.exists():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": f"input_path not found: {src}"}},
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    wcs_path = payload.get("wcs_path")
    copied_wcs = None
    if wcs_path:
        src_wcs = Path(str(wcs_path)).expanduser()
        if src_wcs.exists():
            dst_wcs = dst.with_suffix(".wcs")
            shutil.copy2(src_wcs, dst_wcs)
            copied_wcs = str(dst_wcs)
    return {"output_path": str(dst), "wcs_path": copied_wcs}


@router.get("/pcc/siril/status")
def pcc_siril_status(catalog_dir: str | None = None) -> dict[str, Any]:
    path = Path(catalog_dir).expanduser() if catalog_dir else _default_siril_catalog_dir()
    missing = _missing_siril_chunks(path)
    installed = SIRIL_NUM_CHUNKS - len(missing)
    return {"installed": installed, "total": SIRIL_NUM_CHUNKS, "missing": missing, "catalog_dir": str(path)}


@router.post("/pcc/siril/download-missing", response_model=JobAccepted)
def pcc_siril_download_missing(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    catalog_dir = Path(str(body.get("catalog_dir", _default_siril_catalog_dir()))).expanduser()
    chunk_ids_raw = body.get("chunk_ids")
    max_chunks = int(body.get("max_chunks", 0))

    chunk_ids: list[int] | None = None
    if isinstance(chunk_ids_raw, list):
        chunk_ids = []
        for item in chunk_ids_raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < SIRIL_NUM_CHUNKS:
                chunk_ids.append(idx)

    def _worker(job_id: str) -> None:
        catalog_dir.mkdir(parents=True, exist_ok=True)
        missing = _missing_siril_chunks(catalog_dir)
        if chunk_ids is not None:
            missing = [i for i in missing if i in set(chunk_ids)]
        if max_chunks > 0:
            missing = missing[:max_chunks]

        request.app.state.job_store.merge_data(
            job_id,
            {"catalog_dir": str(catalog_dir), "pending_chunks": missing, "total_chunks": len(missing)},
        )

        for i, chunk in enumerate(missing):
            if _is_job_cancelled(request, job_id):
                return
            request.app.state.job_store.merge_data(job_id, {"current_chunk": chunk, "current_index": i})
            _download_siril_chunk(request, job_id, catalog_dir, chunk)
            request.app.state.job_store.merge_data(job_id, {"completed_chunks": i + 1})

        request.app.state.job_store.merge_data(job_id, {"pending_chunks": [], "missing_after": _missing_siril_chunks(catalog_dir)})

    return _start_custom_job(
        request,
        "pcc_siril_download",
        {"payload": body},
        _worker,
    )


@router.post("/pcc/siril/cancel")
def pcc_siril_cancel(request: Request) -> dict[str, Any]:
    cancelled = False
    for job in request.app.state.job_store.list():
        if job.job_type == "pcc_siril_download" and job.state == "running":
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
    return {"ok": cancelled}


@router.post("/pcc/check-online")
def pcc_check_online() -> dict[str, Any]:
    test_url = (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
        "-source=I/355/gaiadr3&-c=0%200&-c.rd=0.01&-out=RA_ICRS,DE_ICRS,Gmag&-out.max=1"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(test_url, timeout=10) as resp:
            _ = resp.read(2048)
        latency = int((time.perf_counter() - t0) * 1000)
        return {"ok": True, "latency_ms": latency}
    except Exception as exc:
        latency = int((time.perf_counter() - t0) * 1000)
        return {"ok": False, "latency_ms": latency, "error": str(exc)}


@router.post("/pcc/run", response_model=JobAccepted)
def pcc_run(request: Request, payload: dict[str, Any]) -> JobAccepted:
    runtime = request.app.state.runtime
    input_rgb = payload.get("input_rgb")
    output_rgb = payload.get("output_rgb")
    if not input_rgb or not output_rgb:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_rgb and output_rgb are required"}},
        )

    r_scale = float(payload.get("r", 1.0))
    g_scale = float(payload.get("g", 1.0))
    b_scale = float(payload.get("b", 1.0))
    cmd = [
        str(runtime.cli_path),
        "pcc-apply",
        str(input_rgb),
        str(output_rgb),
        "--r",
        str(r_scale),
        "--g",
        str(g_scale),
        "--b",
        str(b_scale),
    ]
    job = request.app.state.job_store.create("pcc_run", {"payload": payload, "command": cmd})
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.post("/pcc/save-corrected")
def pcc_save_corrected(payload: dict[str, Any]) -> dict[str, Any]:
    output_rgb = payload.get("output_rgb")
    if not output_rgb:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "output_rgb is required"}},
        )
    output_channels = payload.get("output_channels", [])
    return {"output_rgb": str(output_rgb), "output_channels": output_channels}


def _start_custom_job(
    request: Request,
    job_type: str,
    data: dict[str, Any],
    worker_fn: Callable[[str], None],
) -> JobAccepted:
    job = request.app.state.job_store.create(job_type, data)
    request.app.state.job_store.set_state(job.job_id, "running")

    def _runner() -> None:
        try:
            worker_fn(job.job_id)
            current = request.app.state.job_store.get(job.job_id)
            if current and current.state != "cancelled":
                request.app.state.job_store.set_state(job.job_id, "ok")
        except _JobCancelled:
            request.app.state.job_store.set_state(job.job_id, "cancelled")
        except Exception as exc:
            request.app.state.job_store.merge_data(job.job_id, {"error": str(exc)})
            current = request.app.state.job_store.get(job.job_id)
            if current and current.state != "cancelled":
                request.app.state.job_store.set_state(job.job_id, "error")

    thread = threading.Thread(target=_runner, name=f"{job_type}-{job.job_id}", daemon=True)
    thread.start()
    return JobAccepted(job_id=job.job_id, state="running")


def _resolve_astap_data_dir(payload: dict[str, Any]) -> Path:
    if payload.get("astap_data_dir"):
        return Path(str(payload["astap_data_dir"])).expanduser()
    return _default_astap_data_dir()


def _resolve_astap_bin(payload: dict[str, Any], astap_data_dir: Path) -> Path | None:
    if payload.get("astap_cli"):
        return Path(str(payload["astap_cli"])).expanduser()
    default = astap_data_dir / "astap_cli"
    if default.exists():
        return default
    for candidate in [shutil.which("astap_cli"), shutil.which("astap")]:
        if candidate:
            return Path(candidate)
    return None


def _default_astap_data_dir() -> Path:
    return Path.home() / ".local/share/tile_compile/astap"


def _default_siril_catalog_dir() -> Path:
    return Path.home() / ".local/share/siril/siril_cat1_healpix8_xpsamp"


def _is_astap_catalog_installed(catalog_dir: Path, catalog_id: str) -> bool:
    if not catalog_dir.exists():
        return False
    pattern = f"{catalog_id}_*"
    return any(catalog_dir.glob(pattern))


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError("unsafe archive entry")
        zf.extractall(dest_dir)


def _find_astap_candidate(data_dir: Path) -> Path | None:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "astap" in name and not name.endswith((".zip", ".deb", ".txt", ".md")):
            return path
    return None


def _extract_deb_catalog(archive_path: Path, data_dir: Path, catalog_id: str) -> None:
    dpkg = shutil.which("dpkg-deb")
    if not dpkg:
        raise RuntimeError("dpkg-deb is required to extract .deb catalog archives")
    with tempfile.TemporaryDirectory(prefix="astap_deb_") as tmp:
        tmp_dir = Path(tmp)
        proc = subprocess.run([dpkg, "-x", str(archive_path), str(tmp_dir)], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"dpkg-deb failed: {proc.stderr.strip()}")
        moved = 0
        for file in tmp_dir.rglob(f"{catalog_id}_*"):
            if file.is_file():
                shutil.move(str(file), str(data_dir / file.name))
                moved += 1
        if moved == 0:
            raise RuntimeError("no catalog files found inside .deb")


def _set_download_progress(request: Request, job_id: str, received: int, total: int) -> None:
    progress = None
    if total > 0:
        progress = max(0.0, min(1.0, float(received) / float(total)))
    request.app.state.job_store.merge_data(job_id, {"bytes_received": int(received), "bytes_total": int(total), "progress": progress})
    if _is_job_cancelled(request, job_id):
        raise _JobCancelled()


def _download_file(
    url: str,
    dest_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
    timeout_s: int = 120,
) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "TileCompileGUI2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        received = 0
        with dest_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if progress_cb:
                    progress_cb(received, total)


def _guess_wcs_path(fits_path: Path) -> Path:
    name = fits_path.name
    lower = name.lower()
    for ext in [".fits.fz", ".fit.fz", ".fts.fz", ".fits", ".fit", ".fts"]:
        if lower.endswith(ext):
            return fits_path.with_name(name[: -len(ext)] + ".wcs")
    return fits_path.with_suffix(fits_path.suffix + ".wcs")


def _missing_siril_chunks(catalog_dir: Path) -> list[int]:
    missing: list[int] = []
    for i in range(SIRIL_NUM_CHUNKS):
        path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{i}.dat"
        if not path.exists():
            missing.append(i)
    return missing


def _download_siril_chunk(request: Request, job_id: str, catalog_dir: Path, chunk_id: int) -> None:
    dat_path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{chunk_id}.dat"
    if dat_path.exists():
        return
    bz2_path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{chunk_id}.dat.bz2"
    url = SIRIL_URL_TEMPLATE.format(chunk=chunk_id)
    try:
        _download_file(
            url,
            bz2_path,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            timeout_s=3600,
        )
        request.app.state.job_store.merge_data(job_id, {"stage": "decompress", "chunk_id": chunk_id})
        with bz2.open(bz2_path, "rb") as src, dat_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 256)
    finally:
        bz2_path.unlink(missing_ok=True)


def _is_job_cancelled(request: Request, job_id: str) -> bool:
    job = request.app.state.job_store.get(job_id)
    return bool(job and job.state == "cancelled")


class _JobCancelled(RuntimeError):
    pass
