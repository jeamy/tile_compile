from __future__ import annotations

import bz2
import math
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
from app.services.command_runner import SecurityPolicyError, launch_background_command, run_command
from app.services.downloads import DownloadAborted, DownloadOptions, download_file_with_retry
from app.services.http_errors import http_from_security_error
from app.services.ui_events import record_ui_event

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
def astrometry_detect(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        astap_data_dir = _resolve_astap_data_dir(payload, runtime)
        catalog_dir = (
            runtime.ensure_path_allowed(Path(str(payload.get("catalog_dir", ""))).expanduser(), label="catalog_dir")
            if payload.get("catalog_dir")
            else astap_data_dir
        )
        astap_bin = _resolve_astap_bin(payload, astap_data_dir, runtime)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc

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


@router.post("/astrometry/install-cli", response_model=JobAccepted, status_code=202)
def astrometry_install(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    runtime = request.app.state.runtime
    try:
        data_dir = _resolve_astap_data_dir(body, runtime)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    url = str(body.get("url") or ASTAP_CLI_URL)
    options = _download_options_from_payload(body, default_timeout_s=1800)
    force_restart = bool(body.get("force_restart", False))

    def _worker(job_id: str) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "astap_cli.zip"
        if force_restart:
            archive_path.unlink(missing_ok=True)
        request.app.state.job_store.merge_data(
            job_id,
            {
                "stage": "download",
                "url": url,
                "data_dir": str(data_dir),
                "resume_enabled": options.resume,
                "retry_count": options.retry_count,
            },
        )
        download_file_with_retry(
            url,
            archive_path,
            options=options,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            state_cb=lambda patch: _set_download_state(request, job_id, patch),
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

    accepted = _start_custom_job(request, "astrometry_install_cli", {"payload": body}, _worker)
    record_ui_event(
        request,
        event="tools.astrometry.install_cli",
        source="tools.astrometry_install",
        job_id=accepted.job_id,
        payload={"data_dir": str(data_dir)},
    )
    return accepted


@router.post("/astrometry/catalog/download", response_model=JobAccepted, status_code=202)
def astrometry_catalog_download(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    catalog_id = str(body.get("catalog_id", "d50")).lower()
    if catalog_id not in ASTAP_CATALOGS:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": f"unknown catalog_id '{catalog_id}'"}},
        )

    runtime = request.app.state.runtime
    try:
        data_dir = _resolve_astap_data_dir(body, runtime)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    filename = ASTAP_CATALOGS[catalog_id]["filename"]
    url = str(body.get("url") or f"{ASTAP_SF_BASE}{filename}/download")
    options = _download_options_from_payload(body, default_timeout_s=1800)
    force_restart = bool(body.get("force_restart", False))

    def _worker(job_id: str) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / filename
        if force_restart:
            archive_path.unlink(missing_ok=True)
        request.app.state.job_store.merge_data(
            job_id,
            {
                "stage": "download",
                "catalog_id": catalog_id,
                "url": url,
                "archive": str(archive_path),
                "resume_enabled": options.resume,
                "retry_count": options.retry_count,
            },
        )
        download_file_with_retry(
            url,
            archive_path,
            options=options,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            state_cb=lambda patch: _set_download_state(request, job_id, patch),
        )

        request.app.state.job_store.merge_data(job_id, {"stage": "extract"})
        if archive_path.suffix.lower() == ".zip":
            _safe_extract_zip(archive_path, data_dir)
        elif archive_path.suffix.lower() == ".deb":
            _extract_deb_catalog(archive_path, data_dir, catalog_id, command_policy=request.app.state.command_policy)
        else:
            raise RuntimeError(f"unsupported archive format: {archive_path.name}")
        archive_path.unlink(missing_ok=True)
        request.app.state.job_store.merge_data(job_id, {"stage": "done", "installed": _is_astap_catalog_installed(data_dir, catalog_id)})

    accepted = _start_custom_job(
        request,
        "astrometry_catalog_download",
        {"payload": body, "catalog_id": catalog_id},
        _worker,
    )
    record_ui_event(
        request,
        event="tools.astrometry.catalog.download",
        source="tools.astrometry_catalog_download",
        job_id=accepted.job_id,
        payload={"catalog_id": catalog_id, "data_dir": str(data_dir)},
    )
    return accepted


@router.post("/astrometry/install-cli/retry", response_model=JobAccepted, status_code=202)
def astrometry_install_retry(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = dict(payload or {})
    body.setdefault("resume", True)
    return astrometry_install(request, body)


@router.post("/astrometry/catalog/download/retry", response_model=JobAccepted, status_code=202)
def astrometry_catalog_download_retry(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = dict(payload or {})
    body.setdefault("resume", True)
    return astrometry_catalog_download(request, body)


@router.post("/astrometry/catalog/cancel")
def astrometry_catalog_cancel(request: Request) -> dict[str, Any]:
    cancelled = False
    for job in request.app.state.job_store.list():
        if job.job_type == "astrometry_catalog_download" and job.state == "running":
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
            record_ui_event(
                request,
                event="tools.astrometry.catalog.cancel",
                source="tools.astrometry_catalog_cancel",
                job_id=job.job_id,
                payload={"ok": True},
            )
    return {"ok": cancelled}


@router.post("/astrometry/solve", response_model=JobAccepted, status_code=202)
def astrometry_solve(request: Request, payload: dict[str, Any]) -> JobAccepted:
    solve_file = str(payload.get("solve_file", "")).strip()
    if not solve_file:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "solve_file is required"}},
        )
    runtime = request.app.state.runtime
    try:
        fits_path = runtime.ensure_path_allowed(Path(solve_file).expanduser(), must_exist=True, label="solve_file")
        astap_data_dir = _resolve_astap_data_dir(payload, runtime)
        astap_bin = _resolve_astap_bin(payload, astap_data_dir, runtime)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
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

    def _worker(job_id: str) -> None:
        result = run_command(
            cmd,
            cwd=runtime.project_root,
            timeout_sec=max(30, int(payload.get("timeout_s", 300))),
            command_policy=request.app.state.command_policy,
        )
        request.app.state.job_store.set_exit_code(job_id, result.exit_code)
        data_patch: dict[str, Any] = {
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "wcs_path": str(wcs_path),
        }
        if result.exit_code != 0:
            request.app.state.job_store.merge_data(job_id, data_patch)
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ASTAP solve failed")
        if not wcs_path.exists():
            request.app.state.job_store.merge_data(job_id, data_patch)
            raise RuntimeError("ASTAP solve completed without producing a WCS file")
        summary = _parse_astrometry_wcs_summary(wcs_path)
        data_patch["result"] = {"wcs_path": str(wcs_path), **summary}
        request.app.state.job_store.merge_data(job_id, data_patch)

    accepted = _start_custom_job(
        request,
        "astrometry_solve",
        {"payload": payload, "command": cmd, "wcs_path": str(wcs_path)},
        _worker,
    )
    record_ui_event(
        request,
        event="tools.astrometry.solve",
        source="tools.astrometry_solve",
        job_id=accepted.job_id,
        payload={"solve_file": str(fits_path), "wcs_path": str(wcs_path)},
    )
    return accepted


@router.post("/astrometry/save-solved")
def astrometry_save_solved(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    input_path = payload.get("input_path")
    output_path = payload.get("output_path")
    if not input_path or not output_path:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_path and output_path are required"}},
        )
    runtime = request.app.state.runtime
    try:
        src = runtime.ensure_path_allowed(Path(str(input_path)).expanduser(), must_exist=True, label="input_path")
        dst = runtime.ensure_path_allowed(Path(str(output_path)).expanduser(), label="output_path")
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    wcs_path = payload.get("wcs_path")
    copied_wcs = None
    if wcs_path:
        src_wcs = runtime.ensure_path_allowed(Path(str(wcs_path)).expanduser(), label="wcs_path")
        if src_wcs.exists():
            dst_wcs = dst.with_suffix(".wcs")
            shutil.copy2(src_wcs, dst_wcs)
            copied_wcs = str(dst_wcs)
    record_ui_event(
        request,
        event="tools.astrometry.save_solved",
        source="tools.astrometry_save_solved",
        payload={"output_path": str(dst), "wcs_path": copied_wcs},
    )
    return {"output_path": str(dst), "wcs_path": copied_wcs}


@router.get("/pcc/siril/status")
def pcc_siril_status(request: Request, catalog_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        path = (
            runtime.ensure_path_allowed(Path(catalog_dir).expanduser(), label="catalog_dir")
            if catalog_dir
            else runtime.ensure_path_allowed(_default_siril_catalog_dir(), label="catalog_dir")
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    missing = _missing_siril_chunks(path)
    installed = SIRIL_NUM_CHUNKS - len(missing)
    return {"installed": installed, "total": SIRIL_NUM_CHUNKS, "missing": missing, "catalog_dir": str(path)}


@router.post("/pcc/siril/download-missing", response_model=JobAccepted, status_code=202)
def pcc_siril_download_missing(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = payload or {}
    runtime = request.app.state.runtime
    try:
        catalog_dir = runtime.ensure_path_allowed(
            Path(str(body.get("catalog_dir", _default_siril_catalog_dir()))).expanduser(),
            label="catalog_dir",
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    chunk_ids_raw = body.get("chunk_ids")
    max_chunks = int(body.get("max_chunks", 0))
    options = _download_options_from_payload(body, default_timeout_s=3600)
    force_restart = bool(body.get("force_restart", False))

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
            {
                "catalog_dir": str(catalog_dir),
                "pending_chunks": missing,
                "total_chunks": len(missing),
                "resume_enabled": options.resume,
                "retry_count": options.retry_count,
            },
        )

        for i, chunk in enumerate(missing):
            if _is_job_cancelled(request, job_id):
                return
            request.app.state.job_store.merge_data(job_id, {"current_chunk": chunk, "current_index": i})
            _download_siril_chunk(
                request,
                job_id,
                catalog_dir,
                chunk,
                options=options,
                force_restart=force_restart,
            )
            request.app.state.job_store.merge_data(job_id, {"completed_chunks": i + 1})

        request.app.state.job_store.merge_data(job_id, {"pending_chunks": [], "missing_after": _missing_siril_chunks(catalog_dir)})

    accepted = _start_custom_job(
        request,
        "pcc_siril_download",
        {"payload": body},
        _worker,
    )
    record_ui_event(
        request,
        event="tools.pcc.siril.download_missing",
        source="tools.pcc_siril_download_missing",
        job_id=accepted.job_id,
        payload={"catalog_dir": str(catalog_dir)},
    )
    return accepted


@router.post("/pcc/siril/download-missing/retry", response_model=JobAccepted, status_code=202)
def pcc_siril_download_missing_retry(request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    body = dict(payload or {})
    body.setdefault("resume", True)
    return pcc_siril_download_missing(request, body)


@router.post("/pcc/siril/cancel")
def pcc_siril_cancel(request: Request) -> dict[str, Any]:
    cancelled = False
    for job in request.app.state.job_store.list():
        if job.job_type == "pcc_siril_download" and job.state == "running":
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
            record_ui_event(
                request,
                event="tools.pcc.siril.cancel",
                source="tools.pcc_siril_cancel",
                job_id=job.job_id,
                payload={"ok": True},
            )
    return {"ok": cancelled}


@router.post("/pcc/check-online")
def pcc_check_online(request: Request) -> dict[str, Any]:
    test_url = (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
        "-source=I/355/gaiadr3&-c=0%200&-c.rd=0.01&-out=RA_ICRS,DE_ICRS,Gmag&-out.max=1"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(test_url, timeout=10) as resp:
            _ = resp.read(2048)
        latency = int((time.perf_counter() - t0) * 1000)
        response = {"ok": True, "latency_ms": latency}
        record_ui_event(
            request,
            event="tools.pcc.check_online",
            source="tools.pcc_check_online",
            payload=response,
        )
        return response
    except Exception as exc:
        latency = int((time.perf_counter() - t0) * 1000)
        response = {"ok": False, "latency_ms": latency, "error": str(exc)}
        record_ui_event(
            request,
            event="tools.pcc.check_online",
            source="tools.pcc_check_online",
            payload=response,
        )
        return response


def _payload_text(payload: dict[str, Any], key: str, *, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _payload_float(payload: dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "BAD_REQUEST", "message": f"invalid float for '{key}'"}},
        ) from exc


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "BAD_REQUEST", "message": f"invalid integer for '{key}'"}},
        ) from exc


def _payload_bool(payload: dict[str, Any], key: str) -> bool | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(
        status_code=422,
        detail={"error": {"code": "BAD_REQUEST", "message": f"invalid boolean for '{key}'"}},
    )


def _download_options_from_payload(payload: dict[str, Any], *, default_timeout_s: int) -> DownloadOptions:
    retry_count = int(payload.get("retry_count", 2))
    retry_backoff_s = float(payload.get("retry_backoff_sec", 1.5))
    resume = bool(payload.get("resume", True))
    timeout_s = int(payload.get("timeout_s", default_timeout_s))
    if retry_count < 0:
        retry_count = 0
    if timeout_s < 1:
        timeout_s = default_timeout_s
    if retry_backoff_s < 0:
        retry_backoff_s = 0.0
    return DownloadOptions(
        timeout_s=timeout_s,
        retry_count=retry_count,
        retry_backoff_s=retry_backoff_s,
        resume=resume,
    )


@router.post("/pcc/run", response_model=JobAccepted, status_code=202)
def pcc_run(request: Request, payload: dict[str, Any]) -> JobAccepted:
    runtime = request.app.state.runtime
    input_rgb = payload.get("input_rgb")
    output_rgb = payload.get("output_rgb")
    wcs_file = payload.get("wcs_file")
    if not input_rgb or not output_rgb or not wcs_file:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_rgb, output_rgb and wcs_file are required"}},
        )

    source = _payload_text(payload, "source", default="auto").lower() or "auto"
    if source not in {"auto", "siril", "vizier_gaia", "vizier_apass"}:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "BAD_REQUEST", "message": f"unsupported pcc source '{source}'"}},
        )

    try:
        input_rgb = str(runtime.ensure_path_allowed(Path(str(input_rgb)).expanduser(), must_exist=True, label="input_rgb"))
        output_rgb = str(runtime.ensure_path_allowed(Path(str(output_rgb)).expanduser(), label="output_rgb"))
        wcs_file = str(runtime.ensure_path_allowed(Path(str(wcs_file)).expanduser(), must_exist=True, label="wcs_file"))
        catalog_dir = _payload_text(payload, "catalog_dir")
        resolved_catalog_dir = (
            str(runtime.ensure_path_allowed(Path(catalog_dir).expanduser(), label="catalog_dir"))
            if catalog_dir
            else ""
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc

    cmd = [
        str(runtime.cli_path),
        "pcc-run",
        str(input_rgb),
        str(output_rgb),
        "--wcs",
        str(wcs_file),
        "--source",
        source,
    ]
    if resolved_catalog_dir:
        cmd.extend(["--siril-catalog-dir", resolved_catalog_dir])

    numeric_options = [
        ("mag_limit", "--mag-limit", _payload_float(payload, "mag_limit")),
        ("mag_bright_limit", "--mag-bright-limit", _payload_float(payload, "mag_bright_limit")),
        ("min_stars", "--min-stars", _payload_int(payload, "min_stars")),
        ("sigma_clip", "--sigma-clip", _payload_float(payload, "sigma_clip")),
        ("aperture_radius_px", "--aperture-radius-px", _payload_float(payload, "aperture_radius_px")),
        ("annulus_inner_px", "--annulus-inner-px", _payload_float(payload, "annulus_inner_px")),
        ("annulus_outer_px", "--annulus-outer-px", _payload_float(payload, "annulus_outer_px")),
        ("chroma_strength", "--chroma-strength", _payload_float(payload, "chroma_strength")),
        ("k_max", "--k-max", _payload_float(payload, "k_max")),
    ]
    for _name, option, value in numeric_options:
        if value is None:
            continue
        cmd.extend([option, str(value)])

    apply_attenuation = _payload_bool(payload, "apply_attenuation")
    if apply_attenuation is not None:
        cmd.extend(["--apply-attenuation", "1" if apply_attenuation else "0"])

    job = request.app.state.job_store.create("pcc_run", {"payload": payload, "command": cmd})
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
        command_policy=request.app.state.command_policy,
    )
    record_ui_event(
        request,
        event="tools.pcc.run",
        source="tools.pcc_run",
        job_id=job.job_id,
        payload={
            "input_rgb": str(input_rgb),
            "output_rgb": str(output_rgb),
            "wcs_file": str(wcs_file),
            "source": source,
            "catalog_dir": resolved_catalog_dir,
        },
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.post("/pcc/save-corrected")
def pcc_save_corrected(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    output_rgb = payload.get("output_rgb")
    if not output_rgb:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "output_rgb is required"}},
        )
    runtime = request.app.state.runtime
    try:
        out_rgb = runtime.ensure_path_allowed(Path(str(output_rgb)).expanduser(), label="output_rgb")
        output_channels_raw = payload.get("output_channels", [])
        output_channels: list[str] = []
        if isinstance(output_channels_raw, list):
            for channel_path in output_channels_raw:
                checked = runtime.ensure_path_allowed(Path(str(channel_path)).expanduser(), label="output_channel")
                output_channels.append(str(checked))
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    response = {"output_rgb": str(out_rgb), "output_channels": output_channels}
    record_ui_event(
        request,
        event="tools.pcc.save_corrected",
        source="tools.pcc_save_corrected",
        payload=response,
    )
    return response


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
        except (DownloadAborted, _JobCancelled):
            request.app.state.job_store.set_state(job.job_id, "cancelled")
        except Exception as exc:
            request.app.state.job_store.merge_data(job.job_id, {"error": str(exc)})
            current = request.app.state.job_store.get(job.job_id)
            if current and current.state != "cancelled":
                request.app.state.job_store.set_state(job.job_id, "error")

    thread = threading.Thread(target=_runner, name=f"{job_type}-{job.job_id}", daemon=True)
    thread.start()
    return JobAccepted(job_id=job.job_id, state="running")


def _resolve_astap_data_dir(payload: dict[str, Any], runtime: Any) -> Path:
    if payload.get("astap_data_dir"):
        return runtime.ensure_path_allowed(Path(str(payload["astap_data_dir"])).expanduser(), label="astap_data_dir")
    return runtime.ensure_path_allowed(_default_astap_data_dir(), label="astap_data_dir")


def _resolve_astap_bin(payload: dict[str, Any], astap_data_dir: Path, runtime: Any) -> Path | None:
    if payload.get("astap_cli"):
        return runtime.ensure_path_allowed(Path(str(payload["astap_cli"])).expanduser(), label="astap_cli")
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


def _extract_deb_catalog(
    archive_path: Path,
    data_dir: Path,
    catalog_id: str,
    *,
    command_policy: Any,
) -> None:
    dpkg = shutil.which("dpkg-deb")
    if not dpkg:
        raise RuntimeError("dpkg-deb is required to extract .deb catalog archives")
    command_policy.validate([dpkg, "-x", str(archive_path), "tmp"])
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
        raise DownloadAborted()


def _set_download_state(request: Request, job_id: str, patch: dict[str, Any]) -> None:
    request.app.state.job_store.merge_data(job_id, patch)
    if _is_job_cancelled(request, job_id):
        raise DownloadAborted()


def _guess_wcs_path(fits_path: Path) -> Path:
    name = fits_path.name
    lower = name.lower()
    for ext in [".fits.fz", ".fit.fz", ".fts.fz", ".fits", ".fit", ".fts"]:
        if lower.endswith(ext):
            return fits_path.with_name(name[: -len(ext)] + ".wcs")
    return fits_path.with_suffix(fits_path.suffix + ".wcs")


def _parse_astrometry_wcs_summary(wcs_path: Path) -> dict[str, Any]:
    values: dict[str, float] = {}
    for line in wcs_path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key_raw, value_raw = line.split("=", 1)
        key = key_raw.strip()
        token = value_raw.split("/", 1)[0].strip().strip("'\"")
        if not key or not token:
            continue
        try:
            values[key] = float(token)
        except ValueError:
            continue

    crval1 = values.get("CRVAL1")
    crval2 = values.get("CRVAL2")
    crpix1 = values.get("CRPIX1", 0.0)
    crpix2 = values.get("CRPIX2", 0.0)
    naxis1_raw = values.get("NAXIS1", crpix1 * 2.0)
    naxis2_raw = values.get("NAXIS2", crpix2 * 2.0)
    naxis1 = int(round(naxis1_raw)) if naxis1_raw else 0
    naxis2 = int(round(naxis2_raw)) if naxis2_raw else 0

    cd11 = values.get("CD1_1")
    cd12 = values.get("CD1_2")
    cd21 = values.get("CD2_1")
    cd22 = values.get("CD2_2")
    if all(v is not None for v in (cd11, cd12, cd21, cd22)):
        scale_x = math.hypot(float(cd11), float(cd21))
        scale_y = math.hypot(float(cd12), float(cd22))
        pixel_scale_arcsec = ((scale_x + scale_y) / 2.0) * 3600.0
        rotation_deg = math.degrees(math.atan2(float(cd21), float(cd11)))
    else:
        cdelt1 = values.get("CDELT1", 0.0)
        cdelt2 = values.get("CDELT2", 0.0)
        pixel_scale_arcsec = ((abs(cdelt1) + abs(cdelt2)) / 2.0) * 3600.0 if (cdelt1 or cdelt2) else 0.0
        rotation_deg = values.get("CROTA2", values.get("CROTA1", 0.0))

    fov_width_deg = (pixel_scale_arcsec * naxis1 / 3600.0) if pixel_scale_arcsec and naxis1 else 0.0
    fov_height_deg = (pixel_scale_arcsec * naxis2 / 3600.0) if pixel_scale_arcsec and naxis2 else 0.0

    return {
        "ra_deg": crval1,
        "dec_deg": crval2,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "rotation_deg": rotation_deg,
        "fov_width_deg": fov_width_deg,
        "fov_height_deg": fov_height_deg,
        "image_width": naxis1,
        "image_height": naxis2,
    }


def _missing_siril_chunks(catalog_dir: Path) -> list[int]:
    missing: list[int] = []
    for i in range(SIRIL_NUM_CHUNKS):
        path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{i}.dat"
        if not path.exists():
            missing.append(i)
    return missing


def _download_siril_chunk(
    request: Request,
    job_id: str,
    catalog_dir: Path,
    chunk_id: int,
    *,
    options: DownloadOptions,
    force_restart: bool = False,
) -> None:
    dat_path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{chunk_id}.dat"
    if dat_path.exists():
        return
    bz2_path = catalog_dir / f"siril_cat1_healpix8_xpsamp_{chunk_id}.dat.bz2"
    if force_restart:
        bz2_path.unlink(missing_ok=True)
    url = SIRIL_URL_TEMPLATE.format(chunk=chunk_id)
    try:
        download_file_with_retry(
            url,
            bz2_path,
            options=options,
            progress_cb=lambda r, t: _set_download_progress(request, job_id, r, t),
            state_cb=lambda patch: _set_download_state(request, job_id, patch),
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
