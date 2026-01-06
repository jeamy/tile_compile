import io
import json
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

import tile_compile_runner as runner


def _write_cfa_fits(path: Path, shape: tuple[int, int] = (32, 32), bayer_pattern: str = "GBRG") -> None:
    data = np.random.normal(100.0, 5.0, shape).astype(np.float32)
    hdr = fits.Header()
    hdr["BAYERPAT"] = str(bayer_pattern)
    fits.writeto(str(path), data, header=hdr, overwrite=True)


def _parse_events(log_text: str) -> list[dict]:
    return [json.loads(line) for line in log_text.splitlines() if line.strip()]


def test_reduced_mode_skips_clustering_and_synthetic(tmp_path, monkeypatch):
    run_id = "test"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]

    frames: list[Path] = []
    for i in range(5):
        p = tmp_path / f"frame_{i:03d}.fit"
        _write_cfa_fits(p)
        frames.append(p)

    def _stub_generate_multi_channel_grid(images, grid_cfg):
        return {k: {"tile_size": grid_cfg["min_tile_size"], "overlap": grid_cfg["overlap"]} for k in images.keys()}

    def _stub_run_siril_script(siril_exe, work_dir, script_path, artifacts_dir, log_name, timeout_s=None, quiet=True):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / log_name).write_text("stub\n", encoding="utf-8")
        if "registration" in log_name:
            for seq in sorted(work_dir.glob("seq*.fit")):
                shutil.copy2(seq, work_dir / f"r_{seq.name}")
            return True, {"returncode": 0}
        if "stacking" in log_name:
            outp = work_dir / "stacked_average_reference.fits"
            data = np.random.normal(100.0, 1.0, (32, 32)).astype(np.float32)
            fits.writeto(str(outp), data, overwrite=True)
            return True, {"returncode": 0}
        return True, {"returncode": 0}

    monkeypatch.setattr(runner, "generate_multi_channel_grid", _stub_generate_multi_channel_grid)
    monkeypatch.setattr(runner, "_run_siril_script", _stub_run_siril_script)

    cfg = {
        "data": {"frames_min": 1, "color_mode": "OSC", "bayer_pattern": "GBRG"},
        "registration": {"engine": "siril", "output_dir": "registered", "registered_filename_pattern": "reg_{index:05d}.fit"},
        "stacking": {
            "engine": "siril",
            "method": "average",
            "input_dir": "registered",
            "input_pattern": "reg_*.fit",
            "output_file": "stacked.fit",
        },
        "assumptions": {"frames_reduced_threshold": 6, "reduced_mode_skip_clustering": True},
    }

    log_fp = io.StringIO()
    ok = runner._run_phases(
        run_id=run_id,
        log_fp=log_fp,
        dry_run=False,
        run_dir=run_dir,
        project_root=project_root,
        cfg=cfg,
        frames=frames,
        siril_exe="siril",
    )
    assert ok is True

    events = _parse_events(log_fp.getvalue())
    clustering_end = next(e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "STATE_CLUSTERING")
    synthetic_end = next(e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "SYNTHETIC_FRAMES")

    assert clustering_end.get("reduced_mode") is True
    assert clustering_end.get("skipped") is True
    assert clustering_end.get("enabled") is False

    assert synthetic_end.get("reduced_mode") is True
    assert synthetic_end.get("skipped") is True
    assert synthetic_end.get("enabled") is False


def test_reduced_mode_uses_reduced_cluster_range_when_not_skipping(tmp_path, monkeypatch):
    run_id = "test"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]

    frames: list[Path] = []
    for i in range(5):
        p = tmp_path / f"frame_{i:03d}.fit"
        _write_cfa_fits(p)
        frames.append(p)

    def _stub_generate_multi_channel_grid(images, grid_cfg):
        return {k: {"tile_size": grid_cfg["min_tile_size"], "overlap": grid_cfg["overlap"]} for k in images.keys()}

    def _stub_run_siril_script(siril_exe, work_dir, script_path, artifacts_dir, log_name, timeout_s=None, quiet=True):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / log_name).write_text("stub\n", encoding="utf-8")
        if "registration" in log_name:
            for seq in sorted(work_dir.glob("seq*.fit")):
                shutil.copy2(seq, work_dir / f"r_{seq.name}")
            return True, {"returncode": 0}
        if "stacking" in log_name:
            outp = work_dir / "stacked_average_reference.fits"
            data = np.random.normal(100.0, 1.0, (32, 32)).astype(np.float32)
            fits.writeto(str(outp), data, overwrite=True)
            return True, {"returncode": 0}
        return True, {"returncode": 0}

    seen: dict = {}

    def _stub_cluster_channels(channels, channel_metrics, clustering_cfg):
        seen["cluster_count_range"] = clustering_cfg.get("cluster_count_range")
        return {"R": {"n_clusters": 5}, "G": {"n_clusters": 5}, "B": {"n_clusters": 5}}

    monkeypatch.setattr(runner, "generate_multi_channel_grid", _stub_generate_multi_channel_grid)
    monkeypatch.setattr(runner, "_run_siril_script", _stub_run_siril_script)
    monkeypatch.setattr(runner, "cluster_channels", _stub_cluster_channels)

    cfg = {
        "data": {"frames_min": 1, "color_mode": "OSC", "bayer_pattern": "GBRG"},
        "registration": {"engine": "siril", "output_dir": "registered", "registered_filename_pattern": "reg_{index:05d}.fit"},
        "synthetic": {"frames_min": 1, "frames_max": 3, "clustering": {"cluster_count_range": [15, 30]}},
        "stacking": {
            "engine": "siril",
            "method": "average",
            "input_dir": "registered",
            "input_pattern": "reg_*.fit",
            "output_file": "stacked.fit",
        },
        "assumptions": {
            "frames_reduced_threshold": 6,
            "reduced_mode_skip_clustering": False,
            "reduced_mode_cluster_range": [5, 10],
        },
    }

    log_fp = io.StringIO()
    ok = runner._run_phases(
        run_id=run_id,
        log_fp=log_fp,
        dry_run=False,
        run_dir=run_dir,
        project_root=project_root,
        cfg=cfg,
        frames=frames,
        siril_exe="siril",
    )
    assert ok is True
    assert seen.get("cluster_count_range") == [5, 10]

    events = _parse_events(log_fp.getvalue())
    clustering_end = next(e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "STATE_CLUSTERING")
    synthetic_end = next(e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "SYNTHETIC_FRAMES")

    assert clustering_end.get("reduced_mode") is True
    assert clustering_end.get("skipped") is False
    assert clustering_end.get("enabled") is True

    assert synthetic_end.get("reduced_mode") is True
    assert synthetic_end.get("skipped") is False
