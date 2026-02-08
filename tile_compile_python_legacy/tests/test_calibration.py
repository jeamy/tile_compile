from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from runner.calibration import build_master_mean, prepare_flat, apply_calibration, bias_correct_dark


def _write_fits(fp: Path, arr: np.ndarray) -> None:
    fits.writeto(str(fp), np.asarray(arr, dtype=np.float32), overwrite=True)


def test_build_master_mean_from_fits(tmp_path: Path) -> None:
    a1 = np.full((4, 4), 10.0, dtype=np.float32)
    a2 = np.full((4, 4), 14.0, dtype=np.float32)
    a3 = np.full((4, 4), 18.0, dtype=np.float32)

    p1 = tmp_path / "b1.fit"
    p2 = tmp_path / "b2.fit"
    p3 = tmp_path / "b3.fit"
    _write_fits(p1, a1)
    _write_fits(p2, a2)
    _write_fits(p3, a3)

    master = build_master_mean([p1, p2, p3])
    assert master is not None
    m, _hdr = master

    expected = (a1 + a2 + a3) / 3.0
    assert m.shape == (4, 4)
    assert np.allclose(m, expected)


def test_prepare_flat_bias_dark_and_normalize(tmp_path: Path) -> None:
    bias = (np.full((2, 2), 2.0, dtype=np.float32), None)
    dark_raw = (np.full((2, 2), 5.0, dtype=np.float32), None)
    dark = bias_correct_dark(dark_raw, bias)
    assert dark is not None

    flat_raw = (np.full((2, 2), 12.0, dtype=np.float32), None)
    flat = prepare_flat(flat_raw, bias, dark)
    assert flat is not None
    flat_arr, _ = flat

    # flat_raw - bias(2) - dark(5-2=3) = 12-2-3=7 everywhere, median=7 => normalized flat = 1
    assert np.allclose(flat_arr, np.ones((2, 2), dtype=np.float32))


def test_apply_calibration_bias_dark_flat(tmp_path: Path) -> None:
    light = np.full((2, 2), 100.0, dtype=np.float32)
    bias_arr = np.full((2, 2), 10.0, dtype=np.float32)
    dark_arr = np.full((2, 2), 20.0, dtype=np.float32)
    flat_arr = np.full((2, 2), 2.0, dtype=np.float32)

    cal = apply_calibration(light, bias_arr, dark_arr, flat_arr)
    # (100 - 10 - 20) / 2 = 35
    assert np.allclose(cal, np.full((2, 2), 35.0, dtype=np.float32))


def test_apply_calibration_flat_zero_protection() -> None:
    light = np.full((2, 2), 10.0, dtype=np.float32)
    flat = np.array([[0.0, 1.0], [1e-12, -1.0]], dtype=np.float32)

    cal = apply_calibration(light, None, None, flat, denom_eps=1e-6)

    # denom elements |d|<eps are replaced with 1.0 -> result equals light there.
    # others divide normally
    expected = np.array(
        [[10.0 / 1.0, 10.0 / 1.0], [10.0 / 1.0, 10.0 / -1.0]],
        dtype=np.float32,
    )
    assert np.allclose(cal, expected)
