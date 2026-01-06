"""
FITS file utilities for the tile-compile runner.

Functions for reading FITS headers, detecting CFA/Bayer patterns, and loading image data.
"""

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def is_fits_image_path(p: Path) -> bool:
    """Check if path has FITS extension."""
    suf = p.suffix.lower()
    return suf in {".fit", ".fits", ".fts"}


def fits_is_cfa(path: Path) -> bool | None:
    """Check if FITS file is CFA/Bayer mosaic."""
    try:
        hdr = fits.getheader(str(path), ext=0)
    except Exception:
        return None
    
    if not isinstance(hdr, fits.Header):
        return None
    
    return hdr.get("BAYERPAT") is not None


def fits_get_bayerpat(path: Path) -> str | None:
    """Get Bayer pattern from FITS header."""
    try:
        hdr = fits.getheader(str(path), ext=0)
    except Exception:
        return None
    
    if not isinstance(hdr, fits.Header):
        return None
    
    bp = hdr.get("BAYERPAT")
    if bp is None:
        return None
    return str(bp).strip().upper() if bp else None


def read_fits_float(path: Path) -> tuple[np.ndarray, Any]:
    """Read FITS file as float32 array with header."""
    hdr = fits.getheader(str(path), ext=0)
    data = fits.getdata(str(path), ext=0)
    if data is None:
        raise RuntimeError(f"no data in FITS: {path}")
    return np.asarray(data).astype("float32", copy=False), hdr


def siril_setext_from_suffix(suffix: str) -> str:
    """Convert file suffix to Siril SETEXT format."""
    s = (suffix or "").lower().lstrip(".")
    if s in {"fit", "fits", "fts"}:
        return "fit"
    return "fit"


def derive_prefix_from_pattern(pattern: str, default_prefix: str) -> str:
    """Derive filename prefix from glob pattern."""
    s = str(pattern or "").strip()
    if not s:
        return default_prefix
    s = s.replace("*", "").replace("?", "")
    if not s:
        return default_prefix
    parts = s.split(".")
    if len(parts) > 1:
        s = ".".join(parts[:-1])
    s = s.strip("_-")
    if not s:
        return default_prefix
    return default_prefix
