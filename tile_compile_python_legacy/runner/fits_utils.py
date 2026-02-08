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


