#!/usr/bin/env python3
"""
Convert RAW images (CR2, NEF, ARW, etc.) to FITS format.
Usage: python3 convert_raw2fits.py [input_dir] [output_dir] [--pattern "*.CR2"]

Dependencies:
    pip install rawpy astropy
"""

import argparse
import sys
from pathlib import Path

try:
    import rawpy
    import numpy as np
    from astropy.io import fits
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install rawpy astropy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None


VALID_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
BAYER_TO_OFFSETS = {
    "RGGB": (0, 0),
    "GRBG": (1, 0),
    "GBRG": (0, 1),
    "BGGR": (1, 1),
}


def detect_bayer_pattern_from_raw(raw) -> str | None:
    """Detect Bayer pattern for the *visible* RAW mosaic.

    Important: raw_pattern alone can be wrong for raw_image_visible when the
    visible area starts with odd top/left margins. Therefore we prefer
    raw_colors_visible (already aligned to visible pixels).
    """
    try:
        color_desc = getattr(raw, "color_desc", None)
        if color_desc is None:
            return None

        if isinstance(color_desc, bytes):
            desc = color_desc.decode("ascii", errors="ignore")
        else:
            desc = str(color_desc)
        desc = "".join(ch for ch in desc.upper() if ch.isalpha())
        if not desc:
            return None

        # Preferred path: directly inspect the visible CFA color index map.
        colors_visible = getattr(raw, "raw_colors_visible", None)
        if colors_visible is not None and getattr(colors_visible, "shape", (0, 0))[0] >= 2 and getattr(colors_visible, "shape", (0, 0))[1] >= 2:
            idx00 = int(colors_visible[0, 0])
            idx01 = int(colors_visible[0, 1])
            idx10 = int(colors_visible[1, 0])
            idx11 = int(colors_visible[1, 1])
            if min(idx00, idx01, idx10, idx11) >= 0 and max(idx00, idx01, idx10, idx11) < len(desc):
                detected = "".join([desc[idx00], desc[idx01], desc[idx10], desc[idx11]])
                if detected in VALID_BAYER_PATTERNS:
                    return detected

        # Fallback: use base raw_pattern (may be wrong when margins shift parity).
        pattern = getattr(raw, "raw_pattern", None)
        if pattern is None:
            return None

        cfa = []
        for y in range(2):
            for x in range(2):
                idx = int(pattern[y][x])
                if idx < 0 or idx >= len(desc):
                    return None
                cfa.append(desc[idx])

        detected = "".join(cfa)
        if detected in VALID_BAYER_PATTERNS:
            return detected
        return None
    except Exception:
        return None


def _bayer_to_cv2_code(pattern: str):
    if cv2 is None:
        return None
    mapping = {
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "BGGR": cv2.COLOR_BayerBG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
    }
    return mapping.get(pattern)


def _raw_reference_bgr(raw) -> np.ndarray | None:
    """Get a robust BGR reference image from RAW without OpenCV CR2 decoding.

    This avoids codec issues like "Old-style JPEG compression support is not
    configured" seen in some OpenCV builds when reading CR2 directly.
    """
    try:
        ref_rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
            half_size=True,
        )
    except Exception:
        return None

    if ref_rgb is None or getattr(ref_rgb, "ndim", 0) != 3:
        return None

    # rawpy postprocess returns RGB; convert to BGR for OpenCV routines.
    return ref_rgb[:, :, ::-1].copy()


def refine_bayer_pattern_with_cr2_preview(
    raw, mosaic_data: np.ndarray, selected_bayer: str
) -> tuple[str, str | None]:
    """Refine Bayer auto-detection by RAW-reference-vs-FITS structural match.

    Some cameras (notably Canon in certain modes) can be ambiguous between
    GBRG and GRBG depending on how decoders expose visible-area parity. We
    compare both candidates against a rawpy-rendered RAW reference and keep
    the better one.
    """
    if cv2 is None:
        return selected_bayer, None
    if selected_bayer not in {"GBRG", "GRBG"}:
        return selected_bayer, None

    preview = _raw_reference_bgr(raw)
    if preview is None or preview.ndim != 3:
        return selected_bayer, None

    alt = "GRBG" if selected_bayer == "GBRG" else "GBRG"

    m = mosaic_data.astype(np.float32)
    m_min = float(np.min(m))
    m_max = float(np.max(m))
    if not np.isfinite(m_min) or not np.isfinite(m_max) or m_max <= m_min:
        return selected_bayer, None
    m16 = ((m - m_min) / (m_max - m_min) * 65535.0).clip(0, 65535).astype(np.uint16)

    # Downsample aggressively for speed (only relative ranking matters).
    scale = 8
    pw = max(32, preview.shape[1] // scale)
    ph = max(32, preview.shape[0] // scale)
    preview_s = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_AREA)
    m16_s = cv2.resize(m16, (max(32, m16.shape[1] // scale), max(32, m16.shape[0] // scale)),
                       interpolation=cv2.INTER_AREA)

    preview_gray = cv2.cvtColor(preview_s, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def pattern_score(pattern: str) -> float:
        code = _bayer_to_cv2_code(pattern)
        if code is None:
            return -1.0
        rgb = cv2.cvtColor(m16_s, code)
        h, w = rgb.shape[:2]
        ph_s, pw_s = preview_gray.shape[:2]
        if h < ph_s or w < pw_s:
            return -1.0
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        corr = cv2.matchTemplate(rgb_gray, preview_gray, cv2.TM_CCOEFF_NORMED)
        return float(np.max(corr)) if corr.size > 0 else -1.0

    s0 = pattern_score(selected_bayer)
    s1 = pattern_score(alt)
    if s1 > s0 + 0.002:
        return alt, f"preview_score {selected_bayer}={s0:.4f}, {alt}={s1:.4f}"
    return selected_bayer, f"preview_score {selected_bayer}={s0:.4f}, {alt}={s1:.4f}"


def convert_raw_to_fits(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.CR2",
    bayer_pattern: str | None = None,
    verify_bayer: bool = False,
):
    """Convert RAW files to FITS format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return False

    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return False

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find RAW files
    raw_files = sorted(input_path.glob(pattern))
    if not raw_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return False

    print(f"Found {len(raw_files)} files to convert...")

    # Cache expensive GBRG/GRBG disambiguation results so we do it once.
    refinement_cache: dict[str, tuple[str, str | None]] = {}

    # Convert each file
    success_count = 0
    for raw_file in raw_files:
        try:
            with rawpy.imread(str(raw_file)) as raw:
                data = raw.raw_image_visible.astype(np.float32)

                selected_bayer = bayer_pattern or detect_bayer_pattern_from_raw(raw)
                if not selected_bayer:
                    selected_bayer = "RGGB"
                    print(f"Warning: Could not detect Bayer pattern from RAW header for {raw_file.name}; falling back to {selected_bayer}")

                # Optional verification mode: disambiguate ambiguous GBRG/GRBG.
                if verify_bayer and bayer_pattern is None:
                    cache_key = selected_bayer if selected_bayer in {"GBRG", "GRBG"} else ""
                    if cache_key and cache_key in refinement_cache:
                        refined_bayer, note = refinement_cache[cache_key]
                    else:
                        refined_bayer, note = refine_bayer_pattern_with_cr2_preview(
                            raw, data, selected_bayer
                        )
                        if cache_key:
                            refinement_cache[cache_key] = (refined_bayer, note)
                    if refined_bayer != selected_bayer:
                        print(
                            f"Info: {raw_file.name}: BAYERPAT adjusted "
                            f"{selected_bayer} -> {refined_bayer} ({note})"
                        )
                    selected_bayer = refined_bayer

                # Create FITS header
                hdr = fits.Header()
                hdr['BAYERPAT'] = selected_bayer
                xoff, yoff = BAYER_TO_OFFSETS.get(selected_bayer, (0, 0))
                hdr['XBAYROFF'] = int(xoff)
                hdr['YBAYROFF'] = int(yoff)
                hdr['BITPIX'] = -32
                hdr['BSCALE'] = 1.0
                hdr['BZERO'] = 0.0
                hdr['INSTRUME'] = 'RAW2FITS'

                # Output filename
                fits_file = output_path / f"{raw_file.stem}.fits"

                # Write FITS file
                fits.writeto(str(fits_file), data, hdr, overwrite=True)
                print(f"{raw_file.name} -> {fits_file.name} (BAYERPAT={selected_bayer})")
                success_count += 1

        except Exception as e:
            print(f"Error converting {raw_file.name}: {e}")

    print(f"\nConversion complete: {success_count}/{len(raw_files)} files converted")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAW images to FITS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all CR2 files in current directory
  python3 convert_raw2fits.py

  # Specify input and output directories
  python3 convert_raw2fits.py /path/to/raws /path/to/fits

  # Use different file pattern (e.g., Nikon NEF files)
  python3 convert_raw2fits.py /path/to/raws /path/to/fits --pattern "*.NEF"

  # Specify different Bayer pattern
  python3 convert_raw2fits.py /path/to/raws /path/to/fits --bayer-pattern "BGGR"

  # Use Bayer pattern from RAW header metadata (default)
  python3 convert_raw2fits.py /path/to/raws /path/to/fits --bayer-pattern AUTO

  # Optional: verify ambiguous GBRG/GRBG using RAW preview correlation (slower)
  python3 convert_raw2fits.py /path/to/raws /path/to/fits --verify-bayer
        """
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Input directory containing RAW files (default: current directory)"
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default="fits_output",
        help="Output directory for FITS files (default: fits_output)"
    )

    parser.add_argument(
        "--pattern",
        default="*.CR2",
        help="File pattern to match (default: *.CR2)"
    )

    parser.add_argument(
        "--bayer-pattern",
        default="AUTO",
        choices=["AUTO", "RGGB", "BGGR", "GRBG", "GBRG"],
        help="Bayer pattern override (default: AUTO = detect from RAW header)"
    )

    parser.add_argument(
        "--verify-bayer",
        action="store_true",
        help="Verify ambiguous GBRG/GRBG with RAW preview correlation (slower)"
    )

    args = parser.parse_args()

    return convert_raw_to_fits(
        args.input_dir,
        args.output_dir,
        args.pattern,
        None if args.bayer_pattern == "AUTO" else args.bayer_pattern,
        args.verify_bayer,
    )


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
