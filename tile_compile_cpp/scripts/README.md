# Scripts

This folder contains small helper scripts used around `tile_compile_cpp`.

## `convert_raw2fits.py`

Convert camera RAW files (e.g. `CR2`, `NEF`, `ARW`) into FITS files (typically for feeding the pipeline with OSC/Bayer data).

### Syntax

```bash
python3 convert_raw2fits.py [input_dir] [output_dir] [--pattern "*.CR2"] [--bayer-pattern AUTO|RGGB|BGGR|GRBG|GBRG] [--verify-bayer]
```

### Notes

- Writes one FITS per input file: `output_dir/<stem>.fits`.
- FITS headers include:
  - `BAYERPAT` (detected or overridden)
  - `XBAYROFF`, `YBAYROFF`
- `--verify-bayer` is intended to disambiguate the common `GBRG` vs `GRBG` ambiguity by comparing against a RAW preview (requires OpenCV `cv2`).

### Examples

```bash
# Convert CR2 files from current directory into ./fits_output
python3 convert_raw2fits.py

# Convert NEF files
python3 convert_raw2fits.py ./raw ./fits --pattern "*.NEF"

# Force a specific Bayer pattern
python3 convert_raw2fits.py ./raw ./fits --bayer-pattern BGGR

# Auto-detect + verification mode (slower)
python3 convert_raw2fits.py ./raw ./fits --bayer-pattern AUTO --verify-bayer
```

## `lco_bulk_download.py`

Download **public** Las Cumbres Observatory (LCO) frames from the LCO Archive API.

The script queries `https://archive-api.lco.global/frames/` and can:

- download `N` frames individually using the per-frame signed `url`, or
- request a single ZIP containing all frames via `https://archive-api.lco.global/frames/zip/`.

### Syntax

```bash
python3 lco_bulk_download.py \
  --object M42|M31|IC434 \
  [--n 50] \
  [--out ./lco_download] \
  [--rlevel 0] \
  [--obstype EXPOSE] \
  [--zip] \
  [--uncompress] \
  [--filters rp,gp,ip] \
  [--extract] \
  [--funpack] \
  [--param KEY=VALUE]...
```

### Important parameters

- `--object`: value for the archive field `OBJECT` (e.g. `M42`, `M31`, `IC434`).
- `--rlevel`:
  - `0` = raw
  - `91` = processed
- `--obstype`: use `EXPOSE` for science frames (avoid `GUIDE`).
- `--param KEY=VALUE`: pass additional query parameters to the frames endpoint, e.g. `--param FILTER=rp`.
- Output files are usually `*.fits.fz` (FITS with Rice compression).
- `--filters rp,gp,ip`: convenience option to download per filter into subfolders.
- `--extract`: if `--zip` is used, extract the zip after download.
- `--funpack`: after download/extract, convert `*.fits.fz` to `*.fits` using system `funpack` (if available).

### Examples

```bash
# Download 50 raw science frames for M42 (individual downloads)
python3 lco_bulk_download.py --object M42 --n 50 --out /tmp/lco

# Download as one zip
python3 lco_bulk_download.py --object IC434 --n 50 --out /tmp/lco --zip

# Restrict by filter
python3 lco_bulk_download.py --object IC434 --n 50 --out /tmp/lco --param FILTER=rp

# Download multiple filters (rp,gp,ip) in one run (creates subfolders)
python3 lco_bulk_download.py --object IC434 --n 50 --out /tmp/lco --filters rp,gp,ip

# Download multiple filters as zip and extract it
python3 lco_bulk_download.py --object IC434 --n 50 --out /tmp/lco --filters rp,gp,ip --zip --extract

# Additionally try to funpack *.fits.fz -> *.fits (requires 'funpack' in PATH)
python3 lco_bulk_download.py --object IC434 --n 50 --out /tmp/lco --filters rp,gp,ip --zip --extract --funpack
```

## `generate_report.py`

Generate an HTML report for a `tile_compile_cpp` run directory (charts + summary tables). It writes into the run’s `artifacts/` directory.

### Syntax

```bash
python3 generate_report.py /path/to/runs/<run_id>
```

### Output

- `<run_dir>/artifacts/report.html`
- `<run_dir>/artifacts/report.css`
- `<run_dir>/artifacts/*.png`

### Examples

```bash
python3 generate_report.py ./runs/2026-02-14_15-40-00
```
