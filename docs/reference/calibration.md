# Calibration (Bias / Dark / Flat)

- Master frames (`bias_master`, `dark_master`, `flat_master`) can be used directly
- Directory-based masters (`bias_dir`, `darks_dir`, `flats_dir`) can be built automatically
- When `use_bias: true` and `use_dark: true`, raw darks are bias-corrected internally unless `dark_already_bias_corrected: true` is set
- `dark_auto_select: true` matches darks by exposure time (±5%)

## External Sources (PCC and Astrometry)

For optional color calibration and astrometric solving, the pipeline can use external data/tools:

### Siril Gaia DR3 XP sampled catalog (for PCC and astrometry fallback)

- Can be reused if already downloaded by Siril.
- Typical local path: `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`
- Upstream source (catalog release): `https://zenodo.org/records/14738271`
- **Download via GUI3**: Tab *Tools → PCC → Download Missing* automatically downloads missing catalog chunks (~2 GB, 48 chunks).
- **Astrometry fallback:** When ASTAP does not produce a valid WCS, the runner matches detected stack stars against this local catalog in-process. This is an offline Tile Compile solver; it does not launch Siril.
- The fallback needs approximate pointing and optics metadata in the RGB stack: `RA`, `DEC`, `FOCALLEN`, and `XPIXSZ` or `YPIXSZ`. It writes a linear TAN/CD WCS without SIP distortion terms.

### ASTAP (for astrometry / WCS plate solving)

- Requires ASTAP plus a star database (e.g., D50 for deep-sky use).
- Official site/downloads: `https://www.hnsky.org/astap.htm`
- **Download via GUI3**: Tab *Tools → Astrometry → Install CLI* and *Download Catalog* download ASTAP binary and star database directly.

If these resources are not installed, core reconstruction still works. ASTROMETRY can succeed through ASTAP alone, or through the local Gaia fallback when the required metadata is available; PCC uses the local catalog or its configured online source. A phase with no applicable solver/catalog may be skipped or fail depending on configuration. BGE (Background Gradient Extraction) works independently of external catalogs.
