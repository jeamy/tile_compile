# Dark processing and frame flow

## Overview
This document explains how **darks** are used in the pipeline and how **light frames** move through the filesystem during a run.

The key rule is:

- Darks are applied **only once**, during **SCAN_INPUT** (calibration).
- After that point, all downstream steps (registration, splitting, normalization, metrics, stacking) operate on the **calibrated light frames**.

## Where darks are applied
### Phase: SCAN_INPUT (calibration)
Code path: `runner/phases_impl.py` → SCAN_INPUT calibration block.

When `calibration.use_dark: true`:

- A dark master is resolved either by:
  - Loading `calibration.dark_master` (if configured), or
  - Building a master from `calibration.darks_dir` (glob `calibration.pattern`).
- Optional: automatic dark selection based on FITS headers (see below).

The actual calibration is applied per light frame via:

- `runner/calibration.py::apply_calibration(img, bias_arr, dark_arr, flat_arr)`

Order:

- Subtract bias (if enabled)
- Subtract dark (if enabled)
- Divide flat (if enabled)

The output of this step is written to:

- `runs/<run_id>/outputs/calibrated/cal_XXXXX.fit`

These `cal_*.fit` files are **the calibrated lights**.

## Dark auto-select + warnings (optional)
Config keys:

- `calibration.dark_auto_select` (default: `true`)
- `calibration.dark_match_exposure_tolerance_percent` (default: `5.0`)
- `calibration.dark_match_use_temp` (default: `false`)
- `calibration.dark_match_temp_tolerance_c` (default: `2.0`)

Behavior:

- The pipeline estimates the typical light exposure (median of first up to 10 lights) from FITS header keys:
  - `EXPTIME`, `EXPOSURE`, `EXPOSURETIME`, `EXP_TIME`, `DURATION`
- If `dark_match_use_temp` is enabled, it also estimates the light CCD temperature from:
  - `CCD-TEMP`, `CCD_TEMP`, `CCD_TEMP_C`, `SENSOR_T`, `SENSORTEMP`, `TEMP`, `TEMPERAT`
- It then selects a subset of darks that match exposure (and optionally temperature) within tolerances.
- If no match is found, it falls back to using all darks and emits warnings.

The selection result and warnings are emitted as `phase_progress` events in `logs/run_events.jsonl` (substep `dark_master`).

## How calibrated lights are used for registration
### Phase: REGISTRATION (Siril engine)
Code path: `runner/phases_impl.py` → `REGISTRATION` branch for Siril.

Input:

- The variable `frames` is the list of input frames for registration.
- If calibration ran, `frames` points to the freshly produced `outputs/calibrated/cal_*.fit`.

Staging directory:

- `runs/<run_id>/work/registration/`

For Siril, the runner creates:

- `work/registration/seq00001.fit`, `seq00002.fit`, ...

These are created using `safe_symlink_or_copy(src, dst)`.

Interpretation:

- `seq*.fit` are **staging inputs for Siril**.
- On most systems they will be symlinks (very small size on disk).

### Siril intermediate files ("lights_*.fit")
Files like:

- `work/registration/lights_00001.fit`

are **not written by the Python runner**.

They are created by the **Siril registration script** (default: `siril_register_osc.ssf` or your custom script) as intermediate products.

They should be treated as:

- Siril-side working files derived from the staged inputs (`seq*.fit`).

### Registered output frames
After Siril finishes, the runner expects Siril to produce registered frames in `work/registration/` with names starting with:

- `r_*.fit*`

The runner then moves/copies those into:

- `runs/<run_id>/outputs/<registration.output_dir>/` (default `outputs/registered/`)

renaming them to the configured pattern:

- `registration.registered_filename_pattern` (default: `reg_{index:05d}.fit`)

So the expected stable registered outputs are:

- `outputs/registered/reg_00001.fit`, `reg_00002.fit`, ...

## Troubleshooting: missing `r_*.fit*` or missing `outputs/registered/`
If you see `work/registration/seq*.fit` and `work/registration/lights_*.fit` but **no** `r_*.fit*`:

- Siril likely did not reach the step that writes `r_*.fit*`.
- Check:
  - `logs/run_events.jsonl` for `phase_end` of `REGISTRATION`
  - The Siril log written by the runner: `siril_registration.log` (path depends on your run directory; typically in the run's artifacts/log area)
  - The used Siril script (default/custom) and what filenames it produces

## Summary
- **Darks are applied only during SCAN_INPUT** when creating `outputs/calibrated/cal_*.fit`.
- Registration consumes those calibrated lights (via `frames` → staged as `work/registration/seq*.fit`).
- `work/registration/lights_*.fit` are Siril script intermediates.
- Final registered frames should appear as `outputs/registered/reg_*.fit` once Siril produces `r_*.fit*`.
