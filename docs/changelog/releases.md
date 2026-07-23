# Release Notes

## v0.4.1 (2026-07-23)

**PI Live Image Editor:**

- **Live Image Editor:** Non-destructive interactive editor for FITS output images. Chat-driven image operations in natural language with AI sidecar or local fallback parser. Operations include brightness, contrast, saturation, sharpen, denoise, bilateral filter, green removal, CLAHE, invert, vibrance, color temperature, unpurple, fixbanding, star desaturation, and dehaze.
- **Session persistence:** Working state saved as `outputs/live_edit.fits` after every operation. Chat and operation history persisted separately in PI storage. Resume after browser refresh or backend restart restores the exact editing state.
- **Undo/Redo:** Deterministic rebuild from original + operation stack — no pixel snapshots needed. Undo/Redo/Repeat all update the operation history and working FITS.
- **+/- Adjust:** Signed operations (brightness, contrast, saturation, vibrance, color temperature) support iterative +/- buttons with deterministic rebuild from the adjust base.
- **Repeat:** Non-adjustable operations (sharpen, denoise, etc.) offer a "repeat" button that reapplies the same parameters without calling the AI.
- **Before/After comparison:** Click the image or badge to toggle between the previous and current preview after each operation.
- **Reset with confirmation:** Restores the original source FITS, deletes chat/operation history, clears undo/redo, and refreshes the run preview immediately.
- **Export:** PNG and FITS export with display metadata (DATAMIN/DATAMAX/CTYPE3) for FITS viewer compatibility.
- **Linear preview rendering:** FITS previews for live-edit files use direct [0,1]→8-bit mapping without robust_range stretch or gamma, matching the editor preview exactly.
- **Run preview integration:** `live_edit.fits` preferred in run-image preview scoring; preview refreshes after editor close or reset.
- **Phase 2 operations:** Vibrance, color temperature, unpurple, fixbanding, star desaturation, and dehaze implemented in backend, fallback parser, AI system prompt, and frontend dropdown.
- **Image-adaptive fallback:** Local fallback parser derives conservative operation strengths from current image statistics (luminance quantiles, mean saturation) when no AI sidecar is available.

## v0.4.0 (2026-07-22)

**PI Run-Chat, AQMH v0.2.1 conformance, documentation reorganisation:**

- **PI Run-Chat:** New run-specific PI chat in Run Monitor and AI-Empfehlung page. The chat provides schema-validated, pipeline-aware recommendations with deterministic action-plan validation, resume-phase computation, and run-artifact context (config, diagnostics, validation, logs). Backend-side PI service with memory store, tool registry, action validator, and context builder. AI sidecar extended with run-chat service, provider rate limiting, and multi-model support.
- **AQMH v0.2.0 → v0.2.1:** Background-gradient penalty in global quality, registration-weight guard, adaptive low-frequency neutralisation, structure-masked detail blending, uniform-control validation gate. Code conformance audit (F-01 through F-10) with all findings resolved: raw AQMH immutability, uniform-control semantics, PSF-FWHM sharpness proxy, common-overlap validation masking, background-penalty invalidity handling, per-metric validation status, default reconciliation, sigmoid temperature extension.
- **Documentation reorganisation:** README shortened from 1560 to ~120 lines, content extracted into focused `docs/guides/`, `docs/reference/`, and `docs/changelog/` structures. MkDocs nav rebuilt with 8 top-level tabs. GitHub Pages landing page updated.
- **Post-stack output phase:** New `runner_phase_post_stack_output` module for cleaner output handling after stacking.
- **Resume hardening:** AQMH reconstruction resume now preserves raw vs. selected output separation. Common-overlap mask threaded through validation. Resume dependencies documented.
- **OpenCL sigma-clip:** OpenCL implementations for median/MAD/percentile and sigma-clipping in reconstruction and stacking paths.
- **CFA mosaic warp:** CFA-safe warp handling improved for OSC data.

## v0.3.B (2026-07-03)

- AutoBGE preview: manual sample points with persistence, guard rejection reasons in UI, HMS preview resume support
- HyperMetric Stretch preview modal added alongside BGE preview, with live parameter editing and preview image rendering.

## v0.3.A (2026-06-28)

- Bug fixes

## v0.3.9 (2026-06-27)

- GPU processing was optimized across PREWARP, AQMH, tile reconstruction, synthetic reconstruction, and stacking with dedicated worker streams and parallel RGB execution.

## v0.3.8 (2026-06-26)

- AutoBGE implementation: native C++ port of the AutoBGE Siril script (polynomial + RBF background model, smart sample point generation with gradient descent, two-stage fit). Based on the AutoBGE Siril script by Adrian Knagg-Baugh.
- AI update: `sky_gradient` metric added to scan-metrics for large-scale background gradient detection. BGE prompt rules updated with `sky_gradient`-based decision thresholds.

## v0.3.7 (2026-06-25)

- Bug fixes.

## v0.3.6 (2026-06-24)

- Bug fixes.

## v0.3.5 (2026-06-22)

**GUI v3 – New Web Frontend:**

- New web frontend (v3) with improved Parameter Studio, Run Monitor, and Input & Scan tab.
- "Save As" dialog with directory browser: save file to selectable directory and filename via modal dialog.
- Warning banner in Run Monitor prominently displays warnings and errors during runs.
- Calibration gain mismatch: Dark/Flat calibration with mismatched gain now produces a warning instead of aborting.

**Astrometry Fix:**

- YAML `~` (null) for `astap_bin` was incorrectly parsed as string "null", causing ASTAP not to be found. Fix: null values are correctly treated as empty string, allowing the default path to be used.

## v0.3.4 (2026-06-16)

- Bug fixes

## v0.3.3 (2026-06-16)

**PI – AI-assisted configuration recommendations:**

- New PI (Parameter Intelligence) module: AI analyses scan results and generates validated parameter recommendations directly in Parameter Studio.
- Frame quality metrics (FWHM, noise, background, roundness, star count) from `scan-metrics` are passed to the AI as measured facts.
- See [PI AI Recommendations](../guides/pi_ai.md) for full documentation.

## v0.3.2 (2026-06-13)

**AQMH-First as default:**

- Top-level `method` field as single source of truth for reconstruction method (AQMH or Classic Tile Compile).
- AQMH is now the default with no migration of existing runs. Missing `method` field automatically defaults to `aqmh`.
- Frontend shows AQMH as pre-selected method in Wizard, Dashboard, and Parameter Studio.
- Classic phases (LOCAL_METRICS, STATE_CLUSTERING, SYNTHETIC_FRAMES) are hidden for AQMH runs.
- Rollback mechanisms: `FORCE_CLASSIC=1` environment variable or `--force-classic` CLI flag.

## v0.3.0 (2026-06-09)

**AQMH as default reconstruction method:**

- AQMH (Adaptive Quality Map Harvesting) is now the default reconstruction path (`aqmh.enabled: true`).
- Classic TILE_RECONSTRUCTION is still available via `aqmh.enabled: false`.
- All example profiles updated with `aqmh:` block.
- Full AQMH parameter documentation added to configuration reference and practical examples.
- `k_artifact` default changed to `3.0`, `frac_artifact_max` default changed to `0.25`.

## v0.2.A (2026-05-26)

- Calibration Bug fixes

## v0.2.9 (2026-05-25)

**Raw Stack preprocessing pipeline:**

- New standalone Raw Stack UI: preprocessing from FITS light frames to final stacked image, separate from the Tile-Compile run studio.
- Pipeline covers: Calibration, CFA/Mono Prep, Registration, Quality Filtering, Stacking, Astrometry, BGE, PCC, HyperMetric Stretch.
- All parameters (sigma-clip, rejection, weighting, BGE, PCC, Astrometry, HMS) are taken from the Parameter Studio config — nothing is hardcoded.
- See [Raw Stack GUI](../guides/raw_stack_gui.md) for GUI documentation.

## v0.2.8 (2026-05-23)

- HMS Bug fixes

## v0.2.7 (2026-05-22)

**HMS:**

- Added VeraLux HyperMetric Stretch (HMS) as a post-PCC pipeline phase.

## v0.2.6 (2026-05-20)

**Build hardening & Frontend cleanup:**

- Hardened web_backend_cpp build with CUDA 13 + OpenCV 4.11 CUDA 13 configuration
- Frontend refactoring: centralized utilities in `src/utils.js`

## v0.2.5 (2026-04-26)

- Documentation system with MkDocs Material + Doxygen
- Configurable BGE sample estimators: `quantile`, `sigma_clipped_median`, `sextractor_mode`, and `biweight`
- BGE autotune now sweeps sample estimators and applies chroma/background-spread guards

## v0.2.4 (2026-04-25)

- Registration performance: anchor-promotion rounds now reuse the parallel worker pool and only retry unresolved frames whose nearest anchor changed after promotion.

## v0.2.3 (2026-04-24)

- More robust registration: deep-chain outlier rejection, doubled anchor density for large-N sessions, "hopping" sequential rescue, ASTAP plate-solving as fallback.

## v0.2.2 (2026-04-24)

- Hot/dead-pixel correction fixed (`cosmetic_correction_cfa`): defective pixels inside star regions were not corrected because `neighbor_threshold` was set too low.

## v0.2.1 (2026-04-23)

- Registration phase: NCC computation made more robust against background subtraction and hot pixels.

## v0.2.0 (2026-04-14)

- Registration for long Alt/Az sessions was expanded substantially: N-scaled multi-anchor reference selection, N-scaled anchor promotion, astrometric registration/rescue.

## v0.1.F (2026-04-07)

- TILE_RECONSTRUCTION performance: replaced the memory-driven worker reduction with frame sub-batching.

## v0.1.E (2026-04-06)

- Calibration/UI follow-up: dark+bias calibration now handles raw darks without double bias subtraction.

## v0.1.D (2026-04-04)

- Added `registration.auto_engine` (default: `true`): automatically detects strong field rotation.

## v0.1.C (2026-04-03)

- Stabilized tile reconstruction after the recent performance optimization rollout.

## v0.1.B (2026-03-31)

- Fixed the late PCC/output path semantics.

## v0.1.A (2026-03-29)

- Stabilized the late RGB/PCC output path after the `v3.3.9` rollout.

## v0.1.9 (2026-03-28)

- Promoted the `v3.3.9` methodology into the active reference state across code, frontend, and documentation.

## v0.1.8 (2026-03-25)

- Improved Linux packaging scripts to bundle all required shared libraries.

## v0.1.7 (2026-03-24)

- Fixed Linux AppImage packaging to export `TILE_COMPILE_INPUT_SEARCH_ROOTS`.

## v0.1.6 (2026-03-24)

- Reworked GUI2 queue/batch handling and run monitoring.

## v0.1.5 (2026-03-23)

- Stabilized `PREWARP` for OpenCL and extended GPU acceleration with OpenCL equivalents.

## v0.1.4 (2026-03-22)

- Added a real artifact-based `STACKING` resume path.

## v0.1.3 (2026-03-21)

- Added per-frame registration provenance and chain-depth tracking.

## v0.1.2 (2026-03-20)

- Fixed Alt/Az registration validation to score warps on the actual common overlap.

## v0.1.1 (2026-03-19)

- Improved GUI2 tool persistence and PCC save handling.

## v0.1.0 (2026-03-18)

- Fixed Astrometry/PCC tool path inputs being overwritten by backend defaults.

## v0.0.F (2026-03-17)

- Promoted the DSO tile-reconstruction methodology to `v3.3.8`.

## v0.0.E (2026-03-15)

- `assumptions.frames_min` is now used by the active C++ runner mode gate.

## v0.0.D (2026-03-15)

- Expanded `TILE_RECONSTRUCTION` boundary diagnostics.

## v0.0.C (2026-03-13)

- GUI2 parameter/config handling synchronized with the current C++ config schema.

## v0.0.B (2026-03-12)

- Added server-side persistence for the GUI2 UI draft state.

## v0.0.A (2026-03-12)

- Bugfixes

## v0.0.9 (2026-03-11)

- Added Linux AppImage generation to the GitHub Actions release workflow.

## v0.0.8 (2026-03-11)

- zero-copy COMMON_OVERLAP
- Scratch reuse in LOCAL_METRICS
- reduced lock contention in tile_weighted-OLA
- faster sigma-clip kernel
- parallel BGE autotune candidate evaluation

## v0.0.7 (2026-03-11)

- Supports: Linux x86_64 with `glibc >= 2.39`, macOS 15, Windows 10 x64+

## v0.0.6 (2026-03-11)

- Completed the productive migration to the Crow/C++ backend.

## v0.0.5 (2026-03-09)

- Promoted GUI2 as the recommended interface.

## v0.0.4 (2026-03-06)

- Fixed Alt/Az registration for datasets with large field rotation.

## v0.0.3 (2026-03-05)

- Improved BGE/PCC pipeline with clearer phase visibility, stronger guardrails.

## v0.0.2 (2026-02-16)

- First release with pre-built packages for Windows, Linux, and macOS.

## v0.0.1 (2026-02-15)

- First public release.
