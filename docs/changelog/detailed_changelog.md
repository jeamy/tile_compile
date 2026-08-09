## Changelog

### (2026-08-08)

**v0.4.6 — AQMH reconstruction quality, registration robustness, PI/AI integration, and run-monitor improvements:**

**AQMH reconstruction and registration:**

- Improved global registration for long or difficult sessions with stronger anchor handling, cascade refinement, affine refinement, smooth local refinement, confidence checks, and safer fallback behavior.
- Added registration-confidence weighting and chain-depth penalties so uncertain or indirectly registered frames contribute less aggressively to AQMH reconstruction.
- Extended AQMH reconstruction candidate handling with immutable raw-AQMH output, uniform-control comparison, adaptive low-frequency neutralisation, structure-masked detail candidates, and validation against both the uniform control and raw baseline.
- Added and aligned AQMH reconstruction controls for asymmetric sigma clipping, structure masks, registration-weight guards, RGB/debayer-first processing, and sequential RGB memory handling.
- Improved AQMH diagnostics and reconstruction artifacts, including raw/selected output separation, validation metadata, cache/backend information, and safer resume dependencies.
- Added regression coverage for AQMH quality maps, reconstruction, validation, debayering, registration cascades, linearity, and GPU/CPU-related behavior.

**Progress, logging, and Run Monitor:**

- AQMH diagnostics now reports visible progress while processing frame/block diagnostics and while writing diagnostic artifacts; the existing `AQMH_DIAGNOSTICS` phase remains resumable and is shown correctly in the v3 Run Monitor.
- Fixed repeated live-log entries caused by periodically reloading the same run-log tail. The frontend now appends only new or non-overlapping log lines.
- Improved phase, progress, warning, and run-status handling during normal runs and resume operations, including restored phase state after reload.
- Added clearer runtime/build and acceleration information for reconstruction and registration phases.

**PI and AI services:**

- Updated `@earendil-works/pi-coding-agent` to `^0.84.1` and raised the agent-service Node.js requirement to `>=22.19.0`.
- Updated the agent-service lockfile with Node 22 and adapted streaming handling to the `0.84.x` delta-only `message_update` event format.
- Expanded AI prompt and provider handling, frame analysis, run chat, model/auth status, and provider payload support.
- Added backend AI/PI routes and run-analysis context handling for configuration, diagnostics, validation artifacts, logs, and resumable phase recommendations.

**Frontend and configuration:**

- Expanded the v3 Parameter Studio with AQMH explanations, parameter guidance, validation-aware controls, calibration controls, and aligned English/German translations.
- Synchronized AQMH configuration defaults, schemas, examples, practical configuration documentation, and methodology references.
- Added build-information generation and runtime version/build metadata to the CMake and runner setup.
- Added sharpness and registration comparison profiles and documented the related M31/AQMH evaluation results.

### (2026-07-25)

**v0.4.5 — PI API key configuration bug fix:**

- Corrected the PI sidecar/frontend status contract: the frontend recognises `auth_storage` as a stored API key and no longer incorrectly displays it as missing.
- After **Save Key**, the UI immediately reloads providers, models, and key status. The sidecar activates the key during saving, so a browser refresh is no longer necessary.
- The API key input field is cleared after a successful save for safety and clarity.

**v0.4.4 — AQMH GPU and MAPS performance optimizations:**

- **AQMH_MAPS:** Bilinear Psi upsampling and logarithmic accumulation were fused into one pass, eliminating the large intermediate upsample matrix.
- **MAPS runtime:** In the native 64-frame A/B test, Psi upsampling/accumulation decreased by 19.5% and total quality-map computation by 6.7%. All 64 Q-map cache files remained bit-identical.
- **CUDA reconstruction:** The CUDA build now links `OpenMP::OpenMP_CUDA`, so the parallel host-preparation loops are compiled and executed with OpenMP.
- **Reconstruction runtime:** In the controlled native 64-frame test with identical chunk geometry, the reconstruction core decreased from 47.31 to 31.70 seconds (33.0%). Host preparation decreased from 30.33 to 14.46 seconds (52.3%).
- **Telemetry:** AQMH artifacts now record separate MAPS-stage timings as well as CUDA host preparation, H2D, kernel, D2H, result commit, and summed worker timings.
- **Regression:** Raw AQMH FITS remained bit-identical with identical `chunk_rows` (0 of 8,507,400 pixels differ). The full suite passed 240 of 241 test cases; the only skipped CUDA unit test had no CUDA-device access in the sandbox process.

### (2026-07-24)

**v0.4.2 — PI Live Image Editor command suite and reusable editing workflows:**

**New GUI commands and deterministic image operations:**

- **Levels:** Added `levels` with validated `black` (`0…1`), `white` (`0…1`), and `gamma` (`0.1…5`) parameters. Black/white ordering is enforced in backend validation and frontend controls.
- **Curves:** Added the GUI-only `curves` operation with 2–32 control points. The editor displays a gamma-style graph, supports click-to-add, drag-to-move, and double-click/right-click removal, and uses the same clamped Catmull-Rom spline in frontend and backend. Curves is excluded from the AI operation prompt.
- **Shadow Recovery:** Added `shadow_recovery` with strength `0…1`.
- **Highlight Recovery:** Added `highlight_recovery` with strength `0…1`. Explicit highlight/spitzlicht requests are routed to the dedicated dialog even when an AI model proposes generic brightness plus recovery operations.
- **Color Balance:** Added global red/green/blue correction plus optional shadow, midtone, and highlight RGB adjustments with luminance-weighted deterministic application.
- **Local Contrast:** Added `local_contrast` with strength `0…1` and radius `0.5…10`.
- **Chroma Denoise:** Added `chroma_denoise` with strength, structure protection, and `soft`/`strong` modes.
- **Crop:** Added deterministic axis-aligned and rotated crop operations. Screen-to-image coordinate conversion remains correct after zoom and pan; crop bounds remain inside the visible image area. Crop bypasses the AI sidecar.

**AI proposal and parameter dialogs:**

- Dialog-based commands are validated but no longer applied by `POST /api/pi/live-image-chat`. The endpoint returns `requires_confirmation` and structured proposed operations.
- Dedicated command-intent routing normalises levels, shadow, highlight, color-balance, local-contrast, and chroma-noise requests. Wrong, legacy, multi-operation, or invalid model responses are replaced by safe deterministic suggestions.
- Added `POST /api/pi/live-image-chat/preview-operation` for non-persistent previews calculated from a clone of the current FITS image.
- Added debounced preview requests, stale-response rejection, pointer lifecycle guards, and validation-aware parameter ranges.
- Parameter and curve dialogs use a shared symmetric theme, can be dragged by their header, and are constrained to the visible browser area.
- Added **Before/Current view** checkbox. Disabling it displays the unchanged canonical current image; enabling it displays the current temporary preview.
- **Apply** commits the operation through the deterministic backend. **Cancel** invalidates pending preview work and restores `currentImageSrc` without changing FITS, history, or undo/redo.

**Timeline, history, and presets:**

- Replaced inverse/rebuild-only undo assumptions with exact pre/post pixel snapshots for every operation. Threshold, crop, denoise, CLAHE, and other lossy operations now undo exactly.
- Added persistent `edit_history` alongside the active `operation_history`. Apply, adjust, undo, and redo actions are recorded explicitly.
- Added `POST /api/pi/live-image-chat/reapply`. Clickable chat entries can execute their stored operation parameters again without AI.
- Added global timeline presets under `.pi_memory/presets`. Backend routes list, create, overwrite, and atomically apply presets. The frontend provides themed selection, **Save**, **Save as**, overwrite confirmation, and **Apply**.
- Presets store the effective operation sequence plus the complete edit timeline and can be applied to any run.

**Preview, reset, compatibility, and documentation:**

- Retained the previous committed preview for click-based Before/Current comparison.
- Reset recreates the canonical `live_edit.fits`, removes stale derived previews, clears chat/edit/undo/redo history after confirmation, and updates the run preview.
- Replaced `cv::getRotationMatrix2D` with an equivalent explicit affine matrix for OpenCV 5/macOS compatibility.
- Added and aligned English/German UI strings for commands, parameter controls, dialogs, previews, presets, crop, comparison, and repeat actions.
- Expanded the English and German Live Image Editor guides with AI proposal semantics, deterministic fallback, live preview, curves, history, and presets.
- Added spline-operation regression coverage and retained complete Live Image Session test coverage.

### (2026-07-23)

**v0.4.1 — PI Live Image Editor:**

**Live Image Editor (backend `web_backend_cpp/`, frontend `web_frontend_v3/`, sidecar `agent_service/`):**

- **Non-destructive editor:** Interactive image editor operating on a canonical working copy (`outputs/live_edit.fits`) derived from the immutable source FITS. The source FITS is never modified. All operations are applied to in-memory float CV_32F BGR data and persisted after each change.
- **Chat-driven operations:** Users issue natural-language commands (e.g. "helle das Bild auf", "erhoehe den Kontrast"). When the PI AI sidecar is available, the request is forwarded with a JPEG vision preview and recent operation history. The AI returns structured operations with parameters. The backend validates and applies them locally. When no sidecar is available, a local fallback parser selects the operation from keywords and derives conservative strengths from image statistics (luminance quantiles, mean saturation).
- **Phase 1 operations:** brightness (midtones/shadows/highlights), contrast, saturation, sharpen (unsharp mask), denoise (Non-Local Means), bilateral filter, green removal (rmgreen), CLAHE (local contrast), threshold, invert, reset.
- **Phase 2 operations:** vibrance, color_temperature, unpurple, fixbanding, star_desaturation, dehaze. Implemented in `pi_image_ops.cpp`, fallback parser, AI system prompt (`liveImageChatService.ts`), and frontend dropdown (`FEATURE_PHASE = 2`).
- **+/- Adjust:** Signed operations (brightness, contrast, saturation, vibrance, color_temperature) support iterative +/- buttons. The adjust mechanism rebuilds deterministically from `original_fits` + base operations + N × adjust_step, ensuring exact reproducibility without accumulating floating-point drift. `adjust_base_size` captures the undo-stack depth at the time `set_adjust_step` is called.
- **Repeat:** Non-adjustable operations offer a "repeat" button (`POST /api/pi/live-image-chat/repeat`) that reapplies the last operation with identical parameters without calling the AI. The repeated operation enters the undo stack and operation history.
- **Undo/Redo:** Deterministic rebuild from `original_fits` + undo_stack entries — no pixel snapshots needed, even for non-invertible operations (CLAHE, bilateral, denoise, threshold). `operation_history` mirrors the current undo_stack at all times. Undo decrements `adjust_count` for adjust-source entries; redo increments it.
- **Before/After comparison:** The previous preview image is retained after each operation. Clicking the image or the VORHER/AKTUELL badge toggles between the previous and current state. Drag detection prevents accidental toggle during pan.
- **Reset with confirmation:** `window.confirm` dialog before reset. Reset copies the original source FITS to `live_edit.fits`, deletes the chat history file, clears undo/redo/operation_history/chat_history, removes stale derived PNG/JPEG previews, and triggers an immediate run-preview refresh via the `onClose` callback.
- **Session persistence and resume:** `live_edit.fits` is the canonical working state. Chat and operation history are persisted in `.pi_memory/live_image_chat/<run_id>_<hash>.json`. On resume (`POST /api/pi/live-image-chat/create`): if `live_edit.fits` exists, it is loaded directly; if only history exists, operations are replayed once and the working FITS is materialized. `LiveImageSessionStore::create` accepts separate `original` and `current` matrices. `persist_live_session` deletes the history file when both histories are empty (post-reset).
- **Linear preview rendering:** `render_fits_preview_png` and `render_fits_preview_png_for_pi` detect `live_edit`/HMS/PCC filenames and render with direct `[0,1]→[0,255]` mapping — no `robust_range` stretch, no gamma. `linear_float_to_u8` helper replaces per-pixel lambda. `render_float_bgr_to_bgr8` and `render_float_planes_to_bgr8` use the same mapping. This ensures the run preview, editor preview, and exported FITS all agree.
- **FITS export metadata:** `write_float_bgr_to_fits` now writes `DATAMIN=0.0`, `DATAMAX=1.0`, and `CTYPE3=RGB` headers so FITS viewers can apply the correct display range.
- **Run preview integration:** `scoreArtifact` in `run-image-preview.js` prefers `live_edit.fits` (score 250) over HMS (score 100). `openLiveEditor` helper passes a refresh callback so the run preview reloads after editor close or reset. A "Live Editor" button is shown next to the artifact path.
- **Session eviction:** `evict_expired(1800, 5)` called on create to enforce 30-minute timeout and max 5 sessions.
- **AI sidecar:** `liveImageChatService.ts` system prompt updated with Phase 2 operation schemas, adjustable vs. repeatable distinction (signed operations → adjustable; one-sided operations → not adjustable; sharpen → repeatable). `repeatable` field added to AI response schema. Operation history trimmed to last 10 entries for prompt context.
- **Frontend:** `live-image-viewer.js` expanded with before/after comparison, repeat button, drag-vs-click detection, `beginOperation`/`commitOperation`/`cancelOperation` lifecycle for preview management. `FEATURE_PHASE` raised to 2. Preset commands reorganized (localContrast and colorDenoise removed from presets; Phase 2 commands added). Chat history restored on resume regardless of `resumed` flag.
- **i18n:** New strings for `confirmReset`, `compareBefore`/`compareAfter`, `repeat`/`repeatDone`, Phase 2 command labels and help texts, `ui.button.live_editor` — in both `de.json` and `en.json`.
- **Documentation:** `docs/guides/live_image_editor_de.md` (58 lines) covering working image, preview rendering, operations, AI usage, and export. `docs/PI/pi_live_image_chat_plan.md` updated to reflect implementation status (Phase 1-2), `live_edit.fits` persistence, deterministic undo/redo without snapshots, reset with confirmation, and resume from working FITS.
- **Tests:** `test_pi_live_image_session.cpp` expanded with operation_history mirror assertions after undo/redo, sharpen +/- rebuild test, reset operation test (clears stacks and chat history), and repeat operation test.

### (2026-07-22)

**v0.4.0 — PI Run-Chat, AQMH v0.2.1 conformance, documentation reorganisation:**

**PI Run-Chat (branch `piai2`, PRs #56 and #57):**

- **Run-specific PI chat:** New run chat in Run Monitor and AI-Empfehlung page. The chat provides schema-validated, pipeline-aware recommendations with deterministic action-plan validation, resume-phase computation, and run-artifact context (config, diagnostics, validation, logs).
- **Backend PI service:** New C++ PI service layer in `web_backend_cpp/` with `pi_routes.cpp` (2357 lines), `pi_memory_store.cpp` (947 lines), `pi_tool_registry.cpp` (373 lines), `pi_action_validator.cpp` (201 lines), `pi_context_builder.cpp` (99 lines), `pi_assistant.cpp` (98 lines), `pi_action_plan.cpp` (80 lines), `pi_ai_request_builder.cpp` (112 lines), and `pi_storage_paths.cpp` (134 lines). New tests: `test_pi_routes.cpp`, `test_pi_memory_store.cpp`, `test_pi_action_plan.cpp`.
- **AI sidecar extensions:** `runChatService.ts` (162 lines) for run-specific chat, `providerRateLimit.ts` (106 lines) for rate limiting, `providerPayload.ts` (70 lines) for multi-provider payload formatting, `piExtensions.ts` (47 lines), `modelService.ts` expanded (+487 lines) for multi-model support, `frameAnalysisService.ts` expanded (+257 lines), `trafficLog.ts` expanded (+118 lines).
- **Frontend integration:** `ai-empfehlung.js` expanded (+1142 lines) with run-chat UI, `run-monitor.js` expanded (+1191 lines) with chat panel and run-image preview, new `run-image-preview.js` component (164 lines), `input-scan.js` updated (+34 lines), `run-history.js` updated (+26 lines).
- **PI documentation:** New `docs/PI/pi_workflow_gui3.md` (53 lines), PI recommendations docs expanded (+45/+47 lines EN/DE). Implementation plans (`pi_run_chat_empfehlungs_chat_datenplan.md`, `pi_integration_architektur_tile_compile_plan.md`) moved to `docs/PI/attic/`.
- **Design principle:** The PI AI may interpret and prioritise, but schema validity, action-plan validity, and resume-phase computation are deterministically validated in the backend. The AI receives config paths, schema constraints, artifact diagnostics, and pipeline phase context to prevent plausible-but-wrong recommendations (e.g. recommending `stretch.star_pressure` which is a diagnostic value, not a config parameter).

**AQMH v0.2.0 → v0.2.1 conformance audit:**

- **F-01 (Raw AQMH immutability):** `AqmhReconstructionPhaseResult` now carries `raw_output` separately from the selected `output`. Raw output is persisted to `outputs/aqmh_reconstructed_raw.fit` before neutralisation/detail-candidate selection. Resume applies the same artifact contract.
- **F-02 (Uniform control semantics):** `compute_aqmh_uniform_control` now separately accumulates the arithmetic mean of finite, reconstruction-support and frame-mask-valid registered samples. Does not inspect Q-map values, global weights, cherry-pick selection, or sigma-clipping. Emits per-pixel validity mask; validation intersects with common overlap.
- **F-03 (PSF-FWHM sharpness proxy):** Methodology formally permits `1/wfwhm` as a post-v0.2.1 cross-infrastructure extension when star metrics are available. AQMH-map `g_sharp` remains the reference/default and fallback. `global_sharpness_source` artifact field added (`"laplacian_variance"` or `"psf_wfwhm_inverted"`).
- **F-04 (Validation mask):** AQMH maps and reconstruction retain reconstruction support for output coverage. Map summaries, diagnostics, all candidate/control/raw validation comparisons, and resume validation now receive a separate common-overlap mask. Non-common pixels excluded from noise, seam, FWHM, star detection, tail, and elongation measurements.
- **F-05 (Background-penalty invalidity):** `calculate_frame_metrics` now accepts optional `frame_valid_mask`. Quadrant medians for sky-gradient only consider bg_mask AND frame-valid pixels. Quadrants with no valid pixels → NaN. `sky_gradient` set to NaN when `background <= 0` or fewer than four valid quadrants.
- **F-06 (Per-metric validation status):** `validation_comparison_json` now emits per-metric status entries (`background_rms`, `fwhm`, `seam_score`, `tail11_abs`, `elongation`), each with `status` (pass/fail/not_applicable), `reason`, `value`, `control`, `regression`, and `threshold`. All four comparison call sites pass the validation config.
- **F-07 (Default reconciliation):** Methodology now distinguishes v0.2.1 reference defaults from current operating defaults: global quality (`g_floor` 0.05→0.03, `g_w_sharp` 0.6→0.55, `g_w_snr` 0.4→0.30, `g_w_background_penalty` 0.3→0.25), structure mask (`low_q` 0.70→0.40, `high_q` 0.97→0.90, `sigma` 2→4), validation thresholds (widened to reduce false rejections).
- **F-08 (Sigmoid temperature):** `g_k_scale` and score clipping `[-8, 8]` formally documented as post-v0.2.1 extension. Formula: `G_f = g_floor + (1 - g_floor) * sigmoid(clamp(g_k_scale * score_f, -8, 8))`. v0.2.1 reference uses implicit scale 1.0 and no clipping.
- **F-09 (Registration guard):** Positive finding — implementation conforms to §4.3. Section 9.4 documentation inconsistency corrected.
- **F-10 (Core invariants):** Positive finding — accumulator rejects invalid pixels, zero maps don't fall back to unweighted mean, separate common and reconstruction masks derived.
- **Methodology documents updated:** `aqmh_methodik_en_v0.2.1.md` (+303 lines) and `aqmh_methodik_de_v0.2.1.md` (+300 lines) with all conformance resolutions, post-v0.2.1 extensions, and default reconciliation.
- **Conformance report:** `docs/AQMH/attic/aqmh_v0.2.1_code_conformance_report.md` (274 lines) documents all findings and resolutions.
- **AQMH tests:** `test_aqmh_validation.cpp` expanded (+226 lines), `test_aqmh_config.cpp` (+21 lines), `test_aqmh_reconstruction.cpp` (+34 lines).

**Runner and pipeline changes:**

- **Post-stack output phase:** New `runner_phase_post_stack_output.cpp` (333 lines) and `.hpp` (120 lines) for cleaner output handling after stacking.
- **Resume hardening:** `runner_resume.cpp` expanded (+554 lines) — AQMH reconstruction resume preserves raw vs. selected output separation, common-overlap mask threaded through validation, resume dependencies documented in `docs/process_flow/resume_dependencies_en.md` (+63 lines) and `_de.md` (+66 lines).
- **Runner shared utilities:** New `runner_shared.cpp` (159 lines) and `.hpp` (32 lines).
- **Pipeline rework:** `runner_pipeline.cpp` (+247 lines), `runner_phase_aqmh_reconstruction.cpp` (+523 lines), `runner_preprocess.cpp` (+55 lines).
- **OpenCL sigma-clip:** OpenCL implementations for median/MAD/percentile and sigma-clipping in reconstruction and stacking paths.
- **CFA mosaic warp:** CFA-safe warp handling improved for OSC data in prewarp phase.
- **AQMH reconstruction core:** `aqmh_reconstruction.cpp` (+99 lines), `aqmh_validation.cpp` (+199 lines), `aqmh_sigma_clip.cpp` (+27 lines), `aqmh_quality_map.cpp` (+16 lines), `aqmh_global_quality.cpp` (+10 lines).

**Documentation reorganisation:**

- **README shortened:** `README.md` from 1560 to ~120 lines, `README_de.md` from 1564 to ~120 lines. Both now contain only overview, quick start, and categorised links.
- **New docs structure:** `docs/guides/` (aqmh_overview, workflow, pi_ai, raw_stack_gui), `docs/reference/` (build, cli, docker, outputs, calibration, project_structure), `docs/changelog/` (releases.md + detailed_changelog.md).
- **MkDocs nav rebuilt:** 8 top-level tabs (Home, Getting Started, User Guides, Configuration, Methodology, Reference, Changelog, API Reference). AQMH v0.2.1 methodology and all process flow phases included.
- **GitHub Pages landing page:** `docs/index.md` rewritten as proper landing page with feature overview, quick start, and categorised documentation links.
- **`docs/about.md` updated** with current feature list and tech stack.
- **PI doc link fixes:** Broken links to old `aqmh_methodik_en.md` fixed in `pi_ai_recommendations_en.md` and `pi_ki_empfehlungen_de.md`.
- **Process flow docs updated:** `phase_4_local_metrics.md` (+276 lines), `phase_5_tile_reconstruction.md` (+319 lines), `phase_6_clustering.md` (+247 lines), `phase_8_aqmh_extensions.md` (+48 lines), `README_en.md` (+213 lines), `README_de.md` (+224 lines).
- **Configuration docs updated:** `configuration_reference_en.md` (+114 lines), `configuration_reference.md` (+118 lines), `configuration_examples_practical_en.md` (+38 lines), `_de.md` (+38 lines).
- **AQMH v0.2.0 Zenodo paper:** New paper bundle in `docs/AQMH/zenodo-0.2.0/` with LaTeX, plots from real run artifacts, and JSON summary.

**Other:**

- `AGENTS.md` (182 lines) added — repository-wide coding standards, build/test instructions, architecture boundaries, and workflow rules.
- `.env.example` expanded (+55 lines) with new PI and Docker variables.
- `agent_service/package.json` updated with new dependencies for run-chat, rate limiting, and multi-model support.

### (2026-07-13)

**AQMH v0.2.0/v0.2.1 implementation (`v0.2.1`):**

- **AQMH pipeline finalized:** Added the complete Adaptive Quality Map Harvesting pipeline (`AQMH_MAPS`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS`) with configuration blocks for `pyramid`, `storage`, `cherry_pick`, `global_quality`, `reconstruction`, `validation`, and `diagnostics`.
- **Global quality weighting:** Per-frame global weights `G_f` are computed from sharpness and SNR summaries via robust z-score sigmoid normalization; `g_floor`, `g_w_sharp`, and `g_w_snr` control the weight distribution.
- **Background-gradient penalty (`v0.2.1`):** Added `aqmh.global_quality.g_w_background_penalty` so frames with strong large-scale sky gradients receive lower global weights.
- **Registration robustness:** NCC computation in `try_method` now clamps negative background values and applies a Gaussian blur before correlation, preventing hot pixels from collapsing NCC for sub-pixel shifts. The near-identity bypass now requires `ncc_identity > 0.7` to avoid false accepts on frames far from the reference.
- **Registration weight guard (`v0.2.1`):** Added `registration_weight_guard`, `registration_weight_floor`, `registration_cc_floor`, `registration_cc_full`, `registration_sequential_factor`, `registration_predicted_factor`, `registration_chain_depth_penalty`, and `registration_chain_depth_max_penalty` to dampen per-pixel weights based on registration correlation and sequential-chain depth.
- **Adaptive neutralization:** AQMH reconstruction evaluates raw, low-frequency-neutralized, structure-masked, and optional control-blend candidates. Every selected candidate must pass against both the uniform control and the immutable raw AQMH baseline; otherwise raw AQMH is preserved.
- **Diagnostics controls:** Added master switch `aqmh.diagnostics.enabled` plus `level`, `per_frame_blocks`, `heatmaps`, `regions`, `format`, and `binary_block_size_px`.
- **Bayer pattern auto detection:** `data.bayer_pattern` default changed to `auto`; FITS header keys `BAYERPAT` and `COLORTYP` now take precedence, with the configured value used only as fallback.
- **Documentation sync:** Updated English and German `configuration_reference` documents with all new AQMH v0.2.1 parameters, correct defaults, and runtime behavior; example configs now use `bayer_pattern: auto` and `memory_budget: 4096`.

### (2026-07-03)

**AutoBGE preview improvements, HMS preview, resume support (`v0.3.B`):**

- **Manual sample points in AutoBGE preview:** Users can now add, clear, and persist manual sample points directly in the BGE preview modal. Points are stored as normalized float coordinates `[0..1]` in the YAML config (`bge.autobge.user_sample_points`), making them resolution-independent. Points survive modal reopenings and resume operations. Manual points are always included in the sample set, bypassing random downselection.
- **Guard rejection reasons in UI:** The BGE preview now displays detailed guard rejection reasons (flatness worsened, background chroma worsened, slope worsened, etc.) with per-channel breakdown and actionable hints. Localization keys added for all guard reasons and hints in both EN and DE.
- **HMS preview:** HyperMetric Stretch preview modal added alongside BGE preview, with live parameter editing and preview image rendering.
- **Resume support:** BGE and HMS preview modals integrate with the resume workflow — applying changes patches the YAML config and triggers a resume from the relevant phase.
- **Frontend formatting:** `bge-preview.js` reformatted for human readability with proper indentation and line breaks.

### (2026-06-27)

**GPU performance optimizations (`v0.3.9`):**

- Added centralized per-worker CUDA stream management across PREWARP, AQMH quality maps, tile and synthetic reconstruction, STACKING, resume, and the preprocessing pipeline.
- Added streaming CUDA acceleration for AQMH_RECONSTRUCTION; frame count no longer determines VRAM usage, with automatic CPU fallback for Cherry-Pick mode or CUDA errors.
- Reduced synchronization and allocation overhead through batched OSC/CFA warps, cached per-worker AQMH CUDA filters, and concurrent RGB-channel stacking.
- Removed the obsolete global OpenCV/OpenCL mutex and added CPU-worker, GPU-usage, and backend information to live progress logs.

### (2026-06-26)

**AutoBGE implementation, AI update (`v0.3.8`):**

- **AutoBGE:** Native C++ implementation of the AutoBGE background gradient extraction, ported from the AutoBGE Siril script (© Adrian Knagg-Baugh, 2025). Two-stage polynomial + RBF background model with smart sample point generation and gradient descent to dimmest local spot. Integrated as `bge.method=autobge` alongside the existing `classic` BGE path.
- **AI update:** New `sky_gradient` metric in `scan-metrics` measures large-scale background gradient (quadrant median diff / overall median), separate from `gradient_energy` (local pixel-scale Sobel). The AI sidecar prompt now uses `sky_gradient` with decision thresholds (≥0.05 → recommend BGE; 0.02–0.05 → moderate; <0.02 → negligible) to provide accurate BGE recommendations. BGE rules updated: prefer `classic` over `autobge`, `poly` over `rbf`, `extended` autotune strategy for nebulosity.

### (2026-06-22)

**GUI v3, calibration improvements, astrometry fix (`v0.3.5`):**

- **New web frontend (v3):** Improved Parameter Studio, Run Monitor, and Input & Scan tab with modern UI components.
- **"Save As" dialog with directory browser:** Modal dialog to select directory and filename for saving config files. Backend `/api/config/presets` now returns subdirectories (`is_dir` field) alongside YAML files.
- **Calibration gain mismatch:** Dark/Flat calibration with mismatched gain now produces a warning instead of aborting the run. `select_dark_inputs` falls back to all available darks when no exact match is found.
- **Warning banner in Run Monitor:** Warnings and errors from the runner are prominently displayed in a dedicated banner during runs.
- **Astrometry fix:** YAML `~` (null) for `astap_bin` was incorrectly parsed as string `"null"` by yaml-cpp, causing `fs::exists("null")` to fail and ASTAP to be reported as "not found". Fix: `IsNull()` check added before parsing `astap_bin` and `astap_data_dir` in `config.cpp`, so null values are treated as empty string and the default path (`astap_data_dir/astap_cli`) is used.

### (2026-06-20)

**PCC fix, AI prompt enhancements, build and smoke test corrections (`v0.3.4`):**

- **PCC green cast fix:** Restored the missing `if (!matrix_is_diagonal)` guard in the adaptive damping logic in `run_pcc` (`photometric_color_cal.cpp`). Without this guard, damping was applied even for diagonal matrices, causing a green cast in the stacked image. The fix ensures damping is only applied for non-diagonal matrices. Validated across multiple test runs (k3-glm, k6-fix) confirming correct PCC matrix and no green cast.
- **AI prompt enhancements:** Enhanced the AI sidecar prompt in `frameAnalysisService.ts` with stricter rules and session context. The AI now receives session geometry (mount type, field rotation estimate, session duration) alongside scan metrics, enabling more context-aware recommendations.
- **Session geometry in `cli_main.cpp`:** Extended FITS header reading to extract RA/DEC and compute session geometry (mount type detection, approximate field rotation from session duration and declination). This data is forwarded to the AI sidecar for analysis.
- **`ai_routes.cpp` fix:** Fixed a bug where `base_config` was stored as a raw string instead of a parsed JSON object. Added `session_geometry` to the analysis context sent to the AI sidecar.
- **Parameter tuning validation:** Conducted systematic comparison runs (k2, k3-glm, k4, k5, k6-fix, m31-classic, m31-default, m31-cl1) to validate PCC fix and identify optimal parameters: `weight_exponent_scale` (1.2 vs 1.5), `chroma_strength` (0.7–0.9), `apply_attenuation` (true/false), `background_neutralization_mode` (auto/always). Results confirmed the PCC fix across all configurations.
- **Windows build fix:** `timegm` is a GNU extension not available under MinGW. Replaced with `#ifdef _WIN32` → `_mkgmtime` / `#else` → `timegm` in `cli_main.cpp` for cross-platform compatibility.
- **Linux smoke test fix:** Smoke tests in `release-tile-compile-gui2.yml` hung for 1h+ because `kill` only terminated the bash script, not the backend process. Applied the following fixes:
  - Set `TILE_COMPILE_AI_AGENT_AUTOSTART=0` (prevents agent service from starting)
  - Used `setsid` + `kill -- -PID` for process group cleanup
  - Added `pkill -f tile_compile_web_backend` as fallback cleanup
- **macOS smoke test fix:** Applied the same fixes in `build_local_macos.sh`. Replaced `setsid` (Linux-only) with `python3` + `os.setsid()` since `setsid` is not available on macOS.

### (2026-06-16)

**PI – AI-assisted configuration recommendations (`v0.3.3`):**

- Implemented the PI (Parameter Intelligence) module: the AI sidecar receives the full scan result, frame quality metrics aggregate (`scan_metrics`: FWHM, noise, background, roundness, star count), all relevant configuration parameters with descriptions, and the complete schema constraints (`min`, `max`, `enum`) — and produces validated, data-driven configuration recommendations.
- `scan_metrics` are now forwarded from the backend to the AI sidecar in both POST and SSE analysis routes (`ai_routes.cpp`).
- `config_schema` sent to the sidecar now includes `minimum` and `maximum` from the JSON schema; the AI prompt formats these as `min:`/`max:` per parameter line and contains an explicit rule: values MUST stay within `[min, max]`.
- Per-update validation in `validate_updates_against_schema()`: if the combined patch fails schema validation, each update is tested individually and cumulatively; only the offending updates are rejected with `config_validation_failed` — valid recommendations are no longer discarded.
- `start_backend.sh`: agent_service sidecar is rebuilt automatically via `npm run build` whenever any `.ts` source file is newer than `dist/server.js`; `npm install` runs automatically if `node_modules` is missing or `package.json` changed.
- Traffic log: `prompt_length` diagnostic entry added before the prompt log to verify which sections are present in the prompt without relying on the truncated log output.
- Full documentation: [docs/PI/pi_ai_recommendations_en.md](../PI/pi_ai_recommendations_en.md)

### (2026-06-13)

**AQMH-First Implementation:**
- Top-level `method` field as single source of truth: `method: aqmh` or `method: classic_tile_compile`.
- Config normalization: Missing `method` automatically defaults to `aqmh`, `aqmh.enabled` is derived from it.
- Schema validation for `method` field with enum values `aqmh` and `classic_tile_compile`.
- Frontend: `currentMethod()` API, AQMH as default in Wizard, Dashboard, Parameter Studio.
- Run Monitor: Classic phases are hidden for AQMH (`LOCAL_METRICS`, `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`).
- Rollback: `FORCE_CLASSIC=1` environment variable and `--force-classic` CLI flag.
- BGE: `tile_metrics_source` set to `aqmh_output` for AQMH runs.
- Reports: Standalone AQMH section with quality map heatmaps.
- History: Method tags for filtering and comparison.
- Documentation: [AQMH First Frontend Transition Plan v0.3.0](../AQMH/attic/aqmh_first_frontend_transition_plan.md)

### (2026-06-09)

**Switch to AQMH as default reconstruction method:**
- AQMH (Adaptive Quality Map Harvesting) is now the default reconstruction path. Set `aqmh.enabled: false` to revert to classic TILE_RECONSTRUCTION.
- Normative specification: [AQMH Methodology v0.1.0](../AQMH/aqmh_methodik_en_v0.2.1.md)
- All example YAML profiles updated with `aqmh:` configuration block.
- Full AQMH parameter documentation added to configuration reference and practical examples.
- `k_artifact` implementation default changed to `3.0`, `frac_artifact_max` to `0.25`.
- Main READMEs restructured; classic TBQR documentation preserved in `README_classic_tile_compile_en.md` / `README_classic_tile_compile_de.md`.

### (2026-05-25)

**Raw Stack preprocessing pipeline (`v0.2.8`):**

- Added a new standalone Raw Stack page in GUI2 for end-to-end preprocessing of FITS light frames through to a stacked and post-processed image, running fully separately from the normal Tile-Compile run studio.
- The pipeline covers all phases: Calibration (Bias/Dark/Flat), CFA/Mono Prep, Registration, Quality Analysis, Frame Filtering, Stacking (Sigma/Median/Winsor), Astrometry (ASTAP), Background Gradient Extraction (BGE), Photometric Color Calibration (PCC), and HyperMetric Stretch.
- All configurable parameters (sigma-clip, rejection method, stacking weighting, BGE, PCC, Astrometry, and HyperMetric Stretch) are taken directly from the Parameter Studio configuration — no hardcoded values.
- Output scaling correctly restores background and scale after stacking to produce accurate pixel values.
- Raw Stack UI cleanup: removed Run Monitor button, added full i18n coverage for all labels and buttons.
- See [docs/raw_stack_gui_en.md](../raw_stack_gui_en.md) for the full GUI reference.

### (2026-05-22) 

**implementationHMS:**
- Added VeraLux HyperMetric Stretch (HMS) as a post-PCC pipeline phase.
- HMS now defaults to enabled in the C++ config defaults, `tile_compile.yaml`, and all example YAML profiles.
- The default mode is `ready_to_use`, using adaptive anchor, Auto LogD, target background `0.2`, and output `outputs/stacked_rgb_hms.fits`.
- `mode: scientific` is implemented for controlled stretch output without ready-to-use final scaling/soft clip and with optional `linear_expansion`.
- Resume supports rerunning HMS directly via `--from-phase HYPERMETRIC_STRETCH` for historical runs with existing PCC artifacts.

### (2026-05-20) 

**Build hardening & Frontend cleanup:**
- Fixed RunnerFrameCache build errors: implemented missing `try_load_normalized` and `store_normalized` methods
- Migrated both C++ projects to C++20 (GCC 13+, Clang 16+)
- Hardened web_backend_cpp build with CUDA 13 + OpenCV 4.11 CUDA 13 configuration
- Backend route_utils: fixed incomplete AppState type errors, hardened path validation
- Frontend refactoring: centralized utilities in `src/utils.js` (escapeHtml, getMessage, getStorageJson, humanizeControlId, etc.)
- Migrated shell.js, parameter-studio-page.js, and tooltips.js to ES6 modules with shared utils.js imports
- Eliminated duplicate I18N functions across frontend scripts (message(), textFor(), activeLocale(), getLocale())
- Removed dead code: `param_editor_index.json` (36KB unused duplicate)
- Updated documentation: unified C++20 requirements, release URLs updated to v0.2.5

### (2026-04-26)

**Documentation system and BGE robustness (`v0.2.5`, 2026-04-26):**

- Added professional documentation system using MkDocs Material with Doxygen integration for C++ API reference
- Updated all GitHub Releases documentation with correct binary filenames (tile_compile_gui2-linux-v0.2.4.zip, etc.)
- Added comprehensive installation instructions for pre-built binaries on Ubuntu/Debian, Fedora/RHEL, and Arch/Manjaro
- Restructured navigation with separate User Guide, Configuration, Methodology, and API Reference sections
- Added configurable `bge.sample_estimator` support in YAML configs, schema files, and Parameter Studio (`quantile`, `sigma_clipped_median`, `sextractor_mode`, `biweight`)
- Extended BGE autotune so it can compare sample estimators and penalize or reject flat models when background/chroma spread indicates a real gradient
- Extended RGB chroma guards across BGE methods, including conservative fallbacks for imbalanced per-channel correction surfaces
- Updated `ic434_background_gradient.example.yaml` with robust RBF/`sextractor_mode` settings for IC434-like red/green background gradients
- Added `docs/reconstruction_audit_2026-04-26.md` with the reconstruction audit checklist and implementation notes
- Hardened reconstruction fallback helpers against mismatched frame/tile shapes and missing or invalid tile weights
- Reworked `reconstruct_tiles_parallel()` to use tile-sized temporary OLA buffers instead of full-frame scratch matrices per tile/sub-batch
- Updated reconstruction memory budgeting so it accounts for global overlap-add accumulators plus per-worker tile scratch
- Removed ineffective reconstruction scheduler/config dead code, including the unused GPU batch field, unused `make_hann_1d()` API, and non-functional underutilization detector

### (2026-04-25)

**Registration performance: parallel anchor-promotion retries (`v0.2.4`):**

- Fixed the direct-registration anchor-promotion loop so promoted-anchor retry passes use the configured parallel registration worker pool instead of falling back to a single-threaded `reg_worker()` call.
- Promotion rounds now build a targeted retry list and only revisit unresolved frames whose nearest active anchor is one of the newly promoted anchors, avoiding repeated full 325-frame passes when the anchor set changes.
- Registration progress now reports the actual job count and worker count for each pass, and `global_registration.json` diagnostics include `reg_promotion_retry_frames` for future runtime analysis.

### (2026-04-24)

**Registration robustness: deep-chain rejection + adaptive anchors + hopping rescue + astrometric fallback (`v0.2.3`):**

- Reject chain-validated frames with `chain_depth > max_blind_chain_depth` and `cc < reject_cc_min_abs` as `deep_chain_low_cc` outliers instead of accepting them; this prevents drift from long sequential chains through cloudy blocks.
- Increased adaptive active-anchor target from `min(21, max(3, (N+59)/60))` to `min(32, max(4, (N+29)/30))`, doubling anchor density for large-N sessions (e.g., 325 frames now use ~12 anchors instead of ~6).
- "Hopping" sequential rescue: when the direct neighbor has low CC or cannot anchor a blind chain, search up to 5 frames (for refine) or 8 frames (for rescue) in each direction for a better anchor with CC > 0.3–0.4, dramatically reducing chain depth in scattered-cloud conditions.
- Astrometric rescue moved to run *after* model-based warp prediction (Section 4b), so ASTAP can now also rescue frames that only have interpolated `model_*` provenances; added `weak_model` condition to `should_try_astrometry` so low-CC model frames are eligible for plate-solving.

### (2026-04-24)

**Hot/dead-pixel correction fix + registration code quality (`v0.2.2`):**

- Fixed `cosmetic_correction_cfa` silently skipping defective pixels inside star regions: `neighbor_threshold` was set to `0.5 × global_threshold`, so star-halo pixels (which sit well above that low bar) were counted as "hot neighbours" and blocked correction of genuine hot pixels nearby. The threshold is now raised to the full global hot-pixel threshold, so only pixels that are themselves hot-pixel candidates count as hot neighbours.
- Added `extreme_outlier` bypass: pixels exceeding `local_median + 5 × local_floor` are replaced unconditionally regardless of neighbourhood support. No real star-PSF pixel reaches that level relative to its same-colour neighbours.
- Added dead/cold-pixel correction: `global_candidate_cold` (`< median − σ_threshold × σ`) and `cold_outlier` (`< local_median − local_floor`) are now also replaced with the local same-colour median.
- All three fixes operate on the raw CFA mosaic before warping and require no dark frames.
- Diagnostic keys in `global_reg_extra` moved into a `diag` sub-object (4.2); downstream-facing keys remain at top level.
- Section headers added to `run_phase_registration_prewarp` to mark the seven major processing phases (4.1).

### (2026-04-23)

**Registration NCC robustness + near-identity guard (`v0.2.1`):**

- NCC computation in `try_method` now clamps negative values and applies a Gaussian blur (σ=1.5) before computing `ncc_identity_overlap` and `ncc_warped`. Raw normalized proxy images carry negative background values and hot pixels that caused NCC to collapse from ~0.88 to ~0.05 for sub-pixel shifts, triggering false near-identity rejections.
- Near-identity bypass condition strengthened with an `ncc_identity > 0.7` guard: a near-zero warp is only accepted as a valid near-identity result when the frame is already close to the reference, preventing false bypasses for frames that simply failed to find a shift.

### (2026-04-14)

**Registration v0.2.0: multi-anchor scaling + astrometric registration/rescue:**

- Global registration no longer relies on rigid `1/3/5` reference buckets. It now uses an N-scaled anchor selection with roughly one requested anchor per 80 frames, forced to odd anchor counts and currently capped at 15.
- Anchor promotion after strong direct matches now scales with `N` as well: the active-anchor target is roughly one anchor per 60 frames, while per-round promotions and the number of extra direct passes grow in a controlled way for long sessions.
- This reduces the classic late-reference failure mode on long Alt/Az datasets, because early and late parts of the sequence can attach directly to nearer temporal anchors instead of being forced through one distant master frame.
- Astrometric registration/rescue in the runner was upgraded in practice: ASTAP-based solves are no longer limited to `cc <= 0`, but can also replace weak or deeply chained results, using the nearest active anchor as the astrometric reference basis.
- New registration telemetry was added to `global_registration.json`, including `requested_ref_frames`, `active_ref_frames`, `reg_target_active_anchor_count`, `reg_promote_limit_per_round`, `reg_max_direct_anchor_rounds`, `reg_direct_anchor_rounds`, and `reg_source_counts`.
- Added the new example profile [tile_compile_cpp/examples/m104.example.yaml](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/m104.example.yaml) for the concrete problem class "Alt/Az, somewhat stronger rotation, poor seeing, weight better frames more strongly"; the DE/EN practical examples and [docs/process_flow/phase_1_registration.md](../process_flow/phase_1_registration.md) were updated to match the current registration flow.

### (2026-04-07)

**TILE_RECONSTRUCTION performance: sub-batch stacking replaces worker reduction (`v0.1.F`):**

- Replaced the memory-budget-driven worker reduction in TILE_RECONSTRUCTION with frame sub-batching. Previously, a 2 GB memory budget capped OSC runs at 3 parallel workers (instead of the configured 8) because the peak RAM estimate assumed all frames loaded simultaneously per worker. Workers now always run at the configured `parallel_workers` count; the budget controls the sub-batch size (frames per batch) instead. For the reference run (610 frames, 475 tiles, 8 workers, 2 GB budget) this yields ~3 batches of ~205 frames each — same quality, ~2.7× faster TILE_RECONSTRUCTION.
- `tile_boundary_diagnostics_enabled` added to `runtime_limits` (default: `false`). Boundary diagnostics are now opt-in; the previous default of always running them added ~5–10 % overhead per production run.
- `tile_grid.json` now includes `estimated_reconstruction_time_s` (calibrated estimate based on tile count, frame count, and worker count) and `coverage_filtered_tiles`.
- `runtime_limits.json` now includes `tile_analysis_to_stack_ratio`; a warning is logged when the ratio exceeds 10.
- `phase_end` event for TILE_RECONSTRUCTION now includes `duration_s`.
- web_backend_cpp code-quality fixes: consolidated three duplicate `utc_now_iso()` implementations into a shared header, fixed SIGKILL being sent on every polling cycle after SIGTERM (now waits ~3 s), fixed FD leak on `fork()` failure, fixed sequential stdout/stderr read deadlock in `run_subprocess()`, and reduced `prune_locked()` call frequency from every mutation to terminal-state transitions only.

### (2026-04-06)

**Calibration fix + Parameter Studio reorganization (`v0.1.E`):**

- Fixed the bias/dark calibration path: when bias and dark are both enabled, a raw dark is now bias-corrected internally before being applied to lights, preventing double subtraction of the bias pedestal.
- Added the new config field `calibration.dark_already_bias_corrected` across runner, schema, docs, defaults, example configs, and GUI2 so pre-bias-corrected master darks can be marked explicitly.
- Reorganized Parameter Studio: selecting a category such as `registration` or `calibration` now shows exactly one consolidated section; missing schema parameters are injected into that same block instead of being rendered in a separate section editor above it.

### (2026-04-05)

**Calibration guardrails and backend persistence for calibration paths:**

- Updated GUI2 calibration-path handling so disabling a calibration stage removes its paths from the active config immediately and re-enabling restores the previously used paths from backend UI state, without using browser storage.
- Added extra calibration guardrails, including warnings for obvious gain mismatches between light frames and calibration files.

### (2026-04-04)

**Auto-engine for Alt/Az field rotation + registration failure fix (`v0.1.D`):**

- Added `registration.auto_engine` (default: `true`): probes a small set of frames before registration starts and automatically overrides the engine to `triangle_star_matching` + `transform_model: affine` when a rotation-blind engine (`robust_phase_ecc`, `hybrid_phase_ecc`) is configured but strong field rotation is detected. Threshold: `auto_engine_rotation_threshold_deg` (default: `0.05°/frame`, covers Alt/Az at any exposure time while staying well below EQ residual rotation).
- Fixed a complete registration failure mode: `engine: robust_phase_ecc` with `allow_rotation: true` on Alt/Az datasets produced NCC ≈ 0 for all frames, causing 469/470 frames to fall back to identity transform (`model_nearest_copy`) with no actual alignment.
- Updated `tile_compile.yaml` default engine to `triangle_star_matching` and `reject_cc_min_abs` to `0.05`.
- Propagated new config fields to all schemas, example configs, and documentation.

### (2026-04-03)

**Tile-reconstruction stabilization after recent optimization rollout (`v0.1.C`):**

- Stabilized tile reconstruction after the recent performance optimization rollout, with follow-up fixes and analysis focused on visible tile-seam artifacts in the final reconstruction output.

### (2026-03-29)

**RGB/PCC output-path stabilization after the `v3.3.9` rollout (`v0.1.A`):**

- Reworked the visible RGB output stretch so it operates luminance-aware and keeps chroma stable instead of exaggerating small background channel offsets into large blue/gray edge bands.
- Added `pcc.background_neutralization_mode = always|auto|off` with a new auto guard that attenuates or suppresses background neutralization when the measured "background" behaves like diffuse field signal rather than neutral sky.
- Synchronized the new PCC control through schema, defaults, reference docs, and all example configurations so the runtime, documentation, and example surface now expose the same behavior.

### (2026-03-28)

**Implementation and rollout of the `v3.3.9` methodology (`v0.1.9`):**

- Moved the key `v3.3.9` methodology changes into the active runtime path: linear reconstruction core without the old pre-OLA tile normalization, cleaner BGE/PCC semantics, more robust support/seam handling, and updated guards/diagnostics.
- Updated the frontend and configuration surface to the current schema/methodology baseline so new `v3.3.9` parameters are exposed more consistently in Parameter Studio and related documentation.
- Refreshed the process-flow, reference, and comparison documents for `v3.3.9`, and additionally hardened Crow/C++ web-backend startup so failures now report a clear fatal error instead of producing a core dump.

### (2026-03-24)

**AppImage packaging fix + file browser navigation enhancement (`v0.1.7`):**

- Fixed Linux AppImage packaging in `packaging/gui2/start_gui2.sh` to export `TILE_COMPILE_INPUT_SEARCH_ROOTS` environment variable, resolving directory scanning failures in packaged releases where relative paths could not be resolved.
- Enhanced GUI2 file browser (`web_frontend/tooltips.js`) to always display parent directory (..) navigation even when the parent path is not yet granted, showing a lock icon (🔒) for restricted paths and triggering the permission grant dialog on click for seamless upward navigation.
- Updated backend file listing route (`web_backend_cpp/src/routes/system_routes.cpp`) to return `parent_allowed` flag alongside `parent` path, enabling frontend to distinguish between accessible and restricted parent directories.

**GUI2 batch/queue run-monitor refresh + docs update (`v0.1.6`):**

- Reworked the GUI2 Run Monitor for queue/batch runs: queue entries now appear as tabs, redundant duplicate batch/filter rows were removed, and top-level batch/structure summary visibility was corrected again for queued runs.
- Enabled batch-targeted post-run actions in the Run Monitor so `Generate Stats`, stats-folder opening, and report opening can operate on the currently selected finished batch tab instead of only the active root/current run.
- Changed unnamed queue-root naming from date-only to `YYYYMMDD_HHMM`, making batch-root directories less collision-prone and aligning the dashboard/wizard path hints with the actual behavior.
- Expanded the EN/DE step-by-step guides with explicit batch/queue usage notes, including the primary MONO multi-filter use case and Run Monitor tab behavior.

### (2026-03-23)

**OpenCL expansion for `PREWARP`, `TILE_RECONSTRUCTION`, and `STACKING` (`v0.1.5`):**

- Stabilized the OpenCL `PREWARP` path for multi-threaded execution by guarding OpenCV OpenCL/T-API access and forcing explicit host copies where needed.
- Extended `tile_compile_cpp/src/core/acceleration.cpp` with OpenCL equivalents for the previously CUDA-only `TILE_RECONSTRUCTION` and `STACKING` paths, including sigma-clipping and overlap-add accumulation/normalization.

### (2026-03-22)

**Real `STACKING` resume + synthetic OLA seam fix (`v0.1.4`):**

- Implemented a true artifact-based `STACKING` resume path in `tile_compile_cpp/apps/runner_resume.cpp`, so `resume --from-phase STACKING` now rebuilds the stacked outputs directly from existing `synthetic_*.fit` plus `canvas_mask.fits` and continues with later phases instead of triggering an in-place full rerun.
- Fixed one overlap-add accumulation failure mode in `tile_compile_cpp/src/core/acceleration.cpp`: zero/invalid tile pixels no longer add Hann weights to `weight_sum`. This removes that specific darkening path, but residual internal seams/lines may still have other causes.

### (2026-03-21)

**Registration provenance/depth diagnostics + resume status visibility (`v0.1.3`):**

- Extended `tile_compile_cpp/apps/runner_phase_registration.cpp` so each frame now carries explicit registration provenance (`direct_global`, `sequential_rescue`, `temporal_rescue`, modeled variants, etc.) plus `chain_depth`, and writes that information into `global_registration.json`.
- Tightened blind sequential chaining: weak `sequential_rescue` frames no longer act as effectively unlimited anchors; anchor reuse is now capped by chain depth unless correlation is strong enough.
- Added aggregate registration diagnostics such as source counts, maximum observed chain depth, and blocked blind-chain-anchor attempts to the registration artifact metadata.
- Fixed GUI2/backend resume status handling so the monitor subtitle and phase state update immediately after `resume`, including the case where the runner has started but the next `resume_start` event has not yet been written to the run log.

### (2026-03-20)

**Registration/field-rotation stabilization for long Alt/Az sessions (`v0.1.2`):**

- Fixed global registration validation in `tile_compile_cpp/` so NCC comparisons are computed only on the actual valid overlap mask of the warped frame instead of on the cropped full-frame canvas. This prevents correct larger-rotation warps from being rejected just because rotated corners fall outside the fixed proxy image.
- Applied the same overlap-masked NCC validation to temporal rescue chaining, so neighbor-to-reference rescue no longer fails for the same cropped-canvas reason.
- Reworked global registration outlier CC filtering for long rotating runs: the `low_cc` reject gate now uses the configured absolute minimum directly instead of a run-global median/MAD threshold that incorrectly rejected many geometrically plausible edge frames.
- Fixed field-rotation model prediction outside the span of real registrations: tail/head frames no longer use unstable local polynomial extrapolation and instead fall back to bounded bridge-style edge prediction, preventing the severe fan-out / wedge artifacts seen on the M66 Alt/Az regression run.

### (2026-03-19)

**GUI2 tool persistence/PCC UX, backend memory guards, and BGE autotune speed-up:**

- Hardened `web_backend_cpp/` against OOM-prone API/tool paths with capped subprocess/stdout capture, bounded scan/report payload retention, streamed event-file inspection, and retained-job limits plus environment-configurable defaults documented for GUI2.
- Added `packaging/gui2/.env.example` and documented the new backend runtime limit environment variables used by GUI2 launchers.
- Fixed GUI2 frontend/backend asset serving and route behavior so `/ui` and direct asset paths resolve reliably instead of producing 404s.
- Improved Astrometry/PCC tool UX: persistent in-progress download state across page switches, corrected download progress calculation, automatic PCC WCS prefill from matching files, and automatic PCC parameter import from a run `config.yaml` with visible traceability in the UI/log.
- Reworked PCC output handling in GUI2 so `Run PCC` writes to a temporary result, `Save Corrected` uses a styled in-app save dialog, copies the RGB result plus `_R/_G/_B` sidecar files from the temp output, and works consistently across Linux/macOS/Windows temp directories.
- Fixed standalone PCC fallback behavior when `canvas_mask` is missing by using a safe full-image fallback instead of aborting the tool run.
- Added BGE phase timing diagnostics to `bge.json` and optimized the real hotspot in `tile_compile_cpp/`: autotune prep now reuses prepared tile analysis across quantile candidates, reducing measured BGE wall time on the IC434 reference run from about `472s` to about `181s` without adding new full-frame memory pressure.

### (2026-03-18)

- Fixed Astrometry data directory input not being respected when user manually changes the path - now uses `shouldKeepAstapSelection` logic to preserve user input.
- Added server-side persistence for Astrometry and PCC tool parameters via UI state API - settings survive server restarts.
- Improved catalog download intelligence: Astrometry catalogs skip download if already installed, PCC Siril only downloads missing chunks.
- Enhanced archive extraction robustness for macOS `.pkg`, Linux `.deb`, and Windows `.exe` formats with better error messages and validation.
- Fixed macOS release bundle library issues by explicitly bundling GCC runtime libraries (`libgcc_s`, `libgfortran`, `libquadmath`, `libgomp`) and preserving `libstdc++` for Homebrew-compiled dependencies.

### (2026-03-17)

**Methodology `v3.3.8` + GUI2 run-name reset (`v0.0.F`):**

- Added new normative methodology documents `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md` and `docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_de.md`.
- Corrected the method specification so it matches the active runtime for operating-mode thresholds, shared-core channel semantics, neighborhood-aware local metric normalization, sigma-clipped tile reconstruction, and affine post-OLA photometric restoration.
- Fixed GUI2 so a changed input directory clears the shared `run_name` across dashboard, wizard, and input-scan.
- Added a short macOS 15 / Sequoia install note for Gatekeeper-blocked `start_gui2.command` launch.
- Changed ASTAP `d80` downloads from the invalid shared ZIP assumption to the real upstream platform packages: Linux `.deb`, macOS `.pkg`, Windows `.exe`.

### (2026-03-15)

**Assumptions runtime/config synchronization (`v0.0.E`):**

- `assumptions.frames_min` is now used by the active C++ runner mode gate instead of the old hardcoded minimum-frame threshold.
- `assumptions.reduced_mode_cluster_range` now affects reduced-mode clustering directly, so the exposed config field is no longer parser-only drift.
- Removed dead assumptions fields from the active config surface: `pipeline_profile`, `frames_optimal`, and `exposure_time_tolerance_percent`.
- Synchronized active C++ config code, generated schemas, example YAMLs, GUI2 Assumptions/Parameter Studio, and DE/EN docs to the remaining runtime-relevant assumptions fields.

### (2026-03-15)

**Boundary diagnostics deepening + GUI2 run-field synchronization:**

- Extended `TILE_RECONSTRUCTION` diagnostics so `tile_reconstruction.json` now exposes raw and normalized tile-boundary metrics separately, plus `tile_norm_bg_r/g/b` and `tile_norm_scale` for direct normalization analysis.
- Corrected tile-boundary analysis to exclude masked `COMMON_OVERLAP` / canvas-invalid zones instead of counting them as valid zero-valued samples.
- Updated the methodology/process/reference/practical docs to reflect the read-only raw/normalized boundary diagnostics and the common-canvas-mask requirement.
- Added `run_name` and `runs_dir` editing to Input&Scan and unified both fields across dashboard, wizard, and input-scan via the shared GUI2 stored state.

### (2026-03-13)

**GUI2 config/studio sync + tile-boundary diagnostics update:**

- Removed the ineffective `stacking.tile_seam_harmonization.*` experiment from the active C++ config surface and replaced it with read-only tile-boundary diagnostics in `TILE_RECONSTRUCTION`.
- Synchronized config code, generated schemas, example configs, and DE/EN reference docs with the active C++ config surface.
- Reworked Parameter Studio so parameter inventory, defaults, ranges, tooltips, and filtering are driven from the current schema/default config instead of stale manual lists.
- Extended GUI2 live-log and run-monitor behavior, including richer phase details, resume config editing/template flows, stored config revisions, and corrected phase status after successful resume.

### (2026-03-12)

**Server-side GUI2 UI-state persistence:**

- Added persistent backend storage plus API access for the GUI2 UI draft state so frontend UX state no longer depends primarily on local browser storage.
- Migrated the major UX-relevant frontend parameters to the shared server-backed UI state, including run naming, preset synchronization, config drafts, validation state, dirty state, queues, and tool path/input settings.
- Restored and synchronized additional tool result state across reloads where useful, while keeping purely ephemeral runtime display state non-persistent.

### (2026-03-11)

**Crow/C++ runtime, release packaging, and PCC update:**

- Finalized the productive GUI2 path around the Crow/C++ backend, including integrated C++ report generation and aligned frontend/backend report handling.
- Updated release packaging, local build/start scripts, and GitHub workflows for Linux, macOS, and Windows, including the documented GUI2 bundle OS baselines.
- Added Linux AppImage creation to the GitHub Actions release workflow so releases now include a portable Linux artifact alongside the ZIP bundle.
- Added date-aware run-directory naming and aligned route/websocket handling plus backend tests for the new naming behavior.
- Reworked PCC background-noise handling and connected UI/report updates so current PCC diagnostics are exposed more consistently in the GUI.

### (2026-03-09)

**GUI2 release + i18n refresh:**

- Promoted the web-based GUI2 stack (`web_frontend/` + `web_backend_cpp/`) to the recommended UI path and updated the top-level docs accordingly.
- Added the dedicated GUI2 release workflow and launcher packaging for Linux, macOS, and Windows under `.github/workflows/release-tile-compile-gui2.yml` and `packaging/gui2/`.
- Expanded frontend localization coverage and parameter-studio translations, with matching backend config contract updates and tests.
- Moved the earlier Qt6 GUI/build-script path into `legacy/` to separate the maintained GUI2 route from the legacy desktop implementation.

### (2026-03-10)

**Python elimination in the productive GUI2 path:**

- Switched GUI2 runtime, packaging, Docker, and CI to the Crow/C++ backend.
- Removed the productive Python dependency for stats/report generation; this now runs via the integrated C++ backend path and CLI support.
- Updated the repository structure and GUI2 documentation to reflect `web_backend_cpp/` as the maintained backend implementation.

### (2026-03-05, later update)

**Strict/Practical runtime unification + verification:**

- Unified the image-processing runtime core path for `assumptions.pipeline_profile: strict|practical`.
- Removed strict-only execution branches in the hot path:
  - no strict-only pre-registration order path,
  - no strict-only reduced/full gate override (`max(200, threshold)`),
  - no strict-only tile re-normalization branch,
  - no strict-only channel re-weighting branch in OSC tile stacking.
- Registration no longer force-overrides `registration.enable_star_pair_fallback=false` in strict mode.
- Updated config reference docs (EN/DE) so profile text matches current runtime behavior.
- Added A/B evidence run pair (`max_frames=80`) confirming same core flow with only minor numeric fit variance.

### (2026-03-05)

**Performance and throughput optimization (large datasets, 1000+ frames):**

- Added adaptive worker selection per phase with I/O-aware caps based on sampled frame size and task count.
- `DiskCacheFrameStore` now uses persistent memory-mapped frame views with rewrite invalidation, reducing repeated open/mmap/unmap overhead for tile access.
- Removed the global PREWARP store mutex so frame-cache writes can proceed concurrently.
- `GLOBAL_METRICS` now runs in a parallel worker pool with thread-safe progress and error aggregation.
- `TILE_RECONSTRUCTION` overlap-add switched from a single global lock to row-stripe locking to reduce contention.
- In OSC tile reconstruction, each valid frame tile is debayered once and cached as R/G/B planes for reuse across channel stacks.
- `LOCAL_METRICS` now skips globally invalid tiles before extraction and limits heavy full-matrix artifact writes for large production runs.

### (2026-03-03)

**Methodology alignment (v3.3.6 strict profile):**

- Added `assumptions.pipeline_profile: practical|strict` to switch between compatibility mode and strict normative behavior.
- In `strict` profile, REGISTRATION/PREWARP is executed before CHANNEL_SPLIT/NORMALIZATION/GLOBAL_METRICS.
- In `strict` profile, reduced/full gating enforces full mode only from `N >= 200`.
- In `strict` profile, phase-7 tile normalization before OLA is always enabled.
- PCC `auto_fwhm` now falls back deterministically to `FWHM=0` when seeing is unavailable.
- Added `registration.enable_star_pair_fallback` (default `true`); strict profile disables it to match the normative cascade order.
- Updated config schema/sample config and v3 reference docs (DE/EN) for these settings.

**BGE/PCC configuration and docs alignment:**

- Restored user-facing BGE fit parameters `bge.fit.robust_loss` and `bge.fit.huber_delta`.
- Added user-facing BGE apply guards `bge.min_valid_sample_fraction_for_apply` and `bge.min_valid_samples_for_apply`.
- Re-enabled parse/serialize/schema support for these keys in the runtime config surface.
- Runner mapping now forwards the configured values (no internal forced override).
- BGE config artifacts in both pipeline and resume paths include `robust_loss` and `huber_delta` again.
- Updated BGE/PCC docs and practical examples (DE/EN) to match current behavior and active parameter set.

### (2026-02-26)

**BGE Phase Visibility / Comparison Outputs:**

- BGE is now emitted as a dedicated pipeline enum phase (`BGE=15`) between `ASTROMETRY` and `PCC`.
- GUI phase progress now shows BGE explicitly, including BGE substep progress updates.
- Added explicit pre-PCC output `outputs/stacked_rgb_bge.fits` for direct BGE-only vs BGE+PCC comparison.
- Configuration docs/examples updated for v3.3.6 option set:
  - `bge.autotune.*` (`enabled`, `strategy`, `max_evals`, `holdout_fraction`, `alpha_flatness`, `beta_roughness`)
  - `pcc.background_model`
  - `pcc.max_condition_number`, `pcc.max_residual_rms`
  - `pcc.radii_mode`
  - `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`

### (2026-02-25)

**Registration / Canvas / Color-Correctness Fixes:**

- **Bayer parity-safe offsets in registration/prewarp path**: Canvas offsets are now handled consistently to preserve CFA parity across expanded/cropped canvases.
- **Output scaling origin fixes**: Scaling calls now use the correct tile/debayer offsets where required, preventing R/G parity mismatches after crop/canvas transforms.
- **Common-overlap and canvas handling clarified** in process-flow docs and aligned with the current phase model.

**PCC (Photometric Color Calibration) Improvements:**

- **Robust log-chromaticity fit** implemented for PCC matrix estimation (instead of the older proportion-only approach).
- **Guardrails on channel scales** added to avoid extreme global color casts.
- **Aperture annulus contamination filter (IQR gate)** added to reject unstable star measurements in nebulous/gradient-heavy fields.

**Documentation Refresh:**

- `docs/process_flow/*` updated to the current production pipeline state, including `PREWARP`, `COMMON_OVERLAP`, canvas/offset propagation, and current enum phase ordering.

**BGE (Background Gradient Extraction):**

- Added optional pre-PCC BGE stage that directly subtracts modeled background from RGB channels.
- `bge.method` is the active selector: `none` (default), `classic`, or `autobge`; legacy `bge.enabled` remains compatibility-only when `method` is absent.
- Added foreground-aware BGE fit method `modeled_mask_mesh` for difficult fields with large diffuse objects (e.g. M31/M42) to reduce color-cloud artifacts before PCC.
- Added `artifacts/bge.json` with per-channel diagnostics (tile samples, grid cells, residual statistics).
- Extended report generation to include a dedicated BGE section with summary plots and residual analysis.

### (2026-02-19)

**Calibration Fixes:**

- **GUI dark calibration propagation fixed**: If `use dark` is enabled and either **Darks dir** or **Dark master** is set, these values are now merged into the effective runtime config and applied by the runner. This fixes cases where dark calibration appeared enabled in the GUI but was not present in the run config (`use_dark: false`, empty dark paths).

### (2026-02-17)

**New Registration Features for Alt/Az Mounts Near Pole:**

- **Temporal-Smoothing Registration**: For field rotation, automatically uses neighbor frames (i-1, i+1) for registration when direct registration to reference fails. Chained warps: `i→(i-1)→ref` or `i→(i+1)→ref`. Useful for continuous field rotation (Alt/Az near pole) and clouds/nebula.

- **Adaptive Star Detection**: When too few stars are detected (< topk/2), automatically performs a second pass with lower threshold (2.5σ instead of 3.5σ). This improves star detection in clouds, nebula, or weak frames.

- **New Registration Engine**: `robust_phase_ecc` with LoG gradient preprocessing, optimized for frames with strong nebulae/clouds.

**Field Rotation Support:**

- **Canvas Expansion for Alt/Az Mounts**: Output canvas is now automatically expanded to contain all rotated frames. Previously, stars at the edges were cropped when using Alt/Az mounts near the pole. The bounding box of all warped frames is computed and the canvas is resized accordingly. Log output shows expansion: `"Field rotation detected: expanding canvas from WxH to W'xH'"`.

**Documentation:**

- **New**: [Practical Configuration Examples & Best Practices](../configuration_examples_practical_en.md) - Comprehensive guide with use cases for different focal lengths, seeing conditions, mount types, and camera setups (DWARF, Seestar, DSLR, Mono CCD). Includes parameter recommendations based on methodology v3.3.4.
