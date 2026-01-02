# tile_compile

Pipeline for **tile-based quality reconstruction** of DSO image series (OSC, **linear**) with:

- **Registration** (currently via **Siril CLI** backend)
- **global frame metrics** (transparency/noise/structure)
- **local tile weighting** (stars or structure)
- reconstruction of **15–30 synthetic frames**
- **final stacking** of synthetic frames (currently via **Siril CLI**, `average`)

**Single Source of Truth:** `tile_basierte_qualitatsrekonstruktion_methodik_en.md` (Methodology v2)

Optional parallelization (RabbitMQ): `tile_compile_parallelisierung_en.md`

## Assumptions

- Data are **linear** (no stretch / asinh / log)
- Registration is complete (translation + rotation)
- Many frames (default gate: `frames_min: 800`)

## Configuration & pipeline control

- **`tile_compile.yaml`**
  - Central parameters (geometry/weights/validation/runtime)
  - Backend selection for registration/stacking
- **`tile_compile.proc`**
  - Process description (Clean Break)
  - Defines step order and which directories are used

## Pipeline (Methodology v2, normative)

Normative phases (see Methodology v2):

1. Registration
2. **Global linear normalization (mandatory, once)**
3. Global frame metrics (median+MAD, clamp, `G_f = exp(Q_f)`)
4. seeing-adaptive tile geometry (FWHM from registered frames)
5. Local tile metrics (stars: `log(FWHM)`, structure: `E/σ`)
6. Tile reconstruction (weighted mean + window + overlap-add)
7. State-based clustering of frames (state vector)
8. Reconstruction of synthetic quality frames
9. Final stacking
10. Validation & abort

## Directory layout (conceptual)

- **Input frames**: arbitrary input directory (project-dependent)
- **Registered frames**: `registration.output_dir` (default: `registered/`)
- **Synthetic frames**: `stacking.input_dir` (default: `synthetic/`)
- **Final stack**: `stacking.output_file` (default: `stacked.fit`)

## Registration (pluggable API)

Registration is modeled as a pluggable step:

- Config:
  - `registration.engine`: `siril | relative | wcs_anchor`
  - `registration.reference`: `auto | frame_index | path`
  - `registration.output_dir`: e.g. `registered`
  - `registration.registered_filename_pattern`: e.g. `"reg_{index:05d}.fit"`

In `tile_compile.proc`, processing switches to registered frames afterwards:

- `LOAD_FRAMES_FROM_DIR registration.output_dir`

## Synthetic frames

Reconstruction produces **15–30** synthetic frames (see methodology document for the formal definition/clustering).

In `tile_compile.proc`:

- `ENSURE_DIR stacking.input_dir`
- `WRITE stacking.input_dir/syn_XX.fits`

## Final stacking (pluggable API)

Stacking is also modeled as a pluggable step (currently Siril):

- Config:
  - `stacking.engine`: `siril`
  - `stacking.method`: `average`
  - `stacking.input_dir`: `synthetic`
  - `stacking.input_pattern`: `syn_*.fits`
  - `stacking.output_file`: `stacked.fit`

In `tile_compile.proc`:

- `CALL stack_frames(engine=stacking.engine, method=stacking.method, ...)`

## Validation

The pipeline includes abort rules/validation (e.g. minimum FWHM improvement, no background degradation, no tile artifacts). See `tile_basierte_qualitatsrekonstruktion_methodik_en.md`.

## Option: RabbitMQ parallelization (Master/Worker)

Parallelization is optional and does **not** change methodology semantics. Determinism is preserved via master aggregation.

- **Local (single host)**
  - RabbitMQ local
  - multiple worker processes/containers
- **Production (main server + remote workers)**
  - central RabbitMQ instance on the main server
  - workers via overlay network (Tailscale/NetBird)
  - binary data not via RabbitMQ but via `tile_data_ref` (shared FS or S3/MinIO)

Details: `tile_compile_parallelisierung_en.md`
