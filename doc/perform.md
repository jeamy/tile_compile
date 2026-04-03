# Tile Reconstruction Performance Findings

## Scope

This document summarizes the current performance findings for tile reconstruction in the C++ pipeline and records the first implementation priorities.

Relevant code paths:

- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `tile_compile_cpp/src/reconstruction/reconstruction.cpp`
- `tile_compile_cpp/src/core/acceleration.cpp`
- `tile_compile_cpp/apps/runner_shared.cpp`
- `tile_compile_cpp/src/image/cfa_processing.cpp`

## Main Findings

### 1. Boundary and weight diagnostics add a full extra post-pass

The current tile reconstruction path computes boundary diagnostics after tile reconstruction and before final artifact serialization.

Cost drivers:

- build of additional diagnostic tile vectors
- overlap pair analysis over reconstructed tiles
- local-weight profile analysis over boundary pairs
- repeated summary reductions for artifact fields

This path is useful for debugging and validation, but it is not required for every production run.

### 2. OSC fallback path repeats expensive work per tile and per frame

When the full-frame RGB cache is disabled, the pipeline currently does the following for each tile/frame pair:

- extract a mosaic tile from `prewarped_frames`
- debayer the extracted tile
- apply common-overlap masking
- push the resulting RGB tiles into channel stacks

This causes repeated allocation/copy/debayer work for the same source frame and becomes expensive when:

- the number of tiles is high
- the overlap is large
- the memory model disables the full-frame RGB cache

### 3. OSC sigma clipping is executed independently for R, G, and B

The current OSC tile reconstruction path reduces each channel separately:

- `sigma_clip_reduce(valid_tiles_R, ...)`
- `sigma_clip_reduce(valid_tiles_G, ...)`
- `sigma_clip_reduce(valid_tiles_B, ...)`

That triples a large part of the expensive per-pixel sigma-clipping workload.

### 4. Overlap-add acceleration still performs host-side staging per tile

The current accelerated overlap-add path still builds temporary host matrices per tile:

- weighted tile buffer on host
- optional weighted mask buffer on host
- upload to device per tile
- add into device ROI

This reduces the benefit of the GPU/OpenCL path, especially for smaller tiles where transfer/setup overhead dominates.

## Implementation Priorities

### Priority 1

Make tile-boundary and local-weight diagnostics optional via config/runtime limits and skip the extra diagnostic pass unless explicitly requested.

### Priority 2

Improve the OSC fallback path so it avoids unnecessary temporary tile copies before debayering and reuses frame-backed data more directly.

### Priority 3

Reduce OSC sigma clipping to one shared outlier decision pass and reuse that keep mask for all three RGB channels.

### Priority 4

Reduce overlap-add staging overhead by batching/precomputing weighted tile contributions and reusing device-side accumulation state more efficiently.

## Implemented So Far

The first implementation steps were limited to behavior-preserving performance work only.

Constraints kept during implementation:

- no algorithm changes
- no change to strict behavior
- no new tile structure
- no new canvas crop/bounds logic

### 1. Overlap-add staging buffer reuse

Implemented in:

- `tile_compile_cpp/src/core/acceleration.cpp`

What changed:

- temporary host-side weighted tile buffers are now reused instead of being recreated for every overlap-add call
- temporary host-side weighted mask buffers are also reused
- temporary CUDA/OpenCL upload buffers are reused per thread in the accelerated overlap-add path

What did not change:

- overlap-add weights
- support-mask handling
- accumulation order inside each overlap-add call
- normalization semantics

### 2. Debayer output buffer reuse for hot OSC paths

Implemented in:

- `tile_compile_cpp/include/tile_compile/image/cfa_processing.hpp`
- `tile_compile_cpp/src/image/cfa_processing.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp`

What changed:

- `debayer_bilinear_into(...)` and `debayer_nearest_neighbor_into(...)` overloads were added so caller-provided output buffers can be filled directly
- existing `DebayerResult` returning functions were kept and now delegate to the new `*_into(...)` variants
- Phase 6 hot paths now reuse debayer output matrices instead of rebuilding temporary `DebayerResult` objects for each iteration

What did not change:

- Bayer parity handling
- interpolation behavior
- mask application order
- RGB tile stacking logic

### 3. OSC full-frame RGB cache worker reuse

Implemented in:

- `tile_compile_cpp/apps/runner_pipeline.cpp`

What changed:

- the OSC full-frame RGB cache worker now keeps reusable debayer output matrices per worker thread

What did not change:

- full-frame cache eligibility logic
- cache contents
- frame masking behavior
- cache storage format

### 4. OSC fallback tile-path copy reduction

Implemented in:

- `tile_compile_cpp/apps/runner_pipeline.cpp`

What changed:

- tile-local OSC fallback now debayers into reusable matrices
- debayered tile planes are moved into the channel tile vectors instead of being carried through a temporary result object

What did not change:

- tile extraction
- common-overlap masking
- per-channel sigma clipping behavior
- tile validity decisions

### 5. Boundary diagnostics raw mono path copy reduction

Implemented in:

- `tile_compile_cpp/apps/runner_pipeline.cpp`

What changed:

- the non-OSC raw boundary diagnostics path now uses `reconstructed_tiles` directly for the raw diagnostics pass instead of first rebuilding an equivalent temporary tile vector

What did not change:

- boundary pair selection
- diagnostic metric definitions
- normalization decision logic
- emitted boundary artifact values

## Explicitly Not Changed Yet

### Shared RGB sigma clipping

This hotspot was reviewed but intentionally not changed yet.

Reason:

- the current OSC path performs independent channel reductions
- replacing that with one shared keep-mask/outlier decision path would risk changing strict behavior
- under the current constraint set, this is not considered safe enough for the first optimization pass

## Expected Impact

### Boundary diagnostics gating

Expected benefit:

- lower post-tile-reconstruction CPU time
- less temporary memory traffic
- lower artifact-generation overhead

### OSC fallback improvements

Expected benefit:

- lower copy/allocation pressure
- faster tile extraction in memory-constrained OSC runs
- better scaling when full-frame RGB caching is disabled

### Shared RGB sigma clipping

Expected benefit:

- significant reduction of per-tile per-pixel clipping work in OSC mode
- better CPU utilization for high-frame-count stacks

### Overlap-add batching/state reuse

Expected benefit:

- lower per-tile overhead in accelerated runs
- better effective device utilization
- reduced host-to-device transfer churn

## Validation Focus

The following outputs should be compared before/after the implementation:

- reconstructed tile support coverage
- `tile_boundary_*` artifact values when diagnostics are enabled
- final RGB output consistency in OSC mode
- background and seam behavior across overlap zones
- runtime and memory usage in Phase 6 / `TILE_RECONSTRUCTION`
