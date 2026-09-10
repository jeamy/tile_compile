#pragma once

// Plan section 19.6 --- the deterministic forward-contribution-list rasterizer.
//
// This is the CPU REFERENCE for the ordering the plan-19 CUDA path must obey:
// materialise every positive-area (source-pixel droplet leaf) x (output cell)
// overlap of one stripe as a record with a canonical key, sort by that key,
// then reduce each (frame, channel, output cell) segment single-threaded with
// double accumulators in record order. By construction this reproduces the
// streaming Uniform reference (stream_forward_drizzle_uniform) bit-for-bit:
// for a fixed output cell the streaming emission order IS
// (source_y, source_x, leaf_order) ascending, which is exactly the canonical
// key order within a segment; and segments are visited in frame-then-cell
// order, matching the streaming per-frame `wx += A` accumulation.
//
// No float atomics, no unordered scatter, no inverse search window (plan 19.3).

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace tile_compile::reconstruction {

// One forward contribution within a single stripe. `frame_order` is the index
// into PreparedDrizzleFrames::frames (the plan's canonical frame order);
// `leaf_order` is 0 for the affine path and 0..n-1 for a subdivided local warp.
struct DrizzleContribKey {
  std::uint32_t frame_order = 0;
  std::uint32_t channel = 0;   // 0..2, always 0 for MONO
  std::uint32_t target_y = 0;  // stripe-local output row
  std::uint32_t target_x = 0;  // output column
  std::uint32_t source_y = 0;
  std::uint32_t source_x = 0;
  std::uint32_t leaf_order = 0;
};

struct DrizzleContrib {
  DrizzleContribKey key;
  double area = 0.0;   // k --- exact polygon/rectangle overlap, always > 0
  double value = 0.0;  // v --- source(source_y, source_x), always finite
};

// Canonical strict-weak ordering over the 7-tuple, in the plan-19.6 field
// order (frame, channel, target_y, target_x, source_y, source_x, leaf_order).
bool contrib_key_less(const DrizzleContribKey &a, const DrizzleContribKey &b);

// §30.81 step-5 (B): the tiled driver's per-column-tile callback. `tile` is a
// reduced ForwardDrizzleUniformAndRawResult of internal width `tile_cols`,
// valid only for the duration of the call; the caller re-inserts it at column
// `tile_x_begin` of the full-width band. When a tile sink is supplied,
// accumulate_pair_by_frame[_cuda] produces + sorts each frame's records ONCE
// per band (a per-band memo) and replays that memo through every column tile,
// instead of the caller re-invoking the whole build per tile. The returned
// result then has empty planes and only `.clipping` populated (summed over
// tiles).
using PairTileSink =
    std::function<void(int tile_x_begin, int tile_cols,
                       const ForwardDrizzleUniformAndRawResult &tile)>;

// FrameQualityRectProvider / to_rect_provider: see forward_drizzle.hpp. The
// CUDA store pair path takes the rect provider; accumulate_pair_by_frame
// adapts a plain FrameQualityProvider to it internally.

// plan 19.6.2: wall-clock split of the hybrid CPU-geometry -> GPU-rasterization
// path, accumulated across every local-warp frame of every band. `cpu_seconds`
// is the CPU leaf geometry (sample_leaves / subdivide_local / fixed-point
// inversion), the (leaf,cell) buffer marshalling and the host record assembly;
// `gpu_raster_seconds` is time inside forward_drizzle_cuda_polygon_rect_area_batch
// (H2D + kernel + D2H, not separable without instrumenting the .cu).
struct HybridPathStats {
  double cpu_seconds = 0.0;
  double gpu_raster_seconds = 0.0;
  // Device polygon-area kernel invocations. INCLUDES retried attempts after a
  // CudaAllocFailure halved the batch, so `leaf_cells / gpu_batch_calls` is not
  // a clean "cells per call" on a run that hit device allocation pressure.
  long long gpu_batch_calls = 0;
  long long leaf_cells = 0;   // (leaf, cell) work items enumerated on the CPU
  long long records = 0;      // positive-area, finite-value records emitted
};

struct DrizzleContribList {
  int width = 0;      // stripe width = canvas_width_native * internal_scale
  int rows = 0;       // stripe height
  int channels = 1;   // 1 (MONO) or 3 (OSC)
  int y_begin = 0;    // stripe origin row on the internal canvas
  std::size_t predicted_count = 0;  // from the pre-count pass; == records.size()
  std::vector<DrizzleContrib> records;  // canonically sorted
};

// The plan-11.9 Uniform accumulators, produced by the segment reduction. These
// are exactly what stream_forward_drizzle_uniform accumulates per stripe:
//   wx[c][i] = sum_f  ( sum_{contrib in (f,c,i)} k*v )
//   w [c][i] = sum_f  ( sum_{contrib in (f,c,i)} k   )
//   w2[c][i] = sum_f  ( sum_{contrib in (f,c,i)} k   )^2
// with the inner sums in canonical record order and the outer sum in frame
// order. `value = wx/w` where `w > 0`.
struct DrizzleUniformAccum {
  int width = 0;
  int rows = 0;
  int channels = 1;
  std::array<std::vector<double>, 3> wx, w, w2;
};

// Whole-stripe form (debug / spec): materialises EVERY frame's contributions
// at once and canonically sorts them in one pass. Two passes over the stripe
// rasterizer: (1) count contributions with a finite source value, overflow-
// safe; (2) reserve exactly `predicted_count` and materialise; then a single
// canonical sort. `mem_budget_bytes` bounds `records * sizeof(DrizzleContrib)`;
// exceeding it throws "DRIZZLE_CONTRIB_LIST_BUDGET". This form does NOT scale to
// full-resolution real data with all profiles --- production uses the per-frame
// accumulator below.
DrizzleContribList build_uniform_contrib_list(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    std::size_t mem_budget_bytes = static_cast<std::size_t>(1) << 32);

// Production form: because `frame_order` leads the canonical key, no segment
// ever spans two frames, so the list can be built, sorted and reduced ONE
// FRAME AT A TIME, accumulating into shared wx/w/w2. The result is bit-identical
// to reduce_uniform_contrib_list(build_uniform_contrib_list(...)) --- the same
// `wx[cell] += A_segment` operations in the same (frame, then cell) order ---
// but peak memory is one frame's records, not the whole stripe's.
// `per_frame_mem_budget_bytes` bounds a single frame's record vector; exceeding
// it throws "DRIZZLE_CONTRIB_LIST_BUDGET" (caller halves the chunk height per
// plan 19.4). Between-segment parallelism (plan 19.6 step 3) is intra-frame and
// is preserved.
DrizzleUniformAccum accumulate_uniform_by_frame(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    std::size_t per_frame_mem_budget_bytes = static_cast<std::size_t>(1) << 32);

DrizzleUniformAccum reduce_uniform_contrib_list(const DrizzleContribList &list);

// Plan 19.6 for the FULL pair path: Uniform (clipped) + Raw + Fine + Medium +
// alpha, for one stripe, via the deterministic per-frame contribution list.
// Each frame is built -> canonically sorted -> segment-reduced into one
// ClipCandidate per (channel, target cell), with the Q_composite / Q_scale0 /
// Q_scale1 / artifact K-averages folded per record (NaN / <= 0 -> 0) in record
// order, exactly as the streaming sink does. The candidate slices come out in
// ascending frame order, and the SAME reduce_pixel_profiles() the streaming
// path uses does the clip + profile + alpha accumulation --- so the result is
// bit-identical to stream_forward_drizzle_uniform_and_raw() over this stripe.
//
// STRIPE-SCOPED REFERENCE. The flat ClipCandidate buffer is
// channels * (W*rows) * frame_count * sizeof(ClipCandidate) --- it does NOT
// self-limit `rows` the way stream_forward_drizzle_uniform_and_raw() does via
// plan_drizzle_memory(). The caller MUST pick `rows` (and, when wiring this
// into the productive path, drive it from plan_cuda_chunking / the §11.13
// pre-plan). `mem_budget_bytes` bounds BOTH a single frame's record vector AND
// that candidate buffer; either overflowing throws "DRIZZLE_CONTRIB_LIST_BUDGET"
// before the allocation. Peak working set is one frame's records + the
// candidate buffer, never the whole stripe's contribution list.
ForwardDrizzleUniformAndRawResult accumulate_pair_by_frame(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff_by_source_index = {},
    const FrameQualityProvider &quality_of = {},
    const MultibandProfileParams &mb = {},
    std::size_t mem_budget_bytes = static_cast<std::size_t>(1) << 32,
    // §30.81 (P6 priority-3 CUDA 2D target tiling): restrict this pair build to
    // the target-column window [target_x_begin, target_x_begin + target_cols) of
    // the internal canvas. `target_cols < 0` => full internal width, i.e. the
    // historical behaviour, byte-for-byte. The record producers still emit
    // full-width records; out-of-window contributions are dropped host-side
    // before the flat ClipCandidate buffer (which is then `target_cols` wide, so
    // the DRIZZLE_CONTRIB_LIST_BUDGET ceiling scales with the tile). The emitted
    // ProfilePlane stripes and `internal_width` report the window width; the
    // caller owns re-inserting the tile at column `target_x_begin`.
    int target_x_begin = 0, int target_cols = -1,
    // §30.81 step-5 (B): when `tile_sink` is non-null and `tile_cols > 0`, this
    // call produces + sorts each frame ONCE, then reduces column tiles of width
    // `tile_cols` from that memo, handing each to `tile_sink`. `target_x_begin`
    // / `target_cols` are ignored in that mode (the tiles span [0, W)).
    const PairTileSink *tile_sink = nullptr, int tile_cols = 0);

// The CUDA counterpart of accumulate_pair_by_frame: the per-frame contribution
// records are produced by the device affine rasterizer
// (forward_drizzle_cuda_affine_frame_contributions); the sort, Q fold, clip and
// profile accumulation are the SAME shared host code, so with -ffp-contract=off
// (host) and --fmad=false (device) on this path the result is bit-identical to
// the CPU accumulate_pair_by_frame.
//
// Affine frames run on the device affine rasterizer. Local-warp frames run the
// plan-19.6.2 HYBRID path: the CPU builds the authoritative leaf geometry
// (displacement, fixed-point inversion, bounds, adaptive subdivision, exact
// corners) and the GPU only rasterizes (exact polygon/cell overlap area) --- no
// GPU evaluation of the displacement field, so no new tolerance and §19.5.1 is
// unchanged. Throws ForwardDrizzleCudaError on a singular/absent affine, a
// device failure, or a leaf exceeding `max_cells_per_pixel` --- the caller then
// re-runs the whole stripe on the CPU path (plan 19.4: no mixed CPU/CUDA within
// a commit). Same stripe-scoped memory contract as accumulate_pair_by_frame.
// `max_batch_items` bounds the (leaf, cell) work items handed to the GPU
// polygon-area kernel at once on the hybrid path (halved down to a floor on
// device allocation pressure); it does not affect the result, only peak
// transfer/scratch. Tests pass a tiny value to exercise flush boundaries.
// `hybrid_stats`, if non-null, is ADDED TO (not reset) with the CPU/GPU
// wall-clock split of the hybrid path for this band --- a no-op when no frame
// takes the hybrid path.
ForwardDrizzleUniformAndRawResult accumulate_pair_by_frame_cuda(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff_by_source_index = {},
    // §30.81 step 3a-2: rect provider (source-rectangle Q maps). A full-source
    // FrameQualityProvider is adapted via to_rect_provider() at the call site.
    const FrameQualityRectProvider &quality_of = {},
    const MultibandProfileParams &mb = {},
    std::size_t mem_budget_bytes = static_cast<std::size_t>(1) << 32,
    int max_cells_per_pixel = 32,
    std::size_t max_batch_items = static_cast<std::size_t>(1) << 20,
    HybridPathStats *hybrid_stats = nullptr,
    // §30.81: see accumulate_pair_by_frame. Same semantics; the CUDA record
    // producer still rasterizes full width, the window is applied host-side.
    int target_x_begin = 0, int target_cols = -1,
    // §30.81 step-5 (B): per-band memo + column-tile replay; see
    // accumulate_pair_by_frame.
    const PairTileSink *tile_sink = nullptr, int tile_cols = 0);

}  // namespace tile_compile::reconstruction
