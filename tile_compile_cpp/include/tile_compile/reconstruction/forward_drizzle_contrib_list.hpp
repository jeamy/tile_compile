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

}  // namespace tile_compile::reconstruction
