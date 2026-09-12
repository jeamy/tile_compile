#pragma once

#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace tile_compile::reconstruction {

// Forward-Drizzle v2 starts as an isolated, record-free affine reference.  It
// deliberately does not participate in the production runner until the
// architecture gates in forward_drizzle_v2_zielarchitektur_2026-09-12_de.md
// have passed.  The output layout and arithmetic contract match
// DrizzleUniformAccum so the current canonical record path can serve as the
// oracle without an adapter.
struct ForwardDrizzleV2Stats {
  std::uint64_t target_cells = 0;
  std::uint64_t source_candidates = 0;
  std::uint64_t leaves_tested = 0;
  std::uint64_t positive_overlaps = 0;
  std::size_t workspace_bytes = 0;
};

struct ForwardDrizzleV2UniformResult {
  DrizzleUniformAccum accum;
  ForwardDrizzleV2Stats stats;
};

// Exact affine target-gather reference for [y_begin, y_begin + rows) in
// internal-canvas coordinates.  Each target cell maps its four corners back to
// source space, expands the conservative box by the droplet half-width, then
// evaluates the shared sample_leaves()/polygon clip primitives in ascending
// (source_y, source_x) order.  Thus every per-cell sum has the same order as
// the canonical source-scatter oracle, while memory is O(target stripe).
//
// Throws invalid_argument for local-warp or invalid affine frames: local
// inversion is a later architecture gate and must not silently fall back to a
// different enumeration here.
ForwardDrizzleV2UniformResult gather_affine_uniform_v2(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision = {});

// Gate-1 CUDA prototype with the same result contract. Returns false when no
// usable CUDA device exists or a device operation fails; it never substitutes
// the CPU result silently.
bool gather_affine_uniform_v2_cuda(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    ForwardDrizzleV2UniformResult &out,
    const ForwardDrizzleSubdivisionParams &subdivision = {});

// One frame's already reduced contribution to one internal cell/channel.
// Keeping B frame-local until after the native-pixel fold preserves the cross
// terms required by B2_out = sum_f(sum_j area_j*B_f,j)^2.
struct ForwardDrizzleV2FrameSubpixel {
  std::size_t frame_order = 0;
  double a = 0.0;
  double b = 0.0;
};

struct ForwardDrizzleV2FoldResult {
  double a = 0.0;
  double b = 0.0;
  double b2 = 0.0;
  double value = 0.0;
  double n_eff = 0.0;
  double supported_area_fraction = 0.0;
  bool source_support = false;
  bool profile_support = false;
};

// Experimental candidate primitive for the native-pixel fold; the fold gate
// (Gate 2) is not closed and this is not a selected production component.
// `subpixels_by_frame` is flattened frame-major and must contain
// frame_count*subpixels_per_native entries.  `area` contains the corresponding
// native-pixel area factors and normally sums to one.  Missing subpixels have
// b=0; available area is renormalized by A/B rather than by an all-subpixel
// support AND.  The returned supported_area_fraction records the lost area for
// confidence and diagnostics.
ForwardDrizzleV2FoldResult fold_native_pixel_v2(
    std::span<const ForwardDrizzleV2FrameSubpixel> subpixels_by_frame,
    std::size_t frame_count, std::span<const double> area);

struct ForwardDrizzleV2RobustCandidate {
  std::size_t frame_order = 0;
  double x = 0.0;
  double b = 0.0;
};

enum class ForwardDrizzleV2RobustState {
  no_source_support,
  too_few_candidates_fallback,
  too_few_groups_fallback,
  degenerate_scale,
  primary_winsorized_mom
};

struct ForwardDrizzleV2RobustConfig {
  int groups = 17;
  int min_candidates = 5;
  int min_groups = 3;
  double winsor_sigma = 4.5;
};

struct ForwardDrizzleV2RobustResult {
  ForwardDrizzleV2RobustState state =
      ForwardDrizzleV2RobustState::no_source_support;
  double a = 0.0;
  double b = 0.0;
  double b2 = 0.0;
  double value = 0.0;
  double n_eff = 0.0;
  double center = 0.0;
  double scale = 0.0;
  std::uint64_t candidates = 0;
  std::uint64_t groups_used = 0;
};

// Experimental candidate primitive for robust reduction; Gate 3 is not closed
// and no reducer has been selected.  A replayable stream supplies candidates
// twice without retaining N values per pixel.  Pass 1 builds a fixed number of
// deterministic balanced group means; pass 2 winsorizes frame values around
// median(group means) and its MAD.  This two-pass replay is the unresolved
// conflict of this candidate: every upstream enumeration must be able to emit
// the identical candidate stream twice, which a single-pass scatter or a
// record-free gather cannot do without a second sweep.  The fallback always
// returns the weighted Uniform estimate when source support exists, so
// statistical robustness cannot create a hole.  Memory is O(groups) and
// independent of the frame count.
using ForwardDrizzleV2CandidateSink =
    std::function<void(const ForwardDrizzleV2RobustCandidate &)>;
using ForwardDrizzleV2CandidateReplay =
    std::function<void(const ForwardDrizzleV2CandidateSink &)>;

ForwardDrizzleV2RobustResult robust_reduce_v2(
    const ForwardDrizzleV2CandidateReplay &replay,
    const ForwardDrizzleV2RobustConfig &cfg = {});

struct ForwardDrizzleV2MemoryInputs {
  int target_width = 0;
  int target_height = 0;
  int channels = 0;
  int robust_groups = 17;
  int frame_count = 0;
  std::size_t device_budget_bytes = 0;
  std::size_t host_budget_bytes = 0;
  std::size_t device_fixed_bytes = 0;
  std::size_t host_fixed_bytes = 0;
  // Conservative host bytes pinned per frame and per target row. This folds
  // source/Q row width, native/internal scale, transform spread and halo into
  // one caller-proved upper bound.
  std::size_t pinned_bytes_per_frame_target_row = 0;
};

struct ForwardDrizzleV2MemoryPlan {
  int tile_cols = 0;
  int band_rows = 0;
  std::size_t device_bytes_per_target_pixel = 0;
  std::size_t device_peak_bytes = 0;
  std::size_t host_peak_bytes = 0;
  bool x_tiled = false;
  bool feasible = false;
};

// Experimental checked RAM/VRAM planner candidate for a fixed-group
// enumeration scheme. Gate 5 is not closed: Gates 1, 3 and 4 have not selected
// geometry, reducer or numerics, so this models a candidate layout only.
// Device pixel storage consists of frame A/B, group A/B, robust A/B/B2,
// centre/scale, support/confidence and output staging. It is independent of N.
// Frame count only limits band_rows through pinned source/Q views; it never
// shrinks tile_cols.
ForwardDrizzleV2MemoryPlan plan_forward_drizzle_v2_memory(
    const ForwardDrizzleV2MemoryInputs &in);

}  // namespace tile_compile::reconstruction
