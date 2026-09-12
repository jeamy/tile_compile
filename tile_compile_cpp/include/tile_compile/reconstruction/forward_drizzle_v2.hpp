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
  double geometry_b = 0.0;
  double source_b = 0.0;
  double estimator_b = 0.0;
  double a = 0.0;
  double b = 0.0;
};

struct ForwardDrizzleV2FoldResult {
  double a = 0.0;
  double b = 0.0;
  double b2 = 0.0;
  double value = 0.0;
  double n_eff = 0.0;
  double geometry_area_fraction = 0.0;
  double source_area_fraction = 0.0;
  double estimator_area_fraction = 0.0;
  double profile_area_fraction = 0.0;
  bool geometry_support = false;
  bool source_support = false;
  bool estimator_support = false;
  bool profile_support = false;
};

// Gate-2-decided scalar native-pixel fold reference. The production batch
// fold over the device-resident scatter planes is Gate 6 scope; this function
// is the exact CPU contract that batch kernel must reproduce.
// `subpixels_by_frame` is flattened frame-major and must contain
// frame_count*subpixels_per_native entries. `area` contains the corresponding
// native-pixel area factors. geometry/source/estimator/profile support remain
// distinct; their positive denominators must imply one another in that order.
// Missing profile subpixels have b=0; available area is normalized by folded
// A/B rather than by an all-subpixel support AND. Four area fractions retain
// the lost-area information for later confidence and delivery decisions.
ForwardDrizzleV2FoldResult fold_native_pixel_v2(
    std::span<const ForwardDrizzleV2FrameSubpixel> subpixels_by_frame,
    std::size_t frame_count, std::span<const double> area);

// Gate-2 CUDA arithmetic oracle for the same scalar fold contract. This is not
// the production batch kernel selected later by Gate 6; it proves that the
// frame-before-square and four-support-layer algebra ports to the device.
bool fold_native_pixel_v2_cuda(
    std::span<const ForwardDrizzleV2FrameSubpixel> subpixels_by_frame,
    std::size_t frame_count, std::span<const double> area,
    ForwardDrizzleV2FoldResult &out);

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
  primary_winsorized_mom,
  primary_mom_median,
  primary_mom_winsorized_groups,
  primary_mom_trimmed_groups,
  primary_uniform,
  primary_reservoir_sigma_clip,
  oracle_sigma_clip
};

// Gate-3 estimator candidates. All selectable candidates must be single-pass
// over the frame stream with O(groups) per-pixel state: the dense scatter
// emits each frame-local plane exactly once. `two_pass_winsorized_frames`
// needs a second pass over the identical candidate stream and is a quality
// reference only.
enum class ForwardDrizzleV2Estimator {
  uniform,
  mom_median,
  mom_winsorized_groups,
  mom_trimmed_groups,
  reservoir_sigma_clip,
  two_pass_winsorized_frames
};

struct ForwardDrizzleV2RobustConfig {
  int groups = 17;
  int min_candidates = 5;
  int min_groups = 3;
  double winsor_sigma = 4.5;
  // reservoir_sigma_clip: deterministic hash reservoir size and seed. Frames
  // are kept iff splitmix64(frame_order ^ seed) < floor(2^64 * R / N) where N
  // is the known stream length (sampling-plan frame count); for N <= R every
  // frame is kept and the result is the exact oracle.
  int reservoir_size = 64;
  std::uint64_t reservoir_seed = 0x9e3779b97f4a7c15ULL;
  int oracle_min_clip_contributors = 5;
  int oracle_passes = 3;
  double oracle_sigma_low = 3.0;
  double oracle_sigma_high = 3.0;
};

enum class ForwardDrizzleV2ConfidenceState {
  no_source_support,
  fallback_n_eff,
  modeled
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
  // Gate-4 confidence sufficient statistics over the estimator's effective
  // set: conf_b = sum b, conf_s = sum b*sigma, conf_c = sum b^2*sigma2.
  // confidence = conf_s^2 / (conf_s^2 + conf_c) when conf_c > 0, else the
  // n_eff/(n_eff+1) fallback.  Never changes support.
  double conf_b = 0.0;
  double conf_s = 0.0;
  double conf_c = 0.0;
  std::uint64_t conf_degraded = 0;
  double confidence = 0.0;
  ForwardDrizzleV2ConfidenceState conf_state =
      ForwardDrizzleV2ConfidenceState::no_source_support;
};

// Gate-4 per-candidate noise model, all inputs in normalized source units:
// sigma2 = sigma_noise^2 + (gx^2 + gy^2) * sigma_reg_px^2 + half^2/3.
// Returns a non-finite value when any input is non-finite or negative; the
// caller maps that to the degraded-sigma contract (sigma2 = 0 + counter).
double forward_drizzle_v2_sigma2_model(double sigma_noise, double grad_x,
                                       double grad_y, double sigma_reg_px,
                                       double droplet_half);

// Gate-3 quality reference, NOT the selected estimator.  A replayable stream
// supplies candidates twice without retaining N values per pixel.  Pass 1
// builds a fixed number of deterministic balanced group means; pass 2
// winsorizes frame values around median(group means) and its MAD.  Gate 3
// rejected it: it needs a second identical replay that single-pass dense
// scatter cannot provide, and it fails the frozen adversarial bound on
// grouped contamination.  Retained only as a measurable reference.  The
// fallback always returns the weighted Uniform estimate when source support
// exists, so statistical robustness cannot create a hole.
using ForwardDrizzleV2CandidateSink =
    std::function<void(const ForwardDrizzleV2RobustCandidate &)>;
using ForwardDrizzleV2CandidateReplay =
    std::function<void(const ForwardDrizzleV2CandidateSink &)>;

ForwardDrizzleV2RobustResult robust_reduce_v2(
    const ForwardDrizzleV2CandidateReplay &replay,
    const ForwardDrizzleV2RobustConfig &cfg = {});

// Gate-3 selected estimator and candidates over a materialized candidate
// list.  Selected: `reservoir_sigma_clip` — a single pass accumulates the
// uniform A/B/B2 totals and a deterministic hash reservoir of at most R=64
// candidates, then the exact production oracle clip runs on the reservoir.
// For N <= R it is bit-identical to the full-list oracle; worst measured
// deviation at production N=600 is 0.036 against the oracle (frozen bound
// 2.0).  The MoM group candidates perform exactly one accumulation pass into
// K group (a,b,b2) slots but failed the grouped-contamination bound; they are
// retained for evidence and possible reuse.  `estimator` must not be
// two_pass_winsorized_frames (that reference lives in robust_reduce_v2).
ForwardDrizzleV2RobustResult robust_reduce_candidates_v2(
    std::span<const ForwardDrizzleV2RobustCandidate> candidates,
    ForwardDrizzleV2Estimator estimator,
    const ForwardDrizzleV2RobustConfig &cfg = {},
    std::uint64_t stream_length = 0,
    std::span<const double> candidate_sigma2 = {});

// Gate-3 full-list oracle mirroring the production iterative weighted
// median/MAD sigma-clip: deterministic (value, frame_order) order, weighted
// median, weighted MAD, asymmetric clip bounds, early stop when the mask no
// longer changes, no epsilon padding of a zero MAD. The surviving candidates
// are combined as the weighted Uniform estimate. `candidate_sigma2` (empty or
// parallel to candidates) feeds the Gate-4 confidence statistics over the
// accepted set.
ForwardDrizzleV2RobustResult robust_frame_oracle_v2(
    std::span<const ForwardDrizzleV2RobustCandidate> candidates,
    int min_clip_contributors = 5, int robust_passes = 3,
    double sigma_low = 3.0, double sigma_high = 3.0,
    std::span<const double> candidate_sigma2 = {});

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

// Experimental checked RAM/VRAM planner candidate for dense scatter with a
// fixed-group reducer. Gate 1 selected affine dense scatter, but Gate 5 is not
// closed: Gates 3 and 4 have not selected reducer or numerics, so this models
// a candidate layout only.
// Device pixel storage consists of frame A/B, group A/B, robust A/B/B2,
// centre/scale, support/confidence and output staging. It is independent of N.
// Frame count only limits band_rows through pinned source/Q views; it never
// shrinks tile_cols.
ForwardDrizzleV2MemoryPlan plan_forward_drizzle_v2_memory(
    const ForwardDrizzleV2MemoryInputs &in);

}  // namespace tile_compile::reconstruction
