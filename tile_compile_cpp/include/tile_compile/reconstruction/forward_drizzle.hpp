#pragma once

// CPU Uniform reference and bounded row streaming. Both coverage and signal
// reconstruction use the same polygon rasterizer. No full-canvas accumulators
// are allocated by the streaming API. The convenience materializer is budgeted.

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/alpha_confidence.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <span>
#include <vector>

namespace tile_compile::reconstruction {

// M2 in-memory stand-in for plan section 11.3's transactional
// DrizzleProfileStore --- one channel plane at internal-canvas resolution.
struct ProfilePlane {
  int width = 0;
  int height = 0;
  std::vector<float> value;      // NaN where channel_support == 0
  std::vector<float> weight_sum; // sum_f w_profile,f,c(q), plan 11.9
  std::vector<float> n_eff;      // plan 11.10
  std::vector<uint8_t> support;  // channel_support_c(q), plan 11.8/11.10

  void allocate(int w, int h);
  bool empty() const { return width <= 0 || height <= 0; }
};

struct ForwardDrizzleDiagnostics {
  size_t estimated_peak_bytes = 0;
  int resolved_chunk_rows = 0;
  int workers_used = 1;
  int workers_requested = 1;
  int workers_budgeted = 1;
  std::size_t worker_scratch_bytes = 0;
  bool reduction_stats_suppressed = false;
  // Local-warp adaptive subdivision (plan section 11.6).
  long long local_model_samples_total = 0;
  long long local_model_samples_discarded =
      0; // failed to converge by max depth
  // frame_id -> discard rate, only for frames that exceeded
  // per_frame_inversion_error_rate_max and were therefore excluded entirely.
  std::vector<std::pair<std::string, double>>
      frames_excluded_subdivision_error_rate;
  // NOT populated by compute_forward_drizzle_uniform() (kept out of the hot
  // per-sample loop deliberately). The affine kernel's area identity (plan
  // 11.6: sum_q K(q,s) == pixfrac^2 * internal_scale^2 * |det J_f|) is
  // instead verified directly by dedicated unit tests
  // (test_forward_drizzle.cpp) across translation, rotation, scale and
  // combined cases. This field is reserved for a future runtime
  // self-check/artifact field, not a currently-active diagnostic.
  double max_affine_area_relative_error = 0.0;
};

struct ForwardDrizzleUniformResult {
  ColorMode color_mode = ColorMode::MONO;
  int internal_width = 0;
  int internal_height = 0;
  // OSC uses R/G/B; MONO uses only L (R/G/B left empty, plan 11.4 --- never
  // filled with copies of L).
  ProfilePlane R, G, B, L;
  ForwardDrizzleDiagnostics diagnostics;
};

// Plan section 11.6's mandatory adaptive-subdivision tolerances for the
// local (non-affine) warp droplet path.
struct ForwardDrizzleSubdivisionParams {
  float position_epsilon_internal_px = 0.05f;
  int max_subdivision_depth = 2;
  float area_relative_epsilon = 0.005f;
  float per_frame_inversion_error_rate_max = 0.001f;
};

// Supplies the normalized CFA source image for a frame by its
// RegistrationSamplingPlan source_index (plan section 10.1's cache --- NEVER
// prewarped_frames, plan section 23 M2 acceptance). Pixel (x, y) is the
// native CFA sample value at that integer source coordinate.
using SourceImageProvider =
    std::function<const Matrix2Df &(std::size_t source_index)>;

// Supplies a frame's frame-local source quality maps (plan sections 13/14, M5
// + M6) in SOURCE geometry: same dimensions as the normalized source, value
// in (0,1] where quality is known, NaN/<=0 where it is a hard veto or has no
// data. Any pointer may be null (that stream is then treated as 1.0
// everywhere). A NaN/<=0 sample contributes 0 to the geometric K-average
// (plan 11.9: a missing Q-map is not an unweighted fallback; Q=0 is an
// explicit per-sample veto) and never vetoes the output pixel (plan 11.7).
// The returned pointers must stay valid until the next call for a different
// source_index.
struct FrameQualityMaps {
  const Matrix2Df *composite = nullptr;  // Q_composite   -> Raw weight
  const Matrix2Df *scale0 = nullptr;     // Q_scale0      -> Fine weight
  const Matrix2Df *scale1 = nullptr;     // Q_scale1      -> Medium weight
  const Matrix2Df *artifact = nullptr;   // artifact_confidence -> A_artifact
  // §30.81 step 3a-2: when the maps cover only a source rectangle (rect
  // provider), a lookup at absolute source (sy, sx) reads element
  // (sy - y_origin, sx - x_origin). Both 0 for full-source-geometry maps.
  int y_origin = 0;
  int x_origin = 0;
};
using FrameQualityProvider =
    std::function<FrameQualityMaps(std::size_t source_index)>;

// §30.81 step 3a-2: a quality provider that decodes only the source rectangle
// [y0, y1) x [x0, x1) (a negative `y1` or `x1` => the full extent on that
// axis; y0 == y1 or x0 == x1 => a pure existence probe, no decode). The
// returned FrameQualityMaps carry the rectangle origin so an absolute
// (source_y, source_x) lookup rebases into it. The plain FrameQualityProvider
// (full source geometry) is unchanged; `to_rect_provider` adapts it.
using FrameQualityRectProvider = std::function<FrameQualityMaps(
    std::size_t source_index, int y0, int y1, int x0, int x1)>;

inline FrameQualityRectProvider to_rect_provider(
    const FrameQualityProvider &full) {
  if (!full) return {};
  return [full](std::size_t si, int, int, int, int) { return full(si); };
}

// Which quality profiles / alpha inputs the drizzle should additionally emit
// (plan 11.9 / 14.1 / 14.4). Fine uses pow(Q_scale0, fine_quality_exponent);
// Medium uses pow(Q_scale1, medium_quality_exponent). emit_alpha_confidence
// computes the per-pixel, channel-min A_separation / A_artifact /
// A_registration maps (plan 14.4) from the accepted frame contributions ---
// it needs the composite AND artifact quality streams. All share the
// Uniform/Raw clip mask unchanged (plan 11.8).
struct MultibandProfileParams {
  bool emit_fine = false;
  bool emit_medium = false;
  bool emit_alpha_confidence = false;
  float fine_quality_exponent = 4.0f;
  float medium_quality_exponent = 2.0f;
  AlphaConfidenceParams alpha_confidence{};
};

// Computes the Uniform-Control profile only (M2 scope, see header note).
ForwardDrizzleUniformResult compute_forward_drizzle_uniform(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision_params = {});

// Sinks consume a complete stripe synchronously; they must not retain its
// buffers. SourceImageProvider may be called again for the next stripe and
// must retain at most one decoded source frame within the phase budget.
using UniformStripeSink =
    std::function<void(int y_begin, const ForwardDrizzleUniformResult &)>;
ForwardDrizzleDiagnostics stream_forward_drizzle_uniform(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const UniformStripeSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    size_t retained_bytes = 0);

struct DrizzleMemoryPlan {
  int width = 0, height = 0, rows = 0;
  size_t budget_bytes = 0, estimated_peak_bytes = 0;
};
DrizzleMemoryPlan
plan_drizzle_memory(const registration::RegistrationSamplingPlan &plan,
                    const config::ReconstructionDrizzleConfig &cfg,
                    size_t bytes_per_pixel, size_t retained_bytes = 0,
                    bool loads_source = true);

struct PreparedDrizzleFrames {
  std::vector<const registration::FrameSamplingTransform *> frames;
  ForwardDrizzleDiagnostics diagnostics;
};
PreparedDrizzleFrames
prepare_drizzle_frames(const registration::RegistrationSamplingPlan &plan,
                       const config::ReconstructionDrizzleConfig &cfg,
                       const ForwardDrizzleSubdivisionParams &subdivision = {});

// index is stripe-local; each contribution is an exact positive area. `leaf` is
// the 0-based order of the transformed source-pixel leaf within this
// (sx, sy) sample (always 0 for the affine path; 0..n-1 for a subdivided local
// warp). It is part of the plan-19.6 canonical contribution key.
using DrizzleAreaSink = std::function<void(int sx, int sy, int channel, int leaf,
                                           size_t index, double area)>;
// §30.81 (P6 priority-3 CUDA 2D target tiling): `x_begin` / `cols` restrict the
// emitted internal cells to the target-column window [x_begin, x_begin + cols)
// (`cols < 0` => full internal width, the historical behaviour). The window also
// bounds the inverse-mapped source scan (R2), so a narrow tile reads only its
// own source footprint. The stripe-local `index` handed to `sink` is rebased to
// the window: `(cell_y - y_begin) * cols + (cell_x - x_begin)`. With the default
// full-width window every emitted cell, its `index`, and the enumeration order
// are byte-identical to the pre-§30.81 path.
void rasterize_drizzle_stripe(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, int internal_scale,
    float pixfrac, int y_begin, int rows, const DrizzleAreaSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision = {}, int x_begin = 0,
    int cols = -1);

// plan 19.6.2 hybrid CPU-geometry -> GPU-rasterization: the shared cell
// enumeration that `rasterize_drizzle_stripe` is built on. For every accepted
// leaf of every source sample in the stripe it emits, once per integer cell in
// the leaf's clamped bounding box, the four exact leaf corners (`leaf_x`,
// `leaf_y`, four doubles each) and the cell's integer origin (`cell_x`,
// `cell_y`; the cell rectangle is [cell_x, cell_y, cell_x+1, cell_y+1] in
// internal-scale pixels, `cell_y` NOT stripe-local). No area is computed and no
// cell is pre-filtered on area: the consumer decides. `channel` is 0 for MONO,
// R/G/B = 0/1/2 for OSC; `leaf_order` is the canonical 19.6 leaf index.
using DrizzleLeafCellSink =
    std::function<void(int sx, int sy, int channel, int leaf_order, int cell_x,
                       int cell_y, const double *leaf_x, const double *leaf_y)>;
// §30.81: `x_begin` / `cols` restrict enumeration to the target-column window
// [x_begin, x_begin + cols) (`cols < 0` => full internal width). `cell_x` is
// still the absolute internal column (NOT window-local), matching `cell_y`.
void enumerate_drizzle_stripe_leaf_cells(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, int internal_scale,
    float pixfrac, int y_begin, int rows, const DrizzleLeafCellSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision = {}, int x_begin = 0,
    int cols = -1);

// One accepted leaf of a (possibly subdivided) source-pixel droplet: a convex
// quadrilateral in internal-canvas coordinates. Exported for the plan-11.14
// geometry cache builder, which must produce byte-identical leaves.
struct Leaf {
  double x[4];
  double y[4];
};

// Plan §30.75 (P6 SAMPLING_GEOMETRY acceleration): the dense-footprint
// touched-cell mask for ONE frame over ONE destination stripe.
//
// The reference footprint pass runs `rasterize_drizzle_stripe` at pixfrac 1.0
// with a sink that only records "this internal cell received positive area from
// some source-pixel square". For an AFFINE frame every unshrunk source-pixel
// square [sx,sx+1]x[sy,sy+1] maps to a parallelogram, and their union over the
// whole source is exactly the single parallelogram affine_f([0,W_src]x
// [0,H_src]) (an affine image of a partition tiles the image). This routine
// classifies each internal cell of the stripe against that parallelogram P:
//   - exterior (cell lies fully outside one edge of P)  -> left untouched;
//   - interior (cell lies at least one mapped-pixel diameter inside every edge
//     of P)                                             -> touched = 1;
//   - boundary (cell straddles or is near an edge of P) -> exact fallback: the
//     same sample_leaves + polygon_rectangle_intersection_area test the
//     reference uses, restricted to the source pixels that can reach the cell.
// The result is byte-identical to the reference footprint pass by construction.
//
// `touched` is resized to internal_width*rows (row-major, stripe-local rows)
// and every entry is written (0 or 1). LOCAL-warp frames, singular/degenerate
// affine frames, and `force_exact` delegate to the reference rasterize
// unchanged. Only used by compute_geometric_coverage; NOT part of the
// rasterize_drizzle_stripe / CUDA contribution path.
void dense_footprint_touched_stripe(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, int internal_scale,
    int y_begin, int rows, std::vector<std::uint8_t> &touched,
    bool force_exact = false);

// The plan-11.6 per-sample leaf set for source pixel (sx, sy): the affine
// droplet, or --- for a local-warp frame --- the adaptively subdivided leaves
// from `invert_local_source_to_canvas`. Returns false (and clears `leaves`)
// when the sample is rejected (inversion / subdivision failure); the caller
// counts that as a discard. Bit-exact and deterministic. Exported so the
// geometry cache builds its store with the SAME evaluation the stripe
// enumerator uses.
bool sample_leaves(const registration::RegistrationSamplingPlan &plan,
                   const registration::FrameSamplingTransform &frame, int sx,
                   int sy, int internal_scale, float pixfrac,
                   const ForwardDrizzleSubdivisionParams &subdivision,
                   std::vector<Leaf> &leaves);

// --- M3 (plan section 11.8): shared robust clipping -------------------------
//
// Status: the algorithm itself (this function) is implemented and tested
// against the plan's 8-step procedure, but it is NOT yet wired into
// compute_forward_drizzle_uniform()/stream_forward_drizzle_uniform(): those
// still produce the unclipped M2 Uniform-Control profile. Plan section 11.8
// mandates that the SAME acceptance mask this function computes ultimately
// applies to Uniform, Raw-Forward-Drizzle and all detail profiles alike ---
// wiring that in (which also requires retaining every frame's x_f,c(q) per
// pixel within a stripe, not just the running sums) is separate, still-open
// M3 integration work. This is the reviewed, tested primitive it will be
// built on, not a claim that M3 is complete.

// One frame's contribution to a single target pixel/channel, plan 11.7:
// x_f,c(q) = A_f,c(q) / B_f,c(q), only constructed for B_f,c(q) > 0.
struct ClipCandidate {
  std::size_t frame_index = 0;  // for the deterministic tie-break, plan 11.8 step 3
  double x = 0.0;                // the value under test
  double b = 0.0;                 // geometric weight B_f,c(q), plan 11.7
  // Frame-local geometric K-averages of the source quality maps over this
  // frame's droplets for this target pixel/channel, plan 11.7:
  //   q  = Q_composite_f,c(q)       -> Raw weight + A_separation
  //   q0 = Q_scale0_f,c(q)          -> Fine weight
  //   q1 = Q_scale1_f,c(q)          -> Medium weight
  //   qa = artifact_confidence K-avg -> A_artifact
  // None of them ever enter the clipping decision (plan 11.8). Each defaults
  // to 1.0 so a caller that supplies no map gets that profile == Uniform
  // scaled by G_eff.
  double q = 1.0;
  double q0 = 1.0;
  double q1 = 1.0;
  double qa = 1.0;
  // True iff at least one droplet sample of this contribution had a FINITE
  // artifact_confidence value. Plan 14.4's "< 8 gueltige Framebeiträge"
  // counts contributions with real artifact data, not merely frame presence.
  bool qa_has_data = false;
};

struct ClipResult {
  // Same order/size as the input candidates vector.
  std::vector<bool> accepted;
  // Plan 11.8 step 8: min_fraction or min_n_eff failed against the
  // geometrically possible frame support --- the pixel/channel is rejected
  // in ALL profiles (channel_support_c(q) = 0), not just this one.
  bool pixel_rejected = false;
};

// Reusable, budgeted clipping scratch (plan 0.2 priority 2, §30.79):
// `accepted` + `order`/`active`/`dev_order` sized to the maximum candidate
// span (== frame count; one candidate per frame/pixel/channel), plus the
// alpha contribution buffer when emitted. Heap-based; the owning scope
// budgets it (streaming path: worker_scratch_bytes). CONTENT IS TRANSIENT:
// every clipping call fully rewrites the used prefix, buffers are never
// shrunk, and one instance must never be shared between concurrent reduce
// calls.
struct DrizzleClipScratch {
  std::vector<std::uint8_t> accepted;  // 0/1 accept flags, size == span
  std::vector<std::size_t> order;      // fixed value order, plan 11.8 step 3
  std::vector<std::size_t> active;     // per-pass valid subset of `order`
  std::vector<std::size_t> dev_order;  // |x - median| order of `active`
  std::vector<AlphaFactorContribution> alpha_contribs;
  std::uint64_t growth_count = 0;      // capacity-growing reserve_for calls

  // Grow-only capacity reservation; counts calls that grow any buffer.
  void reserve_for(std::size_t capacity, bool with_alpha);
};

// Implements plan section 11.8's 8-step procedure exactly, including the
// degenerate-MAD guards (identical values stay valid, no arbitrary epsilon
// widening) and the min_clip_contributors bypass (plan 11.8 step 2, protects
// thin R/B channels at low frame counts from MAD instability). Q-/quality
// weights never enter this decision (plan 11.8: "Q-Gewichte dürfen nicht
// bestimmen, ob ein Sample als Ausreißer gilt") --- only the geometric
// weight `b` is used, exactly as specified.
ClipResult apply_robust_clipping(std::span<const ClipCandidate> candidates,
                                 int min_clip_contributors, int robust_passes,
                                 float clip_sigma_low, float clip_sigma_high,
                                 float min_fraction, float min_n_eff);

// Plan 11.8: "Die resultierende Akzeptanzmaske wird unverändert für Uniform,
// Raw-Forward-Drizzle und alle Detailprofile verwendet." Computes both
// profiles in the same pass, sharing one clipping decision per pixel/channel.
//
// `raw`'s weight is w_raw = B_f,c(q) * G_eff(f) * Q_composite_f,c(q) (plan
// 11.9). Both factors are now wired:
//   - G_eff(f): pass `g_eff_by_source_index` (indexed by
//     FrameSamplingTransform::source_index, values in [0,1] from a
//     QualityFrameWeightPlan); empty => 1.0 for every frame.
//   - Q_composite_f,c(q): pass `quality_of` (M5 source composite Q-maps);
//     null => 1.0 everywhere. Q_composite_f,c(q) is the geometric K-average
//     sum_s K(q,s)*Q_composite_f(s)/B_f,c(q) of the frame's source Q-map over
//     its droplets (plan 11.7); a NaN/<=0 source Q contributes 0 to that
//     average (plan 11.9) and never vetoes the pixel (plan 11.7).
// So `raw` equals `uniform` when neither is supplied, differs by G_eff only
// when just a plan is supplied, and additionally by the per-pixel Q_composite
// when a quality provider is supplied.
// `uniform` here is the clipped profile per 11.8, distinct from
// compute_forward_drizzle_uniform()'s unclipped M2 profile (kept as-is;
// M2's acceptance never required clipping). The clipping decision itself
// only ever uses the geometric weight B (plan 11.8) --- G_eff never enters
// it.
//
// Clipping stores at most one candidate per frame/pixel/channel in a flat
// stripe buffer. Worst-case storage, scratch and both output profiles are
// budgeted before loading a source or allocating image buffers. Streaming
// sinks must consume stripes synchronously without retaining their buffers.
struct ForwardDrizzleClippingDiagnostics {
  long long pixel_channel_evaluations = 0;
  long long pixel_channel_rejected = 0;   // plan 11.8 step 8 veto
  long long candidate_contributions_clipped = 0;  // total false entries across all pixels
};

// Plan 11.8 / 11.9 / 14.4: the per-(channel, target cell) reduction that turns
// one frame-ordered ClipCandidate list into the Uniform / Raw / Fine / Medium
// profile samples plus the alpha-confidence factors. Shared VERBATIM by the
// streaming path (stream_forward_drizzle_uniform_and_raw) and the plan-19.6
// contribution-list path so the two are bit-identical by construction. Q- and
// G_eff weights never enter the clipping decision (only the geometric weight
// B); they only weight the post-clip profile accumulation.
struct DrizzleProfileReduceConfig {
  int min_clip_contributors = 0;
  int robust_passes = 0;
  float clip_sigma_low = 0.0f;
  float clip_sigma_high = 0.0f;
  float min_fraction = 0.0f;
  float min_n_eff = 0.0f;
  bool emit_fine = false;
  bool emit_medium = false;
  bool emit_alpha = false;
  float fine_quality_exponent = 4.0f;
  float medium_quality_exponent = 2.0f;
  AlphaConfidenceParams alpha_confidence{};
};

// `candidates` MUST be in ascending frame order (plan 19.6 step 4's
// "feste Framefolge"). Writes value/weight_sum/n_eff/support at `gi` into each
// non-null profile plane; when the alpha pointers are non-null, folds the
// channel's alpha factors into them via std::min (the caller seeds them to
// +inf and takes the min over active channels). Bumps `diag`
// (pixel_channel_evaluations always; pixel_channel_rejected +
// candidate_contributions_clipped per the clip result).
// `scratch` (optional): reusable clip buffers, never shared between
// concurrent calls; nullptr uses a per-call fallback with identical results.
void reduce_pixel_profiles(
    std::span<const ClipCandidate> candidates,
    const DrizzleProfileReduceConfig &cfg,
    const std::function<double(std::size_t /*source_index*/)> &g_eff_for,
    const std::vector<std::pair<std::uint8_t, float>> &reg_by_source,
    std::size_t gi, ProfilePlane *uniform_c, ProfilePlane *raw_c,
    ProfilePlane *fine_c, ProfilePlane *medium_c, double *ac_sep, double *ac_art,
    double *ac_reg, ForwardDrizzleClippingDiagnostics &diag,
    DrizzleClipScratch *scratch = nullptr);

struct ForwardDrizzleUniformAndRawResult {
  ForwardDrizzleUniformResult uniform;  // clipped (plan 11.8)
  ForwardDrizzleUniformResult raw;      // w_raw = B*G_eff*Q_composite (plan 11.9)
  // Populated only when MultibandProfileParams::emit_fine / emit_medium is
  // set (plan 14.1). Fine weight = B*G_eff*pow(Q_scale0, fine_quality_exponent);
  // Medium weight = B*G_eff*pow(Q_scale1, medium_quality_exponent). Same clip
  // mask as uniform/raw. Empty planes otherwise.
  ForwardDrizzleUniformResult fine;
  ForwardDrizzleUniformResult medium;
  // Per-pixel channel-min adaptive-alpha confidence factors (plan 14.4),
  // populated only when MultibandProfileParams::emit_alpha_confidence is set.
  // Row-major internal geometry; NaN where alpha_confidence_support == 0.
  std::vector<float> a_separation;
  std::vector<float> a_artifact;
  std::vector<float> a_registration;
  std::vector<uint8_t> alpha_confidence_support;
  ForwardDrizzleDiagnostics diagnostics;
  ForwardDrizzleClippingDiagnostics clipping;
};

struct ForwardDrizzlePairDiagnostics {
  ForwardDrizzleDiagnostics diagnostics;
  ForwardDrizzleClippingDiagnostics clipping;
};
using UniformAndRawStripeSink =
    std::function<void(int y_begin, const ForwardDrizzleUniformAndRawResult &)>;
ForwardDrizzlePairDiagnostics stream_forward_drizzle_uniform_and_raw(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const UniformAndRawStripeSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision_params = {},
    const std::vector<float> &g_eff_by_source_index = {},
    size_t retained_bytes = 0,
    const FrameQualityProvider &quality_of = {},
    const MultibandProfileParams &multiband = {},
    // Plan 11.14.5 P3 (Teil 2): partition each stripe's rasterize + candidate
    // gather + reduce into `workers` disjoint output-row bands, computed
    // concurrently. Every canvas cell is still touched by exactly one worker
    // and every source contribution is still added in the unchanged canonical
    // order, so profiles / alpha maps / clipping counters are bit-identical to
    // the serial (`workers == 1`, the default) reference. The source frame is
    // loaded once on the caller thread per frame; the geometry cache reader is
    // shared (its `enumerate_stripe` is const + concurrency-safe). Values < 1
    // are treated as 1. Geometry-stats instrumentation is only collected at
    // `workers == 1` (the process-global registry is not concurrency-safe).
    int workers = 1,
    // §30.81 (P6 priority-3 CUDA 2D target tiling): restrict every stripe to the
    // target-column window [target_x_begin, target_x_begin + target_cols) of the
    // internal canvas. `target_cols < 0` => full internal width, i.e. the
    // historical behaviour, byte-for-byte. When a window is set, all per-stripe
    // buffers and the emitted ProfilePlane stripes are `target_cols` wide and
    // `ForwardDrizzleUniformResult::internal_width` reports the window width; the
    // sink's `y_begin` is still the absolute internal row and the caller owns
    // re-inserting the tile at column `target_x_begin`.
    int target_x_begin = 0, int target_cols = -1);

ForwardDrizzleUniformAndRawResult compute_forward_drizzle_uniform_and_raw(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision_params = {},
    const std::vector<float> &g_eff_by_source_index = {},
    const FrameQualityProvider &quality_of = {},
    const MultibandProfileParams &multiband = {},
    int workers = 1);

// --- exposed for unit tests (plan section 11.6 geometry) -------------------

// Exact area of the intersection of a convex quadrilateral (4 vertices, in
// order around the boundary, either winding) with the axis-aligned rectangle
// [rx0,rx1] x [ry0,ry1]. Sutherland-Hodgman clip + shoelace.
double polygon_rectangle_intersection_area(const double poly_x[4],
                                           const double poly_y[4], double rx0,
                                           double ry0, double rx1, double ry1);

double shoelace_area(const double *x, const double *y, int n);

} // namespace tile_compile::reconstruction
