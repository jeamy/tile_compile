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
  // T4c: per-band source/Q band cache stats. Zero when no caching or no
  // quality/source provider. Counted per accumulate_pair_impl call (one band).
  long long q_band_cache_hits = 0;
  long long q_band_cache_misses = 0;
  long long q_bytes_read = 0;
  long long source_band_cache_hits = 0;
  long long source_band_cache_misses = 0;
  long long source_bytes_read = 0;
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

// A1 (redundant-reload analysis): a source provider that reads only the
// source rectangle [y0, y1) x [x0, x1) a stripe's scan box touches, instead
// of the whole frame. The returned matrix must be exactly (y1 - y0) x
// (x1 - x0); callers rebase absolute source coordinates by (y0, x0). When a
// rect provider is wired the full-frame provider is only still consulted by
// paths that genuinely need the whole source (hybrid local-warp geometry).
using SourceImageRectProvider = std::function<Matrix2Df(
    std::size_t source_index, int y0, int y1, int x0, int x1)>;

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
    const ForwardDrizzleSubdivisionParams &subdivision_params = {},
    const SourceImageRectProvider &source_rect_of = {});

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
    size_t retained_bytes = 0,
    // A3 (redundant-reload analysis): when non-null, only the per-stripe
    // inverse-mapped source box is read through this provider instead of the
    // full frame via `source_of` (see drizzle_source_scan_box). Bit-identical.
    const SourceImageRectProvider &source_rect_of = {});

struct DrizzleMemoryPlan {
  int width = 0, height = 0, rows = 0;
  size_t budget_bytes = 0, estimated_peak_bytes = 0;
};
DrizzleMemoryPlan
plan_drizzle_memory(const registration::RegistrationSamplingPlan &plan,
                    const config::ReconstructionDrizzleConfig &cfg,
                    size_t bytes_per_pixel, size_t retained_bytes = 0,
                    bool loads_source = true);

// Like plan_drizzle_memory, but when the configured budget is too small for
// the retained/source working set, raises `cfg.memory_budget_mb` in 1 GiB
// steps as long as the effective budget can still grow within the
// available-memory headroom (the live 80%-of-available cap), calling `warn`
// with a description of each step. On success the effective budget is stored
// back into `cfg.memory_budget_mb`. Throws the last DRIZZLE_MEMORY_BUDGET
// error when the working set cannot fit the available headroom.
DrizzleMemoryPlan plan_drizzle_memory_autogrow(
    const registration::RegistrationSamplingPlan &plan,
    config::ReconstructionDrizzleConfig &cfg, size_t bytes_per_pixel,
    size_t retained_bytes = 0, bool loads_source = true,
    const std::function<void(const std::string &)> &warn = {});

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

// A1 (redundant-reload analysis): the exact source-pixel box the stripe
// enumerator scans for `frame` over the target window [x_begin, x_begin +
// cols) x [y_begin, y_begin + rows) --- full source extent for local-warp
// frames, the ±1-source-pixel-margined inverse-mapped box for affine frames.
// Throws DRIZZLE_SINGULAR_TRANSFORM for a singular affine, exactly as the
// enumerator does. Shared by the enumerator and every banded source/quality
// read so a rect provider can never under-serve a scanned pixel.
struct DrizzleSourceScanBox {
  int y0 = 0, y1 = 0, x0 = 0, x1 = 0;
};
DrizzleSourceScanBox drizzle_source_scan_box(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, int internal_scale,
    int y_begin, int rows, int x_begin = 0, int cols = -1);

// Tranche 8: canonical ragged affine source-row span. One half-open source
// x interval per active source row; the span list is ordered by ascending
// source_y and is the tight (non-axis-aligned) analog of
// drizzle_source_scan_box: a source pixel is in its row's interval iff its
// pixfrac droplet BBOX (the four-corner box, not just the mapped center)
// can intersect the full native target width x the native band y window.
struct DrizzleAffineSourceSpan {
  int source_y = 0;
  int x_begin = 0;
  int x_end = 0;  // half-open
};

// Returns the active spans for a pure-affine frame and native band
// [band_y_begin_native, +band_rows_native) over the FULL native canvas
// width. Only affine-valid frames produce spans; empty/local/invalid frames
// return an empty list. Membership is exact: pixel (sx, sy) is included iff
//   center_qx=a0*(sx+.5)+a1*(sy+.5)+a2, radius_x=half*(|a0|+|a1|) (same y),
//   qx_max>0 && qx_min<canvas_width && qy_max>band_y0 && qy_min<band_y1
// evaluated per integer source coordinate. Used by the v2 production
// provider for exact ragged source/Q reads and launched-sample accounting.
std::vector<DrizzleAffineSourceSpan> drizzle_affine_source_spans(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, float pixfrac,
    int band_y_begin_native, int band_rows_native);

// Same result written into caller-owned storage (capacity preserved across
// calls) for allocation-free provider hot paths.
void drizzle_affine_source_spans_into(
    const registration::RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &frame, float pixfrac,
    int band_y_begin_native, int band_rows_native,
    std::vector<DrizzleAffineSourceSpan> &out);

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


// --- exposed for unit tests (plan section 11.6 geometry) -------------------

// Exact area of the intersection of a convex quadrilateral (4 vertices, in
// order around the boundary, either winding) with the axis-aligned rectangle
// [rx0,rx1] x [ry0,ry1]. Sutherland-Hodgman clip + shoelace.
double polygon_rectangle_intersection_area(const double poly_x[4],
                                           const double poly_y[4], double rx0,
                                           double ry0, double rx1, double ry1);

double shoelace_area(const double *x, const double *y, int n);

} // namespace tile_compile::reconstruction
