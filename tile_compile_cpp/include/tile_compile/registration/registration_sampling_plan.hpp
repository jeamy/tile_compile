#pragma once

// RegistrationSamplingPlan --- data model for CFA-aware forward drizzle.
//
// This is the milestone-M0 data model from the implementation plan
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md, section 7).
// It carries the per-frame source->canvas sampling geometry that the forward
// drizzle reconstruction consumes instead of a pre-warped signal image.
//
// The plan (section 7) mandates:
//   * warp convention: the existing registration warp maps canvas -> source
//     (s = W(q)); forward drizzle needs q = W^-1(s).
//   * per-frame stable identity (frame_id + source_index) so a resume can match
//     the plan against the normalized-frame cache and the input manifest.
//   * a canonical plan_hash over the *native* sampling geometry only:
//     internal_scale / output_scale are written to the artifact but are NOT part
//     of plan_hash (they belong to the drizzle-geometry hash domain, section 18.3).
//   * the local (non-affine) warp model coordinates (scale + offsets) so a local
//     model is resumable.

#include "tile_compile/core/types.hpp"
#include "tile_compile/registration/global_registration.hpp"  // SmoothLocalWarpModel

#include <cstddef>
#include <string>
#include <vector>

namespace tile_compile::registration {

// Only one convention is supported: the stored warp maps a native canvas
// coordinate q to a source coordinate s (s = W(q)). Forward drizzle inverts it.
enum class SamplingWarpConvention {
    canvas_to_source
};

std::string sampling_warp_convention_to_string(SamplingWarpConvention c);
SamplingWarpConvention string_to_sampling_warp_convention(const std::string& s);

// Per-frame sampling transform. `canvas_to_source` is the existing registration
// warp (2x3 affine, WARP_INVERSE_MAP convention). `source_to_canvas` is its
// checked affine inverse for purely affine frames; when a local model is active
// it is only the affine seed and the non-linear inversion (plan section 7.3)
// must be used instead.
struct FrameSamplingTransform {
    std::string frame_id;                       // stable: input manifest + content identity
    std::size_t source_index = 0;               // canonical ordering index

    bool valid = false;                         // usable for forward drizzle

    WarpMatrix canvas_to_source = WarpMatrix::Identity();
    WarpMatrix source_to_canvas = WarpMatrix::Identity();
    bool source_to_canvas_affine_valid = false;

    bool has_smooth_local_model = false;
    SmoothLocalWarpModel smooth_local_model;

    // Coordinate remap contract of the local model, persisted so the model is
    // resumable without recomputing registration (plan section 7.3).
    float model_coordinate_scale = 1.0f;
    float model_offset_x = 0.0f;
    float model_offset_y = 0.0f;

    // Per-frame effective-weight inputs (plan section 11.9). Persisted here and
    // read-only for GLOBAL_QUALITY so the registration factors are applied
    // exactly once.
    float registration_residual_factor = 1.0f;
    bool  residual_applicable = true;
    float model_prediction_factor = 1.0f;

    bool model_predicted = false;               // provenance only; weight is in the factor
    int  chain_depth = 0;                       // provenance only; effect folded into model_prediction_factor
    std::string provenance;                     // diagnostic string, not hashed
};

struct RegistrationSamplingPlan {
    static constexpr int kSchemaVersion = 2;

    int source_width = 0;
    int source_height = 0;

    int canvas_width_native = 0;
    int canvas_height_native = 0;
    int canvas_offset_x_native = 0;
    int canvas_offset_y_native = 0;

    // Written to the artifact, validated separately on load, but NOT part of
    // plan_hash (plan section 7.4 / 18.3).
    int internal_scale = 1;
    int output_scale = 1;

    ColorMode color_mode = ColorMode::MONO;
    BayerPattern bayer_pattern = BayerPattern::UNKNOWN;

    // Sensor parity of the normalized-cache coordinate (0,0). For OSC this fixes
    // which Bayer colour a given integer source coordinate carries.
    int cfa_origin_x = 0;
    int cfa_origin_y = 0;

    SamplingWarpConvention convention = SamplingWarpConvention::canvas_to_source;

    std::vector<FrameSamplingTransform> frames;

    // Canonical hash over the native sampling geometry only (see compute_plan_hash).
    std::string source_identity_hash; // input manifest and effective calibration/normalization config
    std::string plan_hash;
};

// Convert OpenCV integer-index centers to centers at x+0.5, y+0.5.
WarpMatrix opencv_to_edge_sampling_map(const WarpMatrix& map);

// --- Affine inversion (plan section 7.2) -------------------------------------
//
// Computes the checked 2x3 affine inverse of `canvas_to_source`. Returns false
// (and leaves `out` untouched) on a singular / non-finite matrix or a
// determinant outside [det_min, det_max]. The bounds are the existing
// registration scale-reject bounds squared (a 2x3 affine determinant is the
// squared linear scale for a similarity transform); callers pass
// cfg.registration.reject_scale_min^2 and reject_scale_max^2.
bool invert_affine_2x3(const WarpMatrix& canvas_to_source,
                       float det_min, float det_max,
                       WarpMatrix& out);

// --- Local (non-affine) source->canvas inversion (plan section 7.3) ---------
//
// Solves q + d(q) = u for q by the bounded fixed-point iteration
//   q_0 = u ;  q_{n+1} = u - d(q_n)
// where u = inverse(W_global)(s) and d() is evaluated with the persisted
// model_coordinate_scale / model_offset_{x,y}.
//
// Returns true and writes the converged native canvas coordinate to
// (out_qx, out_qy) on success. Returns false when: the iteration does not reach
// `tol_px` within `max_iter`, a non-finite value appears, or the point leaves
// `safety_margin_px` outside the canvas. `max_iter` <= 6 and `tol_px` == 1e-3
// per the plan; they are parameters here only so tests can exercise edge cases.
struct LocalInversionParams {
    int   max_iter = 6;
    float tol_px = 1.0e-3f;
    float safety_margin_px = 64.0f;
};

bool invert_local_source_to_canvas(const FrameSamplingTransform& frame,
                                   float sx, float sy,
                                   int canvas_width_native,
                                   int canvas_height_native,
                                   const LocalInversionParams& params,
                                   float& out_qx, float& out_qy);

// --- Serialization (plan section 7.4) --------------------------------------
//
// serialize_to_json_string() produces the artifacts/registration_sampling.json
// content (pretty-printed). parse_from_json_string() is its inverse and is
// lossless for every field. Neither call recomputes plan_hash; call
// compute_plan_hash() explicitly and compare.
std::string serialize_to_json_string(const RegistrationSamplingPlan& plan);
bool parse_from_json_string(const std::string& json_text,
                            RegistrationSamplingPlan& out,
                            std::string& error);

// Canonical hash of the native sampling geometry. Uses a fixed field order,
// little-endian byte layout and bit-exact IEEE-754 float encoding --- NOT the
// formatted JSON --- so the hash is stable across platforms and JSON
// float formatting. Excludes: internal_scale, output_scale, chain_depth,
// model_predicted, provenance, and all timestamps (plan section 7.4).
std::string compute_plan_hash(const RegistrationSamplingPlan& plan);

}  // namespace tile_compile::registration
