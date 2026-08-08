#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace tile_compile::registration {

struct StarPoint {
    float x = 0.0f;
    float y = 0.0f;
    float flux = 0.0f;
};

// Result of a conservative affine fine-registration fit on an already warped
// proxy frame. correction_warp uses the same R->M/WARP_INVERSE_MAP convention
// as the global registration warps.
struct AffineStarRefinementResult {
    WarpMatrix correction_warp = WarpMatrix::Zero();
    bool valid = false;
    std::string rejection_reason = "not_attempted";
    int matched_stars = 0;
    int inlier_stars = 0;
    float inlier_ratio = 0.0f;
    float spatial_coverage = 0.0f;
    float median_before_px = 0.0f;
    float p90_before_px = 0.0f;
    float rms_before_px = 0.0f;
    float median_after_px = 0.0f;
    float p90_after_px = 0.0f;
    float rms_after_px = 0.0f;
    float center_displacement_px = 0.0f;
    float rotation_deg = 0.0f;
    float min_scale = 1.0f;
    float max_scale = 1.0f;
};

// Smooth inverse displacement model C(q)=q+d(q) on proxy reference
// coordinates. The fixed 4x4 Gaussian basis keeps the experiment bounded and
// serializable without adding a TPS/OpenCV-shape dependency.
struct SmoothLocalWarpModel {
    bool valid = false;
    int image_rows = 0;
    int image_cols = 0;
    VectorXf coeff_x;
    VectorXf coeff_y;
};

struct SmoothLocalRefinementResult {
    SmoothLocalWarpModel model;
    bool valid = false;
    std::string rejection_reason = "not_attempted";
    int matched_stars = 0;
    int training_stars = 0;
    int validation_stars = 0;
    float spatial_coverage = 0.0f;
    float median_before_px = 0.0f;
    float p90_before_px = 0.0f;
    float rms_before_px = 0.0f;
    float median_after_px = 0.0f;
    float p90_after_px = 0.0f;
    float rms_after_px = 0.0f;
    float validation_median_before_px = 0.0f;
    float validation_p90_before_px = 0.0f;
    float validation_rms_before_px = 0.0f;
    float validation_median_after_px = 0.0f;
    float validation_p90_after_px = 0.0f;
    float validation_rms_after_px = 0.0f;
    float max_displacement_px = 0.0f;
    float min_jacobian_determinant = 1.0f;
    float max_jacobian_determinant = 1.0f;
    float min_local_scale = 1.0f;
    float max_local_scale = 1.0f;
};

struct GlobalRegistrationOutput {
    int ref_idx = 0;
    std::string ref_selection_method; // "global_weight" | "quality_score" | "middle"
    float ref_selection_value = 0.0f;

    float downsample_scale = 1.0f;
    std::string engine_used;
    std::vector<WarpMatrix> warps_fullres;
    std::vector<float> scores;
    std::vector<bool> success;
    std::vector<std::string> errors;
};

GlobalRegistrationOutput register_frames_to_reference(
    const std::vector<Matrix2Df>& frames_fullres,
    ColorMode mode,
    BayerPattern bayer,
    const config::RegistrationConfig& rcfg,
    const std::vector<FrameMetrics>* frame_metrics_opt = nullptr,
    const VectorXf* global_weights_opt = nullptr
);

// Single-frame registration result (canonical cascade output)
struct SingleFrameRegResult {
    RegistrationResult reg;         // warp (R→M direction), correlation, success
    std::string method_used;        // "triangle" | "star_pair" | "akaze" |
                                    // "robust_phase_ecc" | "identity"
    float ncc_identity = 0.0f;      // NCC before warp (identity baseline)
    float ncc_warped   = 0.0f;      // NCC after warp
};

// Canonical single-frame registration cascade.
// Runs all cascade stages with NCC validation.
// mov and ref are proxy-resolution images (already downsampled).
SingleFrameRegResult register_single_frame(
    const Matrix2Df& mov, const Matrix2Df& ref,
    const config::RegistrationConfig& rcfg,
    float min_ncc_improvement = 0.01f);

// Sub-functions (canonical implementations — do NOT duplicate in runner)
Matrix2Df downsample2x2_mean(const Matrix2Df& in);
WarpMatrix scale_translation_warp(const WarpMatrix& w, float scale);
std::vector<StarPoint> detect_stars_simple(
    const Matrix2Df& img, int topk,
    bool enable_local_background_subtraction = false);

AffineStarRefinementResult estimate_affine_star_refinement(
    const std::vector<StarPoint>& ref_stars,
    const std::vector<StarPoint>& warped_stars,
    int image_rows, int image_cols,
    float match_radius_px = 3.0f);

SmoothLocalRefinementResult estimate_smooth_local_star_refinement(
    const std::vector<StarPoint>& ref_stars,
    const std::vector<StarPoint>& warped_stars,
    int image_rows, int image_cols,
    float match_radius_px = 3.0f);

cv::Point2f evaluate_smooth_local_displacement(
    const SmoothLocalWarpModel& model, float x, float y);

void render_smooth_local_displacement(
    const SmoothLocalWarpModel& model, int output_rows, int output_cols,
    float model_coordinate_scale, float model_offset_x, float model_offset_y,
    cv::Mat& displacement_x, cv::Mat& displacement_y);

Matrix2Df apply_smooth_local_correction(
    const Matrix2Df& already_warped,
    const SmoothLocalWarpModel& model,
    const std::string& interpolation = "cubic");

cv::Mat smooth_local_valid_mask(int image_rows, int image_cols,
                                const SmoothLocalWarpModel& model);

RegistrationResult star_registration_similarity(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation,
    int topk_stars, int min_inliers,
    float inlier_tol_px, float dist_bin_px,
    const std::string& transform_model,
    bool enable_local_background_subtraction = false);

RegistrationResult feature_registration_similarity(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation, const std::string& transform_model);

RegistrationResult triangle_star_matching(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation,
    int topk_stars, int min_inliers,
    float inlier_tol_px, const std::string& transform_model,
    bool enable_local_background_subtraction = false,
    float shift_radius_px = 200.0f);

RegistrationResult robust_phase_ecc(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation);

RegistrationResult robust_phase_ecc_seeded(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation, const WarpMatrix& init_warp);

float estimate_rotation_logpolar(const cv::Mat& ref, const cv::Mat& mov);

// §4.13 — Helper für astrometrische Rescue und interne Verwendung
cv::Mat warp_valid_mask(const Matrix2Df& img, const WarpMatrix& warp);
float compute_ncc_masked(const Matrix2Df& a, const Matrix2Df& b,
                         const cv::Mat& mask, int* used_pixels = nullptr);

} // namespace tile_compile::registration
