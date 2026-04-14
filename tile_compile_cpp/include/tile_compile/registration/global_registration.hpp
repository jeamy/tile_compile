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
