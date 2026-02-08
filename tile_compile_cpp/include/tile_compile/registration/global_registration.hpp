#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace tile_compile::registration {

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

// Sub-functions (canonical implementations — do NOT duplicate in runner)
Matrix2Df downsample2x2_mean(const Matrix2Df& in);
WarpMatrix scale_translation_warp(const WarpMatrix& w, float scale);

RegistrationResult star_registration_similarity(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation,
    int topk_stars, int min_inliers,
    float inlier_tol_px, float dist_bin_px);

RegistrationResult feature_registration_similarity(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation);

RegistrationResult triangle_star_matching(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation,
    int topk_stars, int min_inliers,
    float inlier_tol_px);

RegistrationResult hybrid_phase_ecc(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation);

RegistrationResult trail_endpoint_registration(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation, int topk_stars, int min_inliers,
    float inlier_tol_px, float dist_bin_px);

RegistrationResult robust_phase_ecc(
    const Matrix2Df& mov, const Matrix2Df& ref,
    bool allow_rotation);

float estimate_rotation_logpolar(const cv::Mat& ref, const cv::Mat& mov);

} // namespace tile_compile::registration
