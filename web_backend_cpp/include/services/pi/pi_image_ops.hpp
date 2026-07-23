#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

struct ImageOpResult {
    cv::Mat image;
    std::string error;
    bool success = false;
};

// Dispatch: reads "type" and "params" from op, calls the matching function.
// Works on 8-bit BGR display images.
ImageOpResult apply_image_op(const cv::Mat& input, const nlohmann::json& op);

// Dispatch for linear float data (CV_32F, values in [0,1], RGB or 1-channel).
// Operations are applied to the original-data representation before display stretch.
ImageOpResult apply_image_op_fits(const cv::Mat& input, const nlohmann::json& op);

// Compute inverse operation for +/- decrease where parameter negation is safe.
// Undo/redo rebuild deterministically from the operation stack, so
// non-invertible operations do not require pixel snapshots.
nlohmann::json invert_op(const nlohmann::json& op);

// Phase 1 operations
cv::Mat apply_brightness(const cv::Mat& img, double midtones,
                         double shadows, double highlights);
cv::Mat apply_contrast(const cv::Mat& img, double amount);
cv::Mat apply_saturation(const cv::Mat& img, double amount);
cv::Mat apply_sharpen(const cv::Mat& img, double amount, double radius);
cv::Mat apply_denoise(const cv::Mat& img, double strength, bool luminance);
cv::Mat apply_rmgreen(const cv::Mat& img, double strength);
cv::Mat apply_clahe(const cv::Mat& img, double cliplimit, int tilesize);
cv::Mat apply_bilateral(const cv::Mat& img, int d,
                        double sigma_color, double sigma_space);
cv::Mat apply_threshold(const cv::Mat& img, double black_point,
                        double white_point);
cv::Mat apply_invert(const cv::Mat& img);
cv::Mat apply_crop(const cv::Mat& img, int x, int y, int w, int h);

// Phase 2 operations
cv::Mat apply_vibrance(const cv::Mat& img, double amount);
cv::Mat apply_color_temperature(const cv::Mat& img, double amount);
cv::Mat apply_unpurple(const cv::Mat& img, double amount);
cv::Mat apply_fixbanding(const cv::Mat& img, double amount, double sigma);
cv::Mat apply_star_desaturation(const cv::Mat& img, double amount);
cv::Mat apply_dehaze(const cv::Mat& img, double amount);

// Helpers
double clamp_param(double val, double lo, double hi);
nlohmann::json validate_op(const nlohmann::json& op);

} // namespace tile_compile::pi
