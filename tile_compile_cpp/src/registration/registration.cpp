#include "tile_compile/core/types.hpp"
#include "tile_compile/registration/registration.hpp"
#include <opencv2/opencv.hpp>

namespace tile_compile::registration {

Matrix2Df prepare_ecc_image(const Matrix2Df& img) {
    cv::Mat cv_img(img.rows(), img.cols(), CV_32F, const_cast<float*>(img.data()));
    cv::Mat blurred;
    cv::GaussianBlur(cv_img, blurred, cv::Size(0, 0), 1.5);
    
    double minVal, maxVal;
    cv::minMaxLoc(blurred, &minVal, &maxVal);
    cv::Mat normalized = (blurred - minVal) / (maxVal - minVal + 1e-10);
    
    Matrix2Df result(img.rows(), img.cols());
    std::memcpy(result.data(), normalized.data, img.size() * sizeof(float));
    return result;
}

std::pair<float, float> phasecorr_translation(const Matrix2Df& moving, const Matrix2Df& ref) {
    cv::Mat cv_moving(moving.rows(), moving.cols(), CV_32F, const_cast<float*>(moving.data()));
    cv::Mat cv_ref(ref.rows(), ref.cols(), CV_32F, const_cast<float*>(ref.data()));
    
    cv::Point2d shift = cv::phaseCorrelate(cv_moving, cv_ref);
    return {static_cast<float>(shift.x), static_cast<float>(shift.y)};
}

RegistrationResult ecc_warp(const Matrix2Df& moving, const Matrix2Df& ref,
                            bool allow_rotation, const WarpMatrix& init_warp,
                            int max_iterations, float epsilon) {
    cv::Mat cv_moving(moving.rows(), moving.cols(), CV_32F, const_cast<float*>(moving.data()));
    cv::Mat cv_ref(ref.rows(), ref.cols(), CV_32F, const_cast<float*>(ref.data()));
    
    cv::Mat warp_matrix(2, 3, CV_32F);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            warp_matrix.at<float>(i, j) = init_warp(i, j);
        }
    }
    
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                               max_iterations, epsilon);
    
    int motion_type = allow_rotation ? cv::MOTION_EUCLIDEAN : cv::MOTION_TRANSLATION;
    
    RegistrationResult result;
    try {
        result.correlation = cv::findTransformECC(cv_ref, cv_moving, warp_matrix,
                                                   motion_type, criteria);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.warp(i, j) = warp_matrix.at<float>(i, j);
            }
        }
        result.success = true;
    } catch (const cv::Exception& e) {
        result.success = false;
        result.error_message = e.what();
        result.warp = init_warp;
    }
    
    return result;
}

WarpMatrix identity_warp() {
    WarpMatrix warp;
    warp << 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f;
    return warp;
}

Matrix2Df apply_warp(const Matrix2Df& img, const WarpMatrix& warp) {
    cv::Mat cv_img(img.rows(), img.cols(), CV_32F, const_cast<float*>(img.data()));
    cv::Mat warp_matrix(2, 3, CV_32F);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            warp_matrix.at<float>(i, j) = warp(i, j);
        }
    }
    
    cv::Mat warped;
    cv::warpAffine(cv_img, warped, warp_matrix, cv_img.size(),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    
    Matrix2Df result(img.rows(), img.cols());
    std::memcpy(result.data(), warped.data, img.size() * sizeof(float));
    return result;
}

BoundingBox compute_warps_bounding_box(int frame_width, int frame_height,
                                       const std::vector<WarpMatrix>& warps) {
    if (warps.empty()) {
        return BoundingBox{0, 0, frame_width, frame_height};
    }
    
    // Frame corners in input space
    std::vector<std::pair<float, float>> corners = {
        {0.0f, 0.0f},
        {static_cast<float>(frame_width), 0.0f},
        {0.0f, static_cast<float>(frame_height)},
        {static_cast<float>(frame_width), static_cast<float>(frame_height)}
    };
    
    float min_x = 0.0f, min_y = 0.0f;
    float max_x = static_cast<float>(frame_width);
    float max_y = static_cast<float>(frame_height);
    
    // Transform all corners of all frames
    for (const auto& warp : warps) {
        // Warp is R→M (reference to moving) with WARP_INVERSE_MAP
        // To get output coordinates, we need the inverse: M→R
        // For affine: [x', y'] = [a00 a01 tx] [x]
        //                        [a10 a11 ty] [y]
        //                                     [1]
        
        for (const auto& corner : corners) {
            float x = corner.first;
            float y = corner.second;
            
            // Apply warp (this gives us the output position)
            float x_out = warp(0, 0) * x + warp(0, 1) * y + warp(0, 2);
            float y_out = warp(1, 0) * x + warp(1, 1) * y + warp(1, 2);
            
            min_x = std::min(min_x, x_out);
            min_y = std::min(min_y, y_out);
            max_x = std::max(max_x, x_out);
            max_y = std::max(max_y, y_out);
        }
    }
    
    // Round to integers with some padding
    BoundingBox bbox;
    bbox.min_x = static_cast<int>(std::floor(min_x));
    bbox.min_y = static_cast<int>(std::floor(min_y));
    bbox.max_x = static_cast<int>(std::ceil(max_x));
    bbox.max_y = static_cast<int>(std::ceil(max_y));
    
    return bbox;
}

} // namespace tile_compile::registration
