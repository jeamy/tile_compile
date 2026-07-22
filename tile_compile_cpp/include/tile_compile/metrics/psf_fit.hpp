#pragma once

#include <opencv2/core.hpp>

namespace tile_compile::metrics {

struct PsfFit2D {
    float fwhm_major = 0.0f;
    float fwhm_minor = 0.0f;
};

PsfFit2D fit_elliptical_psf_2d(const cv::Mat& patch, float bg);

} // namespace tile_compile::metrics
