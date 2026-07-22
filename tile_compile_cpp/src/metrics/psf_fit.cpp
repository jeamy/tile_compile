#include "tile_compile/metrics/psf_fit.hpp"

#include <algorithm>
#include <cmath>

namespace tile_compile::metrics {

PsfFit2D fit_elliptical_psf_2d(const cv::Mat& patch, float bg) {
    PsfFit2D out;
    if (patch.empty()) return out;

    constexpr double kFwhmScale = 2.3548200450309493;  // 2*sqrt(2*ln(2))

    double sum_w = 0.0;
    double mx = 0.0;
    double my = 0.0;
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            const double w = std::max(0.0, static_cast<double>(row[x]) - static_cast<double>(bg));
            sum_w += w;
            mx += w * static_cast<double>(x);
            my += w * static_cast<double>(y);
        }
    }
    if (!(sum_w > 0.0)) return out;

    mx /= sum_w;
    my /= sum_w;

    double cxx = 0.0;
    double cyy = 0.0;
    double cxy = 0.0;
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            const double w = std::max(0.0, static_cast<double>(row[x]) - static_cast<double>(bg));
            if (!(w > 0.0)) continue;
            const double dx = static_cast<double>(x) - mx;
            const double dy = static_cast<double>(y) - my;
            cxx += w * dx * dx;
            cyy += w * dy * dy;
            cxy += w * dx * dy;
        }
    }

    cxx /= sum_w;
    cyy /= sum_w;
    cxy /= sum_w;

    const double tr = cxx + cyy;
    const double det = cxx * cyy - cxy * cxy;
    const double disc = std::max(0.0, tr * tr - 4.0 * det);
    const double root = std::sqrt(disc);
    const double lambda_major = 0.5 * (tr + root);
    const double lambda_minor = 0.5 * (tr - root);
    if (!(lambda_major > 0.0) || !(lambda_minor > 0.0)) return out;

    const double fwhm_major = kFwhmScale * std::sqrt(lambda_major);
    const double fwhm_minor = kFwhmScale * std::sqrt(lambda_minor);
    if (!(fwhm_major > 0.2 && fwhm_major < 50.0 &&
          fwhm_minor > 0.2 && fwhm_minor < 50.0)) {
        return out;
    }

    out.fwhm_major = static_cast<float>(fwhm_major);
    out.fwhm_minor = static_cast<float>(fwhm_minor);
    return out;
}

} // namespace tile_compile::metrics
