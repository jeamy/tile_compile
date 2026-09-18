#include "tile_compile/reconstruction/luma_denoise.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

float robust_sigma_mad_from_mat(const cv::Mat& m) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    vals.insert(vals.end(), row, row + m.cols);
  }
  return tile_compile::core::robust_sigma_mad(vals);
}

float percentile_from_mat(const cv::Mat& m, float p) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    vals.insert(vals.end(), row, row + m.cols);
  }
  if (vals.empty()) return 0.0f;
  return tile_compile::core::percentile_of(vals, p);
}

cv::Mat soft_threshold_signed(const cv::Mat& src, float tau) {
  if (!(tau > 0.0f)) return src.clone();
  cv::Mat abs_src = cv::abs(src);
  cv::Mat shrunk;
  cv::subtract(abs_src, tau, shrunk);
  cv::threshold(shrunk, shrunk, 0.0, 0.0, cv::THRESH_TOZERO);

  cv::Mat neg_mask;
  cv::compare(src, 0.0f, neg_mask, cv::CMP_LT);
  cv::Mat neg_shrunk;
  cv::subtract(cv::Scalar(0.0f), shrunk, neg_shrunk);
  neg_shrunk.copyTo(shrunk, neg_mask);
  return shrunk;
}

/// @brief Builds the star/structure protection mask for luma denoise.
/// @details Mirrors chroma_denoise's build_protection_mask (star_protection
/// + structure_protection only -- extended_source_protection and
/// large_scale_bias exist there to keep real object COLOR out of a
/// background-bias estimate, which has no equivalent here: smoothing a
/// smooth, extended, low-gradient region is exactly the low-risk case for
/// luma denoise, not a hazard to guard against).
cv::Mat build_protection_mask(const cv::Mat& y,
                              const config::LumaDenoiseConfig& cfg,
                              LumaDenoiseStats* stats) {
  cv::Mat mask = cv::Mat::zeros(y.size(), CV_32F);
  const double pixels = static_cast<double>(y.total());

  if (cfg.star_protection.enabled) {
    const float sigma = robust_sigma_mad_from_mat(y);
    cv::Scalar mean_y;
    cv::Scalar std_y;
    cv::meanStdDev(y, mean_y, std_y);
    const float med_like = static_cast<float>(mean_y[0]);
    const float thr = med_like + cfg.star_protection.threshold_sigma * (sigma + 1.0e-6f);
    cv::Mat stars;
    cv::threshold(y, stars, thr, 1.0, cv::THRESH_BINARY);
    stars.convertTo(stars, CV_32F);
    if (cfg.star_protection.dilate_px > 0) {
      const int k = std::max(1, cfg.star_protection.dilate_px * 2 + 1);
      cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
      cv::dilate(stars, stars, ker);
    }
    if (stats && pixels > 0.0)
      stats->star_protected_fraction = cv::countNonZero(stars > 0.5f) / pixels;
    // Own-scale feather -- see chroma_denoise.cpp's build_protection_mask
    // for why a fixed narrow blur turns the mask boundary into a visible
    // ring, and why the feather must track this component's own dilation
    // radius rather than a combined-mask constant.
    {
      const float star_feather = std::max(
          1.0f, static_cast<float>(cfg.star_protection.dilate_px) / 3.0f);
      cv::GaussianBlur(stars, stars, cv::Size(0, 0), star_feather, star_feather,
                       cv::BORDER_REFLECT_101);
    }
    cv::max(mask, stars, mask);
  }

  if (cfg.structure_protection.enabled) {
    cv::Mat gx, gy, mag;
    cv::Sobel(y, gx, CV_32F, 1, 0, 3);
    cv::Sobel(y, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    const float p = percentile_from_mat(mag, cfg.structure_protection.gradient_percentile);
    cv::Mat structures;
    cv::threshold(mag, structures, p, 1.0, cv::THRESH_BINARY);
    structures.convertTo(structures, CV_32F);
    if (stats && pixels > 0.0)
      stats->structure_protected_fraction =
          cv::countNonZero(structures > 0.5f) / pixels;
    // Own-scale feather -- see chroma_denoise.cpp's build_protection_mask
    // for why this component was left as a hard per-pixel threshold and why
    // that matters here (it catches scattered single/few-pixel gradient
    // spikes, real edges and noise both).
    cv::GaussianBlur(structures, structures, cv::Size(0, 0), 2.0f, 2.0f,
                     cv::BORDER_REFLECT_101);
    cv::max(mask, structures, mask);
  }

  // Final small anti-aliasing pass (star_protection is already feathered at
  // its own scale above; this only smooths the max() seam with
  // structure_protection).
  cv::GaussianBlur(mask, mask, cv::Size(0, 0), 1.0, 1.0, cv::BORDER_REFLECT_101);
  cv::min(mask, 1.0, mask);
  cv::max(mask, 0.0, mask);
  if (stats && pixels > 0.0) {
    stats->combined_protected_fraction = cv::countNonZero(mask > 0.5f) / pixels;
    stats->mean_protection = cv::mean(mask)[0];
  }
  return mask;
}

/// @brief Multi-level Gaussian-pyramid soft-threshold wavelet denoise.
/// @details Identical structure to chroma_denoise's chroma_wavelet stage,
/// applied here to luma instead of a chroma plane.
void denoise_luma_plane_inplace(cv::Mat& y,
                                const config::LumaDenoiseConfig::WaveletConfig& cfg) {
  if (!cfg.enabled) return;
  cv::Mat cur = y.clone();
  const int levels = std::max(1, cfg.levels);
  for (int lvl = 0; lvl < levels; ++lvl) {
    const double sigma = std::pow(2.0, static_cast<double>(lvl)) * 0.75;
    cv::Mat low;
    cv::GaussianBlur(cur, low, cv::Size(0, 0), sigma, sigma, cv::BORDER_REFLECT_101);
    cv::Mat detail = cur - low;
    const float sigma_n = robust_sigma_mad_from_mat(detail);
    const float tau = cfg.threshold_scale * cfg.soft_k * sigma_n;
    cv::Mat shrunk = soft_threshold_signed(detail, tau);
    cur = low + shrunk;
  }
  y = cur;
}

} // namespace

LumaDenoiseStats luma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::LumaDenoiseConfig& cfg,
    Matrix2Df* protection_mask_out) {
  LumaDenoiseStats stats;
  if (!cfg.enabled) return stats;
  if (r.size() <= 0 || g.size() <= 0 || b.size() <= 0) return stats;
  if (r.rows() != g.rows() || r.cols() != g.cols() ||
      r.rows() != b.rows() || r.cols() != b.cols()) {
    return stats;
  }

  cv::Mat R(r.rows(), r.cols(), CV_32F, r.data());
  cv::Mat G(g.rows(), g.cols(), CV_32F, g.data());
  cv::Mat B(b.rows(), b.cols(), CV_32F, b.data());

  cv::Mat Y = 0.25f * R + 0.5f * G + 0.25f * B;
  stats.applied = true;
  stats.valid_pixels = static_cast<std::uint64_t>(r.size());
  stats.input_luma_sigma = robust_sigma_mad_from_mat(Y);

  cv::Mat protect = build_protection_mask(Y, cfg, &stats);
  if (protection_mask_out != nullptr) {
    protection_mask_out->resize(r.rows(), r.cols());
    cv::Mat out_view(r.rows(), r.cols(), CV_32F, protection_mask_out->data());
    protect.copyTo(out_view);
  }

  cv::Mat Y_denoised = Y.clone();
  denoise_luma_plane_inplace(Y_denoised, cfg.wavelet);

  cv::Mat amount_map(Y.size(), CV_32F, cv::Scalar(cfg.blend_amount));
  amount_map = amount_map.mul(1.0f - cfg.luma_guard_strength * protect);
  cv::min(amount_map, cfg.blend_amount, amount_map);
  cv::max(amount_map, 0.0, amount_map);
  stats.mean_denoise_fraction = cv::mean(amount_map)[0];

  cv::Mat one_minus = 1.0f - amount_map;
  cv::Mat Y_mix = Y.mul(one_minus) + Y_denoised.mul(amount_map);

  // Preserve each pixel's original per-channel DIFFERENCE from luma (the
  // chroma, R-Y/G-Y/B-Y) rather than its RATIO to luma. A ratio divides two
  // independently-noisy, correlated quantities (R and Y share the same
  // per-pixel noise realisation in each channel), which blows up exactly
  // where noise is proportionally largest -- faint background, and the
  // partially-protected PSF wings around a star where the blend fades in --
  // producing visible dark-pixel speckle and chroma fringing instead of
  // removing them. Adding the same smooth delta (Y_mix - Y) to every
  // channel is mathematically exact (0.25+0.5+0.25 == 1, so the new
  // weighted luma of R+delta/G+delta/B+delta is exactly Y_mix) and leaves
  // every channel DIFFERENCE, hence all perceived color, completely
  // unchanged -- only brightness moves, with none of the ratio's noise
  // amplification.
  cv::Mat delta = Y_mix - Y;
  cv::Mat R_new = R + delta;
  cv::Mat G_new = G + delta;
  cv::Mat B_new = B + delta;

  R_new.copyTo(R);
  G_new.copyTo(G);
  B_new.copyTo(B);
  return stats;
}

} // namespace tile_compile::reconstruction
