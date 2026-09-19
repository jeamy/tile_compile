#include "tile_compile/reconstruction/luma_denoise.hpp"
#include "tile_compile/core/utils.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

float robust_sigma_mad_from_mat(const cv::Mat& m,
                                const cv::Mat& valid = cv::Mat()) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    const std::uint8_t* vrow = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < m.cols; ++x)
      if ((!vrow || vrow[x] != 0) && std::isfinite(row[x])) vals.push_back(row[x]);
  }
  return tile_compile::core::robust_sigma_mad(vals);
}

float percentile_from_mat(const cv::Mat& m, float p,
                          const cv::Mat& valid = cv::Mat()) {
  if (m.empty()) return 0.0f;
  std::vector<float> vals;
  vals.reserve(m.total());
  for (int y = 0; y < m.rows; ++y) {
    const float* row = m.ptr<float>(y);
    const std::uint8_t* vrow = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < m.cols; ++x)
      if ((!vrow || vrow[x] != 0) && std::isfinite(row[x])) vals.push_back(row[x]);
  }
  if (vals.empty()) return 0.0f;
  return tile_compile::core::percentile_of(vals, p);
}

cv::Mat build_valid_mask(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b,
                         const std::vector<std::uint8_t>* supplied) {
  cv::Mat valid(r.size(), CV_8U, cv::Scalar(0));
  const bool have_supplied = supplied && supplied->size() == r.total();
  for (int y = 0; y < r.rows; ++y) {
    const float* rr = r.ptr<float>(y);
    const float* gg = g.ptr<float>(y);
    const float* bb = b.ptr<float>(y);
    std::uint8_t* vv = valid.ptr<std::uint8_t>(y);
    for (int x = 0; x < r.cols; ++x) {
      const size_t i = static_cast<size_t>(y) * r.cols + x;
      vv[x] = static_cast<std::uint8_t>(
          (!have_supplied || (*supplied)[i] != 0) && std::isfinite(rr[x]) &&
          std::isfinite(gg[x]) && std::isfinite(bb[x]));
    }
  }
  return valid;
}

float high_frequency_sigma(const cv::Mat& src, const cv::Mat& valid) {
  cv::Mat valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::Mat numerator, denominator, low;
  cv::GaussianBlur(src.mul(valid_f), numerator, cv::Size(0, 0), 0.75, 0.75,
                   cv::BORDER_REFLECT_101);
  cv::GaussianBlur(valid_f, denominator, cv::Size(0, 0), 0.75, 0.75,
                   cv::BORDER_REFLECT_101);
  cv::divide(numerator, denominator + 1.0e-12f, low);
  return robust_sigma_mad_from_mat(src - low, valid);
}

cv::Mat masked_gaussian_blur(const cv::Mat& src, const cv::Mat& valid,
                             double sigma) {
  cv::Mat valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::Mat numerator, denominator, out;
  cv::GaussianBlur(src.mul(valid_f), numerator, cv::Size(0, 0), sigma, sigma,
                   cv::BORDER_REFLECT_101);
  cv::GaussianBlur(valid_f, denominator, cv::Size(0, 0), sigma, sigma,
                   cv::BORDER_REFLECT_101);
  cv::divide(numerator, denominator + 1.0e-12f, out);
  src.copyTo(out, denominator < 1.0e-6f);
  return out;
}

void copy_to_matrix(const cv::Mat& src, Matrix2Df& dst) {
  dst.resize(src.rows, src.cols);
  cv::Mat view(src.rows, src.cols, CV_32F, dst.data());
  src.copyTo(view);
}

/// @brief Builds the star/structure protection mask for luma denoise.
/// @details Mirrors chroma_denoise's build_protection_mask (star_protection
/// + structure_protection only. Broad smooth emission remains in the final
/// low-pass remainder of the fixed multiscale decomposition; fine extended
/// detail is protected only when its gradient is significant above the local
/// noise floor. There is no background-bias subtraction in the luma path.
cv::Mat build_protection_mask(const cv::Mat& y,
                              const cv::Mat& valid,
                              const config::LumaDenoiseConfig& cfg,
                              LumaDenoiseStats* stats,
                              LumaDenoiseDiagnostics* diagnostics) {
  cv::Mat mask = cv::Mat::zeros(y.size(), CV_32F);
  const double pixels = static_cast<double>(cv::countNonZero(valid));

  if (cfg.star_protection.enabled) {
    const float sigma = robust_sigma_mad_from_mat(y, valid);
    const float sky_median = percentile_from_mat(y, 50.0f, valid);
    const float thr = sky_median + cfg.star_protection.threshold_sigma *
                                      (sigma + 1.0e-6f);
    cv::Mat stars;
    cv::threshold(y, stars, thr, 1.0, cv::THRESH_BINARY);
    stars.convertTo(stars, CV_32F);
    stars.setTo(0.0f, valid == 0);
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
    if (diagnostics) copy_to_matrix(stars, diagnostics->star_mask);
    cv::max(mask, stars, mask);
  }

  if (cfg.structure_protection.enabled) {
    cv::Mat gx, gy, mag;
    const cv::Mat y_grad = masked_gaussian_blur(y, valid, 0.5);
    cv::Sobel(y_grad, gx, CV_32F, 1, 0, 3);
    cv::Sobel(y_grad, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    const float med = percentile_from_mat(mag, 50.0f, valid);
    const float sig = robust_sigma_mad_from_mat(cv::abs(mag - med), valid);
    const float p = std::max(
        percentile_from_mat(mag, cfg.structure_protection.gradient_percentile,
                            valid),
        med + 3.0f * sig);
    cv::Mat structures;
    cv::threshold(mag, structures, p, 1.0, cv::THRESH_BINARY);
    structures.convertTo(structures, CV_32F);
    structures.setTo(0.0f, valid == 0);
    if (stats && pixels > 0.0)
      stats->structure_protected_fraction =
          cv::countNonZero(structures > 0.5f) / pixels;
    // Own-scale feather -- see chroma_denoise.cpp's build_protection_mask
    // for why this component was left as a hard per-pixel threshold and why
    // that matters here (it catches scattered single/few-pixel gradient
    // spikes, real edges and noise both).
    cv::GaussianBlur(structures, structures, cv::Size(0, 0), 2.0f, 2.0f,
                     cv::BORDER_REFLECT_101);
    if (diagnostics)
      copy_to_matrix(structures, diagnostics->structure_mask);
    cv::max(mask, structures, mask);
  }

  if (cfg.extended_source_protection.enabled) {
    // Heavily blur Y to suppress point sources; only extended smooth
    // emission survives at this scale (sigma ~= 2% of image short axis).
    // Mirrors chroma_denoise's build_protection_mask extended_source_
    // protection block -- see there for the reasoning on each step.
    const float blur_sigma = std::max(
        5.0f, static_cast<float>(std::min(y.rows, y.cols)) * 0.02f);
    cv::Mat y_smooth = masked_gaussian_blur(y, valid, blur_sigma);

    const float sky_med = percentile_from_mat(y_smooth, 50.0f, valid);
    std::vector<float> sky_vals;
    sky_vals.reserve(static_cast<size_t>(y_smooth.total()));
    for (int row = 0; row < y_smooth.rows; ++row) {
      const float* ptr = y_smooth.ptr<float>(row);
      for (int col = 0; col < y_smooth.cols; ++col)
        if (valid.at<std::uint8_t>(row, col) != 0 && std::isfinite(ptr[col]))
          sky_vals.push_back(ptr[col]);
    }
    const float sky_sigma = tile_compile::core::robust_sigma_mad(sky_vals);
    const float thr =
        sky_med + cfg.extended_source_protection.luma_sigma * sky_sigma;

    cv::Mat ext_src;
    cv::threshold(y_smooth, ext_src, static_cast<double>(thr), 1.0,
                  cv::THRESH_BINARY);
    ext_src.convertTo(ext_src, CV_32F);
    ext_src.setTo(0.0f, valid == 0);
    // A large Gaussian turns bright stars into smooth islands as well; keep
    // only connected regions whose area is large at the detection scale so
    // compact sources stay covered by star_protection instead of inflating
    // this mask into broad stellar discs.
    cv::Mat labels, cc_stats, centroids;
    cv::Mat ext_u8;
    ext_src.convertTo(ext_u8, CV_8U, 255.0);
    const int n_labels = cv::connectedComponentsWithStats(
        ext_u8, labels, cc_stats, centroids, 8, CV_32S);
    cv::Mat extended_only = cv::Mat::zeros(ext_src.size(), CV_32F);
    const int min_component_area = std::max(
        64, static_cast<int>(std::ceil(4.0 * CV_PI * blur_sigma * blur_sigma)));
    for (int label = 1; label < n_labels; ++label) {
      if (cc_stats.at<int>(label, cv::CC_STAT_AREA) >= min_component_area)
        extended_only.setTo(1.0f, labels == label);
    }
    ext_src = extended_only;
    if (stats) {
      stats->extended_source_sky_median = sky_med;
      stats->extended_source_sky_sigma = sky_sigma;
      stats->extended_source_threshold = thr;
      if (pixels > 0.0)
        stats->extended_source_raw_fraction =
            cv::countNonZero(ext_src > 0.5f) / pixels;
    }
    if (cfg.extended_source_protection.dilate_px > 0) {
      const int k = std::max(1, cfg.extended_source_protection.dilate_px * 2 + 1);
      cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
      cv::dilate(ext_src, ext_src, ker);
    }
    if (stats && pixels > 0.0)
      stats->extended_source_protected_fraction =
          cv::countNonZero(ext_src > 0.5f) / pixels;
    {
      const float ext_feather = std::max(
          1.0f,
          static_cast<float>(cfg.extended_source_protection.dilate_px) / 3.0f);
      cv::GaussianBlur(ext_src, ext_src, cv::Size(0, 0), ext_feather,
                       ext_feather, cv::BORDER_REFLECT_101);
    }
    if (diagnostics) copy_to_matrix(ext_src, diagnostics->extended_source_mask);
    cv::max(mask, ext_src, mask);
  }

  // Final small anti-aliasing pass (star_protection is already feathered at
  // its own scale above; this only smooths the max() seam with
  // structure_protection).
  cv::GaussianBlur(mask, mask, cv::Size(0, 0), 1.0, 1.0, cv::BORDER_REFLECT_101);
  cv::min(mask, 1.0, mask);
  cv::max(mask, 0.0, mask);
  mask.setTo(0.0f, valid == 0);
  if (stats && pixels > 0.0) {
    stats->combined_protected_fraction = cv::countNonZero(mask > 0.5f) / pixels;
    stats->mean_protection = cv::mean(mask, valid)[0];
  }
  if (diagnostics) copy_to_matrix(mask, diagnostics->combined_mask);
  return mask;
}

/// @brief Multi-level Gaussian-pyramid soft-threshold wavelet denoise.
/// @details Identical structure to chroma_denoise's chroma_wavelet stage,
/// applied here to luma instead of a chroma plane.
void denoise_luma_plane_inplace(cv::Mat& y,
                                const config::LumaDenoiseConfig::WaveletConfig& cfg,
                                const cv::Mat& valid, const cv::Mat& protect) {
  if (!cfg.enabled) return;
  cv::Mat approximation = y.clone();
  std::vector<cv::Mat> details;
  const int levels = std::max(1, cfg.levels);
  for (int lvl = 0; lvl < levels; ++lvl) {
    const double sigma = std::pow(2.0, static_cast<double>(lvl)) * 0.75;
    cv::Mat low;
    low = masked_gaussian_blur(approximation, valid, sigma);
    details.push_back(approximation - low);
    approximation = low;
  }
  cv::Mat noise_mask = valid.clone();
  if (!protect.empty()) noise_mask.setTo(0, protect > 0.1f);
  cv::Mat reconstructed = approximation;
  for (cv::Mat& detail : details) {
    const float sigma_n = robust_sigma_mad_from_mat(detail, noise_mask);
    const float tau = cfg.threshold_scale * sigma_n;
    cv::Mat ratio;
    cv::divide(tau, cv::abs(detail) + 1.0e-12f, ratio);
    cv::pow(ratio, cfg.soft_k, ratio);
    cv::Mat gain = 1.0f - ratio;
    cv::max(gain, 0.0f, gain);
    if (cfg.boost > 0.0f) {
      // Only coefficients clearly above the noise floor (>= 2*tau) get any
      // extra amplification; the ramp is 0 at 2*tau, reaching cfg.boost as
      // |detail| grows well past it -- coefficients near tau stay at the
      // plain denoised gain above, never boosted.
      cv::Mat boost_ratio;
      cv::divide(2.0f * tau, cv::abs(detail) + 1.0e-12f, boost_ratio);
      cv::Mat extra = 1.0f - boost_ratio;
      cv::max(extra, 0.0f, extra);
      gain += extra * cfg.boost;
    }
    reconstructed += detail.mul(gain);
  }
  y = reconstructed;
}

/// @brief Edge-preserving bilateral smoothing, applied after the wavelet
/// stage. Identical structure to chroma_denoise's chroma_bilateral, applied
/// here to luma instead of a chroma plane.
void denoise_luma_bilateral_inplace(
    cv::Mat& y, const config::LumaDenoiseConfig::BilateralConfig& cfg,
    const cv::Mat& valid, const cv::Mat& protect) {
  if (!cfg.enabled) return;
  cv::Mat noise_mask = valid.clone();
  if (!protect.empty()) noise_mask.setTo(0, protect > 0.1f);
  const float sigma_noise = robust_sigma_mad_from_mat(y, noise_mask);
  cv::Mat bilateral_input = y.clone();
  cv::Mat boundary_fill =
      masked_gaussian_blur(y, valid, std::max(0.5f, cfg.sigma_spatial));
  boundary_fill.copyTo(bilateral_input, valid == 0);
  cv::Mat out;
  cv::bilateralFilter(bilateral_input, out, 0,
                      std::max(1.0e-6f, cfg.sigma_range * sigma_noise),
                      cfg.sigma_spatial, cv::BORDER_REFLECT_101);
  y.copyTo(out, valid == 0);
  y = out;
}

} // namespace

LumaDenoiseStats luma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::LumaDenoiseConfig& cfg,
    Matrix2Df* protection_mask_out,
    const std::vector<std::uint8_t>* valid_mask,
    LumaDenoiseDiagnostics* diagnostics) {
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
  const cv::Mat valid = build_valid_mask(R, G, B, valid_mask);
  const auto valid_pixels = static_cast<std::uint64_t>(cv::countNonZero(valid));
  if (valid_pixels == 0) return stats;

  cv::Mat Y = 0.25f * R + 0.5f * G + 0.25f * B;
  stats.applied = true;
  stats.valid_pixels = valid_pixels;
  stats.input_luma_sigma = high_frequency_sigma(Y, valid);

  cv::Mat protect = build_protection_mask(Y, valid, cfg, &stats, diagnostics);
  if (protection_mask_out != nullptr) {
    protection_mask_out->resize(r.rows(), r.cols());
    cv::Mat out_view(r.rows(), r.cols(), CV_32F, protection_mask_out->data());
    protect.copyTo(out_view);
  }

  cv::Mat Y_denoised = Y.clone();
  denoise_luma_plane_inplace(Y_denoised, cfg.wavelet, valid, protect);
  denoise_luma_bilateral_inplace(Y_denoised, cfg.bilateral, valid, protect);

  cv::Mat amount_map(Y.size(), CV_32F, cv::Scalar(cfg.blend_amount));
  amount_map = amount_map.mul(1.0f - cfg.luma_guard_strength * protect);
  cv::min(amount_map, cfg.blend_amount, amount_map);
  cv::max(amount_map, 0.0, amount_map);
  stats.mean_denoise_fraction = cv::mean(amount_map, valid)[0];
  if (diagnostics) copy_to_matrix(amount_map, diagnostics->effective_amount);

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
  // every additive channel DIFFERENCE unchanged. This deliberately does not
  // claim invariant RGB ratios or perceptual saturation: a large common offset
  // changes both, so the blend and protection masks remain part of the
  // scientific contract.
  cv::Mat delta = Y_mix - Y;
  cv::Mat R_new = R + delta;
  cv::Mat G_new = G + delta;
  cv::Mat B_new = B + delta;
  R.copyTo(R_new, valid == 0);
  G.copyTo(G_new, valid == 0);
  B.copyTo(B_new, valid == 0);

  R_new.copyTo(R);
  G_new.copyTo(G);
  B_new.copyTo(B);
  return stats;
}

} // namespace tile_compile::reconstruction
