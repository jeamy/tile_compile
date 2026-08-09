#include "runner_phase_aqmh_reconstruction.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace tile_compile::runner {
namespace {

Matrix2Df gaussian_blur_aqmh(const Matrix2Df &image, float sigma_px) {
  if (image.rows() <= 0 || image.cols() <= 0 || !(sigma_px > 0.0f))
    return image;
  cv::Mat image_view(image.rows(), image.cols(), CV_32F,
                     const_cast<float *>(image.data()),
                     static_cast<size_t>(image.outerStride()) * sizeof(float));
  cv::Mat blurred_view;
  cv::GaussianBlur(image_view, blurred_view, cv::Size(0, 0),
                   static_cast<double>(sigma_px),
                   static_cast<double>(sigma_px), cv::BORDER_REFLECT101);
  Matrix2Df blurred(image.rows(), image.cols());
  for (int y = 0; y < blurred.rows(); ++y) {
    const float *source_row = blurred_view.ptr<float>(y);
    for (int x = 0; x < blurred.cols(); ++x)
      blurred(y, x) = source_row[x];
  }
  return blurred;
}

Matrix2Df low_frequency_neutralized_aqmh(const Matrix2Df &aqmh,
                                         const Matrix2Df &control,
                                         float sigma_px) {
  if (aqmh.rows() != control.rows() || aqmh.cols() != control.cols() ||
      aqmh.rows() <= 0 || aqmh.cols() <= 0 || !(sigma_px > 0.0f)) {
    return aqmh;
  }

  const Matrix2Df low_freq = gaussian_blur_aqmh(aqmh - control, sigma_px);
  Matrix2Df out = aqmh;
  for (int y = 0; y < out.rows(); ++y)
    for (int x = 0; x < out.cols(); ++x)
      out(y, x) -= low_freq(y, x);
  return out;
}

Matrix2Df unsharp_masked_aqmh(const Matrix2Df &base, float sigma_px,
                              float amount) {
  if (base.rows() <= 0 || base.cols() <= 0 || !(sigma_px > 0.0f) ||
      !(amount > 0.0f)) {
    return base;
  }
  const Matrix2Df low_freq = gaussian_blur_aqmh(base, sigma_px);
  Matrix2Df out = base;
  for (int y = 0; y < out.rows(); ++y)
    for (int x = 0; x < out.cols(); ++x)
      out(y, x) += amount * (base(y, x) - low_freq(y, x));
  return out;
}

float quantile_inplace(std::vector<float> &values, float q) {
  if (values.empty()) return 0.0f;
  const size_t idx = static_cast<size_t>(std::clamp(q, 0.0f, 1.0f) *
                                        static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

Matrix2Df structure_masked_aqmh_detail(const Matrix2Df &aqmh,
                                       const Matrix2Df &control,
                                       float low_q, float high_q,
                                       float mask_blur_sigma_px) {
  if (aqmh.rows() != control.rows() || aqmh.cols() != control.cols() ||
      aqmh.rows() <= 0 || aqmh.cols() <= 0) {
    return aqmh;
  }

  cv::Mat control_view(control.rows(), control.cols(), CV_32F,
                       const_cast<float *>(control.data()),
                       static_cast<size_t>(control.outerStride()) *
                           sizeof(float));
  cv::Mat gx, gy, grad;
  cv::Sobel(control_view, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT101);
  cv::Sobel(control_view, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT101);
  cv::magnitude(gx, gy, grad);

  std::vector<float> finite_grad;
  finite_grad.reserve(static_cast<size_t>(grad.rows) * grad.cols);
  for (int y = 0; y < grad.rows; ++y) {
    const float *row = grad.ptr<float>(y);
    for (int x = 0; x < grad.cols; ++x) {
      if (std::isfinite(row[x])) finite_grad.push_back(row[x]);
    }
  }
  std::vector<float> q_values = finite_grad;
  const float lo = quantile_inplace(q_values, low_q);
  q_values = std::move(finite_grad);
  const float hi = quantile_inplace(q_values, high_q);
  const float denom = std::max(hi - lo, std::numeric_limits<float>::epsilon());

  cv::Mat mask(grad.rows, grad.cols, CV_32F);
  for (int y = 0; y < grad.rows; ++y) {
    const float *grad_row = grad.ptr<float>(y);
    float *mask_row = mask.ptr<float>(y);
    for (int x = 0; x < grad.cols; ++x) {
      const float v = std::isfinite(grad_row[x]) ? grad_row[x] : 0.0f;
      mask_row[x] = std::clamp((v - lo) / denom, 0.0f, 1.0f);
    }
  }
  if (mask_blur_sigma_px > 0.0f) {
    cv::GaussianBlur(mask, mask, cv::Size(0, 0),
                     static_cast<double>(mask_blur_sigma_px),
                     static_cast<double>(mask_blur_sigma_px),
                     cv::BORDER_REFLECT101);
  }

  Matrix2Df out = control;
  for (int y = 0; y < out.rows(); ++y) {
    const float *mask_row = mask.ptr<float>(y);
    for (int x = 0; x < out.cols(); ++x) {
      out(y, x) += mask_row[x] * (aqmh(y, x) - control(y, x));
    }
  }
  return out;
}

struct RgbLumaDetailTransfer {
  Matrix2Df candidate_r;
  Matrix2Df candidate_g;
  Matrix2Df candidate_b;
  Matrix2Df raw_luma;
  Matrix2Df candidate_luma;
  bool numerical_ok = false;
  size_t transferred_pixels = 0;
  size_t denominator_floor_pixels = 0;
  size_t gain_clamped_pixels = 0;
  float denominator_epsilon = 0.0f;
  float gain_min = 0.25f;
  float gain_max = 4.0f;
};

RgbLumaDetailTransfer transfer_luma_detail_to_rgb(
    const Matrix2Df &raw_r, const Matrix2Df &raw_g,
    const Matrix2Df &raw_b, const Matrix2Df &raw_aqmh_luma,
    const Matrix2Df &selected_luma,
    const std::vector<uint8_t> &valid_mask) {
  RgbLumaDetailTransfer result;
  const int rows = static_cast<int>(raw_r.rows());
  const int cols = static_cast<int>(raw_r.cols());
  if (rows <= 0 || cols <= 0 || raw_g.rows() != rows ||
      raw_g.cols() != cols || raw_b.rows() != rows ||
      raw_b.cols() != cols || raw_aqmh_luma.rows() != rows ||
      raw_aqmh_luma.cols() != cols || selected_luma.rows() != rows ||
      selected_luma.cols() != cols) {
    return result;
  }

  const size_t pixel_count =
      static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const bool have_mask = valid_mask.size() == pixel_count;
  std::vector<float> finite_scale_samples;
  finite_scale_samples.reserve(std::min<size_t>(pixel_count, 1u << 20));
  const size_t sample_step = std::max<size_t>(1, pixel_count / (1u << 20));
  for (size_t i = 0; i < pixel_count; i += sample_step) {
    if (have_mask && valid_mask[i] == 0u) continue;
    const float value = raw_aqmh_luma.data()[i];
    if (std::isfinite(value) && std::fabs(value) > 0.0f)
      finite_scale_samples.push_back(std::fabs(value));
  }
  const float scale = quantile_inplace(finite_scale_samples, 0.5f);
  result.denominator_epsilon = std::max(
      64.0f * std::numeric_limits<float>::epsilon(), scale * 1.0e-6f);

  result.candidate_r = raw_r;
  result.candidate_g = raw_g;
  result.candidate_b = raw_b;
  result.raw_luma.resize(rows, cols);
  result.candidate_luma.resize(rows, cols);
  for (size_t i = 0; i < pixel_count; ++i) {
    const float r = raw_r.data()[i];
    const float g = raw_g.data()[i];
    const float b = raw_b.data()[i];
    const float rgb_luma = 0.25f * r + 0.50f * g + 0.25f * b;
    result.raw_luma.data()[i] = rgb_luma;
    result.candidate_luma.data()[i] = rgb_luma;
    if (have_mask && valid_mask[i] == 0u) continue;

    const float raw_luma = raw_aqmh_luma.data()[i];
    const float final_luma = selected_luma.data()[i];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b) ||
        !std::isfinite(rgb_luma) || !std::isfinite(raw_luma) ||
        !std::isfinite(final_luma)) {
      return result;
    }

    float gain = 1.0f;
    if (std::fabs(raw_luma) <= result.denominator_epsilon) {
      ++result.denominator_floor_pixels;
    } else {
      const float requested_gain = final_luma / raw_luma;
      if (!std::isfinite(requested_gain)) return result;
      gain = std::clamp(requested_gain, result.gain_min, result.gain_max);
      if (gain != requested_gain) ++result.gain_clamped_pixels;
    }

    const float candidate_r = r * gain;
    const float candidate_g = g * gain;
    const float candidate_b = b * gain;
    const float candidate_luma =
        0.25f * candidate_r + 0.50f * candidate_g + 0.25f * candidate_b;
    if (!std::isfinite(candidate_r) || !std::isfinite(candidate_g) ||
        !std::isfinite(candidate_b) || !std::isfinite(candidate_luma)) {
      return result;
    }
    result.candidate_r.data()[i] = candidate_r;
    result.candidate_g.data()[i] = candidate_g;
    result.candidate_b.data()[i] = candidate_b;
    result.candidate_luma.data()[i] = candidate_luma;
    if (std::fabs(gain - 1.0f) > 8.0f *
            std::numeric_limits<float>::epsilon()) {
      ++result.transferred_pixels;
    }
  }
  result.numerical_ok = true;
  return result;
}

struct RegistrationWeightGuardResult {
  VectorXf weights;
  bool applied = false;
  size_t frames = 0;
  size_t damped_frames = 0;
  float min_factor = 1.0f;
  float median_factor = 1.0f;
  float mean_factor = 1.0f;
  std::map<std::string, int> source_counts;
};

std::string metric_reason(bool applicable, const char *not_applicable_reason,
                          bool ok) {
  if (!applicable) return not_applicable_reason;
  return ok ? "within_threshold" : "regression_exceeds_threshold";
}

nlohmann::json gate_metric_json(bool applicable, bool ok, float value,
                                float control, float regression,
                                float threshold,
                                const char *not_applicable_reason) {
  return {{"status", applicable ? (ok ? "pass" : "fail") : "not_applicable"},
          {"reason", metric_reason(applicable, not_applicable_reason, ok)},
          {"value", value},
          {"control", control},
          {"regression", regression},
          {"threshold", threshold}};
}

nlohmann::json validation_metrics_json(
    const reconstruction::AqmhValidationMetrics &m) {
  return {{"seam_score", m.seam_score},
          {"fwhm", m.fwhm},
          {"background_rms", m.background_rms},
          {"star_count", m.star_count},
          {"tail11_abs_median", m.tail11_abs_median},
          {"tail11_p90", m.tail11_p90},
          {"elongation_median", m.elongation_median}};
}

nlohmann::json validation_comparison_json(
    const reconstruction::AqmhValidationComparison &v,
    const config::AqmhValidationConfig *cfg = nullptr) {
  nlohmann::json j = {{"aqmh", validation_metrics_json(v.aqmh)},
          {"control", validation_metrics_json(v.control)},
          {"seam_score_regression", v.seam_score_regression},
          {"fwhm_regression", v.fwhm_regression},
          {"background_rms_regression", v.background_rms_regression},
          {"tail11_abs_regression", v.tail11_abs_regression},
          {"elongation_regression", v.elongation_regression},
          {"background_rms_applicable", v.background_rms_applicable},
          {"seam_applicable", v.seam_applicable},
          {"fwhm_applicable", v.fwhm_applicable},
          {"tail_applicable", v.tail_applicable},
          {"elongation_applicable", v.elongation_applicable}};
  if (cfg) {
    const auto gates = reconstruction::evaluate_aqmh_validation_gates(v, *cfg);
    j["metrics"] = {
        {"background_rms",
         gate_metric_json(v.background_rms_applicable,
                          gates.background_ok,
                          v.aqmh.background_rms, v.control.background_rms,
                          v.background_rms_regression,
                          cfg->max_background_rms_regression,
                          "control_background_rms_degenerate")},
        {"fwhm",
         gate_metric_json(v.fwhm_applicable,
                          gates.fwhm_ok,
                          v.aqmh.fwhm, v.control.fwhm,
                          v.fwhm_regression,
                          cfg->max_fwhm_regression,
                          "fwhm_not_measurable")},
        {"seam_score",
         gate_metric_json(v.seam_applicable,
                          gates.seam_ok,
                          v.aqmh.seam_score, v.control.seam_score,
                          v.seam_score_regression,
                          cfg->max_seam_score_regression,
                          "control_seam_score_degenerate")},
        {"tail11_abs",
         gate_metric_json(v.tail_applicable,
                          gates.tail_ok,
                          v.aqmh.tail11_abs_median, v.control.tail11_abs_median,
                          v.tail11_abs_regression,
                          cfg->max_tail11_abs_regression,
                          "insufficient_comparable_star_samples")},
        {"elongation",
         gate_metric_json(v.elongation_applicable,
                          gates.elongation_ok,
                          v.aqmh.elongation_median, v.control.elongation_median,
                          v.elongation_regression,
                          cfg->max_elongation_regression,
                          "insufficient_comparable_star_samples")}};
  }
  return j;
}

RegistrationWeightGuardResult apply_registration_weight_guard(
    const std::filesystem::path &run_dir, const VectorXf &base_weights,
    const config::AqmhReconstructionConfig &cfg, std::ostream &log_file) {
  RegistrationWeightGuardResult result;
  result.weights = base_weights;
  if (!cfg.registration_weight_guard || base_weights.size() <= 0) {
    return result;
  }

  const auto artifact_path = run_dir / "artifacts" / "global_registration.json";
  if (!std::filesystem::exists(artifact_path)) {
    log_file << "[AQMH_RECONSTRUCTION] registration weight guard skipped: "
             << artifact_path << " missing" << std::endl;
    return result;
  }

  core::json reg;
  try {
    reg = core::json::parse(core::read_text(artifact_path));
  } catch (const std::exception &e) {
    log_file << "[AQMH_RECONSTRUCTION] registration weight guard skipped: "
             << e.what() << std::endl;
    return result;
  }
  if (!reg.contains("cc") || !reg["cc"].is_array()) {
    return result;
  }

  const auto &cc_arr = reg["cc"];
  const auto source_arr =
      reg.value("source", core::json::array());
  const auto depth_arr =
      reg.value("chain_depth", core::json::array());
  const Eigen::Index n = std::min<Eigen::Index>(
      base_weights.size(), static_cast<Eigen::Index>(cc_arr.size()));
  if (n <= 0) return result;

  std::vector<float> factors;
  factors.reserve(static_cast<size_t>(n));
  double sum_factor = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    float cc = 1.0f;
    if (!cc_arr[static_cast<size_t>(i)].is_null()) {
      cc = cc_arr[static_cast<size_t>(i)].get<float>();
    }
    if (!std::isfinite(cc)) cc = 0.0f;
    std::string source = "unknown";
    if (static_cast<size_t>(i) < source_arr.size() &&
        source_arr[static_cast<size_t>(i)].is_string()) {
      source = source_arr[static_cast<size_t>(i)].get<std::string>();
    }
    result.source_counts[source]++;

    const bool direct_source =
        source == "direct_global" || source == "reference";
    float factor = 1.0f;
    if (direct_source) {
      // Registration already accepted direct solutions geometrically. Do not
      // continuously suppress their signal for ordinary CC variation; retain
      // the hard floor only for genuinely low-confidence direct matches.
      factor = cc >= cfg.registration_cc_floor
                   ? 1.0f
                   : cfg.registration_weight_floor;
    } else {
      const float t = std::clamp(
          (cc - cfg.registration_cc_floor) /
              std::max(cfg.registration_cc_full - cfg.registration_cc_floor,
                       std::numeric_limits<float>::epsilon()),
          0.0f, 1.0f);
      factor = cfg.registration_weight_floor +
               (1.0f - cfg.registration_weight_floor) * t;
    }

    if (source == "sequential_refined") {
      factor *= cfg.registration_sequential_factor;
    } else if (source.find("predicted") != std::string::npos ||
               source.find("interpolated") != std::string::npos ||
               source == "unknown") {
      factor *= cfg.registration_predicted_factor;
    }

    if (!direct_source && cfg.registration_chain_depth_penalty > 0.0f &&
        static_cast<size_t>(i) < depth_arr.size() &&
        !depth_arr[static_cast<size_t>(i)].is_null()) {
      const int depth = depth_arr[static_cast<size_t>(i)].get<int>();
      const float depth_penalty = std::min(
          cfg.registration_chain_depth_max_penalty,
          std::max(0, depth - 1) * cfg.registration_chain_depth_penalty);
      factor *= (1.0f - depth_penalty);
    }
    factor = std::clamp(factor, cfg.registration_weight_floor, 1.0f);

    const float old_w = result.weights[i];
    result.weights[i] =
        std::isfinite(old_w) && old_w > 0.0f ? old_w * factor : 0.0f;
    if (factor < 0.999f && old_w > 0.0f) {
      result.damped_frames++;
    }
    factors.push_back(factor);
    sum_factor += factor;
    result.min_factor = std::min(result.min_factor, factor);
  }
  std::vector<float> sorted = factors;
  std::sort(sorted.begin(), sorted.end());
  result.frames = static_cast<size_t>(n);
  result.median_factor = sorted[sorted.size() / 2];
  result.mean_factor = static_cast<float>(sum_factor / static_cast<double>(n));
  result.applied = true;
  return result;
}

} // namespace

bool run_phase_aqmh_reconstruction(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const std::vector<std::filesystem::path> &frames,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &reconstruction_valid_mask,
    const std::vector<uint8_t> &validation_common_mask,
    int canvas_width, int canvas_height,
    bool osc_mode,
    const DiskCacheFrameStore &prewarped_frames,
    std::unique_ptr<metrics::QualityMapCache> &aqmh_cache,
    const VectorXf &aqmh_global_weights,
    core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    const std::chrono::steady_clock::time_point &phase_started_at,
    int prev_cv_threads,
    AqmhReconstructionPhaseResult &out,
    reconstruction::AqmhPrefetchCoordinator* prefetch_coordinator,
    const DiskCacheFrameStore* prewarped_frames_r,
    const DiskCacheFrameStore* prewarped_frames_g,
    const DiskCacheFrameStore* prewarped_frames_b) {

  const Phase reconstruction_phase = Phase::AQMH_RECONSTRUCTION;

  const auto aqmh_reconstruction_acceleration = acceleration.selection_for(
      core::AccelerationPhase::aqmh_reconstruction);
  log_file << "[AQMH_RECONSTRUCTION] "
           << core::acceleration_selection_summary(
                  aqmh_reconstruction_acceleration)
           << std::endl;
  if (!aqmh_cache) {
    const std::string err =
        "AQMH enabled but AQMH quality-map cache is unavailable";
    cv::setNumThreads(prev_cv_threads);
    emitter.phase_end(run_id, reconstruction_phase, "error",
                      {{"error", err}}, log_file);
    emitter.run_end(run_id, false, "error", log_file,
                    {{"message", std::string("Error during AQMH TILE_RECONSTRUCTION: ") + err}});
    std::cerr << "Error during AQMH TILE_RECONSTRUCTION: " << err
              << std::endl;
    return false;
  }

  reconstruction::AqmhReconstructionConfig aqmh_recon_cfg;
  aqmh_recon_cfg.clip_sigma = cfg.aqmh.reconstruction.clip_sigma;
  aqmh_recon_cfg.clip_sigma_low = cfg.aqmh.reconstruction.clip_sigma_low;
  aqmh_recon_cfg.clip_sigma_high = cfg.aqmh.reconstruction.clip_sigma_high;
  aqmh_recon_cfg.clip_iterations = cfg.aqmh.reconstruction.clip_iterations;
  aqmh_recon_cfg.min_fraction = cfg.aqmh.reconstruction.min_fraction;
  aqmh_recon_cfg.min_n_eff = cfg.aqmh.reconstruction.min_n_eff;
  aqmh_recon_cfg.cherry_pick = cfg.aqmh.cherry_pick.enabled;
  aqmh_recon_cfg.cherry_pick_mode = cfg.aqmh.cherry_pick.mode;
  aqmh_recon_cfg.cherry_pick_k_frac = cfg.aqmh.cherry_pick.k_frac;
  aqmh_recon_cfg.cherry_pick_k_min_required = cfg.aqmh.cherry_pick.k_min_required;
  aqmh_recon_cfg.cherry_pick_margin_min = cfg.aqmh.cherry_pick.margin_min;
  aqmh_recon_cfg.cherry_pick_reject_below_best_fraction =
      cfg.aqmh.cherry_pick.reject_below_best_fraction;
  aqmh_recon_cfg.cherry_pick_min_keep_fraction =
      cfg.aqmh.cherry_pick.min_keep_fraction;
  aqmh_recon_cfg.tiered_k_frac = cfg.aqmh.cherry_pick.tiered_k_frac;
  aqmh_recon_cfg.parallel_workers = std::max(1, cfg.runtime_limits.parallel_workers);
  // Bridge from user-facing config to internal config with fallback logic
  aqmh_recon_cfg.chunk_rows = cfg.aqmh.reconstruction.chunk_rows;
  aqmh_recon_cfg.memory_budget_mb = cfg.aqmh.reconstruction.memory_budget_mb != 0
      ? cfg.aqmh.reconstruction.memory_budget_mb
      : static_cast<size_t>(cfg.runtime_limits.memory_budget);
  aqmh_recon_cfg.compute_uniform_control = true;

  auto aqmh_frame_loader = [&](size_t fi, Matrix2Df &output) -> bool {
    if (fi >= frames.size() || fi >= frame_has_data.size() ||
        frame_has_data[fi] == 0u) {
      return false;
    }
    output = prewarped_frames.load(fi);
    return output.rows() == canvas_height && output.cols() == canvas_width;
  };
  metrics::FrameValidMaskStore aqmh_mask_store(
      run_dir / "cache" / "aqmh_masks", canvas_width, canvas_height);
  auto aqmh_mask_loader = [&](size_t fi, std::vector<uint8_t> &output) -> bool {
    output = aqmh_mask_store.read(fi);
    return output.size() == static_cast<size_t>(canvas_width * canvas_height);
  };
  auto aqmh_frame_region_loader =
      [&](size_t fi, int y0, int rows, Matrix2Df &output) -> bool {
    if (fi >= frames.size() || fi >= frame_has_data.size() ||
        frame_has_data[fi] == 0u) return false;
    return prewarped_frames.extract_tile_into(
        fi, Tile{0, y0, canvas_width, rows}, output);
  };
  auto aqmh_mask_region_loader =
      [&](size_t fi, int y0, int rows,
          std::vector<uint8_t> &output) -> bool {
    output = aqmh_mask_store.read_region(fi, y0, rows);
    return output.size() == static_cast<size_t>(canvas_width * rows);
  };

  int last_reconstruction_progress = 0;
  auto emit_reconstruction_progress =
      [&](int current, const std::string &substep, const std::string &pass) {
        const int monotonic_current = std::max(
            last_reconstruction_progress, std::clamp(current, 0, 99));
        last_reconstruction_progress = monotonic_current;
        emitter.phase_progress_counts(
            run_id, Phase::AQMH_RECONSTRUCTION,
            monotonic_current, 100, substep, pass, log_file);
      };

  // The pixel-wise reconstruction is only the first part of this phase.
  // Reserve the remaining range for RGB reconstruction, candidate validation,
  // quality gates, and artifact generation so the UI never shows 100% while
  // substantial AQMH work is still running.
  auto progress_callback = [&](int rows_done, int rows_total) {
    const int bounded_total = std::max(0, rows_total);
    const int bounded_done = std::clamp(rows_done, 0, bounded_total);
    const int current = bounded_total > 0
        ? 5 + static_cast<int>(50LL * bounded_done / bounded_total)
        : 5;
    emit_reconstruction_progress(
        current,
        "AQMH Kernrekonstruktion: Zeilen " + std::to_string(rows_done) +
            "/" + std::to_string(rows_total),
        "core_rows");
  };

  emit_reconstruction_progress(1, "AQMH Rekonstruktion vorbereiten", "setup");
  bool prefetch_fallback = false;
  if (prefetch_coordinator) {
    emit_reconstruction_progress(
        2, "AQMH Quality-Map-Prefetch abschließen", "qmap_prefetch");
    prefetch_coordinator->wait_all_prefetched();
    prefetch_fallback = !prefetch_coordinator->prefetch_active();
    if (prefetch_fallback) {
      log_file << "[AQMH_RECONSTRUCTION] Q-map prefetch failed; continuing with sequential cache reads" << std::endl;
    }
  }

  reconstruction::AqmhReconstructionResult aqmh_recon;
  const auto registration_guard = apply_registration_weight_guard(
      run_dir, aqmh_global_weights, cfg.aqmh.reconstruction, log_file);
  const VectorXf &effective_aqmh_global_weights =
      registration_guard.applied ? registration_guard.weights : aqmh_global_weights;
  if (registration_guard.applied) {
    log_file << "[AQMH_RECONSTRUCTION] registration weight guard applied: "
             << "frames=" << registration_guard.frames
             << " damped=" << registration_guard.damped_frames
             << " min_factor=" << registration_guard.min_factor
             << " median_factor=" << registration_guard.median_factor
             << " mean_factor=" << registration_guard.mean_factor << std::endl;
  }
  core::AccelerationOps aqmh_reconstruction_ops(
      acceleration, core::AccelerationPhase::aqmh_reconstruction);
  emit_reconstruction_progress(
      5, "AQMH pixelweise Kernrekonstruktion starten", "core_rows");
  std::cout << "[AQMH] Running independent pixel-wise reconstruction for "
            << frames.size() << " frame slots cpu_workers="
            << aqmh_recon_cfg.parallel_workers
            << " backend="
            << core::acceleration_backend_name(
                   aqmh_reconstruction_acceleration.selected)
            << " region_streaming=yes"
            << std::endl;
  const auto reconstruction_core_started_at =
      std::chrono::steady_clock::now();
  aqmh_recon = aqmh_reconstruction_ops.reconstruct_aqmh(
      frames.size(), aqmh_frame_loader, aqmh_cache.get(),
      effective_aqmh_global_weights,
      reconstruction_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
      nullptr, aqmh_mask_loader, aqmh_frame_region_loader,
      aqmh_mask_region_loader, progress_callback);
  const double reconstruction_core_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    reconstruction_core_started_at)
          .count();
  emit_reconstruction_progress(
      55, "AQMH Kernrekonstruktion abgeschlossen", "core_complete");
  double rgb_reconstruction_seconds = 0.0;
  const bool debayer_first_rgb =
      prewarped_frames_r != nullptr && prewarped_frames_g != nullptr &&
      prewarped_frames_b != nullptr;
  if (debayer_first_rgb) {
    auto reconstruct_rgb_plane =
        [&](const DiskCacheFrameStore &plane_store, Matrix2Df &plane_out,
            const char *channel_name, int channel_index) -> bool {
      auto frame_loader = [&](size_t fi, Matrix2Df &output) -> bool {
        if (fi >= frames.size() || fi >= frame_has_data.size() ||
            frame_has_data[fi] == 0u) {
          return false;
        }
        output = plane_store.load(fi);
        return output.rows() == canvas_height && output.cols() == canvas_width;
      };
      auto frame_region_loader =
          [&](size_t fi, int y0, int rows, Matrix2Df &output) -> bool {
        if (fi >= frames.size() || fi >= frame_has_data.size() ||
            frame_has_data[fi] == 0u) {
          return false;
        }
        return plane_store.extract_tile_into(
            fi, Tile{0, y0, canvas_width, rows}, output);
      };
      reconstruction::AqmhReconstructionConfig plane_cfg = aqmh_recon_cfg;
      plane_cfg.compute_uniform_control = false;
      const int channel_base = 55 + 5 * channel_index;
      auto plane_progress = [&](int rows_done, int rows_total) {
        const int bounded_total = std::max(0, rows_total);
        const int bounded_done = std::clamp(rows_done, 0, bounded_total);
        const int current = bounded_total > 0
            ? channel_base +
                  static_cast<int>(5LL * bounded_done / bounded_total)
            : channel_base;
        emit_reconstruction_progress(
            current,
            "AQMH RGB-Kanal " + std::string(channel_name) +
                " rekonstruieren: Zeilen " + std::to_string(rows_done) +
                "/" + std::to_string(rows_total),
            "rgb_" + std::string(channel_name));
      };
      emit_reconstruction_progress(
          channel_base,
          "AQMH RGB-Kanal " + std::string(channel_name) +
              " rekonstruieren",
          "rgb_" + std::string(channel_name));
      std::cout << "[AQMH] Reconstructing debayer-first RGB channel "
                << channel_name << std::endl;
      auto plane_recon = aqmh_reconstruction_ops.reconstruct_aqmh(
          frames.size(), frame_loader, aqmh_cache.get(),
          effective_aqmh_global_weights, reconstruction_valid_mask,
          canvas_width, canvas_height, plane_cfg, nullptr, aqmh_mask_loader,
          frame_region_loader, aqmh_mask_region_loader, plane_progress);
      plane_out = std::move(plane_recon.output);
      return plane_out.rows() == canvas_height &&
             plane_out.cols() == canvas_width;
    };
    const auto rgb_started_at = std::chrono::steady_clock::now();
    const bool rgb_ok =
        reconstruct_rgb_plane(*prewarped_frames_r, out.output_R, "R", 0) &&
        reconstruct_rgb_plane(*prewarped_frames_g, out.output_G, "G", 1) &&
        reconstruct_rgb_plane(*prewarped_frames_b, out.output_B, "B", 2);
    rgb_reconstruction_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      rgb_started_at)
            .count();
    if (!rgb_ok) {
      out.output_R.resize(0, 0);
      out.output_G.resize(0, 0);
      out.output_B.resize(0, 0);
      emitter.warning(
          run_id,
          "AQMH debayer-first RGB reconstruction failed; falling back to "
          "post-stack debayer",
          log_file);
    }
  }
  emit_reconstruction_progress(
      70,
      debayer_first_rgb ? "AQMH RGB-Rekonstruktion abgeschlossen"
                        : "AQMH RGB-Rekonstruktion nicht erforderlich",
      "rgb_complete");
  const size_t expected_control_pixels =
      static_cast<size_t>(canvas_width) * static_cast<size_t>(canvas_height);
  const bool backend_supplied_uniform_control =
      aqmh_recon.uniform_control_output.rows() == canvas_height &&
      aqmh_recon.uniform_control_output.cols() == canvas_width &&
      aqmh_recon.uniform_control_valid_mask.size() == expected_control_pixels;
  double uniform_control_fallback_seconds = 0.0;
  if (!backend_supplied_uniform_control) {
    emit_reconstruction_progress(
        71, "AQMH Uniform-Control-Referenz berechnen", "uniform_control");
    const auto uniform_control_started_at = std::chrono::steady_clock::now();
    auto uniform_control = reconstruction::compute_aqmh_uniform_control(
        frames.size(), aqmh_frame_loader, reconstruction_valid_mask,
        canvas_width, canvas_height, aqmh_mask_loader,
        aqmh_frame_region_loader, aqmh_mask_region_loader);
    aqmh_recon.uniform_control_output = std::move(uniform_control.output);
    aqmh_recon.uniform_control_valid_mask = std::move(uniform_control.valid_mask);
    uniform_control_fallback_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uniform_control_started_at)
            .count();
  }
  emit_reconstruction_progress(
      74,
      backend_supplied_uniform_control
          ? "AQMH Uniform-Control aus Kernrekonstruktion übernehmen"
          : "AQMH Uniform-Control-Referenz abgeschlossen",
      "uniform_control_complete");
  const bool acceleration_used = aqmh_recon.acceleration_used;
  const bool acceleration_fallback = aqmh_recon.acceleration_fallback;
  const std::string execution_backend_str =
      acceleration_used ? "cuda_native_v0_2" : "cpu_exact_v0_2";

  bool low_frequency_neutralization_applied = false;
  bool low_frequency_neutralization_evaluated = false;
  bool structure_masked_detail_applied = false;
  bool star_core_sharpening_applied = false;
  float structure_masked_detail_alpha = 0.0f;
  bool raw_aqmh_preserved_by_guard = false;
  std::string raw_baseline_guard_reason = "not_evaluated";
  std::vector<uint8_t> uniform_control_validation_mask;
  if (aqmh_recon.uniform_control_valid_mask.size() ==
      static_cast<size_t>(canvas_width * canvas_height)) {
    uniform_control_validation_mask.assign(
        static_cast<size_t>(canvas_width * canvas_height), 0u);
    for (size_t i = 0; i < uniform_control_validation_mask.size(); ++i) {
      const bool common_valid = validation_common_mask.empty() ||
          (validation_common_mask.size() == uniform_control_validation_mask.size() &&
           validation_common_mask[i] != 0u);
      uniform_control_validation_mask[i] =
          common_valid && aqmh_recon.uniform_control_valid_mask[i] != 0u ? 1u : 0u;
    }
  } else {
    uniform_control_validation_mask = validation_common_mask;
  }
  reconstruction::AqmhValidationComparison low_frequency_neutralization_validation;
  reconstruction::AqmhValidationComparison structure_masked_detail_validation;
  reconstruction::AqmhValidationComparison star_core_sharpening_validation;
  bool star_core_sharpening_evaluated = false;
  reconstruction::AqmhValidationComparison
      full_structure_masked_detail_vs_raw_validation;
  bool full_structure_masked_detail_vs_raw_evaluated = false;
  bool full_structure_masked_detail_preserves_raw = false;
  int structure_attenuation_evaluations = 0;
  int structure_attenuation_feasible_candidates = 0;
  float structure_attenuation_best_alpha = 0.0f;
  std::string structure_attenuation_strategy = "not_needed";
  emit_reconstruction_progress(
      76, "AQMH Validierungsreferenzen und Rohbaseline berechnen",
      "validation_references");
  const auto validation_started_at = std::chrono::steady_clock::now();
  const Matrix2Df raw_aqmh_output = aqmh_recon.output;
  const Matrix2Df raw_aqmh_weight_sum = aqmh_recon.weight_sum;
  const auto uniform_control_reference =
      reconstruction::prepare_aqmh_validation_reference(
          aqmh_recon.uniform_control_output,
          uniform_control_validation_mask);
  const auto raw_aqmh_reference =
      reconstruction::prepare_aqmh_validation_reference(
          raw_aqmh_output, validation_common_mask);
  const auto raw_control_validation =
      reconstruction::compare_aqmh_to_reference(
          raw_aqmh_output, uniform_control_reference,
          uniform_control_validation_mask);
  if (aqmh_recon.uniform_control_output.rows() == aqmh_recon.output.rows() &&
      aqmh_recon.uniform_control_output.cols() == aqmh_recon.output.cols()) {
    emit_reconstruction_progress(
        79, "AQMH Niederfrequenz-Neutralisierung berechnen und bewerten",
        "low_frequency_validation");
    constexpr float neutralization_sigma_px = 96.0f;
    Matrix2Df neutralized = low_frequency_neutralized_aqmh(
        aqmh_recon.output, aqmh_recon.uniform_control_output,
        neutralization_sigma_px);
    low_frequency_neutralization_validation =
        reconstruction::compare_aqmh_to_reference(
            neutralized, uniform_control_reference,
            uniform_control_validation_mask);
    low_frequency_neutralization_evaluated = true;
    const auto &raw_validation = raw_control_validation;
    const auto neutralized_vs_raw =
        reconstruction::compare_aqmh_to_reference(
            neutralized, raw_aqmh_reference, validation_common_mask);
    const bool neutralized_background_improved =
        raw_validation.background_rms_applicable &&
        low_frequency_neutralization_validation.background_rms_applicable &&
        low_frequency_neutralization_validation.background_rms_regression <
        raw_validation.background_rms_regression;
    const auto neutralized_gates =
        reconstruction::evaluate_aqmh_validation_gates(
            low_frequency_neutralization_validation, cfg.aqmh.validation);
    const bool neutralized_background_ok = neutralized_gates.background_ok;
    // Neutralisation guard: only apply if AQMH has *worse* background than
    // control. If raw AQMH already improves background (regression <= 0),
    // neutralisation would destroy that improvement.
    const bool aqmh_background_worse_than_control =
        raw_validation.background_rms_applicable &&
        raw_validation.background_rms_regression > 0.0f;
    const bool neutralized_selected =
        neutralized_background_improved && aqmh_background_worse_than_control &&
        neutralized_gates.all_ok &&
        reconstruction::evaluate_aqmh_validation_gates(
            neutralized_vs_raw, cfg.aqmh.validation).all_ok;
    low_frequency_neutralization_applied = neutralized_selected;
    const Matrix2Df &neutralization_base =
        neutralized_selected ? neutralized : raw_aqmh_output;
    log_file << "[AQMH_RECONSTRUCTION] adaptive low-frequency neutralization: "
             << "raw_background_regression="
             << raw_validation.background_rms_regression
             << " neutralized_background_regression="
             << low_frequency_neutralization_validation.background_rms_regression
             << " aqmh_background_worse_than_control="
             << (aqmh_background_worse_than_control ? "true" : "false")
             << " selected="
             << (neutralized_selected ? "neutralized" : "raw")
             << " neutralized_background_ok="
             << (neutralized_background_ok ? "true" : "false") << std::endl;

    {
      const float structure_low_q = cfg.aqmh.reconstruction.structure_mask_low_q;
      const float structure_high_q = cfg.aqmh.reconstruction.structure_mask_high_q;
      const float structure_mask_blur_sigma_px = cfg.aqmh.reconstruction.structure_mask_blur_sigma_px;
      emit_reconstruction_progress(
          82, "AQMH Strukturmaske und Detailkandidat berechnen",
          "structure_mask_validation");
      Matrix2Df structure_masked = structure_masked_aqmh_detail(
          neutralization_base, aqmh_recon.uniform_control_output,
          structure_low_q, structure_high_q, structure_mask_blur_sigma_px);
      structure_masked_detail_validation =
          reconstruction::compare_aqmh_to_reference(
              structure_masked, uniform_control_reference,
              uniform_control_validation_mask);
      full_structure_masked_detail_vs_raw_validation =
          reconstruction::compare_aqmh_to_reference(
              structure_masked, raw_aqmh_reference, validation_common_mask);
      full_structure_masked_detail_vs_raw_evaluated = true;
      const auto candidate_gates =
          reconstruction::evaluate_aqmh_validation_gates(
              structure_masked_detail_validation, cfg.aqmh.validation);
      const bool candidate_background_ok = candidate_gates.background_ok;
      const bool candidate_fwhm_ok = candidate_gates.fwhm_ok;
      const bool candidate_seam_ok = candidate_gates.seam_ok;
      const bool candidate_tail_ok = candidate_gates.tail_ok;
      const auto raw_gates =
          reconstruction::evaluate_aqmh_validation_gates(
              raw_validation, cfg.aqmh.validation);
      const auto candidate_raw_guard =
          reconstruction::aqmh_raw_baseline_guard_decision(
          full_structure_masked_detail_vs_raw_validation,
          raw_validation,
          structure_masked_detail_validation,
          cfg.aqmh.validation);
      full_structure_masked_detail_preserves_raw = candidate_raw_guard.ok;
      const bool improves_fwhm =
          structure_masked_detail_validation.fwhm_applicable &&
          structure_masked_detail_validation.aqmh.fwhm > 0.0f &&
          raw_validation.control.fwhm > 0.0f &&
          structure_masked_detail_validation.aqmh.fwhm <
              raw_validation.control.fwhm;
      const bool improves_seam =
          structure_masked_detail_validation.seam_applicable &&
          structure_masked_detail_validation.aqmh.seam_score <
          raw_validation.control.seam_score;
      const bool repairs_raw_background_failure =
          !raw_gates.background_ok &&
          raw_validation.background_rms_applicable &&
          structure_masked_detail_validation.background_rms_applicable &&
          structure_masked_detail_validation.background_rms_regression <
              raw_validation.background_rms_regression;
      const bool repairs_raw_fwhm_failure =
          !raw_gates.fwhm_ok && raw_validation.fwhm_applicable &&
          structure_masked_detail_validation.fwhm_applicable &&
          structure_masked_detail_validation.fwhm_regression <
              raw_validation.fwhm_regression;
      const bool repairs_raw_seam_failure =
          !raw_gates.seam_ok && raw_validation.seam_applicable &&
          structure_masked_detail_validation.seam_applicable &&
          structure_masked_detail_validation.seam_score_regression <
              raw_validation.seam_score_regression;
      const bool repairs_raw_tail_failure =
          !raw_gates.tail_ok &&
          ((raw_validation.tail_applicable &&
            structure_masked_detail_validation.tail_applicable &&
            structure_masked_detail_validation.tail11_abs_regression <
                raw_validation.tail11_abs_regression) ||
           (raw_validation.elongation_applicable &&
            structure_masked_detail_validation.elongation_applicable &&
            structure_masked_detail_validation.elongation_regression <
                raw_validation.elongation_regression));
      const bool repairs_any_raw_gate_failure =
          repairs_raw_background_failure || repairs_raw_fwhm_failure ||
          repairs_raw_seam_failure || repairs_raw_tail_failure;
      if (candidate_background_ok && candidate_fwhm_ok && candidate_seam_ok &&
          candidate_tail_ok && candidate_raw_guard.ok &&
          (improves_fwhm || improves_seam || repairs_any_raw_gate_failure)) {
        aqmh_recon.output = std::move(structure_masked);
        structure_masked_detail_applied = true;
        structure_masked_detail_alpha = 1.0f;
        raw_baseline_guard_reason = candidate_raw_guard.reason;
        emitter.warning(
            run_id,
            "AQMH structure-masked detail applied: background_rms=" +
                std::to_string(structure_masked_detail_validation.aqmh.background_rms) +
                " control_background_rms=" +
                std::to_string(structure_masked_detail_validation.control.background_rms) +
                " fwhm=" +
                std::to_string(structure_masked_detail_validation.aqmh.fwhm) +
                " control_fwhm=" +
                std::to_string(structure_masked_detail_validation.control.fwhm) +
                " seam_score=" +
                std::to_string(structure_masked_detail_validation.aqmh.seam_score) +
                " control_seam_score=" +
                std::to_string(structure_masked_detail_validation.control.seam_score),
            log_file);
      } else if (repairs_any_raw_gate_failure || improves_fwhm ||
                 improves_seam) {
        // Passing both immutable references is not monotonic in alpha: the
        // uniform-control endpoint and the full-detail endpoint can fail
        // different gates while an interior candidate passes. Probe the
        // interval explicitly, then refine only the highest feasible region.
        structure_attenuation_strategy =
            "descending_eighths_then_four_step_refinement";
        reconstruction::AqmhValidationComparison best_validation;
        reconstruction::AqmhValidationComparison best_vs_raw;
        float best_alpha = 0.0f;
        constexpr int coarse_denominator = 8;
        for (int numerator = coarse_denominator - 1; numerator >= 1;
             --numerator) {
          const float alpha =
              static_cast<float>(numerator) /
              static_cast<float>(coarse_denominator);
          emit_reconstruction_progress(
              84,
              "AQMH gedämpften Strukturkandidaten prüfen: alpha=" +
                  std::to_string(alpha),
              "structure_attenuation");
          Matrix2Df attenuated =
              aqmh_recon.uniform_control_output +
              alpha * (structure_masked - aqmh_recon.uniform_control_output);
          const auto validation =
              reconstruction::compare_aqmh_to_reference(
                  attenuated, uniform_control_reference,
                  uniform_control_validation_mask);
          const auto baseline_validation =
              reconstruction::compare_aqmh_to_reference(
                  attenuated, raw_aqmh_reference, validation_common_mask);
          const auto baseline_guard =
              reconstruction::aqmh_raw_baseline_guard_decision(
              baseline_validation, raw_validation, validation,
              cfg.aqmh.validation);
          const bool ok =
              reconstruction::evaluate_aqmh_validation_gates(
              validation, cfg.aqmh.validation).all_ok &&
              baseline_guard.ok;
          ++structure_attenuation_evaluations;
          if (ok) {
            ++structure_attenuation_feasible_candidates;
            best_alpha = alpha;
            best_validation = validation;
            best_vs_raw = baseline_validation;
            break;
          }
        }

        if (best_alpha > 0.0f) {
          float lo = best_alpha;
          float hi = std::min(
              1.0f, best_alpha + 1.0f / coarse_denominator);
          for (int iter = 0; iter < 4; ++iter) {
            const float alpha = 0.5f * (lo + hi);
            emit_reconstruction_progress(
                85,
                "AQMH Strukturdämpfung verfeinern: Schritt " +
                    std::to_string(iter + 1) + "/4, alpha=" +
                    std::to_string(alpha),
                "structure_attenuation_refinement");
            Matrix2Df attenuated =
                aqmh_recon.uniform_control_output +
                alpha * (structure_masked -
                         aqmh_recon.uniform_control_output);
            const auto validation =
                reconstruction::compare_aqmh_to_reference(
                    attenuated, uniform_control_reference,
                    uniform_control_validation_mask);
            const auto baseline_validation =
                reconstruction::compare_aqmh_to_reference(
                    attenuated, raw_aqmh_reference,
                    validation_common_mask);
            const auto baseline_guard =
                reconstruction::aqmh_raw_baseline_guard_decision(
                baseline_validation, raw_validation, validation,
                cfg.aqmh.validation);
            const bool ok =
                reconstruction::evaluate_aqmh_validation_gates(
              validation, cfg.aqmh.validation).all_ok &&
                baseline_guard.ok;
            ++structure_attenuation_evaluations;
            if (ok) {
              ++structure_attenuation_feasible_candidates;
              lo = alpha;
              best_alpha = alpha;
              best_validation = validation;
              best_vs_raw = baseline_validation;
            } else {
              hi = alpha;
            }
          }
          structure_attenuation_best_alpha = best_alpha;

          Matrix2Df attenuated =
              aqmh_recon.uniform_control_output +
              best_alpha *
                  (structure_masked - aqmh_recon.uniform_control_output);
          const bool attenuated_improves_fwhm =
              best_validation.fwhm_applicable &&
              best_validation.aqmh.fwhm > 0.0f &&
              raw_validation.control.fwhm > 0.0f &&
              best_validation.aqmh.fwhm < raw_validation.control.fwhm;
          const bool attenuated_improves_seam =
              best_validation.seam_applicable &&
              best_validation.aqmh.seam_score <
              raw_validation.control.seam_score;
          const bool attenuated_repairs_raw_background_failure =
              !raw_gates.background_ok &&
              raw_validation.background_rms_applicable &&
              best_validation.background_rms_applicable &&
              best_validation.background_rms_regression <
                  raw_validation.background_rms_regression;
          const bool attenuated_repairs_raw_fwhm_failure =
              !raw_gates.fwhm_ok && raw_validation.fwhm_applicable &&
              best_validation.fwhm_applicable &&
              best_validation.fwhm_regression <
                  raw_validation.fwhm_regression;
          const bool attenuated_repairs_raw_seam_failure =
              !raw_gates.seam_ok && raw_validation.seam_applicable &&
              best_validation.seam_applicable &&
              best_validation.seam_score_regression <
                  raw_validation.seam_score_regression;
          const bool attenuated_repairs_raw_tail_failure =
              !raw_gates.tail_ok &&
              ((raw_validation.tail_applicable &&
                best_validation.tail_applicable &&
                best_validation.tail11_abs_regression <
                    raw_validation.tail11_abs_regression) ||
               (raw_validation.elongation_applicable &&
                best_validation.elongation_applicable &&
                best_validation.elongation_regression <
                    raw_validation.elongation_regression));
          const bool attenuated_repairs_any_raw_gate_failure =
              attenuated_repairs_raw_background_failure ||
              attenuated_repairs_raw_fwhm_failure ||
              attenuated_repairs_raw_seam_failure ||
              attenuated_repairs_raw_tail_failure;
          const auto selected_baseline_guard =
              reconstruction::aqmh_raw_baseline_guard_decision(
                  best_vs_raw, raw_validation, best_validation,
                  cfg.aqmh.validation);
          if (reconstruction::evaluate_aqmh_validation_gates(
                  best_validation, cfg.aqmh.validation).all_ok &&
              selected_baseline_guard.ok &&
              (attenuated_improves_fwhm || attenuated_improves_seam ||
               attenuated_repairs_any_raw_gate_failure)) {
            aqmh_recon.output = std::move(attenuated);
            structure_masked_detail_validation = best_validation;
            structure_masked_detail_applied = true;
            structure_masked_detail_alpha = best_alpha;
            raw_baseline_guard_reason = selected_baseline_guard.reason;
            emitter.warning(
                run_id,
                "AQMH attenuated structure-masked detail applied: alpha=" +
                    std::to_string(best_alpha) +
                    " background_rms=" +
                    std::to_string(best_validation.aqmh.background_rms) +
                    " control_background_rms=" +
                    std::to_string(best_validation.control.background_rms) +
                    " fwhm=" +
                    std::to_string(best_validation.aqmh.fwhm) +
                    " control_fwhm=" +
                    std::to_string(best_validation.control.fwhm) +
                    " seam_score=" +
                    std::to_string(best_validation.aqmh.seam_score) +
                    " control_seam_score=" +
                    std::to_string(best_validation.control.seam_score),
                log_file);
          }
        }
      }
    }

    // Star-core sharpening candidate: unsharp mask at the core scale of the
    // stacked CFA mosaic. The Gaussian sigma (2 px) suppresses the 2 px CFA
    // grid so the detail band targets star cores (~6-7 px FWHM), not the
    // Bayer pattern. Same validation contract as the other post-processing
    // candidates: must pass every gate against the uniform control, must
    // preserve the immutable raw AQMH baseline, and must measurably improve
    // FWHM against the control. Otherwise raw AQMH is preserved.
    {
      emit_reconstruction_progress(
          87, "AQMH Sternkern-Schärfung berechnen und bewerten",
          "star_core_validation");
      constexpr float sharpen_sigma_px = 2.0f;
      constexpr float sharpen_amount = 0.6f;
      Matrix2Df sharpened =
          unsharp_masked_aqmh(aqmh_recon.output, sharpen_sigma_px,
                              sharpen_amount);
      star_core_sharpening_validation =
          reconstruction::compare_aqmh_to_reference(
              sharpened, uniform_control_reference,
              uniform_control_validation_mask);
      star_core_sharpening_evaluated = true;
      const auto &sharpened_validation = star_core_sharpening_validation;
      const auto sharpened_vs_raw =
          reconstruction::compare_aqmh_to_reference(
              sharpened, raw_aqmh_reference, validation_common_mask);
      const auto sharpened_raw_guard =
          reconstruction::aqmh_raw_baseline_guard_decision(
              sharpened_vs_raw, raw_control_validation, sharpened_validation,
              cfg.aqmh.validation);
      const bool sharpen_improves_fwhm =
          sharpened_validation.fwhm_applicable &&
          sharpened_validation.aqmh.fwhm > 0.0f &&
          raw_control_validation.control.fwhm > 0.0f &&
          sharpened_validation.aqmh.fwhm <
              raw_control_validation.control.fwhm;
      if (reconstruction::evaluate_aqmh_validation_gates(
              sharpened_validation, cfg.aqmh.validation).all_ok &&
          sharpened_raw_guard.ok && sharpen_improves_fwhm) {
        aqmh_recon.output = std::move(sharpened);
        star_core_sharpening_applied = true;
        raw_baseline_guard_reason = sharpened_raw_guard.reason;
        emitter.warning(
            run_id,
            "AQMH star-core sharpening applied: sigma_px=2.0 amount=0.6"
            " fwhm=" +
                std::to_string(sharpened_validation.aqmh.fwhm) +
                " control_fwhm=" +
                std::to_string(sharpened_validation.control.fwhm) +
                " background_rms=" +
                std::to_string(sharpened_validation.aqmh.background_rms) +
                " control_background_rms=" +
                std::to_string(sharpened_validation.control.background_rms),
            log_file);
      } else {
        log_file << "[AQMH_RECONSTRUCTION] star-core sharpening candidate "
                    "rejected: gate_ok="
                 << (reconstruction::evaluate_aqmh_validation_gates(
                         sharpened_validation, cfg.aqmh.validation).all_ok
                         ? "true"
                         : "false")
                 << " raw_guard=" << sharpened_raw_guard.reason
                 << " improves_fwhm="
                 << (sharpen_improves_fwhm ? "true" : "false")
                 << " fwhm=" << sharpened_validation.aqmh.fwhm
                 << " control_fwhm=" << sharpened_validation.control.fwhm
                 << " fwhm_regression="
                 << sharpened_validation.fwhm_regression
                 << " background_rms_regression="
                 << sharpened_validation.background_rms_regression
                 << " seam_regression="
                 << sharpened_validation.seam_score_regression << std::endl;
      }
    }
  }

  emit_reconstruction_progress(
      90, "AQMH Qualitäts-Gates gegen Uniform-Control prüfen",
      "control_quality_gates");
  out.control_validation =
      reconstruction::compare_aqmh_to_reference(
          aqmh_recon.output, uniform_control_reference,
          uniform_control_validation_mask);
  const auto pre_fallback_control_validation = out.control_validation;
  // A sharpness improvement is diagnostic, not permission to relax unrelated
  // quality limits. Candidate selection must remain within every applicable
  // configured threshold.
  const bool fwhm_clearly_improves =
      pre_fallback_control_validation.fwhm_applicable &&
      pre_fallback_control_validation.fwhm_regression < -0.03f;
  const float effective_background_threshold =
      cfg.aqmh.validation.max_background_rms_regression;
  const float effective_seam_threshold =
      cfg.aqmh.validation.max_seam_score_regression;
  const float effective_tail_threshold =
      cfg.aqmh.validation.max_tail11_abs_regression;
  const float effective_elongation_threshold =
      cfg.aqmh.validation.max_elongation_regression;
  const auto pre_fallback_gates =
      reconstruction::evaluate_aqmh_validation_gates(
          pre_fallback_control_validation, cfg.aqmh.validation);
  const bool aqmh_background_ok = pre_fallback_gates.background_ok;
  const bool aqmh_fwhm_ok = pre_fallback_gates.fwhm_ok;
  const bool aqmh_seam_ok = pre_fallback_gates.seam_ok;
  const bool aqmh_tail_gate_ok = pre_fallback_gates.tail_ok;
  const bool aqmh_control_fallback =
      aqmh_recon.uniform_control_output.rows() == aqmh_recon.output.rows() &&
      aqmh_recon.uniform_control_output.cols() == aqmh_recon.output.cols() &&
      (!aqmh_background_ok || !aqmh_fwhm_ok || !aqmh_seam_ok ||
       !aqmh_tail_gate_ok);
  if (aqmh_control_fallback) {
    // Uniform control is a validation reference, not a reconstruction
    // candidate. Blending toward it can erase diffuse signal while appearing
    // to improve one-sided noise metrics. Preserve the immutable raw AQMH
    // baseline whenever no AQMH-derived post-processing candidate passes.
    aqmh_recon.output = raw_aqmh_output;
    aqmh_recon.weight_sum = raw_aqmh_weight_sum;
    emitter.warning(
        run_id,
        "AQMH postprocessing rejected by quality gate: background_rms=" +
            std::to_string(pre_fallback_control_validation.aqmh.background_rms) +
            " control_background_rms=" +
            std::to_string(pre_fallback_control_validation.control.background_rms) +
            " regression=" +
            std::to_string(pre_fallback_control_validation.background_rms_regression) +
            " max=" +
            std::to_string(cfg.aqmh.validation.max_background_rms_regression) +
            "; preserving raw AQMH because uniform control would regress the "
            "validated AQMH baseline",
        log_file);
    aqmh_recon.output = raw_aqmh_output;
    aqmh_recon.weight_sum = raw_aqmh_weight_sum;
    raw_aqmh_preserved_by_guard = true;
    out.control_validation = raw_control_validation;
  }

  emit_reconstruction_progress(
      91, "AQMH finalen Kandidaten gegen Rohbaseline prüfen",
      "raw_baseline_guard");
  auto final_vs_raw_validation =
      reconstruction::compare_aqmh_to_reference(
          aqmh_recon.output, raw_aqmh_reference, validation_common_mask);
  const auto final_raw_guard = reconstruction::aqmh_raw_baseline_guard_decision(
      final_vs_raw_validation, raw_control_validation, out.control_validation,
      cfg.aqmh.validation);
  if (!final_raw_guard.ok) {
    emitter.warning(
        run_id,
        "AQMH final candidate rejected because it regresses the raw AQMH "
        "baseline; preserving raw AQMH",
        log_file);
    aqmh_recon.output = raw_aqmh_output;
    aqmh_recon.weight_sum = raw_aqmh_weight_sum;
    raw_aqmh_preserved_by_guard = true;
    out.control_validation = raw_control_validation;
    raw_baseline_guard_reason = final_raw_guard.reason;
    final_vs_raw_validation =
        reconstruction::compare_aqmh_to_reference(
            aqmh_recon.output, raw_aqmh_reference, validation_common_mask);
  } else {
    raw_baseline_guard_reason = final_raw_guard.reason;
  }

  emit_reconstruction_progress(
      93, "AQMH RGB-Luminanzdetail übertragen und validieren",
      "rgb_luma_validation");
  bool rgb_detail_transfer_applicable = false;
  bool rgb_detail_transfer_numerical_ok = false;
  bool rgb_detail_transfer_applied = false;
  std::string rgb_detail_transfer_reason =
      debayer_first_rgb ? "rgb_reconstruction_unavailable" : "not_applicable";
  size_t rgb_detail_transfer_pixels = 0;
  size_t rgb_detail_transfer_denominator_floor_pixels = 0;
  size_t rgb_detail_transfer_gain_clamped_pixels = 0;
  float rgb_detail_transfer_denominator_epsilon = 0.0f;
  float rgb_detail_transfer_gain_min = 0.25f;
  float rgb_detail_transfer_gain_max = 4.0f;
  reconstruction::AqmhValidationComparison rgb_raw_vs_control_validation;
  reconstruction::AqmhValidationComparison rgb_candidate_vs_control_validation;
  reconstruction::AqmhValidationComparison rgb_candidate_vs_raw_validation;
  std::string rgb_detail_transfer_raw_guard_reason = "not_evaluated";
  const bool reconstructed_rgb_available =
      out.output_R.rows() == canvas_height &&
      out.output_R.cols() == canvas_width &&
      out.output_G.rows() == canvas_height &&
      out.output_G.cols() == canvas_width &&
      out.output_B.rows() == canvas_height &&
      out.output_B.cols() == canvas_width;
  if (debayer_first_rgb && reconstructed_rgb_available) {
    rgb_detail_transfer_applicable = true;
    auto rgb_transfer = transfer_luma_detail_to_rgb(
        out.output_R, out.output_G, out.output_B, raw_aqmh_output,
        aqmh_recon.output, reconstruction_valid_mask);
    rgb_detail_transfer_numerical_ok = rgb_transfer.numerical_ok;
    rgb_detail_transfer_pixels = rgb_transfer.transferred_pixels;
    rgb_detail_transfer_denominator_floor_pixels =
        rgb_transfer.denominator_floor_pixels;
    rgb_detail_transfer_gain_clamped_pixels =
        rgb_transfer.gain_clamped_pixels;
    rgb_detail_transfer_denominator_epsilon =
        rgb_transfer.denominator_epsilon;
    rgb_detail_transfer_gain_min = rgb_transfer.gain_min;
    rgb_detail_transfer_gain_max = rgb_transfer.gain_max;

    if (rgb_transfer.numerical_ok) {
      const auto raw_rgb_luma_reference =
          reconstruction::prepare_aqmh_validation_reference(
              rgb_transfer.raw_luma, validation_common_mask);
      rgb_raw_vs_control_validation =
          reconstruction::compare_aqmh_to_reference(
              rgb_transfer.raw_luma, uniform_control_reference,
              uniform_control_validation_mask);
      rgb_candidate_vs_control_validation =
          reconstruction::compare_aqmh_to_reference(
              rgb_transfer.candidate_luma, uniform_control_reference,
              uniform_control_validation_mask);
      rgb_candidate_vs_raw_validation =
          reconstruction::compare_aqmh_to_reference(
              rgb_transfer.candidate_luma, raw_rgb_luma_reference,
              validation_common_mask);
      const auto rgb_control_gates =
          reconstruction::evaluate_aqmh_validation_gates(
              rgb_candidate_vs_control_validation, cfg.aqmh.validation);
      const auto rgb_raw_guard =
          reconstruction::aqmh_raw_baseline_guard_decision(
              rgb_candidate_vs_raw_validation,
              rgb_raw_vs_control_validation,
              rgb_candidate_vs_control_validation,
              cfg.aqmh.validation);
      rgb_detail_transfer_raw_guard_reason = rgb_raw_guard.reason;
      if (rgb_control_gates.all_ok && rgb_raw_guard.ok) {
        out.output_R = std::move(rgb_transfer.candidate_r);
        out.output_G = std::move(rgb_transfer.candidate_g);
        out.output_B = std::move(rgb_transfer.candidate_b);
        rgb_detail_transfer_applied = true;
        rgb_detail_transfer_reason =
            rgb_detail_transfer_pixels > 0 ? "validated_transfer"
                                           : "validated_identity";
        log_file << "[AQMH_RECONSTRUCTION] RGB luma detail transfer "
                 << "accepted: pixels=" << rgb_detail_transfer_pixels
                 << " denominator_floor_pixels="
                 << rgb_detail_transfer_denominator_floor_pixels
                 << " gain_clamped_pixels="
                 << rgb_detail_transfer_gain_clamped_pixels
                 << " raw_guard=" << rgb_raw_guard.reason << std::endl;
      } else {
        rgb_detail_transfer_reason =
            !rgb_control_gates.all_ok ? "uniform_control_gate_failed"
                                     : rgb_raw_guard.reason;
        emitter.warning(
            run_id,
            "AQMH RGB luma detail transfer rejected; preserving all three "
            "raw AQMH RGB channels atomically (reason=" +
                rgb_detail_transfer_reason + ")",
            log_file);
      }
    } else {
      rgb_detail_transfer_reason = "numerical_guard_failed";
      emitter.warning(
          run_id,
          "AQMH RGB luma detail transfer hit a non-finite value; preserving "
          "all three raw AQMH RGB channels atomically",
          log_file);
    }
  }
  const double validation_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    validation_started_at)
          .count();

  emit_reconstruction_progress(
      95, "AQMH Postprocessing und Validierung abgeschlossen",
      "validation_complete");
  if (aqmh_recon.cherry_pick_forced_disabled) {
    emitter.warning(
        run_id,
        "AQMH cherry-pick force-disabled: K_nominal_median=" +
            std::to_string(aqmh_recon.k_nominal_median) +
            " is below k_min_required=" +
            std::to_string(cfg.aqmh.cherry_pick.k_min_required),
        log_file);
  } else if (aqmh_recon.cherry_pick_active) {
    emitter.warning(
        run_id,
        "AQMH cherry-pick is active and intentionally breaks the "
        "no-frame-selection invariant at pixel level",
        log_file);
  }
  if (aqmh_recon.missing_map_samples > 0) {
    emitter.warning(run_id,
                    "AQMH reconstruction encountered missing, stale, or "
                    "mask-incompatible quality-map samples: " +
                        std::to_string(aqmh_recon.missing_map_samples),
                    log_file);
  }
  if (aqmh_recon.numerical_guard_pixels > 0) {
    emitter.warning(run_id,
                    "AQMH reconstruction rejected pixels through the "
                    "post-clipping numerical guard: " +
                        std::to_string(aqmh_recon.numerical_guard_pixels),
                    log_file);
  }

  out.recon = aqmh_recon;
  out.raw_output = raw_aqmh_output;
  out.output = aqmh_recon.output;
  out.weight_sum = aqmh_recon.weight_sum;
  out.osc_rgb_cleared = osc_mode && !debayer_first_rgb;

  cv::setNumThreads(prev_cv_threads);

  const auto aqmh_cache_stats = aqmh_cache->stats();
  emit_reconstruction_progress(
      96, "AQMH Rekonstruktionsartefakt zusammenstellen",
      "artifact_assembly");
  core::json artifact;
  artifact["method"] = "aqmh";
  artifact["acceleration"] = core::acceleration_selection_to_json(
      aqmh_reconstruction_acceleration);
  artifact["execution_backend"] = execution_backend_str;
  artifact["region_streaming"] = true;
  artifact["debayer_first_rgb"] = debayer_first_rgb;
  artifact["rgb_q_map_mode"] =
      debayer_first_rgb ? cfg.aqmh.reconstruction.rgb_q_map_mode
                        : std::string("not_applicable");
  artifact["rgb_memory_strategy"] =
      debayer_first_rgb ? cfg.aqmh.reconstruction.rgb_memory_strategy
                        : std::string("not_applicable");
  artifact["rgb_luma_detail_transfer"] = {
      {"applicable", rgb_detail_transfer_applicable},
      {"numerical_ok", rgb_detail_transfer_numerical_ok},
      {"applied", rgb_detail_transfer_applied},
      {"atomic_rgb_fallback", rgb_detail_transfer_applicable &&
                                  !rgb_detail_transfer_applied},
      {"reason", rgb_detail_transfer_reason},
      {"luma_coefficients", {0.25f, 0.50f, 0.25f}},
      {"transferred_pixels", rgb_detail_transfer_pixels},
      {"denominator_floor_pixels",
       rgb_detail_transfer_denominator_floor_pixels},
      {"gain_clamped_pixels", rgb_detail_transfer_gain_clamped_pixels},
      {"denominator_epsilon", rgb_detail_transfer_denominator_epsilon},
      {"gain_min", rgb_detail_transfer_gain_min},
      {"gain_max", rgb_detail_transfer_gain_max},
      {"raw_baseline_guard_reason",
       rgb_detail_transfer_raw_guard_reason}};
  if (rgb_detail_transfer_applicable && rgb_detail_transfer_numerical_ok) {
    artifact["rgb_luma_detail_transfer"]["raw_rgb_luma_vs_uniform_control"] =
        validation_comparison_json(rgb_raw_vs_control_validation,
                                   &cfg.aqmh.validation);
    artifact["rgb_luma_detail_transfer"]
            ["candidate_rgb_luma_vs_uniform_control"] =
        validation_comparison_json(rgb_candidate_vs_control_validation,
                                   &cfg.aqmh.validation);
    artifact["rgb_luma_detail_transfer"]["candidate_rgb_luma_vs_raw_rgb_luma"] =
        validation_comparison_json(rgb_candidate_vs_raw_validation,
                                   &cfg.aqmh.validation);
  }
  artifact["uniform_control_same_pass"] =
      acceleration_used && backend_supplied_uniform_control;
  artifact["uniform_control_mode"] =
      backend_supplied_uniform_control ? "fused_unweighted_mean"
                                       : "cpu_unweighted_mean_fallback";
  artifact["timing_seconds"] = {
      {"reconstruction_core", reconstruction_core_seconds},
      {"rgb_reconstruction", rgb_reconstruction_seconds},
      {"uniform_control_fallback", uniform_control_fallback_seconds},
      {"postprocessing_and_validation", validation_seconds}};
  artifact["cuda_pipeline_timing_seconds"] = {
      {"host_region_load_and_pack", aqmh_recon.cuda_host_prepare_seconds},
      {"host_chunk_setup", aqmh_recon.cuda_host_chunk_setup_seconds},
      {"host_to_device", aqmh_recon.cuda_h2d_seconds},
      {"kernel", aqmh_recon.cuda_kernel_seconds},
      {"device_to_host", aqmh_recon.cuda_d2h_seconds},
      {"result_commit", aqmh_recon.cuda_result_commit_seconds}};
  artifact["cuda_host_worker_timing_seconds"] = {
      {"frame_region_read", aqmh_recon.cuda_host_frame_read_worker_seconds},
      {"quality_map_read", aqmh_recon.cuda_host_q_map_read_worker_seconds},
      {"valid_mask_read", aqmh_recon.cuda_host_mask_read_worker_seconds},
      {"pixel_major_pack", aqmh_recon.cuda_host_pack_worker_seconds}};
  artifact["cuda_host_worker_timing_semantics"] =
      "sum_of_parallel_worker_wall_times";
  artifact["sample_layout"] = "pixel_major_soa";
  artifact["sample_bytes_per_frame_pixel"] = 8;
  artifact["persistent_mmap_cache_views"] = true;
  artifact["weighted_selection"] =
      acceleration_used ? "thread_sequential_shellsort_reused_ordering"
                        : "cpu_aqmh_sigma_clip";
  artifact["chunk_rows"] = aqmh_recon.chunk_rows;
  artifact["chunk_count"] = aqmh_recon.chunk_count;
  artifact["cuda_free_bytes"] = aqmh_recon.cuda_free_bytes;
  artifact["cuda_total_bytes"] = aqmh_recon.cuda_total_bytes;
  artifact["cuda_device_budget_bytes"] = aqmh_recon.cuda_device_budget_bytes;
  artifact["cuda_bytes_per_row"] = aqmh_recon.cuda_bytes_per_row;
  artifact["cuda_auto_chunk_rows_initial"] =
      aqmh_recon.cuda_auto_chunk_rows_initial;
  artifact["cuda_allocation_retries"] = aqmh_recon.cuda_allocation_retries;
  artifact["num_frames"] = static_cast<int>(frames.size());
  artifact["canvas_width"] = canvas_width;
  artifact["canvas_height"] = canvas_height;
  artifact["map_stream_id"] = aqmh_cache->map_stream_id();
  artifact["cache_dir"] = aqmh_cache->cache_dir().string();
  artifact["unsupported_pixels"] = aqmh_recon.unsupported_pixels;
  artifact["zero_veto_pixels"] = aqmh_recon.zero_veto_pixels;
  artifact["finite_map_samples"] = aqmh_recon.finite_map_samples;
  artifact["missing_map_samples"] = aqmh_recon.missing_map_samples;
  artifact["acceleration_used"] = acceleration_used;
  artifact["acceleration_fallback"] = acceleration_fallback;
  artifact["gpu_reconstruction_available"] =
      aqmh_reconstruction_acceleration.selected == core::AccelerationBackend::cuda;
  artifact["selected_backend"] =
      core::acceleration_backend_name(aqmh_reconstruction_acceleration.selected);
  artifact["prefetch_fallback"] = prefetch_fallback;
  artifact["prefetch_strategy"] = "parallel_region_on_demand";
  artifact["clip_sigma"] = aqmh_recon_cfg.clip_sigma;
  artifact["clip_sigma_low"] = aqmh_recon_cfg.clip_sigma_low;
  artifact["clip_sigma_high"] = aqmh_recon_cfg.clip_sigma_high;
  artifact["clip_iterations"] = aqmh_recon_cfg.clip_iterations;
  artifact["min_fraction"] = aqmh_recon_cfg.min_fraction;
  artifact["min_n_eff"] = aqmh_recon_cfg.min_n_eff;
  artifact["classic_tile_weights_used"] = false;
  artifact["fallback_to_classic"] = false;
  artifact["registration_weight_guard"] = {
      {"enabled", cfg.aqmh.reconstruction.registration_weight_guard},
      {"applied", registration_guard.applied},
      {"frames", registration_guard.frames},
      {"damped_frames", registration_guard.damped_frames},
      {"min_factor", registration_guard.min_factor},
      {"median_factor", registration_guard.median_factor},
      {"mean_factor", registration_guard.mean_factor},
      {"weight_floor", cfg.aqmh.reconstruction.registration_weight_floor},
      {"cc_floor", cfg.aqmh.reconstruction.registration_cc_floor},
      {"cc_full", cfg.aqmh.reconstruction.registration_cc_full},
      {"sequential_factor",
       cfg.aqmh.reconstruction.registration_sequential_factor},
      {"predicted_factor",
       cfg.aqmh.reconstruction.registration_predicted_factor},
      {"chain_depth_penalty",
       cfg.aqmh.reconstruction.registration_chain_depth_penalty},
      {"chain_depth_max_penalty",
       cfg.aqmh.reconstruction.registration_chain_depth_max_penalty}};
  artifact["registration_weight_guard"]["source_counts"] =
      registration_guard.source_counts;
  artifact["raw_aqmh_validation"] =
      validation_comparison_json(raw_control_validation, &cfg.aqmh.validation);
  artifact["final_vs_raw_aqmh_validation"] =
      validation_comparison_json(final_vs_raw_validation, &cfg.aqmh.validation);
  artifact["low_frequency_neutralization_evaluated"] =
      low_frequency_neutralization_evaluated;
  artifact["low_frequency_neutralization_applied"] =
      low_frequency_neutralization_applied;
  artifact["structure_masked_detail_applied"] =
      structure_masked_detail_applied;
  artifact["structure_masked_detail_alpha"] = structure_masked_detail_alpha;
  artifact["raw_baseline_guard"] = {
      {"relaxed_used", false},
      {"reason", raw_baseline_guard_reason}};
  if (low_frequency_neutralization_evaluated) {
    artifact["low_frequency_neutralization"] =
        validation_comparison_json(low_frequency_neutralization_validation, &cfg.aqmh.validation);
    artifact["low_frequency_neutralization"]["sigma_px"] = 96.0f;
  }
  artifact["structure_masked_detail"] =
      validation_comparison_json(structure_masked_detail_validation, &cfg.aqmh.validation);
  artifact["structure_masked_detail"].update({
      {"low_q", cfg.aqmh.reconstruction.structure_mask_low_q},
      {"high_q", cfg.aqmh.reconstruction.structure_mask_high_q},
      {"mask_blur_sigma_px", cfg.aqmh.reconstruction.structure_mask_blur_sigma_px},
      {"applied", structure_masked_detail_applied},
      {"alpha", structure_masked_detail_alpha},
      {"aqmh_background_rms",
       structure_masked_detail_validation.aqmh.background_rms},
      {"control_background_rms",
       structure_masked_detail_validation.control.background_rms},
      {"aqmh_fwhm", structure_masked_detail_validation.aqmh.fwhm},
      {"control_fwhm", structure_masked_detail_validation.control.fwhm},
      {"aqmh_seam_score", structure_masked_detail_validation.aqmh.seam_score},
      {"control_seam_score",
       structure_masked_detail_validation.control.seam_score}});
  if (full_structure_masked_detail_vs_raw_evaluated) {
    artifact["structure_masked_detail"]["full_candidate_vs_raw_aqmh"] =
        validation_comparison_json(
            full_structure_masked_detail_vs_raw_validation,
            &cfg.aqmh.validation);
    artifact["structure_masked_detail"]["full_candidate_preserves_raw_aqmh"] =
        full_structure_masked_detail_preserves_raw;
  }
  artifact["structure_masked_detail"]["attenuation_search"] = {
      {"strategy", structure_attenuation_strategy},
      {"evaluations", structure_attenuation_evaluations},
      {"feasible_candidates", structure_attenuation_feasible_candidates},
      {"best_alpha", structure_attenuation_best_alpha}};
  artifact["star_core_sharpening_applied"] = star_core_sharpening_applied;
  if (star_core_sharpening_evaluated) {
    artifact["star_core_sharpening"] =
        validation_comparison_json(star_core_sharpening_validation,
                                   &cfg.aqmh.validation);
    artifact["star_core_sharpening"].update({
        {"sigma_px", 2.0f},
        {"amount", 0.6f},
        {"applied", star_core_sharpening_applied}});
  }
  artifact["uniform_control_gate_triggered"] = aqmh_control_fallback;
  artifact["raw_aqmh_preserved_by_guard"] = raw_aqmh_preserved_by_guard;
  artifact["selected_candidate"] =
      raw_aqmh_preserved_by_guard
          ? "raw_aqmh"
          : (star_core_sharpening_applied
                 ? "star_core_sharpening"
                 : (structure_masked_detail_applied
                        ? "structure_masked_detail"
                        : (low_frequency_neutralization_applied
                               ? "low_frequency_neutralized"
                               : "raw_aqmh")));
  artifact["uniform_control_gate"] = {
      {"background_rms_ok", aqmh_background_ok},
      {"fwhm_ok", aqmh_fwhm_ok},
      {"seam_score_ok", aqmh_seam_ok},
      {"tail_ok", aqmh_tail_gate_ok},
      {"background_rms",
       gate_metric_json(pre_fallback_control_validation.background_rms_applicable,
                        aqmh_background_ok,
                        pre_fallback_control_validation.aqmh.background_rms,
                        pre_fallback_control_validation.control.background_rms,
                        pre_fallback_control_validation.background_rms_regression,
                        effective_background_threshold,
                        "control_background_rms_degenerate")},
      {"fwhm",
       gate_metric_json(pre_fallback_control_validation.fwhm_applicable,
                        aqmh_fwhm_ok,
                        pre_fallback_control_validation.aqmh.fwhm,
                        pre_fallback_control_validation.control.fwhm,
                        pre_fallback_control_validation.fwhm_regression,
                        cfg.aqmh.validation.max_fwhm_regression,
                        "fwhm_not_measurable")},
      {"seam_score",
       gate_metric_json(pre_fallback_control_validation.seam_applicable,
                        aqmh_seam_ok,
                        pre_fallback_control_validation.aqmh.seam_score,
                        pre_fallback_control_validation.control.seam_score,
                        pre_fallback_control_validation.seam_score_regression,
                        effective_seam_threshold,
                        "control_seam_score_degenerate")},
      {"tail11_abs",
       gate_metric_json(pre_fallback_control_validation.tail_applicable,
                        !pre_fallback_control_validation.tail_applicable ||
                            pre_fallback_control_validation.tail11_abs_regression <=
                                effective_tail_threshold,
                        pre_fallback_control_validation.aqmh.tail11_abs_median,
                        pre_fallback_control_validation.control.tail11_abs_median,
                        pre_fallback_control_validation.tail11_abs_regression,
                        effective_tail_threshold,
                        "insufficient_comparable_star_samples")},
      {"elongation",
       gate_metric_json(pre_fallback_control_validation.elongation_applicable,
                        !pre_fallback_control_validation.elongation_applicable ||
                            pre_fallback_control_validation.elongation_regression <=
                                effective_elongation_threshold,
                        pre_fallback_control_validation.aqmh.elongation_median,
                        pre_fallback_control_validation.control.elongation_median,
                        pre_fallback_control_validation.elongation_regression,
                        effective_elongation_threshold,
                        "insufficient_comparable_star_samples")},
      {"aqmh_background_rms", pre_fallback_control_validation.aqmh.background_rms},
      {"control_background_rms",
       pre_fallback_control_validation.control.background_rms},
      {"background_rms_regression",
       pre_fallback_control_validation.background_rms_regression},
      {"aqmh_fwhm", pre_fallback_control_validation.aqmh.fwhm},
      {"control_fwhm", pre_fallback_control_validation.control.fwhm},
      {"fwhm_regression", pre_fallback_control_validation.fwhm_regression},
      {"aqmh_seam_score", pre_fallback_control_validation.aqmh.seam_score},
      {"control_seam_score", pre_fallback_control_validation.control.seam_score},
      {"seam_score_regression",
       pre_fallback_control_validation.seam_score_regression},
      {"aqmh_tail11_abs_median",
       pre_fallback_control_validation.aqmh.tail11_abs_median},
      {"control_tail11_abs_median",
       pre_fallback_control_validation.control.tail11_abs_median},
      {"tail11_abs_regression",
       pre_fallback_control_validation.tail11_abs_regression},
      {"aqmh_elongation_median",
       pre_fallback_control_validation.aqmh.elongation_median},
      {"control_elongation_median",
       pre_fallback_control_validation.control.elongation_median},
      {"elongation_regression",
       pre_fallback_control_validation.elongation_regression},
      {"max_background_rms_regression",
       cfg.aqmh.validation.max_background_rms_regression},
      {"max_fwhm_regression", cfg.aqmh.validation.max_fwhm_regression},
      {"max_seam_score_regression",
       cfg.aqmh.validation.max_seam_score_regression},
      {"max_tail11_abs_regression",
       cfg.aqmh.validation.max_tail11_abs_regression},
      {"max_elongation_regression",
       cfg.aqmh.validation.max_elongation_regression},
      {"fwhm_clearly_improves", fwhm_clearly_improves},
      {"effective_background_threshold", effective_background_threshold},
      {"effective_seam_threshold", effective_seam_threshold},
      {"effective_tail_threshold", effective_tail_threshold},
      {"effective_elongation_threshold", effective_elongation_threshold},
      {"background_rms_applicable",
       pre_fallback_control_validation.background_rms_applicable},
      {"seam_applicable", pre_fallback_control_validation.seam_applicable},
      {"tail_applicable", pre_fallback_control_validation.tail_applicable},
      {"elongation_applicable", pre_fallback_control_validation.elongation_applicable},
      {"fwhm_applicable", pre_fallback_control_validation.fwhm_applicable}};
  // Cherry-pick diagnostics
  artifact["cherry_pick_enabled"] = cfg.aqmh.cherry_pick.enabled;
  if (cfg.aqmh.cherry_pick.enabled) {
    artifact["cherry_pick_mode"] = cfg.aqmh.cherry_pick.mode;
    artifact["cherry_pick_k_min_required"] = cfg.aqmh.cherry_pick.k_min_required;
    artifact["cherry_pick_k_frac_cfg"] = cfg.aqmh.cherry_pick.k_frac;
    artifact["cherry_pick_reject_below_best_fraction"] =
        cfg.aqmh.cherry_pick.reject_below_best_fraction;
    artifact["cherry_pick_min_keep_fraction"] =
        cfg.aqmh.cherry_pick.min_keep_fraction;
    artifact["cherry_pick_per_pixel_mode"] = aqmh_recon.cherry_pick_per_pixel_mode;
    artifact["cherry_pick_active_frac"] = aqmh_recon.cherry_pick_active_frac;
    artifact["cherry_pick_mean_k"] = aqmh_recon.cherry_pick_mean_k;
    artifact["cherry_pick_median_k"] = aqmh_recon.cherry_pick_median_k;
    artifact["cherry_pick_k_min_observed"] = aqmh_recon.cherry_pick_k_min_observed;
    artifact["cherry_pick_k_max_observed"] = aqmh_recon.cherry_pick_k_max_observed;
    artifact["cherry_pick_forced_disabled"] = aqmh_recon.cherry_pick_forced_disabled;
    artifact["cherry_pick_active"] = aqmh_recon.cherry_pick_active;
    artifact["k_nominal_median"] = aqmh_recon.k_nominal_median;
    artifact["k_effective_p10"] = aqmh_recon.k_effective_p10;
    artifact["k_effective_p50"] = aqmh_recon.k_effective_p50;
    artifact["k_effective_p90"] = aqmh_recon.k_effective_p90;
    artifact["low_rank_separation"] = aqmh_recon.low_rank_separation;
    emit_reconstruction_progress(
        97, "AQMH Cherry-Pick-K-Heatmap berechnen",
        "cherry_pick_heatmap");
    // Downsampled K-map for visualization: emit a compact flat array at
    // 1/8 linear resolution (max 200x200 grid) so the JSON stays small.
    const int kmap_divisor = std::max(1, std::max(canvas_width, canvas_height) / 200);
    const int kmap_w = std::max(1, (canvas_width  + kmap_divisor - 1) / kmap_divisor);
    const int kmap_h = std::max(1, (canvas_height + kmap_divisor - 1) / kmap_divisor);
    if (!aqmh_recon.cherry_pick_k_map.size()) {
      artifact["cherry_pick_k_heatmap"] = nullptr;
    } else {
      core::json kmap_arr = core::json::array();
      for (int oy = 0; oy < kmap_h; ++oy) {
        for (int ox = 0; ox < kmap_w; ++ox) {
          double sum = 0.0;
          int cnt = 0;
          for (int y = oy * kmap_divisor;
               y < std::min(canvas_height, (oy + 1) * kmap_divisor); ++y) {
            for (int x = ox * kmap_divisor;
                 x < std::min(canvas_width, (ox + 1) * kmap_divisor); ++x) {
              const float v = aqmh_recon.cherry_pick_k_map(y, x);
              if (v > 0.0f) { sum += v; ++cnt; }
            }
          }
          kmap_arr.push_back(cnt > 0 ? static_cast<float>(sum / cnt) : 0.0f);
        }
      }
      artifact["cherry_pick_k_heatmap"] = {
          {"width", kmap_w}, {"height", kmap_h},
          {"divisor", kmap_divisor}, {"values", std::move(kmap_arr)}
      };
    }
  }
  artifact["cache_stats"] = {
      {"bytes_written", aqmh_cache_stats.bytes_written},
      {"bytes_read", aqmh_cache_stats.bytes_read},
      {"write_count", aqmh_cache_stats.write_count},
      {"read_count", aqmh_cache_stats.read_count},
      {"cache_hits", aqmh_cache_stats.cache_hits},
      {"cache_misses", aqmh_cache_stats.cache_misses},
      {"max_resident_maps_observed",
       static_cast<uint64_t>(aqmh_cache_stats.max_resident_maps_observed)}};
  emit_reconstruction_progress(
      99, "AQMH Rekonstruktionsartefakt schreiben", "artifact_write");
  core::write_text(run_dir / "artifacts" / "aqmh_reconstruction.json",
                   artifact.dump(2));

  emitter.phase_end(
      run_id, reconstruction_phase, "ok",
      {
          {"method", "aqmh"},
          {"duration_s",
           std::chrono::duration<double>(
               std::chrono::steady_clock::now() -
               phase_started_at)
               .count()},
          {"output",
           (run_dir / "outputs" / "reconstructed_L.fit").string()},
          {"unsupported_pixels", aqmh_recon.unsupported_pixels},
          {"zero_veto_pixels", aqmh_recon.zero_veto_pixels},
          {"missing_map_samples", aqmh_recon.missing_map_samples},
          {"execution_backend", execution_backend_str},
          {"acceleration_used", acceleration_used},
          {"acceleration_fallback", acceleration_fallback},
          {"uniform_control_gate_triggered", aqmh_control_fallback},
          {"raw_aqmh_preserved_by_guard", raw_aqmh_preserved_by_guard},
          {"classic_tile_weights_used", false},
          {"cherry_pick_enabled", cfg.aqmh.cherry_pick.enabled},
          {"cherry_pick_active_frac", aqmh_recon.cherry_pick_active_frac},
          {"cherry_pick_mean_k", aqmh_recon.cherry_pick_mean_k},
      },
      log_file);
  return true;
}

} // namespace tile_compile::runner
