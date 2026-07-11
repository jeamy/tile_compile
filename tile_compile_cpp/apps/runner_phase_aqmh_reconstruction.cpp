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
#include <vector>

namespace tile_compile::runner {
namespace {

Matrix2Df low_frequency_neutralized_aqmh(const Matrix2Df &aqmh,
                                         const Matrix2Df &control,
                                         float sigma_px) {
  if (aqmh.rows() != control.rows() || aqmh.cols() != control.cols() ||
      aqmh.rows() <= 0 || aqmh.cols() <= 0 || !(sigma_px > 0.0f)) {
    return aqmh;
  }

  Matrix2Df residual = aqmh - control;
  cv::Mat residual_view(residual.rows(), residual.cols(), CV_32F,
                        residual.data(),
                        static_cast<size_t>(residual.outerStride()) *
                            sizeof(float));
  cv::Mat low_freq;
  cv::GaussianBlur(residual_view, low_freq, cv::Size(0, 0),
                   static_cast<double>(sigma_px),
                   static_cast<double>(sigma_px), cv::BORDER_REFLECT101);

  Matrix2Df out = aqmh;
  for (int y = 0; y < out.rows(); ++y) {
    const float *low_row = low_freq.ptr<float>(y);
    for (int x = 0; x < out.cols(); ++x) {
      out(y, x) -= low_row[x];
    }
  }
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
    const float t = std::clamp(
        (cc - cfg.registration_cc_floor) /
            std::max(cfg.registration_cc_full - cfg.registration_cc_floor,
                     std::numeric_limits<float>::epsilon()),
        0.0f, 1.0f);
    float factor =
        cfg.registration_weight_floor +
        (1.0f - cfg.registration_weight_floor) * t;

    std::string source = "unknown";
    if (static_cast<size_t>(i) < source_arr.size() &&
        source_arr[static_cast<size_t>(i)].is_string()) {
      source = source_arr[static_cast<size_t>(i)].get<std::string>();
    }
    result.source_counts[source]++;
    if (source == "sequential_refined") {
      factor *= cfg.registration_sequential_factor;
    } else if (source.find("predicted") != std::string::npos ||
               source.find("interpolated") != std::string::npos ||
               source == "unknown") {
      factor *= cfg.registration_predicted_factor;
    }

    if (cfg.registration_chain_depth_penalty > 0.0f &&
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
    const std::vector<uint8_t> &common_valid_mask,
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
    reconstruction::AqmhPrefetchCoordinator* prefetch_coordinator) {

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
    emitter.run_end(run_id, false, "error", log_file);
    std::cerr << "Error during AQMH TILE_RECONSTRUCTION: " << err
              << std::endl;
    return false;
  }

  reconstruction::AqmhReconstructionConfig aqmh_recon_cfg;
  aqmh_recon_cfg.clip_sigma = cfg.aqmh.reconstruction.clip_sigma;
  aqmh_recon_cfg.clip_iterations = cfg.aqmh.reconstruction.clip_iterations;
  aqmh_recon_cfg.min_fraction = cfg.aqmh.reconstruction.min_fraction;
  aqmh_recon_cfg.min_n_eff = cfg.aqmh.reconstruction.min_n_eff;
  aqmh_recon_cfg.cherry_pick = cfg.aqmh.cherry_pick.enabled;
  aqmh_recon_cfg.cherry_pick_k_frac = cfg.aqmh.cherry_pick.k_frac;
  aqmh_recon_cfg.cherry_pick_k_min_required = cfg.aqmh.cherry_pick.k_min_required;
  aqmh_recon_cfg.cherry_pick_margin_min = cfg.aqmh.cherry_pick.margin_min;
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

  // Progress callback - defined once for all backends
  auto progress_callback = [&](int rows_done, int rows_total) {
    emitter.phase_progress_counts(
        run_id, Phase::AQMH_RECONSTRUCTION, rows_done, rows_total,
        "AQMH reconstruction rows " + std::to_string(rows_done) + "/" +
            std::to_string(rows_total),
        "rows", log_file);
  };

  bool prefetch_fallback = false;
  if (prefetch_coordinator) {
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
  std::cout << "[AQMH] Running independent pixel-wise reconstruction for "
            << frames.size() << " frame slots cpu_workers="
            << aqmh_recon_cfg.parallel_workers
            << " backend="
            << core::acceleration_backend_name(
                   aqmh_reconstruction_acceleration.selected)
            << " region_streaming=yes"
            << std::endl;
  aqmh_recon = aqmh_reconstruction_ops.reconstruct_aqmh(
      frames.size(), aqmh_frame_loader, aqmh_cache.get(),
      effective_aqmh_global_weights,
      common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
      nullptr, aqmh_mask_loader, aqmh_frame_region_loader,
      aqmh_mask_region_loader, progress_callback);
  const bool acceleration_used = aqmh_recon.acceleration_used;
  const bool acceleration_fallback = aqmh_recon.acceleration_fallback;
  const std::string execution_backend_str =
      acceleration_used ? "cuda_native_v0_2" : "cpu_exact_v0_2";

  bool low_frequency_neutralization_applied = false;
  bool structure_masked_detail_applied = false;
  float structure_masked_detail_alpha = 0.0f;
  reconstruction::AqmhValidationComparison low_frequency_neutralization_validation;
  reconstruction::AqmhValidationComparison structure_masked_detail_validation;
  if (aqmh_recon.uniform_control_output.rows() == aqmh_recon.output.rows() &&
      aqmh_recon.uniform_control_output.cols() == aqmh_recon.output.cols()) {
    constexpr float neutralization_sigma_px = 96.0f;
    Matrix2Df neutralized = low_frequency_neutralized_aqmh(
        aqmh_recon.output, aqmh_recon.uniform_control_output,
        neutralization_sigma_px);
    low_frequency_neutralization_validation =
        reconstruction::compare_aqmh_to_uniform_control(
            neutralized, aqmh_recon.uniform_control_output);
    const auto raw_validation =
        reconstruction::compare_aqmh_to_uniform_control(
            aqmh_recon.output, aqmh_recon.uniform_control_output);
    const bool background_better =
        low_frequency_neutralization_validation.background_rms_regression <
        raw_validation.background_rms_regression;
    const bool fwhm_not_regressed =
        low_frequency_neutralization_validation.fwhm_regression <=
        cfg.aqmh.validation.max_fwhm_regression;
    const bool seam_not_regressed =
        low_frequency_neutralization_validation.seam_score_regression <=
        cfg.aqmh.validation.max_seam_score_regression;
    if (background_better && fwhm_not_regressed && seam_not_regressed) {
      aqmh_recon.output = std::move(neutralized);
      low_frequency_neutralization_applied = true;
      emitter.warning(
          run_id,
          "AQMH low-frequency neutralization applied: background_rms=" +
              std::to_string(low_frequency_neutralization_validation.aqmh.background_rms) +
              " raw_background_rms=" +
              std::to_string(raw_validation.aqmh.background_rms) +
              " control_background_rms=" +
              std::to_string(low_frequency_neutralization_validation.control.background_rms) +
              " fwhm=" +
              std::to_string(low_frequency_neutralization_validation.aqmh.fwhm) +
              " seam_score=" +
              std::to_string(low_frequency_neutralization_validation.aqmh.seam_score),
          log_file);
    } else {
      constexpr float structure_low_q = 0.70f;
      constexpr float structure_high_q = 0.97f;
      constexpr float structure_mask_blur_sigma_px = 2.0f;
      Matrix2Df structure_masked = structure_masked_aqmh_detail(
          aqmh_recon.output, aqmh_recon.uniform_control_output,
          structure_low_q, structure_high_q, structure_mask_blur_sigma_px);
      structure_masked_detail_validation =
          reconstruction::compare_aqmh_to_uniform_control(
              structure_masked, aqmh_recon.uniform_control_output);
      const bool candidate_background_ok =
          structure_masked_detail_validation.background_rms_regression <=
          cfg.aqmh.validation.max_background_rms_regression;
      const bool candidate_fwhm_ok =
          structure_masked_detail_validation.fwhm_regression <=
          cfg.aqmh.validation.max_fwhm_regression;
      const bool candidate_seam_ok =
          structure_masked_detail_validation.seam_score_regression <=
          cfg.aqmh.validation.max_seam_score_regression;
      const bool improves_fwhm =
          structure_masked_detail_validation.aqmh.fwhm > 0.0f &&
          raw_validation.control.fwhm > 0.0f &&
          structure_masked_detail_validation.aqmh.fwhm <
              raw_validation.control.fwhm;
      const bool improves_seam =
          structure_masked_detail_validation.aqmh.seam_score <
          raw_validation.control.seam_score;
      if (candidate_background_ok && candidate_fwhm_ok && candidate_seam_ok &&
          (improves_fwhm || improves_seam)) {
        aqmh_recon.output = std::move(structure_masked);
        structure_masked_detail_applied = true;
        structure_masked_detail_alpha = 1.0f;
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
      } else if (candidate_fwhm_ok && candidate_seam_ok &&
                 (improves_fwhm || improves_seam)) {
        float lo = 0.0f;
        float hi = 1.0f;
        reconstruction::AqmhValidationComparison attenuated_validation;
        for (int iter = 0; iter < 12; ++iter) {
          const float alpha = 0.5f * (lo + hi);
          Matrix2Df attenuated =
              aqmh_recon.uniform_control_output +
              alpha * (structure_masked - aqmh_recon.uniform_control_output);
          const auto validation =
              reconstruction::compare_aqmh_to_uniform_control(
                  attenuated, aqmh_recon.uniform_control_output);
          const bool ok =
              validation.background_rms_regression <=
                  cfg.aqmh.validation.max_background_rms_regression &&
              validation.fwhm_regression <=
                  cfg.aqmh.validation.max_fwhm_regression &&
              validation.seam_score_regression <=
                  cfg.aqmh.validation.max_seam_score_regression;
          if (ok) {
            lo = alpha;
            attenuated_validation = validation;
          } else {
            hi = alpha;
          }
        }
        if (lo > 0.0f) {
          Matrix2Df attenuated =
              aqmh_recon.uniform_control_output +
              lo * (structure_masked - aqmh_recon.uniform_control_output);
          attenuated_validation =
              reconstruction::compare_aqmh_to_uniform_control(
                  attenuated, aqmh_recon.uniform_control_output);
          const bool attenuated_improves_fwhm =
              attenuated_validation.aqmh.fwhm > 0.0f &&
              raw_validation.control.fwhm > 0.0f &&
              attenuated_validation.aqmh.fwhm < raw_validation.control.fwhm;
          const bool attenuated_improves_seam =
              attenuated_validation.aqmh.seam_score <
              raw_validation.control.seam_score;
          if (attenuated_validation.background_rms_regression <=
                  cfg.aqmh.validation.max_background_rms_regression &&
              attenuated_validation.fwhm_regression <=
                  cfg.aqmh.validation.max_fwhm_regression &&
              attenuated_validation.seam_score_regression <=
                  cfg.aqmh.validation.max_seam_score_regression &&
              (attenuated_improves_fwhm || attenuated_improves_seam)) {
            aqmh_recon.output = std::move(attenuated);
            structure_masked_detail_validation = attenuated_validation;
            structure_masked_detail_applied = true;
            structure_masked_detail_alpha = lo;
            emitter.warning(
                run_id,
                "AQMH attenuated structure-masked detail applied: alpha=" +
                    std::to_string(lo) +
                    " background_rms=" +
                    std::to_string(attenuated_validation.aqmh.background_rms) +
                    " control_background_rms=" +
                    std::to_string(attenuated_validation.control.background_rms) +
                    " fwhm=" +
                    std::to_string(attenuated_validation.aqmh.fwhm) +
                    " control_fwhm=" +
                    std::to_string(attenuated_validation.control.fwhm) +
                    " seam_score=" +
                    std::to_string(attenuated_validation.aqmh.seam_score) +
                    " control_seam_score=" +
                    std::to_string(attenuated_validation.control.seam_score),
                log_file);
          }
        }
      }
    }
  }

  out.control_validation =
      reconstruction::compare_aqmh_to_uniform_control(
          aqmh_recon.output, aqmh_recon.uniform_control_output);
  const auto pre_fallback_control_validation = out.control_validation;
  const bool aqmh_background_ok =
      pre_fallback_control_validation.background_rms_regression <=
      cfg.aqmh.validation.max_background_rms_regression;
  const bool aqmh_fwhm_ok =
      pre_fallback_control_validation.fwhm_regression <=
      cfg.aqmh.validation.max_fwhm_regression;
  const bool aqmh_seam_ok =
      pre_fallback_control_validation.seam_score_regression <=
      cfg.aqmh.validation.max_seam_score_regression;
  const bool aqmh_control_fallback =
      aqmh_recon.uniform_control_output.rows() == aqmh_recon.output.rows() &&
      aqmh_recon.uniform_control_output.cols() == aqmh_recon.output.cols() &&
      (!aqmh_background_ok || !aqmh_fwhm_ok || !aqmh_seam_ok);
  bool aqmh_blend_accepted = false;
  float aqmh_control_blend_alpha = 0.0f;
  reconstruction::AqmhValidationComparison blend_validation;
  if (aqmh_control_fallback) {
    const Matrix2Df aqmh_candidate = aqmh_recon.output;
    const Matrix2Df &control_candidate = aqmh_recon.uniform_control_output;
    float lo = 0.0f;
    float hi = 1.0f;
    for (int iter = 0; iter < 10; ++iter) {
      const float alpha = 0.5f * (lo + hi);
      Matrix2Df blended =
          control_candidate + alpha * (aqmh_candidate - control_candidate);
      const auto candidate_validation =
          reconstruction::compare_aqmh_to_uniform_control(blended,
                                                          control_candidate);
      const bool candidate_ok =
          candidate_validation.background_rms_regression <=
              cfg.aqmh.validation.max_background_rms_regression &&
          candidate_validation.fwhm_regression <=
              cfg.aqmh.validation.max_fwhm_regression &&
          candidate_validation.seam_score_regression <=
              cfg.aqmh.validation.max_seam_score_regression;
      if (candidate_ok) {
        lo = alpha;
        blend_validation = candidate_validation;
      } else {
        hi = alpha;
      }
    }
    if (lo > 0.0f) {
      Matrix2Df blended =
          control_candidate + lo * (aqmh_candidate - control_candidate);
      blend_validation =
          reconstruction::compare_aqmh_to_uniform_control(blended,
                                                          control_candidate);
      const bool improves_fwhm =
          blend_validation.aqmh.fwhm > 0.0f &&
          pre_fallback_control_validation.control.fwhm > 0.0f &&
          blend_validation.aqmh.fwhm <
              pre_fallback_control_validation.control.fwhm;
      const bool improves_seam =
          blend_validation.aqmh.seam_score <
          pre_fallback_control_validation.control.seam_score;
      const bool blend_ok =
          blend_validation.background_rms_regression <=
              cfg.aqmh.validation.max_background_rms_regression &&
          blend_validation.fwhm_regression <=
              cfg.aqmh.validation.max_fwhm_regression &&
          blend_validation.seam_score_regression <=
              cfg.aqmh.validation.max_seam_score_regression &&
          (improves_fwhm || improves_seam);
      if (blend_ok) {
        aqmh_recon.output = std::move(blended);
        aqmh_recon.weight_sum.setConstant(1.0f);
        out.control_validation = blend_validation;
        aqmh_control_blend_alpha = lo;
        aqmh_blend_accepted = true;
        emitter.warning(
            run_id,
            "AQMH attenuated by uniform-control quality gate: alpha=" +
                std::to_string(aqmh_control_blend_alpha) +
                " background_rms=" +
                std::to_string(blend_validation.aqmh.background_rms) +
                " control_background_rms=" +
                std::to_string(blend_validation.control.background_rms) +
                " fwhm=" + std::to_string(blend_validation.aqmh.fwhm) +
                " control_fwhm=" +
                std::to_string(blend_validation.control.fwhm) +
                " seam_score=" +
                std::to_string(blend_validation.aqmh.seam_score) +
                " control_seam_score=" +
                std::to_string(blend_validation.control.seam_score),
            log_file);
      }
    }
  }
  if (aqmh_control_fallback && !aqmh_blend_accepted) {
    emitter.warning(
        run_id,
        "AQMH rejected by uniform-control background gate: background_rms=" +
            std::to_string(pre_fallback_control_validation.aqmh.background_rms) +
            " control_background_rms=" +
            std::to_string(pre_fallback_control_validation.control.background_rms) +
            " regression=" +
            std::to_string(pre_fallback_control_validation.background_rms_regression) +
            " max=" +
            std::to_string(cfg.aqmh.validation.max_background_rms_regression) +
            "; using uniform-control reconstruction output",
        log_file);
    aqmh_recon.output = aqmh_recon.uniform_control_output;
    aqmh_recon.weight_sum.setConstant(1.0f);
    out.control_validation =
        reconstruction::compare_aqmh_to_uniform_control(
            aqmh_recon.output, aqmh_recon.uniform_control_output);
  }

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
  out.output = aqmh_recon.output;
  out.weight_sum = aqmh_recon.weight_sum;
  out.osc_rgb_cleared = osc_mode;

  cv::setNumThreads(prev_cv_threads);

  const auto aqmh_cache_stats = aqmh_cache->stats();
  core::json artifact;
  artifact["method"] = "aqmh";
  artifact["acceleration"] = core::acceleration_selection_to_json(
      aqmh_reconstruction_acceleration);
  artifact["execution_backend"] = execution_backend_str;
  artifact["region_streaming"] = true;
  artifact["uniform_control_same_pass"] = true;
  artifact["sample_layout"] = "pixel_major_soa";
  artifact["sample_bytes_per_frame_pixel"] = 8;
  artifact["persistent_mmap_cache_views"] = true;
  artifact["weighted_selection"] = "deterministic_linear_quickselect";
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
  artifact["clip_sigma"] = aqmh_recon_cfg.clip_sigma;
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
  artifact["low_frequency_neutralization_applied"] =
      low_frequency_neutralization_applied;
  artifact["structure_masked_detail_applied"] =
      structure_masked_detail_applied;
  artifact["structure_masked_detail_alpha"] = structure_masked_detail_alpha;
  if (low_frequency_neutralization_applied) {
    artifact["low_frequency_neutralization"] = {
        {"sigma_px", 96.0f},
        {"aqmh_background_rms",
         low_frequency_neutralization_validation.aqmh.background_rms},
        {"control_background_rms",
         low_frequency_neutralization_validation.control.background_rms},
        {"background_rms_regression",
         low_frequency_neutralization_validation.background_rms_regression},
        {"aqmh_fwhm", low_frequency_neutralization_validation.aqmh.fwhm},
        {"control_fwhm", low_frequency_neutralization_validation.control.fwhm},
        {"fwhm_regression",
         low_frequency_neutralization_validation.fwhm_regression},
        {"aqmh_seam_score",
         low_frequency_neutralization_validation.aqmh.seam_score},
        {"control_seam_score",
         low_frequency_neutralization_validation.control.seam_score},
        {"seam_score_regression",
         low_frequency_neutralization_validation.seam_score_regression}};
  }
  artifact["structure_masked_detail"] = {
      {"low_q", 0.70f},
      {"high_q", 0.97f},
      {"mask_blur_sigma_px", 2.0f},
      {"applied", structure_masked_detail_applied},
      {"alpha", structure_masked_detail_alpha},
      {"aqmh_background_rms",
       structure_masked_detail_validation.aqmh.background_rms},
      {"control_background_rms",
       structure_masked_detail_validation.control.background_rms},
      {"background_rms_regression",
       structure_masked_detail_validation.background_rms_regression},
      {"aqmh_fwhm", structure_masked_detail_validation.aqmh.fwhm},
      {"control_fwhm", structure_masked_detail_validation.control.fwhm},
      {"fwhm_regression", structure_masked_detail_validation.fwhm_regression},
      {"aqmh_seam_score", structure_masked_detail_validation.aqmh.seam_score},
      {"control_seam_score",
       structure_masked_detail_validation.control.seam_score},
      {"seam_score_regression",
       structure_masked_detail_validation.seam_score_regression}};
  artifact["fallback_to_uniform_control"] = aqmh_control_fallback;
  artifact["uniform_control_blend_accepted"] = aqmh_blend_accepted;
  artifact["uniform_control_blend_alpha"] = aqmh_control_blend_alpha;
  artifact["uniform_control_gate"] = {
      {"background_rms_ok", aqmh_background_ok},
      {"fwhm_ok", aqmh_fwhm_ok},
      {"seam_score_ok", aqmh_seam_ok},
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
      {"max_background_rms_regression",
       cfg.aqmh.validation.max_background_rms_regression},
      {"max_fwhm_regression", cfg.aqmh.validation.max_fwhm_regression},
      {"max_seam_score_regression",
       cfg.aqmh.validation.max_seam_score_regression}};
  if (aqmh_blend_accepted) {
    artifact["uniform_control_blend_validation"] = {
        {"aqmh_background_rms", blend_validation.aqmh.background_rms},
        {"control_background_rms", blend_validation.control.background_rms},
        {"background_rms_regression", blend_validation.background_rms_regression},
        {"aqmh_fwhm", blend_validation.aqmh.fwhm},
        {"control_fwhm", blend_validation.control.fwhm},
        {"fwhm_regression", blend_validation.fwhm_regression},
        {"aqmh_seam_score", blend_validation.aqmh.seam_score},
        {"control_seam_score", blend_validation.control.seam_score},
        {"seam_score_regression", blend_validation.seam_score_regression}};
  }
  // Cherry-pick diagnostics
  artifact["cherry_pick_enabled"] = cfg.aqmh.cherry_pick.enabled;
  if (cfg.aqmh.cherry_pick.enabled) {
    artifact["cherry_pick_k_min_required"] = cfg.aqmh.cherry_pick.k_min_required;
    artifact["cherry_pick_k_frac_cfg"] = cfg.aqmh.cherry_pick.k_frac;
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
          {"fallback_to_uniform_control", aqmh_control_fallback},
          {"uniform_control_blend_accepted", aqmh_blend_accepted},
          {"uniform_control_blend_alpha", aqmh_control_blend_alpha},
          {"classic_tile_weights_used", false},
          {"cherry_pick_enabled", cfg.aqmh.cherry_pick.enabled},
          {"cherry_pick_active_frac", aqmh_recon.cherry_pick_active_frac},
          {"cherry_pick_mean_k", aqmh_recon.cherry_pick_mean_k},
      },
      log_file);
  return true;
}

} // namespace tile_compile::runner
