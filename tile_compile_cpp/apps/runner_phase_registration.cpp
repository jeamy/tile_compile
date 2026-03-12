#include "runner_phase_registration.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"
#include "tile_compile/runner/registration_outlier_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace registration = tile_compile::registration;

namespace {

constexpr float kPi = 3.14159265358979323846f;

float wrap_angle_near(float angle, float reference) {
  while (angle - reference > kPi) {
    angle -= 2.0f * kPi;
  }
  while (angle - reference < -kPi) {
    angle += 2.0f * kPi;
  }
  return angle;
}

std::vector<float> unwrap_angle_sequence(const std::vector<float> &angles) {
  if (angles.empty()) {
    return {};
  }
  std::vector<float> out = angles;
  for (size_t i = 1; i < out.size(); ++i) {
    out[i] = wrap_angle_near(out[i], out[i - 1]);
  }
  return out;
}

struct TemporalWarpSample {
  float fi = 0.0f;
  float ang = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  float cc = 1.0f;
};

struct ScalarPolyFit {
  Eigen::VectorXf coeffs;
  float max_abs_residual = std::numeric_limits<float>::infinity();
  float rms_residual = std::numeric_limits<float>::infinity();
  bool ok = false;
};

struct WarpPredictionCandidate {
  bool ok = false;
  float ang = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  float score = std::numeric_limits<float>::infinity();
  float res_ang_deg = std::numeric_limits<float>::infinity();
  float res_tx = std::numeric_limits<float>::infinity();
  float res_ty = std::numeric_limits<float>::infinity();
  int support = 0;
  float span = 0.0f;
};

ScalarPolyFit fit_weighted_poly(const std::vector<float> &xs,
                                const std::vector<float> &ys,
                                const std::vector<float> &weights,
                                int degree) {
  ScalarPolyFit out;
  const int n = static_cast<int>(xs.size());
  if (n <= 0 || ys.size() != xs.size() || weights.size() != xs.size()) {
    return out;
  }
  const int deg = std::max(0, std::min(degree, n - 1));
  Eigen::MatrixXf A(n, deg + 1);
  Eigen::VectorXf b(n);
  for (int i = 0; i < n; ++i) {
    const float w = std::sqrt(std::max(weights[static_cast<size_t>(i)], 1.0e-6f));
    float xpow = 1.0f;
    for (int j = 0; j <= deg; ++j) {
      A(i, j) = w * xpow;
      xpow *= xs[static_cast<size_t>(i)];
    }
    b(i) = w * ys[static_cast<size_t>(i)];
  }
  Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(A);
  if (qr.rank() <= 0) {
    return out;
  }
  out.coeffs = qr.solve(b);
  if (out.coeffs.size() != deg + 1) {
    return out;
  }
  float max_abs = 0.0f;
  float sum_sq = 0.0f;
  for (int i = 0; i < n; ++i) {
    float pred = 0.0f;
    float xpow = 1.0f;
    for (int j = 0; j < out.coeffs.size(); ++j) {
      pred += out.coeffs(j) * xpow;
      xpow *= xs[static_cast<size_t>(i)];
    }
    const float r = pred - ys[static_cast<size_t>(i)];
    max_abs = std::max(max_abs, std::fabs(r));
    sum_sq += r * r;
  }
  out.max_abs_residual = max_abs;
  out.rms_residual = std::sqrt(sum_sq / static_cast<float>(n));
  out.ok = true;
  return out;
}

struct WarpBounds {
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;

  int width() const { return max_x - min_x; }
  int height() const { return max_y - min_y; }
};

bool invert_affine_warp(const WarpMatrix &w, WarpMatrix &inv) {
  const float a = w(0, 0);
  const float b = w(0, 1);
  const float c = w(1, 0);
  const float d = w(1, 1);
  const float tx = w(0, 2);
  const float ty = w(1, 2);
  const float det = a * d - b * c;
  if (std::fabs(det) < 1.0e-12f) {
    return false;
  }
  const float inv_det = 1.0f / det;
  inv(0, 0) = d * inv_det;
  inv(0, 1) = -b * inv_det;
  inv(1, 0) = -c * inv_det;
  inv(1, 1) = a * inv_det;
  inv(0, 2) = -(inv(0, 0) * tx + inv(0, 1) * ty);
  inv(1, 2) = -(inv(1, 0) * tx + inv(1, 1) * ty);
  return true;
}

WarpBounds compute_warps_bounds(int width, int height,
                                const std::vector<WarpMatrix> &warps) {
  WarpBounds b;
  if (width <= 0 || height <= 0 || warps.empty()) {
    b.max_x = std::max(0, width);
    b.max_y = std::max(0, height);
    return b;
  }

  const float corners_x[4] = {0.0f, static_cast<float>(width), 0.0f,
                              static_cast<float>(width)};
  const float corners_y[4] = {0.0f, 0.0f, static_cast<float>(height),
                              static_cast<float>(height)};

  bool init = false;
  float min_xf = 0.0f;
  float min_yf = 0.0f;
  float max_xf = 0.0f;
  float max_yf = 0.0f;
  for (const auto &w : warps) {
    // Warps are used with cv::WARP_INVERSE_MAP (dst -> src). For output canvas
    // bounds we need source -> dst, i.e. inverse affine.
    WarpMatrix fwd;
    if (!invert_affine_warp(w, fwd)) {
      continue;
    }
    for (int i = 0; i < 4; ++i) {
      const float x = corners_x[i];
      const float y = corners_y[i];
      const float tx = fwd(0, 0) * x + fwd(0, 1) * y + fwd(0, 2);
      const float ty = fwd(1, 0) * x + fwd(1, 1) * y + fwd(1, 2);
      if (!init) {
        min_xf = max_xf = tx;
        min_yf = max_yf = ty;
        init = true;
      } else {
        min_xf = std::min(min_xf, tx);
        min_yf = std::min(min_yf, ty);
        max_xf = std::max(max_xf, tx);
        max_yf = std::max(max_yf, ty);
      }
    }
  }

  if (!init) {
    b.max_x = std::max(0, width);
    b.max_y = std::max(0, height);
    return b;
  }

  b.min_x = static_cast<int>(std::floor(min_xf));
  b.min_y = static_cast<int>(std::floor(min_yf));
  b.max_x = static_cast<int>(std::ceil(max_xf));
  b.max_y = static_cast<int>(std::ceil(max_yf));
  return b;
}

} // namespace

bool run_phase_registration_prewarp(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, int height, int width,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    const std::vector<image::NormalizationScales> &norm_scales,
    const std::vector<FrameMetrics> &frame_metrics,
    const VectorXf &global_weights, const io::FitsHeader &first_header,
    core::EventEmitter &emitter, std::ostream &log_file,
    PhaseRegistrationContext &out) {
  config::RegistrationConfig registration_cfg = cfg.registration;

  auto load_frame_normalized =
      [&](size_t frame_index) -> std::pair<Matrix2Df, io::FitsHeader> {
    auto frame_pair = io::read_fits_float(frames[frame_index]);
    Matrix2Df img = frame_pair.first;
    image::apply_normalization_inplace(img, norm_scales[frame_index],
                                       detected_mode, detected_bayer_str, 0, 0);
    return {img, frame_pair.second};
  };

  emitter.phase_start(run_id, Phase::REGISTRATION, "REGISTRATION", log_file);

  std::vector<WarpMatrix> global_frame_warps(frames.size(),
                                             registration::identity_warp());
  std::vector<float> global_frame_cc(frames.size(), 0.0f);
  std::string global_reg_status = "skipped";
  core::json global_reg_extra;
  const int temporal_center_idx =
      frames.empty() ? 0 : static_cast<int>(frames.size() / 2);
  int global_ref_idx = temporal_center_idx;
  std::string ref_frame_strategy = "temporal_center";

  // Pick a reference frame that is both high quality and temporally central.
  // This is more stable for long Alt/Az sessions with strong field rotation
  // than selecting the single highest-weight frame near sequence edges.
  if (!frame_metrics.empty()) {
    struct RefCandidate {
      int idx = 0;
      float score = 0.0f;
      float quality = 0.0f;
    };

    std::vector<RefCandidate> candidates;
    candidates.reserve(frame_metrics.size());
    std::vector<float> scores;
    scores.reserve(frame_metrics.size());

    for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
      float score = (i < global_weights.size()) ? global_weights[i]
                                                : frame_metrics[i].quality_score;
      if (!std::isfinite(score)) {
        score = frame_metrics[i].quality_score;
      }
      candidates.push_back(
          {i, score, static_cast<float>(frame_metrics[i].quality_score)});
      scores.push_back(score);
    }

    const int n = static_cast<int>(candidates.size());
    const int top_count = std::max(1, std::min(n, std::max(16, n / 5)));
    std::nth_element(scores.begin(),
                     scores.begin() + static_cast<long>(top_count - 1),
                     scores.end(), std::greater<float>());
    const float top_cutoff = scores[static_cast<size_t>(top_count - 1)];

    bool found_top_centered = false;
    int best_dist = std::numeric_limits<int>::max();
    float best_score = -std::numeric_limits<float>::infinity();
    for (const auto &c : candidates) {
      if (c.score < top_cutoff) {
        continue;
      }
      const int d = std::abs(c.idx - temporal_center_idx);
      if (!found_top_centered || d < best_dist ||
          (d == best_dist && c.score > best_score)) {
        found_top_centered = true;
        best_dist = d;
        best_score = c.score;
        global_ref_idx = c.idx;
      }
    }

    if (found_top_centered) {
      ref_frame_strategy = "quality_topk_centered";
    }

    if (!found_top_centered || best_score < 0.05f) {
      bool found_quality = false;
      int quality_dist = std::numeric_limits<int>::max();
      float best_quality = -std::numeric_limits<float>::infinity();
      for (const auto &c : candidates) {
        const int d = std::abs(c.idx - temporal_center_idx);
        if (!found_quality || c.quality > best_quality ||
            (c.quality == best_quality && d < quality_dist)) {
          found_quality = true;
          best_quality = c.quality;
          quality_dist = d;
          global_ref_idx = c.idx;
        }
      }
      if (found_quality) {
        ref_frame_strategy = "quality_score_fallback";
      } else {
        global_ref_idx = temporal_center_idx;
        ref_frame_strategy = "temporal_center_fallback";
      }
    }
  }

  if (global_ref_idx < 0 || global_ref_idx >= static_cast<int>(frames.size())) {
    global_ref_idx = temporal_center_idx;
    ref_frame_strategy = "temporal_center_bounds_fallback";
  }
  global_reg_extra["ref_frame"] = global_ref_idx;
  global_reg_extra["ref_frame_center"] = temporal_center_idx;
  global_reg_extra["ref_frame_strategy"] = ref_frame_strategy;

  if (!frames.empty()) {
    try {
      Matrix2Df ref_full;
      auto pair = load_frame_normalized(static_cast<size_t>(global_ref_idx));
      ref_full = std::move(pair.first);
      if (ref_full.size() <= 0) {
        global_reg_status = "error";
        global_reg_extra["error"] = "ref_frame_empty";
      } else {
        Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
                                ? image::cfa_green_proxy_downsample2x2(
                                      ref_full, detected_bayer_str)
                                : registration::downsample2x2_mean(ref_full);
        float global_reg_scale = 1.0f;
        if (ref_reg.rows() > 0) {
          int full_h2 = ref_full.rows() - (ref_full.rows() % 2);
          global_reg_scale =
              static_cast<float>(full_h2) / static_cast<float>(ref_reg.rows());
        }
        Matrix2Df ref_ecc = registration::prepare_ecc_image(ref_reg);

        // Diagnostic: proxy image stats
        {
          float rmin = ref_reg.minCoeff();
          float rmax = ref_reg.maxCoeff();
          float rmean = ref_reg.mean();
          std::cout << "[REG-DIAG] ref_reg " << ref_reg.rows() << "x" << ref_reg.cols()
                    << " min=" << rmin << " max=" << rmax << " mean=" << rmean
                    << std::endl;
        }

        const int reg_workers = compute_adaptive_worker_count(
            cfg, frames.size(), frames, WorkerParallelProfile::MixedIo);
        std::cout << "[REGISTRATION] Using " << reg_workers
                  << " parallel workers for " << frames.size() << " frames"
                  << std::endl;
        std::mutex reg_log_mutex;
        std::mutex reg_progress_mutex;
        std::atomic<size_t> reg_next{0};
        std::atomic<size_t> reg_done{0};
        std::atomic<bool> reg_failed{false};
        std::string reg_error;

        auto reg_worker = [&]() {
          while (true) {
            const size_t fi = reg_next.fetch_add(1);
            if (fi >= frames.size()) {
              break;
            }
            try {
              if (static_cast<int>(fi) == global_ref_idx) {
                global_frame_warps[fi] = registration::identity_warp();
                global_frame_cc[fi] = 1.0f;
              } else {
                Matrix2Df mov_full;
                {
                  auto pair = load_frame_normalized(fi);
                  mov_full = std::move(pair.first);
                }
                if (mov_full.size() <= 0) {
                  global_frame_warps[fi] = registration::identity_warp();
                  global_frame_cc[fi] = 0.0f;
                } else {
                  Matrix2Df mov_reg = (detected_mode == ColorMode::OSC)
                                          ? image::cfa_green_proxy_downsample2x2(
                                                mov_full, detected_bayer_str)
                                          : registration::downsample2x2_mean(
                                                mov_full);
                  // Diagnostic: first few moving frames
                  if (fi < 3) {
                    float mmin = mov_reg.minCoeff();
                    float mmax = mov_reg.maxCoeff();
                    float mmean = mov_reg.mean();
                    std::lock_guard<std::mutex> lock(reg_log_mutex);
                    std::cout << "[REG-DIAG] mov_reg[" << fi << "] "
                              << mov_reg.rows() << "x" << mov_reg.cols()
                              << " min=" << mmin << " max=" << mmax
                              << " mean=" << mmean << std::endl;
                  }

                  if (mov_reg.rows() != ref_reg.rows() ||
                      mov_reg.cols() != ref_reg.cols()) {
                    global_frame_warps[fi] = registration::identity_warp();
                    global_frame_cc[fi] = 0.0f;
                  } else {
                    // Delegate to canonical cascade in module
                    auto sfr = registration::register_single_frame(
                        mov_reg, ref_reg, registration_cfg);

                    if (sfr.reg.success) {
                      global_frame_cc[fi] = sfr.reg.correlation;
                      WarpMatrix w_full = sfr.reg.warp;
                      w_full(0, 2) *= global_reg_scale;
                      w_full(1, 2) *= global_reg_scale;
                      global_frame_warps[fi] = w_full;
                    } else {
                      global_frame_warps[fi] = registration::identity_warp();
                      global_frame_cc[fi] = 0.0f;
                    }

                    // Per-frame logging
                    if (fi < 5 || fi == frames.size() - 1 || (fi % 50 == 0)) {
                      std::lock_guard<std::mutex> lock(reg_log_mutex);
                      std::cout << "[REG] frame " << fi << "/" << frames.size()
                                << " method=" << sfr.method_used
                                << " ncc_id=" << sfr.ncc_identity
                                << " cc=" << global_frame_cc[fi] << std::endl;
                    }
                  }
                }
              }
            } catch (const std::exception &e) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = e.what();
              }
            } catch (...) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = "unknown_error";
              }
            }

            const size_t done = reg_done.fetch_add(1) + 1;
            if (done % 5 == 0 || done == frames.size()) {
              const float p = frames.empty()
                                  ? 1.0f
                                  : static_cast<float>(done) /
                                        static_cast<float>(frames.size());
              std::lock_guard<std::mutex> lock(reg_progress_mutex);
              emitter.phase_progress(
                  run_id, Phase::REGISTRATION, p,
                  "global_reg " + std::to_string(done) + "/" +
                      std::to_string(frames.size()) + " workers=" +
                      std::to_string(reg_workers),
                  log_file);
            }
          }
        };

        if (reg_workers > 1) {
          std::vector<std::thread> workers;
          workers.reserve(static_cast<size_t>(reg_workers));
          for (int w = 0; w < reg_workers; ++w) {
            workers.emplace_back(reg_worker);
          }
          for (auto &worker : workers) {
            if (worker.joinable()) {
              worker.join();
            }
          }
        } else {
          reg_worker();
        }

        if (reg_failed.load(std::memory_order_relaxed)) {
          throw std::runtime_error(reg_error.empty() ? "registration_failed"
                                                     : reg_error);
        }

        global_reg_status = "ok";
        try {
          core::json j;
          j["num_frames"] = static_cast<int>(frames.size());
          j["scale"] = global_reg_scale;
          j["ref_frame"] = global_ref_idx;
          j["cc"] = core::json::array();
          j["warps"] = core::json::array();
          j["dithering"] = {
              {"enabled", cfg.dithering.enabled},
              {"min_shift_px", cfg.dithering.min_shift_px},
              {"detected_fraction", 0.0},
          };
          int shifts_detected = 0;
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            const auto &w = global_frame_warps[fi];
            j["cc"].push_back(global_frame_cc[fi]);
            const float shift_mag =
                std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
            if (cfg.dithering.enabled && shift_mag >= cfg.dithering.min_shift_px) {
              shifts_detected++;
            }
            j["warps"].push_back({
                {"a00", w(0, 0)},
                {"a01", w(0, 1)},
                {"tx", w(0, 2)},
                {"a10", w(1, 0)},
                {"a11", w(1, 1)},
                {"ty", w(1, 2)},
                {"shift_px", shift_mag},
            });
          }
          if (!frames.empty()) {
            j["dithering"]["detected_fraction"] =
                static_cast<double>(shifts_detected) /
                static_cast<double>(frames.size());
            j["dithering"]["detected_count"] = shifts_detected;
            j["dithering"]["total_frames"] = static_cast<int>(frames.size());
          }
          core::write_text(run_dir / "artifacts" / "global_registration.json",
                           j.dump(2));

          if (cfg.output.write_registered_frames) {
            fs::create_directories(run_dir / cfg.output.registered_dir);
            // first_header is available from outer scope

            for (size_t fi = 0; fi < frames.size(); ++fi) {
              if (static_cast<size_t>(fi) >= global_frame_warps.size())
                continue;
              const auto &w = global_frame_warps[fi];

              auto pair = load_frame_normalized(fi);
              Matrix2Df img = std::move(pair.first);
              if (img.size() <= 0)
                continue;

              Matrix2Df out_img =
                  image::apply_global_warp(img, w, detected_mode);

              std::ostringstream name;
              name << "frame_" << std::setw(4) << std::setfill('0') << fi
                   << ".fits";
              io::write_fits_float(run_dir / cfg.output.registered_dir /
                                       name.str(),
                                   out_img, first_header);
            }
          }

        } catch (...) {
        }
      }
    } catch (const std::exception &e) {
      global_reg_status = "error";
      global_reg_extra["error"] = e.what();
    } catch (...) {
      global_reg_status = "error";
      global_reg_extra["error"] = "unknown_error";
    }
  } else {
    global_reg_extra["reason"] = "no_frames";
  }

  // Reject implausible global registration outliers before downstream phases.
  // These outliers can pass NCC but still produce heavy tile/grid artifacts.
  int reg_reject_orientation_outliers = 0;
  int reg_reject_reflection_outliers = 0;
  int reg_reject_scale_outliers = 0;
  int reg_reject_cc_outliers = 0;
  int reg_reject_shift_outliers = 0;
  core::json reg_rejected_frames = core::json::array();
  std::vector<uint8_t> reg_rejected_mask(frames.size(), 0);
  if (cfg.registration.reject_outliers) {
    std::vector<float> cc_positive;
    cc_positive.reserve(frames.size());
    std::vector<float> normal_shift_mags_positive;
    normal_shift_mags_positive.reserve(frames.size());
    std::vector<float> half_turn_shift_mags_positive;
    half_turn_shift_mags_positive.reserve(frames.size());
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f) {
        continue;
      }
      cc_positive.push_back(global_frame_cc[fi]);
      const auto &w = global_frame_warps[fi];
      const auto shift_diag = registration_shift_diagnostics(w, width, height);
      if (shift_diag.half_turn_family) {
        half_turn_shift_mags_positive.push_back(shift_diag.shift_magnitude);
      } else {
        normal_shift_mags_positive.push_back(shift_diag.shift_magnitude);
      }
    }

    auto robust_median = [](std::vector<float> values) -> float {
      if (values.empty()) {
        return 0.0f;
      }
      const size_t mid = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + static_cast<long>(mid),
                       values.end());
      float med = values[mid];
      if (values.size() % 2 == 0 && mid > 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1),
                         values.end());
        med = 0.5f * (med + values[mid - 1]);
      }
      return med;
    };

    const float cc_median = robust_median(cc_positive);
    float cc_mad = 0.0f;
    if (!cc_positive.empty()) {
      std::vector<float> abs_dev;
      abs_dev.reserve(cc_positive.size());
      for (float v : cc_positive) {
        abs_dev.push_back(std::fabs(v - cc_median));
      }
      cc_mad = robust_median(abs_dev);
    }
    const float cc_threshold_abs = cfg.registration.reject_cc_min_abs;
    const float cc_threshold_robust =
        cc_median - cfg.registration.reject_cc_mad_multiplier * cc_mad;
    const float cc_min_keep = std::max(cc_threshold_abs, cc_threshold_robust);

    const float normal_shift_median = robust_median(normal_shift_mags_positive);
    const float normal_shift_limit =
        std::max(cfg.registration.reject_shift_px_min,
                 cfg.registration.reject_shift_median_multiplier * normal_shift_median);
    const float half_turn_shift_median = robust_median(half_turn_shift_mags_positive);
    const float half_turn_shift_limit =
        std::max(cfg.registration.reject_shift_px_min,
                 cfg.registration.reject_shift_median_multiplier * half_turn_shift_median);

    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f)
        continue;
      const auto &w = global_frame_warps[fi];
      const auto shift_diag = registration_shift_diagnostics(w, width, height);

      bool reject = false;
      std::vector<std::string> reject_reasons;
      // Accept both 0° and ~180° rotations (trace can be positive or negative).
      // But reject mirror/reflection solutions (det < 0), which cause
      // characteristic mirrored ghost artifacts in the final stack.
      const float det = w(0, 0) * w(1, 1) - w(0, 1) * w(1, 0);
      if (det < 0.0f) {
        reject = true;
        ++reg_reject_reflection_outliers;
        reject_reasons.push_back("reflection");
      }

      if (!reject) {
        const float scale = std::sqrt(std::fabs(det));
        if (scale < cfg.registration.reject_scale_min ||
            scale > cfg.registration.reject_scale_max) {
          reject = true;
          ++reg_reject_scale_outliers;
          reject_reasons.push_back("scale");
        }
      }

      if (!reject) {
        const float cc = global_frame_cc[fi];
        if (cc < cc_min_keep) {
          reject = true;
          ++reg_reject_cc_outliers;
          reject_reasons.push_back("low_cc");
        }
      }

      if (!reject) {
        const float shift_mag = shift_diag.shift_magnitude;
        const float shift_limit = shift_diag.half_turn_family
                                      ? half_turn_shift_limit
                                      : normal_shift_limit;
        if (shift_mag > shift_limit) {
          reject = true;
          ++reg_reject_shift_outliers;
          reject_reasons.push_back("shift_outlier");
        }
      }

      if (reject) {
        reg_rejected_mask[fi] = 1;
        core::json rej = {
            {"frame_index", static_cast<int>(fi)},
            {"frame_name", frames[fi].filename().string()},
            {"cc", global_frame_cc[fi]},
            {"reasons", reject_reasons},
            {"a00", w(0, 0)},
            {"a01", w(0, 1)},
            {"tx", w(0, 2)},
            {"a10", w(1, 0)},
            {"a11", w(1, 1)},
            {"ty", w(1, 2)},
        };
        reg_rejected_frames.push_back(rej);
        std::ostringstream msg;
        msg << "REGISTRATION outlier rejected: frame=" << fi << " ("
            << frames[fi].filename().string() << ") cc=" << global_frame_cc[fi]
            << " reasons=" << core::join(reject_reasons, ",")
            << " tx=" << w(0, 2) << " ty=" << w(1, 2);
        emitter.warning(run_id, msg.str(), log_file);
        std::cout << "[REG-FILTER] " << msg.str() << std::endl;

        global_frame_warps[fi] = registration::identity_warp();
        global_frame_cc[fi] = 0.0f;
      }
    }

    // Predict warps for rejected frames using a polynomial field rotation
    // model fitted to the valid registrations.  For alt-az mounts the warp
    // parameters (angle, tx, ty) follow smooth trajectories that are well
    // approximated by a low-degree polynomial over the session duration.
    // This retains ALL frames (methodology v3.2.2 §1.2) while providing
    // physically plausible geometry for frames where registration failed.
    {
      std::vector<float> vfi, vang_raw, vtx, vty, vcc;
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        if (!reg_rejected_mask[fi] && global_frame_cc[fi] > 0.0f) {
          const auto &w = global_frame_warps[fi];
          vfi.push_back(static_cast<float>(fi));
          vang_raw.push_back(std::atan2(w(0, 1), w(0, 0)));
          vtx.push_back(w(0, 2));
          vty.push_back(w(1, 2));
          vcc.push_back(global_frame_cc[fi]);
        }
      }

      const int nv = static_cast<int>(vfi.size());
      int reg_model_predicted = 0;
      int reg_model_local_refined = 0;
      int reg_model_interpolated = 0;
      int reg_model_blended = 0;

      if (nv >= 3) {
        const std::vector<float> vang = unwrap_angle_sequence(vang_raw);
        // Normalise frame indices to [0,1] for numerical stability.
        const float fi_lo = vfi.front();
        const float fi_hi = vfi.back();
        const float fi_span = std::max(1.0f, fi_hi - fi_lo);

        // Degree-2 Vandermonde
        Eigen::MatrixXf V(nv, 3);
        Eigen::VectorXf ya(nv), yx(nv), yy(nv);
        for (int i = 0; i < nv; ++i) {
          const float t = (vfi[static_cast<size_t>(i)] - fi_lo) / fi_span;
          V(i, 0) = 1.0f;
          V(i, 1) = t;
          V(i, 2) = t * t;
          ya(i) = vang[static_cast<size_t>(i)];
          yx(i) = vtx[static_cast<size_t>(i)];
          yy(i) = vty[static_cast<size_t>(i)];
        }

        auto qr = V.householderQr();
        Eigen::VectorXf ca = qr.solve(ya);
        Eigen::VectorXf cx = qr.solve(yx);
        Eigen::VectorXf cy = qr.solve(yy);

        // Residual stats for diagnostics
        const float res_ang =
            (V * ca - ya).cwiseAbs().maxCoeff() * 57.29577951f;
        const float res_tx = (V * cx - yx).cwiseAbs().maxCoeff();
        const float res_ty = (V * cy - yy).cwiseAbs().maxCoeff();

        std::vector<TemporalWarpSample> valid_samples;
        valid_samples.reserve(static_cast<size_t>(nv));
        for (int i = 0; i < nv; ++i) {
          valid_samples.push_back(
              {vfi[static_cast<size_t>(i)], vang[static_cast<size_t>(i)],
               vtx[static_cast<size_t>(i)], vty[static_cast<size_t>(i)],
               vcc[static_cast<size_t>(i)]});
        }

        auto build_local_candidate = [&](size_t fi, int support_count)
            -> WarpPredictionCandidate {
          WarpPredictionCandidate out;
          if (valid_samples.empty()) {
            return out;
          }
          std::vector<std::pair<float, size_t>> by_dist;
          by_dist.reserve(valid_samples.size());
          for (size_t i = 0; i < valid_samples.size(); ++i) {
            by_dist.emplace_back(
                std::fabs(valid_samples[i].fi - static_cast<float>(fi)), i);
          }
          std::sort(by_dist.begin(), by_dist.end(),
                    [](const auto &a, const auto &b) {
                      if (a.first != b.first) {
                        return a.first < b.first;
                      }
                      return a.second < b.second;
                    });
          if (by_dist.empty()) {
            return out;
          }
          const int take_n =
              std::max(1, std::min(support_count, static_cast<int>(by_dist.size())));
          std::vector<size_t> chosen;
          chosen.reserve(static_cast<size_t>(take_n));
          for (int i = 0; i < take_n; ++i) {
            chosen.push_back(by_dist[static_cast<size_t>(i)].second);
          }
          std::sort(chosen.begin(), chosen.end());

          float max_abs_dist = 0.0f;
          for (size_t idx : chosen) {
            max_abs_dist = std::max(
                max_abs_dist,
                std::fabs(valid_samples[idx].fi - static_cast<float>(fi)));
          }
          max_abs_dist = std::max(max_abs_dist, 1.0f);

          std::vector<float> xs;
          std::vector<float> ws;
          std::vector<float> ys_ang;
          std::vector<float> ys_tx;
          std::vector<float> ys_ty;
          xs.reserve(chosen.size());
          ws.reserve(chosen.size());
          ys_ang.reserve(chosen.size());
          ys_tx.reserve(chosen.size());
          ys_ty.reserve(chosen.size());
          for (size_t idx : chosen) {
            const auto &s = valid_samples[idx];
            const float x = (s.fi - static_cast<float>(fi)) / max_abs_dist;
            const float u = std::min(1.0f, std::fabs(x));
            const float tricube =
                std::pow(std::max(0.0f, 1.0f - u * u * u), 3.0f);
            const float w = std::max(1.0e-3f, tricube * std::max(0.05f, s.cc));
            xs.push_back(x);
            ws.push_back(w);
            ys_ang.push_back(s.ang);
            ys_tx.push_back(s.tx);
            ys_ty.push_back(s.ty);
          }

          const int degree =
              (take_n >= 5) ? 2 : ((take_n >= 2) ? 1 : 0);
          const auto fit_ang = fit_weighted_poly(xs, ys_ang, ws, degree);
          const auto fit_tx = fit_weighted_poly(xs, ys_tx, ws, degree);
          const auto fit_ty = fit_weighted_poly(xs, ys_ty, ws, degree);
          if (!fit_ang.ok || !fit_tx.ok || !fit_ty.ok ||
              fit_ang.coeffs.size() == 0 || fit_tx.coeffs.size() == 0 ||
              fit_ty.coeffs.size() == 0) {
            return out;
          }

          out.ok = true;
          out.ang = fit_ang.coeffs(0);
          out.tx = fit_tx.coeffs(0);
          out.ty = fit_ty.coeffs(0);
          out.res_ang_deg = fit_ang.max_abs_residual * 57.29577951f;
          out.res_tx = fit_tx.max_abs_residual;
          out.res_ty = fit_ty.max_abs_residual;
          out.support = take_n;
          out.span = max_abs_dist;
          out.score = out.res_ang_deg / 0.35f + out.res_tx / 20.0f +
                      out.res_ty / 20.0f + 0.05f * max_abs_dist;
          return out;
        };

        auto build_bridge_candidate = [&](size_t fi) -> WarpPredictionCandidate {
          WarpPredictionCandidate out;
          if (valid_samples.empty()) {
            return out;
          }
          int right = -1;
          for (int i = 0; i < nv; ++i) {
            if (valid_samples[static_cast<size_t>(i)].fi >= static_cast<float>(fi)) {
              right = i;
              break;
            }
          }
          if (right >= 0 && right < nv &&
              valid_samples[static_cast<size_t>(right)].fi ==
                  static_cast<float>(fi)) {
            out.ok = true;
            out.ang = valid_samples[static_cast<size_t>(right)].ang;
            out.tx = valid_samples[static_cast<size_t>(right)].tx;
            out.ty = valid_samples[static_cast<size_t>(right)].ty;
            out.support = 1;
            out.span = 0.0f;
            out.score = 0.0f;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (right > 0 && right < nv) {
            const auto &l = valid_samples[static_cast<size_t>(right - 1)];
            const auto &r = valid_samples[static_cast<size_t>(right)];
            const float denom = std::max(1.0f, r.fi - l.fi);
            const float alpha = (static_cast<float>(fi) - l.fi) / denom;
            out.ok = true;
            out.ang = l.ang + alpha * (r.ang - l.ang);
            out.tx = l.tx + alpha * (r.tx - l.tx);
            out.ty = l.ty + alpha * (r.ty - l.ty);
            out.support = 2;
            out.span = r.fi - l.fi;
            out.score = 0.5f + 0.05f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (nv >= 2 && right == 0) {
            const auto &s0 = valid_samples[0];
            const auto &s1 = valid_samples[1];
            const float denom = std::max(1.0f, s1.fi - s0.fi);
            const float delta = static_cast<float>(fi) - s0.fi;
            out.ok = true;
            out.ang = s0.ang + delta * (s1.ang - s0.ang) / denom;
            out.tx = s0.tx + delta * (s1.tx - s0.tx) / denom;
            out.ty = s0.ty + delta * (s1.ty - s0.ty) / denom;
            out.support = 2;
            out.span = std::fabs(delta);
            out.score = 1.0f + 0.08f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (nv >= 2 && right < 0) {
            const auto &s0 = valid_samples[static_cast<size_t>(nv - 2)];
            const auto &s1 = valid_samples[static_cast<size_t>(nv - 1)];
            const float denom = std::max(1.0f, s1.fi - s0.fi);
            const float delta = static_cast<float>(fi) - s1.fi;
            out.ok = true;
            out.ang = s1.ang + delta * (s1.ang - s0.ang) / denom;
            out.tx = s1.tx + delta * (s1.tx - s0.tx) / denom;
            out.ty = s1.ty + delta * (s1.ty - s0.ty) / denom;
            out.support = 2;
            out.span = std::fabs(delta);
            out.score = 1.0f + 0.08f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          return out;
        };

        // Predict warps for rejected frames and frames with cc=0
        // (completely failed registration, not caught by outlier filter)
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!reg_rejected_mask[fi] && global_frame_cc[fi] > 0.0f) {
            continue;
          }

          const float t = (static_cast<float>(fi) - fi_lo) / fi_span;
          WarpPredictionCandidate global_candidate;
          global_candidate.ok = true;
          global_candidate.ang = ca(0) + ca(1) * t + ca(2) * t * t;
          global_candidate.tx = cx(0) + cx(1) * t + cx(2) * t * t;
          global_candidate.ty = cy(0) + cy(1) * t + cy(2) * t * t;
          global_candidate.score =
              res_ang / 0.35f + res_tx / 20.0f + res_ty / 20.0f + 5.0f;

          WarpPredictionCandidate best_local;
          for (int support_count : {6, 8, 12, 16, 24, 32, 48}) {
            if (support_count > nv) {
              continue;
            }
            const auto cand = build_local_candidate(fi, support_count);
            if (!cand.ok || cand.score >= best_local.score) {
              continue;
            }
            best_local = cand;
          }
          const auto bridge_candidate = build_bridge_candidate(fi);

          WarpPredictionCandidate chosen = global_candidate;
          if (best_local.ok && bridge_candidate.ok) {
            chosen.ok = true;
            const float local_conf = 1.0f / (1.0f + std::max(0.0f, best_local.score));
            const float bridge_conf =
                1.0f / (1.0f + std::max(0.0f, bridge_candidate.score));
            const float norm = std::max(1.0e-6f, local_conf + bridge_conf);
            const float wl = local_conf / norm;
            const float wb = bridge_conf / norm;
            chosen.ang = bridge_candidate.ang +
                         wl * (wrap_angle_near(best_local.ang, bridge_candidate.ang) -
                               bridge_candidate.ang);
            chosen.tx = wl * best_local.tx + wb * bridge_candidate.tx;
            chosen.ty = wl * best_local.ty + wb * bridge_candidate.ty;
            chosen.score = std::min(best_local.score, bridge_candidate.score);
            chosen.res_ang_deg = best_local.res_ang_deg;
            chosen.res_tx = best_local.res_tx;
            chosen.res_ty = best_local.res_ty;
            chosen.support = std::max(best_local.support, bridge_candidate.support);
            chosen.span = std::max(best_local.span, bridge_candidate.span);
            ++reg_model_blended;
          } else if (best_local.ok) {
            chosen = best_local;
            ++reg_model_local_refined;
          } else if (bridge_candidate.ok) {
            chosen = bridge_candidate;
            ++reg_model_interpolated;
          }

          WarpMatrix w;
          w(0, 0) = std::cos(chosen.ang);
          w(0, 1) = std::sin(chosen.ang);
          w(1, 0) = -std::sin(chosen.ang);
          w(1, 1) = std::cos(chosen.ang);
          w(0, 2) = chosen.tx;
          w(1, 2) = chosen.ty;

          global_frame_warps[fi] = w;
          // Small positive cc → included in prewarp but lower than valid
          // frames. Downstream tile-level quality metrics handle weighting.
          global_frame_cc[fi] = 1.0e-4f;
          ++reg_model_predicted;
        }

        {
          std::ostringstream msg;
          msg << "REGISTRATION field-rotation model: predicted "
              << reg_model_predicted << " rejected frames from " << nv
              << " valid warps (max residual: angle="
              << std::fixed << std::setprecision(2) << res_ang
              << "deg tx=" << res_tx << "px ty=" << res_ty << "px"
              << ", local_refined=" << reg_model_local_refined
              << ", interpolated=" << reg_model_interpolated
              << ", blended=" << reg_model_blended << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
      } else if (nv >= 1) {
        // Too few points for polynomial — copy nearest valid warp.
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!reg_rejected_mask[fi] && global_frame_cc[fi] > 0.0f) {
            continue;
          }
          // Find nearest valid frame
          int best = -1;
          int best_dist = std::numeric_limits<int>::max();
          for (int k = 0; k < nv; ++k) {
            int d = std::abs(static_cast<int>(fi) -
                             static_cast<int>(vfi[static_cast<size_t>(k)]));
            if (d < best_dist) {
              best_dist = d;
              best = static_cast<int>(vfi[static_cast<size_t>(k)]);
            }
          }
          if (best >= 0) {
            global_frame_warps[fi] =
                global_frame_warps[static_cast<size_t>(best)];
            global_frame_cc[fi] = 1.0e-4f;
            ++reg_model_predicted;
          }
        }
        if (reg_model_predicted > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION nearest-copy fallback: predicted "
              << reg_model_predicted << " rejected frames from " << nv
              << " valid warp(s)";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
      }

      global_reg_extra["reg_model_predicted"] = reg_model_predicted;
      global_reg_extra["reg_model_local_refined"] = reg_model_local_refined;
      global_reg_extra["reg_model_interpolated"] = reg_model_interpolated;
      global_reg_extra["reg_model_blended"] = reg_model_blended;
    }
  }
  if (reg_reject_orientation_outliers > 0 ||
      reg_reject_reflection_outliers > 0 ||
      reg_reject_scale_outliers > 0 ||
      reg_reject_cc_outliers > 0 ||
      reg_reject_shift_outliers > 0) {
    std::cout << "[REG-FILTER] rejected outlier warps: orientation="
              << reg_reject_orientation_outliers
              << " reflection=" << reg_reject_reflection_outliers
              << " scale=" << reg_reject_scale_outliers
              << " cc=" << reg_reject_cc_outliers
              << " shift=" << reg_reject_shift_outliers << std::endl;
  }
  global_reg_extra["reg_reject_orientation_outliers"] =
      reg_reject_orientation_outliers;
  global_reg_extra["reg_reject_reflection_outliers"] =
      reg_reject_reflection_outliers;
  global_reg_extra["reg_reject_scale_outliers"] = reg_reject_scale_outliers;
  global_reg_extra["reg_reject_cc_outliers"] = reg_reject_cc_outliers;
  global_reg_extra["reg_reject_shift_outliers"] = reg_reject_shift_outliers;
  global_reg_extra["reg_rejected_frames"] = reg_rejected_frames;

  // All frames now have warps (valid registration or polynomial prediction).
  // Tile-level quality metrics handle downstream weighting (v3.2.2 §1.2).
  int n_cc_positive = 0;
  int n_cc_zero = 0;
  int n_cc_negative = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (global_frame_cc[fi] > 0.0f) {
      ++n_cc_positive;
    } else if (global_frame_cc[fi] < 0.0f) {
      ++n_cc_negative;
    } else {
      ++n_cc_zero;
    }
  }
  std::cout << "[REG] cc>0: " << n_cc_positive << ", cc==0: " << n_cc_zero
            << ", cc<0: " << n_cc_negative << std::endl;
  global_reg_extra["frames_cc_positive"] = n_cc_positive;
  global_reg_extra["frames_cc_zero"] = n_cc_zero;
  global_reg_extra["frames_cc_negative"] = n_cc_negative;

  emitter.phase_end(run_id, Phase::REGISTRATION, global_reg_status,
                    global_reg_extra, log_file);

  emitter.phase_start(run_id, Phase::PREWARP, "PREWARP", log_file);

  // Compute bounding box for field rotation: output canvas must be large enough
  // to contain all rotated frames (Alt/Az mounts near pole).
  // All warps are now meaningful (valid registration or polynomial prediction).
  WarpBounds bbox = compute_warps_bounds(width, height, global_frame_warps);
  
  // Round canvas to even dimensions for CFA (Bayer) compatibility.
  // warp_cfa_mosaic_via_subplanes works on half-resolution subplanes, so
  // canvas must be even in both dimensions to avoid size mismatch in store().
  int canvas_width = (bbox.width() + 1) & ~1;   // round up to even
  int canvas_height = (bbox.height() + 1) & ~1; // round up to even
  
  // Offset to shift all frames into positive coordinate space
  int offset_x = -bbox.min_x;
  int offset_y = -bbox.min_y;
  
  // For OSC mode: ensure offsets are even to preserve Bayer pattern alignment.
  // If offset is odd, the entire CFA mosaic shifts by one pixel and R/G swap.
  if (detected_mode == ColorMode::OSC) {
    if ((offset_x & 1) != 0) {
      offset_x = (offset_x + 1) & ~1;  // round up to even
      canvas_width += 1;
      canvas_width = (canvas_width + 1) & ~1;  // keep canvas even
    }
    if ((offset_y & 1) != 0) {
      offset_y = (offset_y + 1) & ~1;  // round up to even
      canvas_height += 1;
      canvas_height = (canvas_height + 1) & ~1;  // keep canvas even
    }
  }
  
  // Apply offset correction to all warps
  if (offset_x != 0 || offset_y != 0) {
    const float ox = static_cast<float>(offset_x);
    const float oy = static_cast<float>(offset_y);
    for (auto& w : global_frame_warps) {
      // Compose destination-space translation q = p + offset into an inverse-map
      // warp M (src = M * p) => M' = M * T(-offset).
      w(0, 2) -= w(0, 0) * ox + w(0, 1) * oy;
      w(1, 2) -= w(1, 0) * ox + w(1, 1) * oy;
    }
  }
  
  // Log canvas expansion for field rotation
  if (canvas_width > width || canvas_height > height) {
    std::ostringstream msg;
    msg << "Field rotation detected: expanding canvas from " << width << "x" << height
        << " to " << canvas_width << "x" << canvas_height
        << " (bbox: [" << bbox.min_x << "," << bbox.min_y << "] to ["
        << bbox.max_x << "," << bbox.max_y << "], offset: ["
        << offset_x << "," << offset_y << "])";
    emitter.warning(run_id, msg.str(), log_file);
    std::cout << "[PREWARP] " << msg.str() << std::endl;
  }

  // Pre-warp all frames at full resolution before tile extraction.
  // Applying rotation warps to small tile ROIs is fundamentally broken:
  // warpAffine needs source pixels outside the tile boundary that don't
  // exist, causing CFA pattern corruption (colored tile rectangles).
  //
  // Disk-backed: frames are written as raw float binaries and mmap'd on
  // demand, so RAM usage is bounded by OS page cache rather than N*W*H*4.
  DiskCacheFrameStore prewarped_frames(
      run_dir / ".prewarped_cache", frames.size(), canvas_height, canvas_width);
  std::vector<uint8_t> frame_has_data(frames.size(), 0);
  const int prewarp_workers = compute_adaptive_worker_count(
      cfg, frames.size(), frames, WorkerParallelProfile::IoHeavy);
  std::cout << "[PREWARP] Using " << prewarp_workers
            << " parallel workers for " << frames.size() << " frames"
            << std::endl;
  std::mutex prewarp_log_mutex;
  std::mutex prewarp_progress_mutex;
  std::atomic<size_t> prewarp_next{0};
  std::atomic<size_t> prewarp_done{0};
  std::atomic<int> n_frames_with_data{0};
  std::atomic<bool> prewarp_failed{false};
  std::string prewarp_error;

  auto prewarp_worker = [&]() {
    while (true) {
      const size_t fi = prewarp_next.fetch_add(1);
      if (fi >= frames.size()) {
        break;
      }
      try {
        auto pair = load_frame_normalized(fi);
        Matrix2Df img = std::move(pair.first);
        if (img.size() <= 0) {
          continue;
        }
        if (cfg.stacking.per_frame_cosmetic_correction) {
          if (detected_mode == ColorMode::OSC) {
            img = image::cosmetic_correction_cfa(
                img, cfg.stacking.per_frame_cosmetic_correction_sigma, true, 0,
                0);
          } else {
            img = image::cosmetic_correction(
                img, cfg.stacking.per_frame_cosmetic_correction_sigma, true);
          }
        }
        const auto &w = global_frame_warps[fi];
        const float eps = 1.0e-6f;
        const bool is_identity =
            std::fabs(w(0, 0) - 1.0f) < eps && std::fabs(w(0, 1)) < eps &&
            std::fabs(w(1, 0)) < eps && std::fabs(w(1, 1) - 1.0f) < eps &&
            std::fabs(w(0, 2)) < eps && std::fabs(w(1, 2)) < eps;
        Matrix2Df warped;
        if (is_identity) {
          // For identity warp, we still need to expand canvas if bbox requires it
          if (canvas_width > width || canvas_height > height) {
            // Create expanded canvas and place image at offset position
            warped = Matrix2Df::Zero(canvas_height, canvas_width);
            int src_h = img.rows();
            int src_w = img.cols();
            int dst_y = offset_y;
            int dst_x = offset_x;
            // Ensure we don't write outside canvas bounds
            int copy_h = std::min(src_h, canvas_height - dst_y);
            int copy_w = std::min(src_w, canvas_width - dst_x);
            if (copy_h > 0 && copy_w > 0) {
              warped.block(dst_y, dst_x, copy_h, copy_w) = img.block(0, 0, copy_h, copy_w);
            }
          } else {
            warped = std::move(img);
          }
        } else {
          warped = image::apply_global_warp(img, w, detected_mode, canvas_height, canvas_width);
        }
        if (warped.size() > 0) {
          prewarped_frames.store(fi, warped);
          const bool has_data = prewarped_frames.has_data(fi);
          if (has_data) {
            frame_has_data[fi] = 1;
            n_frames_with_data.fetch_add(1, std::memory_order_relaxed);
          }
        }
      } catch (const std::exception &e) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = e.what();
        }
      } catch (...) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = "unknown_error";
        }
      }

      const size_t done = prewarp_done.fetch_add(1) + 1;
      if (done % 5 == 0 || done == frames.size()) {
        std::lock_guard<std::mutex> lock(prewarp_progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::PREWARP, static_cast<int>(done),
            static_cast<int>(frames.size()), "prewarp workers=" +
                                              std::to_string(prewarp_workers),
            "frames", log_file);
      }
    }
  };

  if (prewarp_workers > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(prewarp_workers));
    for (int w = 0; w < prewarp_workers; ++w) {
      workers.emplace_back(prewarp_worker);
    }
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    prewarp_worker();
  }

  if (prewarp_failed.load(std::memory_order_relaxed)) {
    emitter.phase_end(run_id, Phase::PREWARP, "error",
                      {{"error", prewarp_error.empty() ? "unknown_error"
                                                         : prewarp_error}},
                      log_file);
    std::cerr << "Error during PREWARP: "
              << (prewarp_error.empty() ? "unknown_error" : prewarp_error)
              << std::endl;
    emitter.run_end(run_id, false, "error", log_file);
    return false;
  }

  out.n_usable_frames = n_frames_with_data.load(std::memory_order_relaxed);
  out.canvas_width = canvas_width;
  out.canvas_height = canvas_height;
  out.tile_offset_x = offset_x;
  out.tile_offset_y = offset_y;
  
  emitter.phase_end(run_id, Phase::PREWARP, "ok",
                    {{"num_frames", static_cast<int>(frames.size())},
                     {"num_frames_with_data", out.n_usable_frames},
                     {"canvas_width", canvas_width},
                     {"canvas_height", canvas_height},
                     {"tile_offset_x", offset_x},
                     {"tile_offset_y", offset_y},
                     {"workers", prewarp_workers}},
                    log_file);

  out.frame_has_data = std::move(frame_has_data);
  out.prewarped_frames = std::move(prewarped_frames);
  out.min_valid_frames = 1;
  return true;
}

} // namespace tile_compile::runner
