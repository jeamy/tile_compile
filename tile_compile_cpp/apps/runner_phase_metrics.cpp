#include "runner_phase_metrics.hpp"
#include "runner_shared.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace metrics = tile_compile::metrics;

namespace {

/// @brief Extracts exposure seconds.
/// @details Part of the channel metadata, normalization, and global-metrics phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> extract_exposure_seconds(const io::FitsHeader &header) {
  auto read_positive = [&](const char *key) -> std::optional<double> {
    if (auto value = header.get_double(key);
        value && std::isfinite(*value) && *value > 0.0) {
      return value;
    }
    if (auto value = header.get_int(key); value && *value > 0) {
      return static_cast<double>(*value);
    }
    if (auto value = header.get_string(key)) {
      char *end = nullptr;
      errno = 0;
      const double d = std::strtod(value->c_str(), &end);
      if (errno == 0 && end != value->c_str() && *end == '\0' &&
          std::isfinite(d) && d > 0.0) {
        return d;
      }
    }
    return std::nullopt;
  };

  for (const char *key : {"EXPTIME", "EXPOSURE", "EXPOSURETIME",
                           "EXPOSURE_TIME", "EXP_TIME", "DURATION", "EXPOS"}) {
    if (auto value = read_positive(key)) {
      return value;
    }
  }
  return std::nullopt;
}

} // namespace

/// @brief Runs phase channel split normalization global metrics.
/// @details Part of the channel metadata, normalization, and global-metrics phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool run_phase_channel_split_normalization_global_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, ColorMode detected_mode,
    const std::string &detected_bayer_str, core::EventEmitter &emitter,
    std::ostream &log_file, PhaseMetricsContext &out) {
  // Phase 1: CHANNEL_SPLIT (metadata-only; actual split happens later)
  emitter.phase_start(run_id, Phase::CHANNEL_SPLIT, "CHANNEL_SPLIT", log_file);

  core::json extra;
  if (detected_mode == ColorMode::OSC) {
    extra["mode"] = "OSC";
    extra["channels"] = core::json::array({"R", "G", "B"});
    extra["bayer_pattern"] = detected_bayer_str;
    extra["note"] = "deferred_to_tile_processing";
  } else {
    extra["mode"] = "MONO";
    extra["channels"] = core::json::array({"L"});
  }
  emitter.phase_end(run_id, Phase::CHANNEL_SPLIT, "ok", extra, log_file);

  // Phase 2: NORMALIZATION (// Methodik v3 §3)
  emitter.phase_start(run_id, Phase::NORMALIZATION, "NORMALIZATION", log_file);

  if (!cfg.normalization.enabled) {
    emitter.phase_end(
        run_id, Phase::NORMALIZATION, "error",
        {{"error", "NORMALIZATION: disabled but required"}}, log_file);
    emitter.run_end(run_id, false, "error", log_file);
    return false;
  }

  const float eps_b = 1.0e-6f;
  out.norm_scales.assign(frames.size(), {});
  auto &norm_scales = out.norm_scales;
  std::vector<float> B_mono(frames.size(), 0.0f);
  std::vector<float> B_r(frames.size(), 0.0f);
  std::vector<float> B_g(frames.size(), 0.0f);
  std::vector<float> B_b(frames.size(), 0.0f);
  std::vector<float> P_mono(frames.size(), 1.0f);
  std::vector<float> P_r(frames.size(), 1.0f);
  std::vector<float> P_g(frames.size(), 1.0f);
  std::vector<float> P_b(frames.size(), 1.0f);
  out.frame_cache.reset();

  std::vector<float> frame_photometric_scale(
      frames.size(), 1.0f);
  std::string photometric_scale_method = "identity_fallback";
  if (!frames.empty()) {
    std::vector<float> exposure_seconds(
        frames.size(), std::numeric_limits<float>::quiet_NaN());
    bool all_exposures_available = true;
    for (size_t i = 0; i < frames.size(); ++i) {
      try {
        const auto header = io::read_fits_header(frames[i]);
        const auto exposure = extract_exposure_seconds(header);
        if (!exposure || !std::isfinite(*exposure) || *exposure <= 0.0) {
          all_exposures_available = false;
          break;
        }
        exposure_seconds[i] = static_cast<float>(*exposure);
      } catch (...) {
        all_exposures_available = false;
        break;
      }
    }
    const float ref_exposure =
        exposure_seconds.empty() ? std::numeric_limits<float>::quiet_NaN()
                                 : exposure_seconds.front();
    if (all_exposures_available && std::isfinite(ref_exposure) &&
        ref_exposure > 0.0f) {
      for (size_t i = 0; i < frames.size(); ++i) {
        const float exposure = exposure_seconds[i];
        frame_photometric_scale[i] =
            (std::isfinite(exposure) && exposure > 0.0f)
                ? (exposure / ref_exposure)
                : 1.0f;
      }
      photometric_scale_method = "exposure_ratio";
    }
  }

  if (!frames.empty()) {
    try {
      const auto [cache_width, cache_height, cache_naxis] =
          io::get_fits_dimensions(frames.front());
      (void)cache_naxis;
      if (cache_width > 0 && cache_height > 0) {
        out.frame_cache = std::make_shared<RunnerFrameCache>(
            run_dir / "cache" / "normalized_frames", frames.size(), cache_height,
            cache_width);
      }
    } catch (const std::exception &e) {
      emitter.warning(run_id,
                      std::string("FRAME_CACHE disabled: ") + e.what(),
                      log_file);
    }
  }

  auto load_or_build_normalized = [&](size_t frame_index) -> Matrix2Df {
    if (out.frame_cache && out.frame_cache->has_normalized(frame_index)) {
      return out.frame_cache->load_normalized(frame_index);
    }
    Matrix2Df img = io::read_fits_pixels_float(frames[frame_index]);
    if (img.size() > 0) {
      image::apply_normalization_inplace(img, norm_scales[frame_index],
                                         detected_mode, detected_bayer_str, 0,
                                         0);
      if (out.frame_cache) {
        out.frame_cache->store_normalized(frame_index, img);
        out.frame_cache->store_registration_proxy(
            frame_index,
            build_registration_proxy(img, detected_mode, detected_bayer_str));
      }
    }
    return img;
  };

  const int normalization_workers = compute_adaptive_worker_count(
      cfg, frames.size(), frames, WorkerParallelProfile::MixedIo);

  std::cout << "[NORMALIZATION] Using " << normalization_workers
            << " parallel workers for " << frames.size() << " frames"
            << std::endl;

  std::atomic<size_t> norm_next{0};
  std::atomic<size_t> norm_done{0};
  std::atomic<bool> norm_failed{false};
  std::mutex norm_error_mutex;
  std::mutex norm_progress_mutex;
  std::string norm_error;

  auto normalization_worker = [&]() {
    // Limit OpenCV's internal thread pool to avoid N_workers × N_cores
    // oversubscription. Each worker gets an equal share of hardware threads.
    const int cv_threads_per_worker = std::max(1,
        static_cast<int>(std::thread::hardware_concurrency()) /
        std::max(1, normalization_workers));
    cv::setNumThreads(cv_threads_per_worker);

    std::vector<float> r_samples;
    std::vector<float> g_samples;
    std::vector<float> b_samples;
    std::vector<float> mono_samples;
    std::vector<float> shared_samples;
    auto reset_samples = [](std::vector<float> &samples, size_t reserve_count) {
      samples.clear();
      if (samples.capacity() < reserve_count) {
        samples.reserve(reserve_count);
      }
    };

    while (true) {
      const size_t i = norm_next.fetch_add(1);
      if (i >= frames.size()) {
        break;
      }
      const auto &path = frames[i];
      try {
        Matrix2Df img = io::read_fits_pixels_float(path);

        image::NormalizationScales s;
        {
          const size_t pixel_count = static_cast<size_t>(img.size());
          cv::Mat coarse_cv(img.rows(), img.cols(), CV_32F, img.data());
          const cv::Mat1b bg_mask =
              metrics::build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);
          const bool median_mode = (cfg.normalization.mode == "median");
          const bool per_channel_norm =
              (detected_mode != ColorMode::OSC) || cfg.normalization.per_channel;
          const float p_frame = (i < frame_photometric_scale.size() &&
                                 std::isfinite(frame_photometric_scale[i]) &&
                                 frame_photometric_scale[i] > 0.0f)
                                    ? frame_photometric_scale[i]
                                    : 1.0f;

          auto estimate_center =
              [&](std::vector<float> &samples,
                  const auto &fallback_fill) -> float {
            if (median_mode) {
              if (samples.empty()) {
                fallback_fill();
              }
              return samples.empty() ? 0.0f : core::median_of(samples);
            }

            float center = samples.empty() ? 0.0f : core::median_of(samples);
            if (!std::isfinite(center)) {
              fallback_fill();
              center = core::estimate_background_sigma_clip(samples);
            }
            return center;
          };

          if (detected_mode == ColorMode::OSC) {
            s.is_osc = true;
            int r_row, r_col, b_row, b_col;
            image::bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);

            reset_samples(r_samples, pixel_count / 4);
            reset_samples(g_samples, pixel_count / 2);
            reset_samples(b_samples, pixel_count / 4);
            reset_samples(shared_samples, pixel_count);

            for (int y = 0; y < img.rows(); ++y) {
              const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
              const int py = y & 1;
              for (int x = 0; x < img.cols(); ++x) {
                const float v = img(y, x);
                if (!std::isfinite(v)) {
                  continue;
                }
                const bool use_sample = median_mode || (mrow[x] != 0);
                if (!use_sample) {
                  continue;
                }
                const int px = x & 1;
                shared_samples.push_back(v);
                if (py == r_row && px == r_col) {
                  r_samples.push_back(v);
                } else if (py == b_row && px == b_col) {
                  b_samples.push_back(v);
                } else {
                  g_samples.push_back(v);
                }
              }
            }

            auto refill_shared = [&]() {
              reset_samples(shared_samples, pixel_count);
              for (Eigen::Index k = 0; k < img.size(); ++k) {
                const float v = img.data()[k];
                if (std::isfinite(v)) {
                  shared_samples.push_back(v);
                }
              }
            };
            auto refill_r = [&]() {
              reset_samples(r_samples, pixel_count / 4);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (py == r_row && px == r_col && std::isfinite(img(y, x))) {
                    r_samples.push_back(img(y, x));
                  }
                }
              }
            };
            auto refill_g = [&]() {
              reset_samples(g_samples, pixel_count / 2);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (!((py == r_row && px == r_col) ||
                        (py == b_row && px == b_col)) &&
                      std::isfinite(img(y, x))) {
                    g_samples.push_back(img(y, x));
                  }
                }
              }
            };
            auto refill_b = [&]() {
              reset_samples(b_samples, pixel_count / 4);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (py == b_row && px == b_col && std::isfinite(img(y, x))) {
                    b_samples.push_back(img(y, x));
                  }
                }
              }
            };

            float br = 0.0f;
            float bg = 0.0f;
            float bb = 0.0f;
            if (per_channel_norm) {
              br = estimate_center(r_samples, refill_r);
              bg = estimate_center(g_samples, refill_g);
              bb = estimate_center(b_samples, refill_b);
            } else {
              const float shared_center =
                  estimate_center(shared_samples, refill_shared);
              br = shared_center;
              bg = shared_center;
              bb = shared_center;
            }

            if (!std::isfinite(br) || !std::isfinite(bg) || !std::isfinite(bb)) {
              throw std::runtime_error(
                  "NORMALIZATION: invalid background estimate");
            }

            const float pr = p_frame;
            const float pg = p_frame;
            const float pb = p_frame;

            s.background_r = br;
            s.background_g = bg;
            s.background_b = bb;
            s.scale_r = 1.0f / std::max(pr, eps_b);
            s.scale_g = 1.0f / std::max(pg, eps_b);
            s.scale_b = 1.0f / std::max(pb, eps_b);
            B_r[i] = br;
            B_g[i] = bg;
            B_b[i] = bb;
            P_r[i] = pr;
            P_g[i] = pg;
            P_b[i] = pb;
          } else {
            reset_samples(mono_samples, pixel_count);
            for (int y = 0; y < img.rows(); ++y) {
              const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
              for (int x = 0; x < img.cols(); ++x) {
                if (mrow[x] != 0)
                  mono_samples.push_back(img(y, x));
              }
            }
            float b = mono_samples.empty() ? 0.0f : core::median_of(mono_samples);
            if (median_mode) {
              if (!std::isfinite(b)) {
                reset_samples(mono_samples, pixel_count);
                for (Eigen::Index k = 0; k < img.size(); ++k) {
                  const float v = img.data()[k];
                  if (std::isfinite(v)) {
                    mono_samples.push_back(v);
                  }
                }
                b = mono_samples.empty() ? 0.0f : core::median_of(mono_samples);
              }
            } else if (!std::isfinite(b)) {
              reset_samples(mono_samples, pixel_count);
              for (Eigen::Index k = 0; k < img.size(); ++k) {
                const float v = img.data()[k];
                if (std::isfinite(v)) {
                  mono_samples.push_back(v);
                }
              }
              b = core::estimate_background_sigma_clip(mono_samples);
            }
            if (!std::isfinite(b)) {
              throw std::runtime_error(
                  "NORMALIZATION: invalid background estimate");
            }
            const float p = p_frame;
            s.background_mono = b;
            s.scale_mono = 1.0f / std::max(p, eps_b);
            B_mono[i] = b;
            P_mono[i] = p;
          }
        }
        norm_scales[i] = s;
        if (out.frame_cache && img.size() > 0) {
          image::apply_normalization_inplace(img, s, detected_mode,
                                             detected_bayer_str, 0, 0);
          out.frame_cache->store_normalized(i, img);
          out.frame_cache->store_registration_proxy(
              i, build_registration_proxy(img, detected_mode,
                                          detected_bayer_str));
        }
      } catch (const std::exception &e) {
        norm_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(norm_error_mutex);
        if (norm_error.empty()) {
          norm_error = e.what();
        }
      } catch (...) {
        norm_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(norm_error_mutex);
        if (norm_error.empty()) {
          norm_error = "unknown_error";
        }
      }

      const size_t done = norm_done.fetch_add(1) + 1;
      if (done % 2 == 0 || done == frames.size()) {
        const float progress =
            frames.empty() ? 1.0f
                           : static_cast<float>(done) /
                                 static_cast<float>(frames.size());
        std::lock_guard<std::mutex> lock(norm_progress_mutex);
        emitter.phase_progress(run_id, Phase::NORMALIZATION, progress,
                               "normalize " + std::to_string(done) + "/" +
                                   std::to_string(frames.size()) +
                                   " workers=" +
                                   std::to_string(normalization_workers),
                               log_file);
      }
    }
  };

  if (normalization_workers > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(normalization_workers));
    for (int w = 0; w < normalization_workers; ++w) {
      workers.emplace_back(normalization_worker);
    }
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    normalization_worker();
  }

  if (norm_failed.load(std::memory_order_relaxed)) {
    emitter.phase_end(run_id, Phase::NORMALIZATION, "error",
                      {{"error", norm_error.empty() ? "unknown_error"
                                                     : norm_error}},
                      log_file);
    emitter.run_end(run_id, false, "error", log_file);
    std::cerr << "Error during NORMALIZATION: "
              << (norm_error.empty() ? "unknown_error" : norm_error)
              << std::endl;
    return false;
  }

  {
    core::json artifact;
    artifact["mode"] = (detected_mode == ColorMode::OSC) ? "OSC" : "MONO";
    artifact["bayer_pattern"] = detected_bayer_str;
    artifact["normalization_mode"] = cfg.normalization.mode;
    artifact["normalization_per_channel"] = cfg.normalization.per_channel;
    artifact["B_mono"] = core::json::array();
    artifact["B_r"] = core::json::array();
    artifact["B_g"] = core::json::array();
    artifact["B_b"] = core::json::array();
    artifact["P_mono"] = core::json::array();
    artifact["P_r"] = core::json::array();
    artifact["P_g"] = core::json::array();
    artifact["P_b"] = core::json::array();
    artifact["photometric_scale_method"] = photometric_scale_method;
    for (size_t i = 0; i < frames.size(); ++i) {
      artifact["B_mono"].push_back(B_mono[i]);
      artifact["B_r"].push_back(B_r[i]);
      artifact["B_g"].push_back(B_g[i]);
      artifact["B_b"].push_back(B_b[i]);
      artifact["P_mono"].push_back(P_mono[i]);
      artifact["P_r"].push_back(P_r[i]);
      artifact["P_g"].push_back(P_g[i]);
      artifact["P_b"].push_back(P_b[i]);
    }
    core::write_text(run_dir / "artifacts" / "normalization.json",
                     artifact.dump(2));
  }

  out.output_pedestal = 0.0f;
  out.output_scale_mono = core::median_finite_positive(P_mono, 1.0f);
  out.output_scale_r = core::median_finite_positive(P_r, 1.0f);
  out.output_scale_g = core::median_finite_positive(P_g, 1.0f);
  out.output_scale_b = core::median_finite_positive(P_b, 1.0f);
  out.output_bg_mono = core::median_finite(B_mono, 0.0f);
  out.output_bg_r = core::median_finite(B_r, 0.0f);
  out.output_bg_g = core::median_finite(B_g, 0.0f);
  out.output_bg_b = core::median_finite(B_b, 0.0f);

  emitter.phase_end(run_id, Phase::NORMALIZATION, "ok",
                    {
                        {"num_frames", static_cast<int>(frames.size())},
                        {"normalization_mode", cfg.normalization.mode},
                        {"normalization_per_channel",
                         cfg.normalization.per_channel},
                        {"photometric_scale_method", photometric_scale_method},
                    },
                    log_file);

  // Phase 3: GLOBAL_METRICS
  // For AQMH this phase is not exposed as a pipeline stage (AQMH computes its
  // own frame-quality factor G in AQMH_GLOBAL_QUALITY), but the metrics are
  // still needed for downstream BGE validation and the registration weight
  // penalty. Therefore we keep the computation but skip the phase events.
  const bool expose_global_metrics = !cfg.aqmh.enabled;
  if (expose_global_metrics) {
    emitter.phase_start(run_id, Phase::GLOBAL_METRICS, "GLOBAL_METRICS",
                        log_file);
  }

  out.frame_metrics.assign(frames.size(), {});
  auto &frame_metrics = out.frame_metrics;
  std::vector<metrics::FrameStarMetrics> frame_star_metrics;
  frame_star_metrics.resize(frames.size());
  int ref_star_count = 0;
  const int global_metrics_workers = compute_adaptive_worker_count(
      cfg, frames.size(), frames, WorkerParallelProfile::MixedIo);

  std::cout << "[GLOBAL_METRICS] Using " << global_metrics_workers
            << " parallel workers for " << frames.size() << " frames"
            << std::endl;

  std::atomic<size_t> gm_next{0};
  std::atomic<size_t> gm_done{0};
  std::atomic<bool> gm_failed{false};
  std::mutex gm_error_mutex;
  std::mutex gm_log_mutex;
  std::mutex gm_progress_mutex;
  std::string gm_error;

  auto global_metrics_worker = [&]() {
    const int cv_threads_per_worker = std::max(1,
        static_cast<int>(std::thread::hardware_concurrency()) /
        std::max(1, global_metrics_workers));
    cv::setNumThreads(cv_threads_per_worker);

    while (true) {
      const size_t i = gm_next.fetch_add(1);
      if (i >= frames.size()) {
        break;
      }
      try {
        Matrix2Df img = load_or_build_normalized(i);
        if (img.size() <= 0) {
          {
            std::lock_guard<std::mutex> lock(gm_log_mutex);
            emitter.warning(run_id,
                            "GLOBAL_METRICS: empty frame for " +
                                frames[i].filename().string(),
                            log_file);
          }
          FrameMetrics m;
          m.background = 0.0f;
          m.noise = 0.0f;
          m.gradient_energy = 0.0f;
          m.sky_gradient = 0.0f;
          m.quality_score = 1.0f;
          frame_metrics[i] = m;
          frame_star_metrics[i] = metrics::FrameStarMetrics{};
        } else {
          FrameMetrics m = metrics::calculate_frame_metrics(img);
          // Methodik v3: for the global background metric B_f, use the raw
          // (pre-normalization) background estimate from the normalization
          // stage.
          if (detected_mode == ColorMode::OSC) {
            const float b_raw = 0.25f * B_r[i] + 0.5f * B_g[i] + 0.25f * B_b[i];
            if (std::isfinite(b_raw))
              m.background = b_raw;
          } else {
            const float b_raw = B_mono[i];
            if (std::isfinite(b_raw))
              m.background = b_raw;
          }
          frame_metrics[i] = m;
          frame_star_metrics[i] = metrics::measure_frame_stars(img, 0);
        }
      } catch (const std::exception &e) {
        gm_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(gm_error_mutex);
        if (gm_error.empty()) {
          gm_error = e.what();
        }
      } catch (...) {
        gm_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(gm_error_mutex);
        if (gm_error.empty()) {
          gm_error = "unknown_error";
        }
      }

      const size_t done = gm_done.fetch_add(1) + 1;
      if (done % 2 == 0 || done == frames.size()) {
        const float progress =
            frames.empty()
                ? 1.0f
                : static_cast<float>(done) / static_cast<float>(frames.size());
        if (expose_global_metrics) {
          std::lock_guard<std::mutex> lock(gm_progress_mutex);
          emitter.phase_progress(run_id, Phase::GLOBAL_METRICS, progress,
                                 "metrics " + std::to_string(done) + "/" +
                                     std::to_string(frames.size()) +
                                     " workers=" +
                                     std::to_string(global_metrics_workers),
                                 log_file);
        }
      }
    }
  };

  if (global_metrics_workers > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(global_metrics_workers));
    for (int w = 0; w < global_metrics_workers; ++w) {
      workers.emplace_back(global_metrics_worker);
    }
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    global_metrics_worker();
  }

  if (gm_failed.load(std::memory_order_relaxed)) {
    if (expose_global_metrics) {
      emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "error",
                        {{"error", gm_error.empty() ? "unknown_error"
                                                     : gm_error}},
                        log_file);
    }
    emitter.run_end(run_id, false, "error", log_file);
    std::cerr << "Error during GLOBAL_METRICS: "
              << (gm_error.empty() ? "unknown_error" : gm_error) << std::endl;
    return false;
  }

  // Determine reference star count (max) and recompute wFWHM
  for (const auto &sm : frame_star_metrics) {
    if (sm.star_count > ref_star_count)
      ref_star_count = sm.star_count;
  }
  if (ref_star_count > 0) {
    for (auto &sm : frame_star_metrics) {
      if (sm.star_count > 0 && sm.fwhm > 0) {
        sm.wfwhm = sm.fwhm * static_cast<float>(ref_star_count) /
                   static_cast<float>(sm.star_count);
      }
    }
  }

  out.frame_star_metrics = frame_star_metrics;
  out.global_weights = metrics::calculate_global_weights_with_stars(
      frame_metrics, frame_star_metrics, cfg.global_metrics.weights.background,
      cfg.global_metrics.weights.noise, cfg.global_metrics.weights.gradient,
      cfg.global_metrics.weights.fwhm, cfg.global_metrics.weights.roundness,
      cfg.global_metrics.weights.star_count, cfg.global_metrics.clamp[0],
      cfg.global_metrics.clamp[1], cfg.global_metrics.adaptive_weights,
      cfg.global_metrics.weight_exponent_scale);
  auto &global_weights = out.global_weights;

  {
    core::json artifact;
    artifact["metrics"] = core::json::array();
    for (size_t i = 0; i < frame_metrics.size(); ++i) {
      core::json m;
      m["background"] = frame_metrics[i].background;
      m["noise"] = frame_metrics[i].noise;
      m["gradient_energy"] = frame_metrics[i].gradient_energy;
      m["sky_gradient"] = frame_metrics[i].sky_gradient;
      m["quality_score"] = frame_metrics[i].quality_score;
      m["global_weight"] = (i < static_cast<size_t>(global_weights.size()))
                               ? global_weights[static_cast<int>(i)]
                               : 0.0f;
      m["fwhm"] = frame_star_metrics[i].fwhm;
      m["fwhm_x"] = frame_star_metrics[i].fwhm_x;
      m["fwhm_y"] = frame_star_metrics[i].fwhm_y;
      m["roundness"] = frame_star_metrics[i].roundness;
      m["wfwhm"] = frame_star_metrics[i].wfwhm;
      m["star_count"] = frame_star_metrics[i].star_count;
      artifact["metrics"].push_back(m);
    }

    artifact["weights"] = {
        {"background", cfg.global_metrics.weights.background},
        {"noise", cfg.global_metrics.weights.noise},
        {"gradient", cfg.global_metrics.weights.gradient},
        {"fwhm", cfg.global_metrics.weights.fwhm},
        {"roundness", cfg.global_metrics.weights.roundness},
        {"star_count", cfg.global_metrics.weights.star_count}};
    artifact["clamp"] = {cfg.global_metrics.clamp[0],
                          cfg.global_metrics.clamp[1]};
    artifact["adaptive_weights"] = cfg.global_metrics.adaptive_weights;
    artifact["adaptive_weighting_method"] =
        "leave_one_out_positive_correlation_squared";
    artifact["adaptive_weighting_target"] =
        "For each metric, predict the leave-one-out consensus of the other two "
        "higher-is-better normalized signals using static weights renormalized "
        "over the remaining metrics.";
    artifact["adaptive_weighting_tie_break"] =
        "If utilities are degenerate or nearly tied, keep the static weights. "
        "Otherwise clip adaptive weights to [0.1,0.7] and renormalize.";
    core::write_text(run_dir / "artifacts" / "global_metrics.json",
                     artifact.dump(2));
  }

  if (expose_global_metrics) {
    emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "ok",
                      {
                          {"num_frames", static_cast<int>(frame_metrics.size())},
                      },
                      log_file);
  }

  return true;
}

} // namespace tile_compile::runner
