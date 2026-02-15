#include "runner_phase_metrics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace metrics = tile_compile::metrics;

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

  int normalization_workers = cfg.runtime_limits.parallel_workers;
  if (normalization_workers < 1) {
    normalization_workers = 1;
  }
  int normalization_cpu_cores =
      static_cast<int>(std::thread::hardware_concurrency());
  if (normalization_cpu_cores > 0) {
    normalization_workers = std::min(normalization_workers, normalization_cpu_cores);
  }
  if (!frames.empty()) {
    normalization_workers = std::min(
        normalization_workers,
        static_cast<int>(std::max<size_t>(1, frames.size())));
  }
  normalization_workers = std::max(1, normalization_workers);

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
    while (true) {
      const size_t i = norm_next.fetch_add(1);
      if (i >= frames.size()) {
        break;
      }
      const auto &path = frames[i];
      try {
        auto frame_pair = io::read_fits_float(path);
        const Matrix2Df &img = frame_pair.first;

        image::NormalizationScales s;
        {
          std::vector<float> all;
          all.reserve(static_cast<size_t>(img.size()));
          for (Eigen::Index k = 0; k < img.size(); ++k) {
            all.push_back(img.data()[k]);
          }
          const float b0 = core::median_of(all);
          const float coarse_scale = (b0 > eps_b) ? (1.0f / b0) : 1.0f;
          Matrix2Df coarse_norm = img * coarse_scale;
          cv::Mat coarse_cv(coarse_norm.rows(), coarse_norm.cols(), CV_32F,
                            coarse_norm.data());
          const cv::Mat1b bg_mask =
              metrics::build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);

          if (detected_mode == ColorMode::OSC) {
            s.is_osc = true;
            int r_row, r_col, b_row, b_col;
            image::bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);

            std::vector<float> pr_bg;
            std::vector<float> pg_bg;
            std::vector<float> pb_bg;
            pr_bg.reserve(static_cast<size_t>(img.size()) / 4);
            pg_bg.reserve(static_cast<size_t>(img.size()) / 2);
            pb_bg.reserve(static_cast<size_t>(img.size()) / 4);

            for (int y = 0; y < img.rows(); ++y) {
              const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
              const int py = y & 1;
              for (int x = 0; x < img.cols(); ++x) {
                if (mrow[x] == 0)
                  continue;
                const int px = x & 1;
                const float v = img(y, x);
                if (py == r_row && px == r_col) {
                  pr_bg.push_back(v);
                } else if (py == b_row && px == b_col) {
                  pb_bg.push_back(v);
                } else {
                  pg_bg.push_back(v);
                }
              }
            }

            float br = pr_bg.empty() ? 0.0f : core::median_of(pr_bg);
            float bg = pg_bg.empty() ? 0.0f : core::median_of(pg_bg);
            float bb = pb_bg.empty() ? 0.0f : core::median_of(pb_bg);

            if (!(br > eps_b)) {
              std::vector<float> pr;
              pr.reserve(static_cast<size_t>(img.size()) / 4);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (py == r_row && px == r_col)
                    pr.push_back(img(y, x));
                }
              }
              br = core::estimate_background_sigma_clip(std::move(pr));
            }
            if (!(bg > eps_b)) {
              std::vector<float> pg;
              pg.reserve(static_cast<size_t>(img.size()) / 2);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (!((py == r_row && px == r_col) ||
                        (py == b_row && px == b_col)))
                    pg.push_back(img(y, x));
                }
              }
              bg = core::estimate_background_sigma_clip(std::move(pg));
            }
            if (!(bb > eps_b)) {
              std::vector<float> pb;
              pb.reserve(static_cast<size_t>(img.size()) / 4);
              for (int y = 0; y < img.rows(); ++y) {
                const int py = y & 1;
                for (int x = 0; x < img.cols(); ++x) {
                  const int px = x & 1;
                  if (py == b_row && px == b_col)
                    pb.push_back(img(y, x));
                }
              }
              bb = core::estimate_background_sigma_clip(std::move(pb));
            }

            if (!(br > eps_b) || !(bg > eps_b) || !(bb > eps_b)) {
              throw std::runtime_error(
                  "NORMALIZATION: invalid background estimate");
            }

            s.scale_r = 1.0f / br;
            s.scale_g = 1.0f / bg;
            s.scale_b = 1.0f / bb;
            B_r[i] = br;
            B_g[i] = bg;
            B_b[i] = bb;
          } else {
            std::vector<float> p_bg;
            p_bg.reserve(static_cast<size_t>(img.size()));
            for (int y = 0; y < img.rows(); ++y) {
              const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
              for (int x = 0; x < img.cols(); ++x) {
                if (mrow[x] != 0)
                  p_bg.push_back(img(y, x));
              }
            }
            float b = p_bg.empty() ? 0.0f : core::median_of(p_bg);
            if (!(b > eps_b)) {
              std::vector<float> p;
              p.reserve(static_cast<size_t>(img.size()));
              for (Eigen::Index k = 0; k < img.size(); ++k) {
                p.push_back(img.data()[k]);
              }
              b = core::estimate_background_sigma_clip(std::move(p));
            }
            if (!(b > eps_b)) {
              throw std::runtime_error(
                  "NORMALIZATION: invalid background estimate");
            }
            s.scale_mono = 1.0f / b;
            B_mono[i] = b;
          }
        }
        norm_scales[i] = s;
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
    artifact["B_mono"] = core::json::array();
    artifact["B_r"] = core::json::array();
    artifact["B_g"] = core::json::array();
    artifact["B_b"] = core::json::array();
    for (size_t i = 0; i < frames.size(); ++i) {
      artifact["B_mono"].push_back(B_mono[i]);
      artifact["B_r"].push_back(B_r[i]);
      artifact["B_g"].push_back(B_g[i]);
      artifact["B_b"].push_back(B_b[i]);
    }
    core::write_text(run_dir / "artifacts" / "normalization.json",
                     artifact.dump(2));
  }

  out.output_pedestal = 0.0f;
  out.output_bg_mono = core::median_finite_positive(B_mono, 1.0f);
  out.output_bg_r = core::median_finite_positive(B_r, 1.0f);
  out.output_bg_g = core::median_finite_positive(B_g, 1.0f);
  out.output_bg_b = core::median_finite_positive(B_b, 1.0f);

  emitter.phase_end(run_id, Phase::NORMALIZATION, "ok",
                    {
                        {"num_frames", static_cast<int>(frames.size())},
                    },
                    log_file);

  // Phase 3: GLOBAL_METRICS
  emitter.phase_start(run_id, Phase::GLOBAL_METRICS, "GLOBAL_METRICS",
                      log_file);

  out.frame_metrics.assign(frames.size(), {});
  auto &frame_metrics = out.frame_metrics;
  std::vector<metrics::FrameStarMetrics> frame_star_metrics;
  frame_star_metrics.resize(frames.size());
  int ref_star_count = 0;

  // First pass: compute metrics + star metrics for all frames
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto &path = frames[i];
    try {
      auto frame_pair = io::read_fits_float(path);
      Matrix2Df img = frame_pair.first;
      if (img.size() <= 0) {
        emitter.warning(run_id,
                        "GLOBAL_METRICS: empty frame for " +
                            path.filename().string(),
                        log_file);
        FrameMetrics m;
        m.background = 0.0f;
        m.noise = 0.0f;
        m.gradient_energy = 0.0f;
        m.quality_score = 1.0f;
        frame_metrics[i] = m;
        frame_star_metrics[i] = metrics::FrameStarMetrics{};
      } else {
        image::apply_normalization_inplace(img, norm_scales[i], detected_mode,
                                           detected_bayer_str, 0, 0);
        FrameMetrics m = metrics::calculate_frame_metrics(img);
        // Methodik v3: for the global background metric B_f, use the raw
        // (pre-normalization) background estimate from the normalization stage.
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
      emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "error",
                        {{"error", e.what()}}, log_file);
      emitter.run_end(run_id, false, "error", log_file);
      std::cerr << "Error during GLOBAL_METRICS: " << e.what() << std::endl;
      return false;
    }

    const float progress =
        frames.empty()
            ? 1.0f
            : static_cast<float>(i + 1) / static_cast<float>(frames.size());
    emitter.phase_progress(run_id, Phase::GLOBAL_METRICS, progress,
                           "metrics " + std::to_string(i + 1) + "/" +
                               std::to_string(frames.size()),
                           log_file);
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

  out.global_weights = metrics::calculate_global_weights(
      frame_metrics, cfg.global_metrics.weights.background,
      cfg.global_metrics.weights.noise, cfg.global_metrics.weights.gradient,
      cfg.global_metrics.clamp[0], cfg.global_metrics.clamp[1],
      cfg.global_metrics.adaptive_weights,
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
        {"gradient", cfg.global_metrics.weights.gradient}};
    artifact["clamp"] = {cfg.global_metrics.clamp[0],
                          cfg.global_metrics.clamp[1]};
    artifact["adaptive_weights"] = cfg.global_metrics.adaptive_weights;
    core::write_text(run_dir / "artifacts" / "global_metrics.json",
                     artifact.dump(2));
  }

  emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "ok",
                    {
                        {"num_frames", static_cast<int>(frame_metrics.size())},
                    },
                    log_file);

  return true;
}

} // namespace tile_compile::runner
