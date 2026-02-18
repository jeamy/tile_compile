#include "runner_phase_local_metrics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace core = tile_compile::core;
namespace metrics = tile_compile::metrics;

bool run_phase_local_metrics(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<Tile> &tiles_phase56,
    const DiskCacheFrameStore &prewarped_frames, core::EventEmitter &emitter,
    std::ostream &log_file, std::vector<std::vector<TileMetrics>> &local_metrics,
    std::vector<std::vector<float>> &local_weights,
    std::vector<float> &tile_quality_median, std::vector<uint8_t> &tile_is_star,
    std::vector<float> &tile_fwhm_median, int tile_offset_x, int tile_offset_y) {
  auto compute_worker_count = [&](size_t task_count) -> int {
    int workers = cfg.runtime_limits.parallel_workers;
    if (workers < 1) {
      workers = 1;
    }
    int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (cpu_cores > 0) {
      workers = std::min(workers, cpu_cores);
    }
    if (task_count > 0) {
      workers =
          std::min(workers, static_cast<int>(std::max<size_t>(1, task_count)));
    }
    return std::max(1, workers);
  };

    // Phase 5: LOCAL_METRICS (compute tile metrics per frame)
    emitter.phase_start(run_id, Phase::LOCAL_METRICS, "LOCAL_METRICS",
                        log_file);

    local_metrics.assign(frames.size(), {});
    local_weights.assign(frames.size(), {});

    const int local_metrics_workers = compute_worker_count(frames.size());
    std::cout << "[LOCAL_METRICS] Using " << local_metrics_workers
              << " parallel workers for " << frames.size() << " frames"
              << std::endl;
    std::atomic<size_t> lm_next{0};
    std::atomic<size_t> lm_done{0};
    std::mutex lm_progress_mutex;
    std::mutex lm_error_mutex;
    std::atomic<bool> lm_failed{false};
    std::string lm_error;

    auto make_zero_metrics = [&]() -> TileMetrics {
      TileMetrics z;
      z.fwhm = 0.0f;
      z.roundness = 0.0f;
      z.contrast = 0.0f;
      z.sharpness = 0.0f;
      z.background = 0.0f;
      z.noise = 0.0f;
      z.gradient_energy = 0.0f;
      z.star_count = 0;
      z.type = TileType::STRUCTURE;
      z.quality_score = 0.0f;
      return z;
    };

    auto local_metrics_worker = [&]() {
      while (true) {
        const size_t fi = lm_next.fetch_add(1);
        if (fi >= frames.size()) {
          break;
        }
        try {
          local_metrics[fi].reserve(tiles_phase56.size());
          local_weights[fi].reserve(tiles_phase56.size());

          if (!frame_has_data[fi]) {
            for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
              local_metrics[fi].push_back(make_zero_metrics());
              local_weights[fi].push_back(0.0f);
            }
          } else {
            for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
              const Tile &t = tiles_phase56[ti];
              Matrix2Df tile_img = prewarped_frames.extract_tile(fi, t, tile_offset_x, tile_offset_y);

              if (tile_img.size() <= 0) {
                local_metrics[fi].push_back(make_zero_metrics());
                local_weights[fi].push_back(0.0f);
                continue;
              }

              TileMetrics tm = metrics::calculate_tile_metrics(tile_img);
              local_metrics[fi].push_back(tm);
              local_weights[fi].push_back(1.0f);
            }
          }
        } catch (const std::exception &e) {
          lm_failed.store(true, std::memory_order_relaxed);
          std::lock_guard<std::mutex> lock(lm_error_mutex);
          if (lm_error.empty()) {
            lm_error = e.what();
          }
        } catch (...) {
          lm_failed.store(true, std::memory_order_relaxed);
          std::lock_guard<std::mutex> lock(lm_error_mutex);
          if (lm_error.empty()) {
            lm_error = "unknown_error";
          }
        }

        const size_t done = lm_done.fetch_add(1) + 1;
        if (done % 2 == 0 || done == frames.size()) {
          const float p = frames.empty() ? 1.0f
                                         : static_cast<float>(done) /
                                               static_cast<float>(frames.size());
          std::lock_guard<std::mutex> lock(lm_progress_mutex);
          emitter.phase_progress(
              run_id, Phase::LOCAL_METRICS, p,
              "local_metrics " + std::to_string(done) + "/" +
                  std::to_string(frames.size()) + " workers=" +
                  std::to_string(local_metrics_workers),
              log_file);
        }
      }
    };

    if (local_metrics_workers > 1) {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<size_t>(local_metrics_workers));
      for (int w = 0; w < local_metrics_workers; ++w) {
        workers.emplace_back(local_metrics_worker);
      }
      for (auto &worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    } else {
      local_metrics_worker();
    }

    if (lm_failed.load(std::memory_order_relaxed)) {
      emitter.phase_end(run_id, Phase::LOCAL_METRICS, "error",
                        {{"error", lm_error.empty() ? "unknown_error" : lm_error}},
                        log_file);
      emitter.run_end(run_id, false, "error", log_file);
      std::cerr << "Error during LOCAL_METRICS: "
                << (lm_error.empty() ? "unknown_error" : lm_error)
                << std::endl;
      return false;
    }

    std::vector<uint8_t> tile_star_flags(tiles_phase56.size(), 0);
    {
      // robust_tilde is now core::robust_zscore (canonical module function)

      auto clip3 = [&](float x) -> float {
        return std::min(std::max(x, cfg.local_metrics.clamp[0]),
                        cfg.local_metrics.clamp[1]);
      };

      const int star_thr = cfg.tile.star_min_count;
      const float eps = 1.0e-12f;

      const size_t n_frames = local_metrics.size();
      const size_t n_tiles = tiles_phase56.size();
      for (size_t ti = 0; ti < n_tiles; ++ti) {
        std::vector<float> fwhm;
        std::vector<float> roundness;
        std::vector<float> contrast;
        std::vector<float> bg;
        std::vector<float> energy_over_noise;
        std::vector<float> star_counts;

        fwhm.reserve(n_frames);
        roundness.reserve(n_frames);
        contrast.reserve(n_frames);
        bg.reserve(n_frames);
        energy_over_noise.reserve(n_frames);
        star_counts.reserve(n_frames);

        // Collect metrics from frames that actually have image data
        std::vector<size_t> usable_indices;
        usable_indices.reserve(n_frames);
        for (size_t fi = 0; fi < n_frames; ++fi) {
          if (!frame_has_data[fi])
            continue;
          usable_indices.push_back(fi);
          const TileMetrics &tm = local_metrics[fi][ti];
          // STAR mode uses FWHM directly (no log transform)
          fwhm.push_back(tm.fwhm);
          roundness.push_back(tm.roundness);
          contrast.push_back(tm.contrast);
          bg.push_back(tm.background);
          // STRUCTURE mode uses robust z-score of (E / σ)
          const float denom = tm.noise;
          const float ratio = (denom > eps) ? (tm.gradient_energy / denom) : 0.0f;
          energy_over_noise.push_back(ratio);
          star_counts.push_back(static_cast<float>(tm.star_count));
        }

        std::vector<float> sc_tmp = star_counts;
        float sc_med = sc_tmp.empty() ? 0.0f : core::median_of(sc_tmp);
        const TileType tile_type = (sc_med >= static_cast<float>(star_thr))
                                       ? TileType::STAR
                                       : TileType::STRUCTURE;
        tile_star_flags[ti] = (tile_type == TileType::STAR) ? 1 : 0;

        std::vector<float> fwhm_t, r_t, c_t, b_t, en_t;
        core::robust_zscore(fwhm, fwhm_t);
        core::robust_zscore(roundness, r_t);
        core::robust_zscore(contrast, c_t);
        core::robust_zscore(bg, b_t);
        core::robust_zscore(energy_over_noise, en_t);

        // Assign z-score-based weights to usable frames
        for (size_t ui = 0; ui < usable_indices.size(); ++ui) {
          size_t fi = usable_indices[ui];
          TileMetrics &tm = local_metrics[fi][ti];
          tm.type = tile_type;

          float q = 0.0f;
          if (tile_type == TileType::STAR) {
            q = cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_t[ui]) +
                cfg.local_metrics.star_mode.weights.roundness * (r_t[ui]) +
                cfg.local_metrics.star_mode.weights.contrast * (c_t[ui]);
          } else {
            q = cfg.local_metrics.structure_mode.metric_weight * (en_t[ui]) +
                cfg.local_metrics.structure_mode.background_weight * (-b_t[ui]);
          }

          q = clip3(q);
          tm.quality_score = q;
          local_weights[fi][ti] = std::exp(q);
        }
      }

      core::json artifact;
      artifact["num_frames"] = static_cast<int>(frames.size());
      artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
      artifact["tile_metrics"] = core::json::array();

      for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
        core::json fm = core::json::array();
        for (size_t ti = 0; ti < local_metrics[fi].size(); ++ti) {
          const auto &m = local_metrics[fi][ti];
          fm.push_back({
              {"fwhm", m.fwhm},
              {"roundness", m.roundness},
              {"contrast", m.contrast},
              {"sharpness", m.sharpness},
              {"background", m.background},
              {"noise", m.noise},
              {"gradient_energy", m.gradient_energy},
              {"star_count", m.star_count},
              {"tile_type", (m.type == TileType::STAR) ? "STAR" : "STRUCTURE"},
              {"quality_score", m.quality_score},
              {"local_weight", local_weights[fi][ti]},
          });
        }
        artifact["tile_metrics"].push_back(fm);
      }

      core::write_text(run_dir / "artifacts" / "local_metrics.json",
                       artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::LOCAL_METRICS, "ok",
                      {
                          {"num_frames", static_cast<int>(frames.size())},
                          {"num_tiles", static_cast<int>(tiles_phase56.size())},
                      },
                      log_file);

    // Precompute per-tile median quality and type (for Wiener denoise gating)
    tile_quality_median.assign(tiles_phase56.size(), 0.0f);
    tile_is_star = tile_star_flags;
    if (!local_metrics.empty()) {
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        std::vector<float> qs;
        qs.reserve(local_metrics.size());
        for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          if (ti < local_metrics[fi].size()) {
            qs.push_back(local_metrics[fi][ti].quality_score);
          }
        }
        tile_quality_median[ti] = qs.empty() ? 0.0f : core::median_of(qs);
      }
    }

    // Precompute per-tile median FWHM (for FWHM heatmap validation artifact)
    tile_fwhm_median.assign(tiles_phase56.size(), 0.0f);
    if (!local_metrics.empty()) {
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        std::vector<float> fwhms;
        fwhms.reserve(local_metrics.size());
        for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          if (ti < local_metrics[fi].size()) {
            fwhms.push_back(local_metrics[fi][ti].fwhm);
          }
        }
        tile_fwhm_median[ti] = fwhms.empty() ? 0.0f : core::median_of(fwhms);
      }
    }


  return true;
}

} // namespace tile_compile::runner
