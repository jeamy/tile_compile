#include "runner_phase_local_metrics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/reconstruction/local_weight_regularization.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
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
    const std::vector<uint8_t> &common_valid_mask,
    int common_mask_width, int common_mask_height,
    const std::vector<uint8_t> &tile_common_valid,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    bool apply_normalization_to_tiles, core::EventEmitter &emitter,
    std::ostream &log_file, std::vector<std::vector<TileMetrics>> &local_metrics,
    std::vector<std::vector<float>> &local_weights,
    std::vector<float> &tile_quality_median, std::vector<uint8_t> &tile_is_star,
    std::vector<float> &tile_fwhm_median, int tile_offset_x,
    int tile_offset_y) {
  (void)tile_offset_x;
  (void)tile_offset_y;

    // Phase 5: LOCAL_METRICS (compute tile metrics per frame)
    emitter.phase_start(run_id, Phase::LOCAL_METRICS, "LOCAL_METRICS",
                        log_file);

    local_metrics.assign(frames.size(), {});
    local_weights.assign(frames.size(), {});

    const int local_metrics_workers = compute_adaptive_worker_count(
        cfg, frames.size(), frames, WorkerParallelProfile::CpuBound);
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
          Matrix2Df tile_img;
          local_metrics[fi].reserve(tiles_phase56.size());
          local_weights[fi].reserve(tiles_phase56.size());

          if (!frame_has_data[fi]) {
            for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
              local_metrics[fi].push_back(make_zero_metrics());
              local_weights[fi].push_back(0.0f);
            }
          } else {
            for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
              if (ti >= tile_common_valid.size() || tile_common_valid[ti] == 0) {
                local_metrics[fi].push_back(make_zero_metrics());
                local_weights[fi].push_back(0.0f);
                continue;
              }
              const Tile &t = tiles_phase56[ti];
              // PREWARP already composes canvas offset into the stored frames.
              // Tiles are defined in canvas coordinates, so no extra tile offset
              // must be applied here.
              if (!prewarped_frames.extract_tile_into(fi, t, tile_img, 0, 0)) {
                local_metrics[fi].push_back(make_zero_metrics());
                local_weights[fi].push_back(0.0f);
                continue;
              }

              if (apply_normalization_to_tiles && fi < norm_scales.size() &&
                  tile_img.size() > 0) {
                image::apply_normalization_inplace(
                    tile_img, norm_scales[fi], detected_mode,
                    detected_bayer_str, t.x, t.y);
              }

              if (tile_img.size() <= 0) {
                local_metrics[fi].push_back(make_zero_metrics());
                local_weights[fi].push_back(0.0f);
                continue;
              }

              // Enforce COMMON_OVERLAP support before metric extraction so
              // canvas pixels are excluded from all downstream BGE weighting.
              if (!apply_common_overlap_to_tile_inplace_and_check_nonzero(
                      tile_img, t, common_valid_mask, common_mask_width,
                      common_mask_height)) {
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
    std::vector<std::vector<float>> local_quality_scores(
        frames.size(), std::vector<float>(tiles_phase56.size(), 0.0f));
    reconstruction::LocalWeightRegularizationSummary
        local_weight_regularization_summary;
    double neighborhood_q_delta_sum = 0.0;
    size_t neighborhood_q_delta_count = 0;
    float neighborhood_q_delta_p95 = 0.0f;
    {
      // robust_tilde is now core::robust_zscore (canonical module function)

      auto clip3 = [&](float x) -> float {
        return std::min(std::max(x, cfg.local_metrics.clamp[0]),
                        cfg.local_metrics.clamp[1]);
      };

      const int star_thr = cfg.tile.star_min_count;
      const float eps = 1.0e-12f;
      const bool neighborhood_enabled =
          cfg.local_metrics.neighborhood_normalization.enabled &&
          cfg.local_metrics.neighborhood_normalization.radius > 0 &&
          cfg.local_metrics.neighborhood_normalization.blend > 0.0f;
      const int neighborhood_radius =
          cfg.local_metrics.neighborhood_normalization.radius;
      const float neighborhood_blend =
          cfg.local_metrics.neighborhood_normalization.blend;
      auto tile_key = [](int row, int col) -> uint64_t {
        return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) ^
               static_cast<uint32_t>(col);
      };
      std::unordered_map<uint64_t, size_t> tile_by_grid;
      tile_by_grid.reserve(tiles_phase56.size());
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        tile_by_grid.emplace(
            tile_key(tiles_phase56[ti].row, tiles_phase56[ti].col), ti);
      }
      std::vector<std::vector<size_t>> tile_neighbors(tiles_phase56.size());
      if (neighborhood_enabled) {
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          const auto &tile = tiles_phase56[ti];
          auto &neighbors = tile_neighbors[ti];
          for (int dy = -neighborhood_radius; dy <= neighborhood_radius; ++dy) {
            for (int dx = -neighborhood_radius; dx <= neighborhood_radius; ++dx) {
              if (dx == 0 && dy == 0) {
                continue;
              }
              if (std::abs(dx) + std::abs(dy) > neighborhood_radius) {
                continue;
              }
              auto it =
                  tile_by_grid.find(tile_key(tile.row + dy, tile.col + dx));
              if (it != tile_by_grid.end()) {
                neighbors.push_back(it->second);
              }
            }
          }
        }
      }
      auto robust_location_scale = [](const std::vector<float> &values) {
        std::pair<float, float> out{0.0f, 1.0f};
        if (values.empty()) {
          return out;
        }
        std::vector<float> tmp = values;
        const float center = core::median_of(tmp);
        for (float &v : tmp) {
          v = std::fabs(v - center);
        }
        const float mad = core::median_of(tmp);
        out.first = center;
        out.second = std::max(1.4826f * mad, 1.0e-6f);
        return out;
      };
      auto blended_zscores = [&](const std::vector<float> &local_values,
                                 const std::vector<float> &pooled_values,
                                 std::vector<float> *out_local_z,
                                 std::vector<float> *out_blended_z) {
        out_local_z->assign(local_values.size(), 0.0f);
        out_blended_z->assign(local_values.size(), 0.0f);
        if (local_values.empty()) {
          return;
        }
        core::robust_zscore(local_values, *out_local_z);
        if (!neighborhood_enabled || pooled_values.empty()) {
          *out_blended_z = *out_local_z;
          return;
        }
        const auto [center, scale] = robust_location_scale(pooled_values);
        for (size_t i = 0; i < local_values.size(); ++i) {
          const float z_pooled = (local_values[i] - center) / scale;
          (*out_blended_z)[i] =
              (1.0f - neighborhood_blend) * (*out_local_z)[i] +
              neighborhood_blend * z_pooled;
        }
      };

      std::vector<float> neighborhood_q_deltas;
      neighborhood_q_deltas.reserve(
          frames.size() * std::max<size_t>(1, tiles_phase56.size() / 4));

      const size_t n_frames = local_metrics.size();
      const size_t n_tiles = tiles_phase56.size();
      for (size_t ti = 0; ti < n_tiles; ++ti) {
        std::vector<float> fwhm;
        std::vector<float> roundness;
        std::vector<float> contrast;
        std::vector<float> bg;
        std::vector<float> energy_over_noise;
        std::vector<float> star_counts;
        std::vector<float> pooled_fwhm;
        std::vector<float> pooled_roundness;
        std::vector<float> pooled_contrast;
        std::vector<float> pooled_bg;
        std::vector<float> pooled_energy_over_noise;

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
        if (neighborhood_enabled) {
          for (size_t fi = 0; fi < n_frames; ++fi) {
            if (!frame_has_data[fi]) {
              continue;
            }
            for (size_t ni : tile_neighbors[ti]) {
              if (ni >= local_metrics[fi].size()) {
                continue;
              }
              const TileMetrics &nm = local_metrics[fi][ni];
              pooled_fwhm.push_back(nm.fwhm);
              pooled_roundness.push_back(nm.roundness);
              pooled_contrast.push_back(nm.contrast);
              pooled_bg.push_back(nm.background);
              const float denom = nm.noise;
              pooled_energy_over_noise.push_back(
                  (denom > eps) ? (nm.gradient_energy / denom) : 0.0f);
            }
          }
        }

        std::vector<float> sc_tmp = star_counts;
        float sc_med = sc_tmp.empty() ? 0.0f : core::median_of(sc_tmp);
        const TileType tile_type = (sc_med >= static_cast<float>(star_thr))
                                       ? TileType::STAR
                                       : TileType::STRUCTURE;
        tile_star_flags[ti] = (tile_type == TileType::STAR) ? 1 : 0;

        std::vector<float> fwhm_local_t, r_local_t, c_local_t, b_local_t,
            en_local_t;
        std::vector<float> fwhm_t, r_t, c_t, b_t, en_t;
        blended_zscores(fwhm, pooled_fwhm, &fwhm_local_t, &fwhm_t);
        blended_zscores(roundness, pooled_roundness, &r_local_t, &r_t);
        blended_zscores(contrast, pooled_contrast, &c_local_t, &c_t);
        blended_zscores(bg, pooled_bg, &b_local_t, &b_t);
        blended_zscores(energy_over_noise, pooled_energy_over_noise,
                        &en_local_t, &en_t);

        // Assign z-score-based weights to usable frames
        for (size_t ui = 0; ui < usable_indices.size(); ++ui) {
          size_t fi = usable_indices[ui];
          TileMetrics &tm = local_metrics[fi][ti];
          tm.type = tile_type;
          float q_before = 0.0f;
          if (tile_type == TileType::STAR) {
            q_before =
                cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_local_t[ui]) +
                cfg.local_metrics.star_mode.weights.roundness * (r_local_t[ui]) +
                cfg.local_metrics.star_mode.weights.contrast * (c_local_t[ui]);
          } else {
            q_before = cfg.local_metrics.structure_mode.metric_weight *
                           (en_local_t[ui]) +
                       cfg.local_metrics.structure_mode.background_weight *
                           (-b_local_t[ui]);
          }
          q_before = clip3(q_before);

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
          local_quality_scores[fi][ti] = q;
          const float abs_delta = std::fabs(q - q_before);
          if (abs_delta > 0.0f) {
            neighborhood_q_delta_sum += static_cast<double>(abs_delta);
            ++neighborhood_q_delta_count;
            neighborhood_q_deltas.push_back(abs_delta);
          }
        }
      }
      if (!neighborhood_q_deltas.empty()) {
        std::sort(neighborhood_q_deltas.begin(), neighborhood_q_deltas.end());
        neighborhood_q_delta_p95 =
            core::percentile_from_sorted(neighborhood_q_deltas, 95.0f);
      }

      reconstruction::LocalWeightRegularizationConfig regularization_cfg;
      regularization_cfg.enabled =
          cfg.local_metrics.spatial_regularization.enabled;
      regularization_cfg.lambda =
          cfg.local_metrics.spatial_regularization.lambda;
      regularization_cfg.passes =
          cfg.local_metrics.spatial_regularization.passes;
      local_weight_regularization_summary =
          reconstruction::regularize_local_quality_scores(
              tiles_phase56, tile_common_valid, frame_has_data,
              regularization_cfg, &local_quality_scores);

      for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
        if (!frame_has_data[fi]) {
          continue;
        }
        for (size_t ti = 0; ti < local_metrics[fi].size(); ++ti) {
          if (ti >= tile_common_valid.size() || tile_common_valid[ti] == 0u) {
            local_metrics[fi][ti].quality_score = 0.0f;
            local_weights[fi][ti] = 0.0f;
            continue;
          }
          const float q = clip3(local_quality_scores[fi][ti]);
          local_metrics[fi][ti].quality_score = q;
          local_weights[fi][ti] = std::exp(q);
        }
      }

      const size_t artifact_entries = frames.size() * tiles_phase56.size();
      constexpr size_t kLocalMetricsArtifactMaxEntries = 120000;
      const bool write_full_local_metrics_artifact =
          (cfg.pipeline.mode != "production") ||
          (artifact_entries <= kLocalMetricsArtifactMaxEntries);

      core::json artifact;
      artifact["num_frames"] = static_cast<int>(frames.size());
      artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
      artifact["entry_count"] = static_cast<uint64_t>(artifact_entries);
      artifact["full_tile_metrics_written"] = write_full_local_metrics_artifact;
      artifact["entry_limit_full_write"] =
          static_cast<uint64_t>(kLocalMetricsArtifactMaxEntries);
      artifact["neighborhood_normalization_enabled"] =
          cfg.local_metrics.neighborhood_normalization.enabled;
      artifact["neighborhood_normalization_radius"] =
          cfg.local_metrics.neighborhood_normalization.radius;
      artifact["neighborhood_normalization_blend"] =
          cfg.local_metrics.neighborhood_normalization.blend;
      artifact["neighborhood_normalization_mean_abs_q_delta"] =
          neighborhood_q_delta_count > 0
              ? static_cast<float>(neighborhood_q_delta_sum /
                                   static_cast<double>(neighborhood_q_delta_count))
              : 0.0f;
      artifact["neighborhood_normalization_p95_abs_q_delta"] =
          neighborhood_q_delta_p95;
      artifact["spatial_regularization_enabled"] =
          cfg.local_metrics.spatial_regularization.enabled;
      artifact["spatial_regularization_lambda"] =
          cfg.local_metrics.spatial_regularization.lambda;
      artifact["spatial_regularization_passes"] =
          cfg.local_metrics.spatial_regularization.passes;
      artifact["spatial_regularization_tile_edge_count"] =
          static_cast<uint64_t>(local_weight_regularization_summary.tile_edge_count);
      artifact["spatial_regularization_adjusted_entries"] =
          static_cast<uint64_t>(local_weight_regularization_summary.adjusted_entries);
      artifact["spatial_regularization_mean_abs_q_delta"] =
          local_weight_regularization_summary.mean_abs_q_delta;
      artifact["spatial_regularization_p95_abs_q_delta"] =
          local_weight_regularization_summary.p95_abs_q_delta;

      if (write_full_local_metrics_artifact) {
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
      } else {
        artifact["reason"] = "omitted_large_dataset";
      }

      core::write_text(run_dir / "artifacts" / "local_metrics.json", artifact.dump(2));
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
