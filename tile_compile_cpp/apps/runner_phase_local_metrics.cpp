#include "runner_phase_local_metrics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/metrics/aqmh_quality_map.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/reconstruction/local_weight_regularization.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tile_compile::runner {

namespace core = tile_compile::core;
namespace metrics = tile_compile::metrics;

namespace {

std::array<float, 3> aqmh_quantiles(std::vector<float> &values) {
  std::array<float, 3> out{};
  constexpr std::array<float, 3> quantiles{0.10f, 0.50f, 0.90f};
  struct Request {
    size_t rank;
    size_t quantile;
    bool upper;
  };
  std::array<Request, 6> requests{};
  for (size_t qi = 0; qi < quantiles.size(); ++qi) {
    const float pos = quantiles[qi] * static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    requests[2 * qi] = {lo, qi, false};
    requests[2 * qi + 1] = {std::min(values.size() - 1, lo + 1), qi, true};
  }
  std::sort(requests.begin(), requests.end(), [](const Request &a,
                                                  const Request &b) {
    return a.rank < b.rank;
  });

  std::array<float, 3> lo_values{};
  std::array<float, 3> hi_values{};
  size_t selected_begin = 0;
  size_t previous_rank = std::numeric_limits<size_t>::max();
  float selected_value = 0.0f;
  for (const Request &request : requests) {
    if (request.rank != previous_rank) {
      std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(selected_begin),
                       values.begin() + static_cast<std::ptrdiff_t>(request.rank),
                       values.end());
      selected_value = values[request.rank];
      previous_rank = request.rank;
      selected_begin = request.rank + 1;
    }
    (request.upper ? hi_values : lo_values)[request.quantile] = selected_value;
  }
  for (size_t qi = 0; qi < quantiles.size(); ++qi) {
    const float pos = quantiles[qi] * static_cast<float>(values.size() - 1);
    const float t = pos - std::floor(pos);
    out[qi] = lo_values[qi] * (1.0f - t) + hi_values[qi] * t;
  }
  return out;
}

struct AqmhFrameDiag {
  size_t frame_index = 0;
  bool written = false;
  float map_mean = std::numeric_limits<float>::quiet_NaN();
  float map_p10 = std::numeric_limits<float>::quiet_NaN();
  float map_p50 = std::numeric_limits<float>::quiet_NaN();
  float map_p90 = std::numeric_limits<float>::quiet_NaN();
  float artifact_frac = std::numeric_limits<float>::quiet_NaN();
  float sharpness_p50 = std::numeric_limits<float>::quiet_NaN();
  float snr_p50 = std::numeric_limits<float>::quiet_NaN();
  bool scene_dependent_snr = false;
  std::vector<int> omitted_scales;
};

AqmhFrameDiag summarize_aqmh_map(size_t fi, const Matrix2Df &q_map,
                                 const std::vector<uint8_t> &valid_mask,
                                 int mask_width, int mask_height,
                                 float tau_artifact) {
  AqmhFrameDiag diag;
  diag.frame_index = fi;
  std::vector<float> values;
  values.reserve(static_cast<size_t>(q_map.size()));
  double sum = 0.0;
  size_t artifact_count = 0;
  for (int y = 0; y < q_map.rows(); ++y) {
    for (int x = 0; x < q_map.cols(); ++x) {
      const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(mask_width) +
                         static_cast<size_t>(x);
      if (x >= mask_width || y >= mask_height || idx >= valid_mask.size() ||
          valid_mask[idx] == 0) {
        continue;
      }
      const float v = q_map(y, x);
      if (!std::isfinite(v))
        continue;
      values.push_back(v);
      sum += v;
      if (v <= tau_artifact)
        ++artifact_count;
    }
  }
  if (values.empty())
    return diag;
  const auto quantiles = aqmh_quantiles(values);
  diag.map_mean = static_cast<float>(sum / values.size());
  diag.map_p10 = quantiles[0];
  diag.map_p50 = quantiles[1];
  diag.map_p90 = quantiles[2];
  diag.artifact_frac =
      static_cast<float>(artifact_count) / static_cast<float>(values.size());
  return diag;
}

} // namespace

/// @brief Runs phase local metrics.
/// @details Part of the local tile-metrics and local-weights phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
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
    bool apply_normalization_to_tiles,
    core::AccelerationContext &acceleration, core::EventEmitter &emitter,
    std::ostream &log_file, std::vector<std::vector<TileMetrics>> &local_metrics,
    std::vector<std::vector<float>> &local_weights,
    std::vector<float> &tile_quality_median, std::vector<uint8_t> &tile_is_star,
    std::vector<float> &tile_fwhm_median,
    std::unique_ptr<metrics::QualityMapCache> &out_aqmh_cache, int tile_offset_x,
    int tile_offset_y) {
  (void)tile_offset_x;
  (void)tile_offset_y;
  out_aqmh_cache.reset();
  const bool compute_classic_local_metrics = !cfg.aqmh.enabled;

  const std::string phase_display_name =
      compute_classic_local_metrics ? "LOCAL_METRICS" : "AQMH_QUALITY_MAPS";

  // Phase 5: Classic local metrics or AQMH quality maps.
  emitter.phase_start(run_id, Phase::LOCAL_METRICS, phase_display_name,
                      log_file);

    local_metrics.assign(frames.size(), {});
    local_weights.assign(frames.size(), {});

  if (compute_classic_local_metrics) {
    const int local_metrics_workers = compute_adaptive_worker_count(
        cfg, frames.size(), frames, WorkerParallelProfile::CpuBound);
    std::cout << "[LOCAL_METRICS] Using " << local_metrics_workers
              << " parallel workers for " << frames.size() << " frames"
              << " cpu_workers=" << local_metrics_workers
              << " gpu=no backend=cpu"
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
  } else {
    local_metrics.clear();
    local_weights.clear();
    emitter.phase_progress(
        run_id, Phase::LOCAL_METRICS, 0.0f,
        "classic_local_metrics skipped: aqmh_independent_reconstruction",
        log_file);
  }

    std::vector<AqmhFrameDiag> aqmh_frame_diag(frames.size());
    if (cfg.aqmh.enabled) {
      const auto aqmh_acceleration =
          acceleration.selection_for(core::AccelerationPhase::aqmh_maps);
      log_file << "[AQMH] "
               << core::acceleration_selection_summary(aqmh_acceleration)
               << std::endl;
      if (!aqmh_acceleration.request_honored &&
          !aqmh_acceleration.fallback_reason.empty()) {
        emitter.warning(run_id,
                        "AQMH acceleration fallback: " +
                            aqmh_acceleration.fallback_reason,
                        log_file);
      }
      if (common_mask_width <= 0 || common_mask_height <= 0 ||
          common_valid_mask.size() !=
              static_cast<size_t>(common_mask_width) *
                  static_cast<size_t>(common_mask_height)) {
        const std::string error = "AQMH requires a valid full-canvas common mask";
        emitter.phase_end(run_id, Phase::LOCAL_METRICS, "error",
                          {{"error", error}}, log_file);
        emitter.run_end(run_id, false, "error", log_file);
        std::cerr << "Error during AQMH map computation: " << error
                  << std::endl;
        return false;
      }

      if (cfg.aqmh.cherry_pick.enabled) {
        log_file << "[AQMH] WARNING: cherry_pick mode enabled; AQMH maps are "
                    "computed independently of Classic tile weights."
                 << std::endl;
      }

      const std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(
          common_valid_mask, common_mask_width, common_mask_height);
      out_aqmh_cache = std::make_unique<metrics::QualityMapCache>(
          run_dir / "cache" / "aqmh", "luma", common_mask_width,
          common_mask_height, cfg.aqmh.pyramid, cfg.aqmh.storage, mask_hash,
          core::acceleration_backend_name(aqmh_acceleration.selected));

      const int aqmh_workers = compute_adaptive_worker_count(
          cfg, frames.size(), frames, WorkerParallelProfile::CpuBound);
      const int aqmh_effective_workers = aqmh_workers;
      std::cout << "[AQMH] Using " << aqmh_effective_workers
                << " parallel workers for quality-map computation"
                << " cpu_workers=" << aqmh_effective_workers
                << " gpu=" << (aqmh_acceleration.using_gpu ? "yes" : "no")
                << " backend="
                << core::acceleration_backend_name(aqmh_acceleration.selected)
                << std::endl;

      emitter.phase_progress(run_id, Phase::LOCAL_METRICS, 0.0f,
                             "aqmh_maps starting: 0/" +
                                 std::to_string(frames.size()) + " workers=" +
                                 std::to_string(aqmh_effective_workers) +
                                 " cpu_workers=" +
                                 std::to_string(aqmh_effective_workers) +
                                 " gpu=" +
                                 (aqmh_acceleration.using_gpu ? "yes" : "no") +
                                 " backend=" + core::acceleration_backend_name(
                                                    aqmh_acceleration.selected),
                             log_file);

      std::atomic<size_t> aqmh_next{0};
      std::atomic<size_t> aqmh_done{0};
      std::atomic<size_t> aqmh_written{0};
      std::atomic<size_t> aqmh_gpu_frames{0};
      std::atomic<size_t> aqmh_gpu_fallbacks{0};
      std::atomic<bool> aqmh_failed{false};
      std::mutex aqmh_error_mutex;
      std::mutex aqmh_progress_mutex;
      std::string aqmh_error;

      core::WorkerCudaStreams aqmh_streams(
          aqmh_acceleration.selected == core::AccelerationBackend::opencv_cuda,
          static_cast<size_t>(aqmh_effective_workers));

      auto aqmh_worker = [&](int worker_idx) {
        while (true) {
          const size_t fi = aqmh_next.fetch_add(1);
          if (fi >= frames.size())
            break;
          try {
            AqmhFrameDiag diag;
            diag.frame_index = fi;
            if (frame_has_data[fi]) {
              Matrix2Df frame = prewarped_frames.load(fi);
              if (frame.rows() == common_mask_height &&
                  frame.cols() == common_mask_width) {
                if (apply_normalization_to_tiles && fi < norm_scales.size() &&
                    frame.size() > 0) {
                  image::apply_normalization_inplace(
                      frame, norm_scales[fi], detected_mode,
                      detected_bayer_str, 0, 0);
                }
                cv::cuda::Stream *stream_ptr =
                    aqmh_streams.get(static_cast<size_t>(worker_idx));
                const auto aqmh_result = metrics::compute_aqmh_quality_map(
                    frame, common_valid_mask, common_mask_width,
                    common_mask_height, cfg.aqmh.pyramid,
                    aqmh_acceleration.selected, stream_ptr);
                if (aqmh_result.diagnostics.acceleration_used)
                  aqmh_gpu_frames.fetch_add(1, std::memory_order_relaxed);
                if (aqmh_result.diagnostics.acceleration_fallback)
                  aqmh_gpu_fallbacks.fetch_add(1, std::memory_order_relaxed);
                // Each frame has a distinct cache path. QualityMapCache keeps
                // only its shared statistics/LRU state under a short lock.
                out_aqmh_cache->write(fi, aqmh_result.q_map);
                diag = summarize_aqmh_map(
                    fi, aqmh_result.q_map, common_valid_mask,
                    common_mask_width, common_mask_height,
                    cfg.aqmh.diagnostics.tau_artifact);
                diag.written = true;
                diag.sharpness_p50 = aqmh_result.diagnostics.sharpness_p50;
                diag.snr_p50 = aqmh_result.diagnostics.snr_p50;
                diag.scene_dependent_snr =
                    aqmh_result.diagnostics.scene_dependent_snr;
                diag.omitted_scales = aqmh_result.diagnostics.omitted_scales;
                aqmh_written.fetch_add(1, std::memory_order_relaxed);
              }
            }
            aqmh_frame_diag[fi] = std::move(diag);
          } catch (const std::exception &e) {
            aqmh_failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(aqmh_error_mutex);
            if (aqmh_error.empty())
              aqmh_error = e.what();
          } catch (...) {
            aqmh_failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(aqmh_error_mutex);
            if (aqmh_error.empty())
              aqmh_error = "unknown_error";
          }

          const size_t done = aqmh_done.fetch_add(1) + 1;
          if (done % 2 == 0 || done == frames.size()) {
            const float p =
                frames.empty() ? 1.0f
                               : static_cast<float>(done) /
                                     static_cast<float>(frames.size());
            std::lock_guard<std::mutex> lock(aqmh_progress_mutex);
            emitter.phase_progress(
                run_id, Phase::LOCAL_METRICS, p,
                "aqmh_maps " + std::to_string(done) + "/" +
                    std::to_string(frames.size()) + " written=" +
                    std::to_string(aqmh_written.load(std::memory_order_relaxed)) +
                    " workers=" + std::to_string(aqmh_effective_workers) +
                    " cpu_workers=" +
                    std::to_string(aqmh_effective_workers) + " gpu=" +
                    (aqmh_acceleration.using_gpu ? "yes" : "no") +
                    " backend=" + core::acceleration_backend_name(
                                       aqmh_acceleration.selected),
                log_file);
          }
        }
      };

      if (aqmh_effective_workers > 1) {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(aqmh_effective_workers));
        for (int w = 0; w < aqmh_effective_workers; ++w)
          workers.emplace_back(aqmh_worker, w);
        for (auto &worker : workers) {
          if (worker.joinable())
            worker.join();
        }
      } else {
        aqmh_worker(0);
      }

      if (aqmh_failed.load(std::memory_order_relaxed)) {
        emitter.phase_end(
            run_id, Phase::LOCAL_METRICS, "error",
            {{"error", aqmh_error.empty() ? "unknown_error" : aqmh_error}},
            log_file);
        emitter.run_end(run_id, false, "error", log_file);
        std::cerr << "Error during AQMH map computation: "
                  << (aqmh_error.empty() ? "unknown_error" : aqmh_error)
                  << std::endl;
        return false;
      }

      core::json aqmh_artifact;
      aqmh_artifact["enabled"] = true;
      aqmh_artifact["acceleration"] =
          core::acceleration_selection_to_json(aqmh_acceleration);
      aqmh_artifact["acceleration"]["gpu_frames"] =
          aqmh_gpu_frames.load(std::memory_order_relaxed);
      aqmh_artifact["acceleration"]["gpu_fallbacks"] =
          aqmh_gpu_fallbacks.load(std::memory_order_relaxed);
      aqmh_artifact["map_stream_id"] = out_aqmh_cache->map_stream_id();
      aqmh_artifact["cache_dir"] = out_aqmh_cache->cache_dir().string();
      aqmh_artifact["full_width"] = out_aqmh_cache->full_width();
      aqmh_artifact["full_height"] = out_aqmh_cache->full_height();
      aqmh_artifact["stored_width"] = out_aqmh_cache->stored_width();
      aqmh_artifact["stored_height"] = out_aqmh_cache->stored_height();
      aqmh_artifact["dtype"] = cfg.aqmh.storage.dtype;
      aqmh_artifact["resolution_divisor"] =
          cfg.aqmh.storage.resolution_divisor;
      aqmh_artifact["frames_total"] = static_cast<uint64_t>(frames.size());
      aqmh_artifact["frames_written"] =
          static_cast<uint64_t>(aqmh_written.load(std::memory_order_relaxed));
      aqmh_artifact["diagnostics"] = core::json::array();
      for (const auto &diag : aqmh_frame_diag) {
        core::json jd;
        jd["frame_index"] = static_cast<uint64_t>(diag.frame_index);
        jd["written"] = diag.written;
        jd["map_mean"] = diag.map_mean;
        jd["map_p10"] = diag.map_p10;
        jd["map_p50"] = diag.map_p50;
        jd["map_p90"] = diag.map_p90;
        jd["artifact_frac"] = diag.artifact_frac;
        jd["sharpness_p50"] = diag.sharpness_p50;
        jd["snr_p50"] = diag.snr_p50;
        jd["scene_dependent_snr"] = diag.scene_dependent_snr;
        jd["omitted_scales"] = diag.omitted_scales;
        aqmh_artifact["diagnostics"].push_back(jd);
      }
      const auto cache_stats = out_aqmh_cache->stats();
      aqmh_artifact["cache_stats"] = {
          {"bytes_written", cache_stats.bytes_written},
          {"bytes_read", cache_stats.bytes_read},
          {"write_count", cache_stats.write_count},
          {"read_count", cache_stats.read_count},
          {"cache_hits", cache_stats.cache_hits},
          {"cache_misses", cache_stats.cache_misses},
          {"max_resident_maps_observed",
           static_cast<uint64_t>(cache_stats.max_resident_maps_observed)}};
      std::filesystem::create_directories(run_dir / "artifacts");
      core::write_text(run_dir / "artifacts" / "aqmh_metrics.json",
                       aqmh_artifact.dump(2));
    }

    if (!compute_classic_local_metrics) {
      emitter.phase_end(run_id, Phase::LOCAL_METRICS, "ok",
                        {
                            {"num_frames", static_cast<int>(frames.size())},
                            {"num_tiles", static_cast<int>(tiles_phase56.size())},
                            {"classic_tile_metrics_used", false},
                            {"aqmh_enabled", cfg.aqmh.enabled},
                        },
                        log_file);
      tile_quality_median.assign(tiles_phase56.size(), 0.0f);
      tile_is_star.assign(tiles_phase56.size(), 0u);
      tile_fwhm_median.assign(tiles_phase56.size(), 0.0f);
      return true;
    }

    std::vector<uint8_t> tile_star_flags(tiles_phase56.size(), 0);
    std::vector<std::vector<float>> local_quality_scores(
        frames.size(), std::vector<float>(tiles_phase56.size(), 0.0f));
    std::vector<std::vector<float>> local_quality_confidence(
        frames.size(), std::vector<float>(tiles_phase56.size(), 0.0f));
    std::vector<float> tile_star_support_blend(tiles_phase56.size(), 0.0f);
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
      const float star_soft_count =
          static_cast<float>(std::max(1, cfg.tile.star_soft_count));
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
        const float eta =
            std::clamp(sc_med / std::max(star_soft_count, 1.0f), 0.0f, 1.0f);
        const TileType tile_type =
            (eta >= 0.5f || sc_med >= static_cast<float>(star_thr))
                ? TileType::STAR
                : TileType::STRUCTURE;
        tile_star_flags[ti] = (tile_type == TileType::STAR) ? 1 : 0;
        tile_star_support_blend[ti] = eta;

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
          const bool star_available =
              std::isfinite(tm.fwhm) && tm.fwhm > 0.0f &&
              std::isfinite(tm.roundness) && std::isfinite(tm.contrast);
          const float energy_ratio =
              (std::isfinite(tm.noise) && std::fabs(tm.noise) > eps)
                  ? (tm.gradient_energy / tm.noise)
                  : 0.0f;
          const bool struct_available =
              std::isfinite(tm.background) && std::isfinite(energy_ratio);

          float q_before_star =
              cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_local_t[ui]) +
              cfg.local_metrics.star_mode.weights.roundness * (r_local_t[ui]) +
              cfg.local_metrics.star_mode.weights.contrast * (c_local_t[ui]);
          float q_before_struct =
              cfg.local_metrics.structure_mode.metric_weight * (en_local_t[ui]) +
              cfg.local_metrics.structure_mode.background_weight * (-b_local_t[ui]);
          float q_star =
              cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_t[ui]) +
              cfg.local_metrics.star_mode.weights.roundness * (r_t[ui]) +
              cfg.local_metrics.star_mode.weights.contrast * (c_t[ui]);
          float q_struct =
              cfg.local_metrics.structure_mode.metric_weight * (en_t[ui]) +
              cfg.local_metrics.structure_mode.background_weight * (-b_t[ui]);

          float q_before = 0.0f;
          if (star_available && struct_available) {
            q_before = eta * q_before_star + (1.0f - eta) * q_before_struct;
          } else if (star_available) {
            q_before = q_before_star;
          } else {
            q_before = q_before_struct;
          }
          q_before = clip3(q_before);

          float q = 0.0f;
          if (star_available && struct_available) {
            q = eta * q_star + (1.0f - eta) * q_struct;
          } else if (star_available) {
            q = q_star;
          } else {
            q = q_struct;
          }
          q = clip3(q);
          local_quality_scores[fi][ti] = q;
          int valid_metric_count = 0;
          if (std::isfinite(tm.fwhm) && tm.fwhm > 0.0f) {
            ++valid_metric_count;
          }
          if (std::isfinite(tm.roundness)) {
            ++valid_metric_count;
          }
          if (std::isfinite(tm.contrast)) {
            ++valid_metric_count;
          }
          if (std::isfinite(tm.background)) {
            ++valid_metric_count;
          }
          if (std::isfinite(energy_ratio)) {
            ++valid_metric_count;
          }
          local_quality_confidence[fi][ti] =
              std::clamp(static_cast<float>(valid_metric_count) / 5.0f, 0.0f,
                         1.0f);
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
      regularization_cfg.tau_local =
          cfg.local_metrics.spatial_regularization.tau_local;
      local_weight_regularization_summary =
          reconstruction::regularize_local_quality_scores(
              tiles_phase56, tile_common_valid, frame_has_data,
              regularization_cfg, &local_quality_scores,
              &local_quality_confidence);

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
          local_weights[fi][ti] = std::exp(cfg.local_metrics.k_local * q);
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
      artifact["spatial_regularization_tau_local"] =
          cfg.local_metrics.spatial_regularization.tau_local;
      artifact["tile_star_support_blend"] = core::json::array();
      for (float eta : tile_star_support_blend) {
        artifact["tile_star_support_blend"].push_back(eta);
      }
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
