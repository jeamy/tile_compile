#include "runner_phase_aqmh_reconstruction.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"

// Only include GPU headers if available
#if TILE_COMPILE_WITH_CUDA
#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"
#endif

#include "tile_compile/reconstruction/aqmh_reconstruction_opencl.hpp"

#include <opencv2/core.hpp>

#include <iostream>

namespace tile_compile::runner {

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

  // Rollout safety gate for GPU reconstruction (Item 4.1.4)
  const std::string gpu_recon_mode = cfg.aqmh.reconstruction.gpu_reconstruction;
  bool use_gpu_reconstruction = false;
  if (gpu_recon_mode == "force") {
    use_gpu_reconstruction = true;
  } else if (gpu_recon_mode == "auto") {
    use_gpu_reconstruction = aqmh_reconstruction_acceleration.request_honored &&
                           aqmh_reconstruction_acceleration.selected != core::AccelerationBackend::cpu;
  }

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

  // Backend dispatch for AQMH reconstruction (Item 4.1.4 / 4.2.1)
  reconstruction::AqmhReconstructionResult aqmh_recon;
  std::string execution_backend_str;
  bool acceleration_used = false;
  bool acceleration_fallback = aqmh_reconstruction_acceleration.using_gpu;

  if (use_gpu_reconstruction) {
#if TILE_COMPILE_WITH_CUDA
    if (aqmh_reconstruction_acceleration.selected == core::AccelerationBackend::cuda &&
        aqmh_reconstruction_acceleration.request_honored) {
      log_file << "[AQMH_RECONSTRUCTION] Using native CUDA backend" << std::endl;
      try {
        aqmh_recon = reconstruction::reconstruct_aqmh_weighted_cuda(
            frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
            common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
            aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
            progress_callback);
        if (aqmh_recon.output.rows() == canvas_height && aqmh_recon.output.cols() == canvas_width) {
          execution_backend_str = "cuda_native";
          acceleration_used = true;
          acceleration_fallback = false;
        } else {
          // CUDA failed, fall back to CPU
          log_file << "[AQMH_RECONSTRUCTION] CUDA backend failed, falling back to CPU" << std::endl;
          aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
              frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
              common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
              aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
              progress_callback);
          execution_backend_str = "cuda_native_fallback";
          acceleration_used = true;
          acceleration_fallback = true;
        }
      } catch (...) {
        log_file << "[AQMH_RECONSTRUCTION] CUDA backend exception, falling back to CPU" << std::endl;
        aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
            frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
            common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
            aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
            progress_callback);
        execution_backend_str = "cuda_native_fallback";
        acceleration_used = true;
        acceleration_fallback = true;
      }
    } else
#endif // TILE_COMPILE_WITH_CUDA
    if (aqmh_reconstruction_acceleration.selected == core::AccelerationBackend::opencv_opencl &&
               aqmh_reconstruction_acceleration.request_honored) {
      log_file << "[AQMH_RECONSTRUCTION] Using OpenCL backend" << std::endl;
      try {
        aqmh_recon = reconstruction::reconstruct_aqmh_weighted_opencl(
            frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
            common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
            aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
            progress_callback);
        if (aqmh_recon.output.rows() == canvas_height && aqmh_recon.output.cols() == canvas_width) {
          execution_backend_str = "opencl";
          acceleration_used = true;
          acceleration_fallback = false;
        } else {
          // OpenCL failed, fall back to CPU
          log_file << "[AQMH_RECONSTRUCTION] OpenCL backend failed, falling back to CPU" << std::endl;
          aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
              frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
              common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
              aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
              progress_callback);
          execution_backend_str = "opencl_fallback";
          acceleration_used = true;
          acceleration_fallback = true;
        }
      } catch (...) {
        log_file << "[AQMH_RECONSTRUCTION] OpenCL backend exception, falling back to CPU" << std::endl;
        aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
            frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
            common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
            aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
            progress_callback);
        execution_backend_str = "opencl_fallback";
        acceleration_used = true;
        acceleration_fallback = true;
      }
    } else {
      log_file << "[AQMH_RECONSTRUCTION] GPU backend selected but not available, using CPU" << std::endl;
      aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
          frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
          common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
          aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
          progress_callback);
      execution_backend_str = "cpu_exact_v0_2";
      acceleration_used = false;
      acceleration_fallback = true;
    }
  } else {
    // GPU reconstruction disabled or not requested - use CPU path
    std::cout << "[AQMH] Running independent pixel-wise reconstruction for "
              << frames.size() << " frame slots cpu_workers="
              << aqmh_recon_cfg.parallel_workers
              << " backend=cpu_exact_v0_2 region_streaming=yes"
              << std::endl;
    aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
        frames.size(), aqmh_frame_loader, aqmh_cache.get(), aqmh_global_weights,
        common_valid_mask, canvas_width, canvas_height, aqmh_recon_cfg,
        aqmh_mask_loader, aqmh_frame_region_loader, aqmh_mask_region_loader,
        progress_callback);
    execution_backend_str = "cpu_exact_v0_2";
    acceleration_used = false;
    acceleration_fallback = aqmh_reconstruction_acceleration.using_gpu;
  }

  // Continue with original code: out.control_validation = 
  out.control_validation =
      reconstruction::compare_aqmh_to_uniform_control(
          aqmh_recon.output, aqmh_recon.uniform_control_output);

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
  artifact["prefetch_fallback"] = prefetch_fallback;
  artifact["clip_sigma"] = aqmh_recon_cfg.clip_sigma;
  artifact["clip_iterations"] = aqmh_recon_cfg.clip_iterations;
  artifact["min_fraction"] = aqmh_recon_cfg.min_fraction;
  artifact["min_n_eff"] = aqmh_recon_cfg.min_n_eff;
  artifact["classic_tile_weights_used"] = false;
  artifact["fallback_to_classic"] = false;
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
          {"classic_tile_weights_used", false},
          {"cherry_pick_enabled", cfg.aqmh.cherry_pick.enabled},
          {"cherry_pick_active_frac", aqmh_recon.cherry_pick_active_frac},
          {"cherry_pick_mean_k", aqmh_recon.cherry_pick_mean_k},
      },
      log_file);
  return true;
}

} // namespace tile_compile::runner
