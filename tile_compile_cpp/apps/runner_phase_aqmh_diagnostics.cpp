#include "runner_phase_aqmh_diagnostics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tile_compile::runner {

namespace {

// Compute percentile from a sorted vector (linear interpolation, §1.4).
float sorted_percentile(const std::vector<float> &sorted, double q) {
  if (sorted.empty()) return std::numeric_limits<float>::quiet_NaN();
  const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  const float t = static_cast<float>(pos - static_cast<double>(lo));
  return sorted[lo] * (1.0f - t) + sorted[hi] * t;
}

// Build per-tile block-level diagnostics and heatmap arrays for a single
// Q-map frame. Block grid size is given by block_size_px (canvas pixels).
// Returns JSON object with aqmh_q_median, aqmh_q_p10, aqmh_q_p90,
// aqmh_artifact_frac, and q_map_heatmap/artifact_heatmap arrays.
core::json compute_block_diagnostics(
    const Matrix2Df &q_map,
    const std::vector<uint8_t> &canvas_mask,
    int canvas_width, int canvas_height,
    float tau_artifact, int block_size_px) {

  const int bw = std::max(1, (canvas_width  + block_size_px - 1) / block_size_px);
  const int bh = std::max(1, (canvas_height + block_size_px - 1) / block_size_px);

  core::json q_med_arr  = core::json::array();
  core::json q_p10_arr  = core::json::array();
  core::json q_p90_arr  = core::json::array();
  core::json art_arr    = core::json::array();
  core::json heatmap_arr = core::json::array();  // mean Q per block
  core::json art_heat_arr = core::json::array(); // artifact_frac per block

  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      const int y0 = by * block_size_px;
      const int x0 = bx * block_size_px;
      const int y1 = std::min(canvas_height, y0 + block_size_px);
      const int x1 = std::min(canvas_width,  x0 + block_size_px);

      std::vector<float> vals;
      vals.reserve(static_cast<size_t>((y1 - y0) * (x1 - x0)));
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const size_t idx = static_cast<size_t>(y * canvas_width + x);
          if (!canvas_mask.empty() && !canvas_mask[idx]) continue;
          const float v = q_map(y, x);
          if (std::isfinite(v)) vals.push_back(v);
        }
      }

      if (vals.empty()) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        q_med_arr.push_back(nan); q_p10_arr.push_back(nan);
        q_p90_arr.push_back(nan); art_arr.push_back(nan);
        heatmap_arr.push_back(nan); art_heat_arr.push_back(nan);
        continue;
      }

      std::sort(vals.begin(), vals.end());
      const float med  = sorted_percentile(vals, 0.50);
      const float p10  = sorted_percentile(vals, 0.10);
      const float p90  = sorted_percentile(vals, 0.90);
      const float mean = static_cast<float>(
          std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size());
      const float afrac = static_cast<float>(
          std::count_if(vals.begin(), vals.end(),
                        [tau_artifact](float v) { return v < tau_artifact; })) /
          static_cast<float>(vals.size());

      q_med_arr.push_back(med);  q_p10_arr.push_back(p10);
      q_p90_arr.push_back(p90); art_arr.push_back(afrac);
      heatmap_arr.push_back(mean); art_heat_arr.push_back(afrac);
    }
  }

  core::json result;
  result["block_grid_width"]  = bw;
  result["block_grid_height"] = bh;
  result["block_size_px"]     = block_size_px;
  result["aqmh_q_median"]     = std::move(q_med_arr);
  result["aqmh_q_p10"]        = std::move(q_p10_arr);
  result["aqmh_q_p90"]        = std::move(q_p90_arr);
  result["aqmh_artifact_frac"]= std::move(art_arr);
  result["q_map_heatmap"]     = std::move(heatmap_arr);
  result["artifact_heatmap"]  = std::move(art_heat_arr);
  return result;
}

} // namespace

bool run_phase_aqmh_diagnostics(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const reconstruction::AqmhReconstructionResult &reconstruction,
    metrics::QualityMapCache *q_map_cache,
    const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &frame_has_data,
    int canvas_width, int canvas_height,
    core::EventEmitter &emitter, std::ostream &log_file) {

  emitter.phase_start(run_id, Phase::AQMH_DIAGNOSTICS,
                      "AQMH_DIAGNOSTICS", log_file);
  const auto path = run_dir / "artifacts" / "aqmh_metrics.json";
  core::json artifact = core::json::object();
  try {
    if (std::filesystem::exists(path))
      artifact = core::json::parse(core::read_text(path));
  } catch (const std::exception &e) {
    emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "error",
                      {{"error", e.what()}}, log_file);
    return false;
  }

  // Run-level cherry-pick fields (§6.1 run-level).
  artifact["cherry_pick_configured"]     = cfg.aqmh.cherry_pick.enabled;
  artifact["cherry_pick_forced_disabled"]= reconstruction.cherry_pick_forced_disabled;
  artifact["cherry_pick_active"]         = reconstruction.cherry_pick_active;
  artifact["k_nominal_median"]           = reconstruction.k_nominal_median;
  artifact["k_effective_p10"]            = reconstruction.k_effective_p10;
  artifact["k_effective_p50"]            = reconstruction.k_effective_p50;
  artifact["k_effective_p90"]            = reconstruction.k_effective_p90;
  artifact["low_rank_separation"]        = reconstruction.low_rank_separation;
  artifact["unsupported_pixels"]         = reconstruction.unsupported_pixels;
  artifact["zero_veto_pixels"]           = reconstruction.zero_veto_pixels;
  artifact["missing_map_samples"]        = reconstruction.missing_map_samples;
  artifact["numerical_guard_pixels"]     = reconstruction.numerical_guard_pixels;

  // Block-level diagnostics and heatmaps (§6.2, §6.3): iterate over all
  // frames that have a cached Q-map and augment per-frame entries.
  if (q_map_cache && canvas_width > 0 && canvas_height > 0) {
    const float tau_artifact = cfg.aqmh.diagnostics.tau_artifact;
    // Use a ~64px block for the heatmap grid (configurable via block_size_px
    // derived from canvas dimensions; no new config parameter needed).
    const int block_size_px = std::max(
        16, std::min(canvas_width, canvas_height) / 32);

    auto &frames_arr = artifact["frames"];
    if (!frames_arr.is_array()) frames_arr = core::json::array();

    for (size_t fi = 0; fi < frame_has_data.size(); ++fi) {
      if (!frame_has_data[fi] || !q_map_cache->has(fi)) continue;
      try {
        const Matrix2Df q_map = q_map_cache->read(fi);
        if (q_map.rows() != canvas_height || q_map.cols() != canvas_width)
          continue;

        const core::json blk = compute_block_diagnostics(
            q_map, canvas_mask, canvas_width, canvas_height,
            tau_artifact, block_size_px);

        // Find or create the per-frame entry and merge block diagnostics.
        bool found = false;
        for (auto &jf : frames_arr) {
          if (jf.contains("frame_index") &&
              jf["frame_index"].get<size_t>() == fi) {
            jf["block_diagnostics"] = blk;
            found = true;
            break;
          }
        }
        if (!found) {
          core::json jf;
          jf["frame_index"] = fi;
          jf["block_diagnostics"] = blk;
          frames_arr.push_back(std::move(jf));
        }
      } catch (const std::exception &) {
        // Skip frames with unreadable or missing maps silently.
      }
    }
  }

  core::write_text(path, artifact.dump(2));
  emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "ok",
                    {{"cherry_pick_active", reconstruction.cherry_pick_active},
                     {"unsupported_pixels", reconstruction.unsupported_pixels}},
                    log_file);
  return true;
}

} // namespace tile_compile::runner
