#include "runner_phase_aqmh_diagnostics.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_binary_diagnostics.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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

// Stream the final aqmh_metrics.json: write the base artifact (without frames)
// and append the per-frame block diagnostics from a JSONL temp file.
// This avoids holding the entire per-frame block-diagnostic tree in memory.
void stream_aqmh_metrics_json(
    const std::filesystem::path &out_path,
    const core::json &artifact,
    const std::filesystem::path &frames_jsonl_path,
    bool has_frames) {
  std::ofstream out(out_path, std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open aqmh_metrics.json for streaming write");
  }

  core::json base = artifact;
  base.erase("frames");
  std::string base_str = base.dump(2);
  if (base_str.empty() || base_str.back() != '}') {
    throw std::runtime_error("malformed base artifact JSON");
  }
  base_str.pop_back();  // remove trailing '}'
  while (!base_str.empty() &&
         (base_str.back() == '\n' || base_str.back() == '\r')) {
    base_str.pop_back();
  }

  out << base_str;
  if (has_frames) {
    out << (base.empty() ? "\n  \"frames\": [\n" : ",\n  \"frames\": [\n");

    std::ifstream in(frames_jsonl_path, std::ios::in);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (!first) out << ",\n";
      first = false;
      out << "    " << line;
    }
    out << "\n  ]\n}";
  } else {
    out << "}\n";
  }
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

  // Early exit when disabled
  if (!cfg.aqmh.diagnostics.enabled || cfg.aqmh.diagnostics.level == "none") {
    emitter.phase_start(run_id, Phase::AQMH_DIAGNOSTICS,
                        "AQMH_DIAGNOSTICS", log_file);
    emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "skipped",
                      {{"reason", "disabled_by_config"}}, log_file);
    return true;
  }

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

  const bool full_mode = cfg.aqmh.diagnostics.level == "full";

  // Block-level diagnostics and heatmaps (§6.2, §6.3): iterate over all
  // frames that have a cached Q-map and augment per-frame entries.
  // Only compute when in full mode AND per_frame_blocks is enabled.
  const auto frames_jsonl_path = run_dir / "cache" / "aqmh_block_diagnostics.jsonl";
  bool has_streamed_frames = false;

  if (full_mode && cfg.aqmh.diagnostics.per_frame_blocks &&
      q_map_cache && canvas_width > 0 && canvas_height > 0) {
    const float tau_artifact = cfg.aqmh.diagnostics.tau_artifact;
    // Use r_morph_canvas_px as block size, or fall back to derived value
    const int block_size_px = cfg.aqmh.diagnostics.r_morph_canvas_px > 0
        ? cfg.aqmh.diagnostics.r_morph_canvas_px
        : std::max(16, std::min(canvas_width, canvas_height) / 32);

    std::ofstream frames_jsonl(frames_jsonl_path,
                             std::ios::out | std::ios::trunc);
    if (!frames_jsonl) {
      emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "error",
                        {{"error", "cannot open block diagnostics temp file"}},
                        log_file);
      return false;
    }

    for (size_t fi = 0; fi < frame_has_data.size(); ++fi) {
      if (!frame_has_data[fi] || !q_map_cache->has(fi)) continue;
      try {
        const Matrix2Df q_map = q_map_cache->read(fi);
        if (q_map.rows() != canvas_height || q_map.cols() != canvas_width)
          continue;

        // Only include heatmaps if enabled
        const bool emit_heatmaps = cfg.aqmh.diagnostics.heatmaps;
        const core::json blk = compute_block_diagnostics(
            q_map, canvas_mask, canvas_width, canvas_height,
            tau_artifact, block_size_px);

        // Remove heatmap arrays if heatmaps flag is false
        core::json blk_to_store = emit_heatmaps ? blk : [&]() {
          core::json filtered = blk;
          filtered.erase("q_map_heatmap");
          filtered.erase("artifact_heatmap");
          return filtered;
        }();

        core::json jf;
        jf["frame_index"] = fi;
        jf["block_diagnostics"] = std::move(blk_to_store);
        frames_jsonl << jf.dump() << '\n';
        has_streamed_frames = true;
      } catch (const std::exception &) {
        // Skip frames with unreadable or missing maps silently.
      }
    }
    frames_jsonl.close();
  }

  // Write output based on format setting (Item 4.4.2)
  if (cfg.aqmh.diagnostics.format == "binary") {
    // Convert artifact to binary format and write
    metrics::AqmhBinaryDiagnostics bin_diag;
    bin_diag.header.canvas_width = canvas_width;
    bin_diag.header.canvas_height = canvas_height;
    bin_diag.header.frame_count = static_cast<uint32_t>(frame_has_data.size());
    bin_diag.header.block_size_px = cfg.aqmh.diagnostics.binary_block_size_px > 0
        ? cfg.aqmh.diagnostics.binary_block_size_px
        : cfg.aqmh.diagnostics.r_morph_canvas_px;
    bin_diag.header.has_heatmaps = (full_mode && cfg.aqmh.diagnostics.per_frame_blocks &&
                                   cfg.aqmh.diagnostics.heatmaps) ? 1 : 0;

    // Convert per-frame summary records from the maps-phase diagnostics array
    // (block-level diagnostics are kept in JSON only and not materialized for binary).
    if (artifact.contains("diagnostics") && artifact["diagnostics"].is_array()) {
      const auto &diag_arr = artifact["diagnostics"];
      bin_diag.frame_records.reserve(diag_arr.size());
      for (const auto &frame_json : diag_arr) {
        metrics::AqmhBinaryFrameRecord record;
        record.frame_index = frame_json.value("frame_index", 0u);
        record.map_mean = frame_json.value("map_mean", 0.0f);
        record.map_p10 = frame_json.value("map_p10", 0.0f);
        record.map_p50 = frame_json.value("map_p50", 0.0f);
        record.map_p90 = frame_json.value("map_p90", 0.0f);
        record.artifact_frac = frame_json.value("artifact_frac", 0.0f);
        record.sharpness_p50 = frame_json.value("sharpness_p50", 0.0f);
        record.snr_p50 = frame_json.value("snr_p50", 0.0f);
        record.scene_dependent_snr = frame_json.value("scene_dependent_snr", 0.0f);
        record.n_regions = frame_json.value("n_regions", 0u);
        record.global_quality = frame_json.value("global_quality", 0.0f);
        record.global_sharpness_input = frame_json.value("global_sharpness_input", 0.0f);
        record.global_snr_input = frame_json.value("global_snr_input", 0.0f);
        record.global_summary_invalid = frame_json.value("global_summary_invalid", 0u);
        bin_diag.frame_records.push_back(record);
      }
    }

    // Heatmap arrays are produced by the JSON streaming path; the binary format
    // intentionally stores only the much smaller summary records for now.

    const auto binary_path = run_dir / "artifacts" / "aqmh_metrics.bin";
    metrics::write_binary_diagnostics(binary_path, bin_diag);
  } else {
    // Default JSON format: stream the base artifact and append per-frame block
    // diagnostics from the temporary JSONL file, keeping the intermediate tree
    // out of RAM.
    stream_aqmh_metrics_json(path, artifact, frames_jsonl_path, has_streamed_frames);
  }
  emitter.phase_end(run_id, Phase::AQMH_DIAGNOSTICS, "ok",
                    {{"cherry_pick_active", reconstruction.cherry_pick_active},
                     {"unsupported_pixels", reconstruction.unsupported_pixels},
                     {"level", cfg.aqmh.diagnostics.level}},
                    log_file);
  return true;
}

} // namespace tile_compile::runner
