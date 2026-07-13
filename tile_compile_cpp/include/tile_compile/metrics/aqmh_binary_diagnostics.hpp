#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

namespace tile_compile::metrics {

/// Binary diagnostics format (AQDB - AQMH Diagnostics Binary)
/// Compact binary alternative to JSON diagnostics to reduce I/O and load time
/// Item 4.4: Replace JSON Diagnostics with Compact Binary Format

struct AqmhBinaryDiagnosticsHeader {
  char magic[4] = {'A', 'Q', 'D', 'B'};  // "AQDB"
  uint32_t version = 1;
  uint32_t frame_count = 0;
  uint32_t canvas_width = 0;
  uint32_t canvas_height = 0;
  uint32_t block_size_px = 6;  // Default, can be overridden
  uint8_t has_heatmaps = 0;     // 0 or 1
  uint8_t reserved[3] = {0, 0, 0};
};

struct AqmhBinaryFrameRecord {
  uint32_t frame_index;
  float map_mean;
  float map_p10;
  float map_p50;
  float map_p90;
  float artifact_frac;
  float sharpness_p50;
  float snr_p50;
  float scene_dependent_snr;
  uint32_t n_regions;
  float global_quality;
  float global_sharpness_input;
  float global_snr_input;
  uint8_t global_summary_invalid;
  uint8_t reserved[3] = {0, 0, 0};
};

struct AqmhBinaryDiagnostics {
  AqmhBinaryDiagnosticsHeader header;
  std::vector<AqmhBinaryFrameRecord> frame_records;
  // Heatmap arrays (if has_heatmaps): each is block_grid_width * block_grid_height float32
  std::vector<std::vector<float>> heatmap_arrays;
  // Heatmap array names in order:
  // 0: aqmh_q_median
  // 1: aqmh_q_p10
  // 2: aqmh_q_p90
  // 3: aqmh_artifact_frac
  // 4: q_map_heatmap
  // 5: artifact_heatmap
};

/// Write binary diagnostics to file
void write_binary_diagnostics(
    const std::filesystem::path &path,
    const AqmhBinaryDiagnostics &diag);

/// Read binary diagnostics from file
AqmhBinaryDiagnostics read_binary_diagnostics(
    const std::filesystem::path &path);

/// Compute block grid dimensions from canvas dimensions and block size
std::pair<int, int> compute_block_grid(
    int canvas_width, int canvas_height, int block_size_px);

} // namespace tile_compile::metrics
