// Binary diagnostics format implementation
// Item 4.4: Replace JSON Diagnostics with Compact Binary Format

#include "tile_compile/metrics/aqmh_binary_diagnostics.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace tile_compile::metrics {

namespace {

// Number of heatmap arrays in AQDB format
constexpr int NUM_HEATMAP_ARRAYS = 6;

// Heatmap array names (for documentation)
constexpr const char* HEATMAP_ARRAY_NAMES[NUM_HEATMAP_ARRAYS] = {
    "aqmh_q_median",
    "aqmh_q_p10",
    "aqmh_q_p90",
    "aqmh_artifact_frac",
    "q_map_heatmap",
    "artifact_heatmap"
};

} // namespace

std::pair<int, int> compute_block_grid(
    int canvas_width, int canvas_height, int block_size_px) {
  const int block_grid_width = (canvas_width + block_size_px - 1) / block_size_px;
  const int block_grid_height = (canvas_height + block_size_px - 1) / block_size_px;
  return {block_grid_width, block_grid_height};
}

void write_binary_diagnostics(
    const std::filesystem::path &path,
    const AqmhBinaryDiagnostics &diag) {

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Unable to open file for writing: " + path.string());
  }

  // Write header
  out.write(reinterpret_cast<const char*>(&diag.header), sizeof(AqmhBinaryDiagnosticsHeader));

  // Write frame records
  const uint32_t num_frames = static_cast<uint32_t>(diag.frame_records.size());
  out.write(reinterpret_cast<const char*>(&num_frames), sizeof(uint32_t));
  for (const auto &record : diag.frame_records) {
    out.write(reinterpret_cast<const char*>(&record), sizeof(AqmhBinaryFrameRecord));
  }

  // Write heatmap arrays
  if (diag.header.has_heatmaps) {
    const uint8_t num_heatmaps = static_cast<uint8_t>(diag.heatmap_arrays.size());
    out.write(reinterpret_cast<const char*>(&num_heatmaps), sizeof(uint8_t));
    
    const auto block_grid = compute_block_grid(
        diag.header.canvas_width, diag.header.canvas_height, diag.header.block_size_px);
    const uint32_t grid_width = static_cast<uint32_t>(block_grid.first);
    const uint32_t grid_height = static_cast<uint32_t>(block_grid.second);
    
    for (const auto &array : diag.heatmap_arrays) {
      // Write dimensions
      const uint32_t arr_width = static_cast<uint32_t>(block_grid.first);
      const uint32_t arr_height = static_cast<uint32_t>(block_grid.second);
      out.write(reinterpret_cast<const char*>(&arr_width), sizeof(uint32_t));
      out.write(reinterpret_cast<const char*>(&arr_height), sizeof(uint32_t));
      
      // Write data
      const size_t data_bytes = array.size() * sizeof(float);
      out.write(reinterpret_cast<const char*>(array.data()), data_bytes);
    }
  } else {
    const uint8_t no_heatmaps = 0;
    out.write(reinterpret_cast<const char*>(&no_heatmaps), sizeof(uint8_t));
  }

  if (!out) {
    throw std::runtime_error("Error writing binary diagnostics to: " + path.string());
  }
}

AqmhBinaryDiagnostics read_binary_diagnostics(
    const std::filesystem::path &path) {

  AqmhBinaryDiagnostics diag;
  
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open file for reading: " + path.string());
  }

  // Read and verify header
  in.read(reinterpret_cast<char*>(&diag.header), sizeof(AqmhBinaryDiagnosticsHeader));
  
  if (diag.header.magic[0] != 'A' || diag.header.magic[1] != 'Q' ||
      diag.header.magic[2] != 'D' || diag.header.magic[3] != 'B') {
    throw std::runtime_error("Invalid AQDB magic in file: " + path.string());
  }
  
  if (diag.header.version != 1) {
    throw std::runtime_error("Unsupported AQDB version: " + std::to_string(diag.header.version));
  }

  // Read frame records
  uint32_t num_frames = 0;
  in.read(reinterpret_cast<char*>(&num_frames), sizeof(uint32_t));
  diag.frame_records.resize(num_frames);
  for (auto &record : diag.frame_records) {
    in.read(reinterpret_cast<char*>(&record), sizeof(AqmhBinaryFrameRecord));
  }

  // Read heatmap arrays
  uint8_t num_heatmaps = 0;
  in.read(reinterpret_cast<char*>(&num_heatmaps), sizeof(uint8_t));
  
  if (num_heatmaps > 0) {
    diag.header.has_heatmaps = 1;
    diag.heatmap_arrays.resize(num_heatmaps);
    
    for (auto &array : diag.heatmap_arrays) {
      uint32_t arr_width = 0, arr_height = 0;
      in.read(reinterpret_cast<char*>(&arr_width), sizeof(uint32_t));
      in.read(reinterpret_cast<char*>(&arr_height), sizeof(uint32_t));
      
      const size_t num_elements = static_cast<size_t>(arr_width) * arr_height;
      array.resize(num_elements);
      in.read(reinterpret_cast<char*>(array.data()), num_elements * sizeof(float));
    }
  }

  if (!in) {
    throw std::runtime_error("Error reading binary diagnostics from: " + path.string());
  }

  return diag;
}

} // namespace tile_compile::metrics
