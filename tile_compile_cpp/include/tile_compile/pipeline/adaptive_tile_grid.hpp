#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <functional>
#include <ostream>
#include <vector>

namespace tile_compile::pipeline {

struct WarpGradientField {
    // Coarse gradient magnitude grid (grid_h x grid_w)
    Matrix2Df grid;
    int probe_window = 256;
    int step = 128;
    int grid_h = 0;
    int grid_w = 0;
    std::vector<int> probe_indices;

    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean_val = 0.0f;
};

WarpGradientField compute_warp_gradient_field(const std::vector<fs::path>& frame_paths,
                                             int image_width,
                                             int image_height,
                                             int probe_window,
                                             int num_probe_frames,
                                             std::ostream* progress_out = nullptr,
                                             std::function<void(float)> progress_cb = nullptr);

std::vector<Tile> build_initial_tile_grid(int image_width,
                                         int image_height,
                                         int tile_size,
                                         float overlap_fraction);

std::vector<Tile> build_adaptive_tile_grid(int image_width,
                                          int image_height,
                                          const config::Config& cfg,
                                          const WarpGradientField* gradient_field);

std::vector<Tile> build_hierarchical_tile_grid(int image_width,
                                              int image_height,
                                              const config::Config& cfg,
                                              const WarpGradientField* gradient_field);

} // namespace tile_compile::pipeline
