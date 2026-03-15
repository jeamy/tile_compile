#include "tile_compile/reconstruction/local_weight_regularization.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>

namespace tile_compile::reconstruction {

namespace {

uint64_t tile_grid_key(int row, int col) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) ^
         static_cast<uint32_t>(col);
}

} // namespace

LocalWeightRegularizationSummary regularize_local_quality_scores(
    const std::vector<Tile> &tiles, const std::vector<uint8_t> &tile_valid,
    const std::vector<uint8_t> &frame_has_data,
    const LocalWeightRegularizationConfig &cfg,
    std::vector<std::vector<float>> *quality_scores) {
  LocalWeightRegularizationSummary out;
  if (!quality_scores || !cfg.enabled || cfg.passes <= 0 || tiles.empty() ||
      tile_valid.size() != tiles.size()) {
    return out;
  }

  auto &scores = *quality_scores;
  if (scores.empty() || scores.size() != frame_has_data.size()) {
    return out;
  }

  std::unordered_map<uint64_t, size_t> tile_by_grid;
  tile_by_grid.reserve(tiles.size());
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    if (tile_valid[ti] == 0u) {
      continue;
    }
    tile_by_grid.emplace(tile_grid_key(tiles[ti].row, tiles[ti].col), ti);
  }

  std::vector<std::vector<size_t>> neighbors(tiles.size());
  auto link_neighbors = [&](size_t lhs, size_t rhs) {
    if (lhs >= tiles.size() || rhs >= tiles.size() || lhs == rhs ||
        tile_valid[lhs] == 0u || tile_valid[rhs] == 0u) {
      return;
    }
    neighbors[lhs].push_back(rhs);
    neighbors[rhs].push_back(lhs);
    ++out.tile_edge_count;
  };
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    if (tile_valid[ti] == 0u) {
      continue;
    }
    const auto &tile = tiles[ti];
    auto right_it = tile_by_grid.find(tile_grid_key(tile.row, tile.col + 1));
    if (right_it != tile_by_grid.end()) {
      link_neighbors(ti, right_it->second);
    }
    auto down_it = tile_by_grid.find(tile_grid_key(tile.row + 1, tile.col));
    if (down_it != tile_by_grid.end()) {
      link_neighbors(ti, down_it->second);
    }
  }

  if (out.tile_edge_count == 0u) {
    return out;
  }

  std::vector<float> abs_deltas;
  for (size_t fi = 0; fi < scores.size(); ++fi) {
    if (frame_has_data[fi] == 0u || scores[fi].size() != tiles.size()) {
      continue;
    }
    std::vector<float> current = scores[fi];
    std::vector<float> next = current;
    for (int pass = 0; pass < cfg.passes; ++pass) {
      for (size_t ti = 0; ti < tiles.size(); ++ti) {
        if (tile_valid[ti] == 0u || !std::isfinite(current[ti]) ||
            neighbors[ti].empty()) {
          continue;
        }
        double neighbor_sum = 0.0;
        size_t neighbor_count = 0u;
        for (size_t ni : neighbors[ti]) {
          if (ni >= current.size() || tile_valid[ni] == 0u ||
              !std::isfinite(current[ni])) {
            continue;
          }
          neighbor_sum += current[ni];
          ++neighbor_count;
        }
        if (neighbor_count == 0u) {
          continue;
        }
        const float neighbor_mean =
            static_cast<float>(neighbor_sum / static_cast<double>(neighbor_count));
        next[ti] = (1.0f - cfg.lambda) * current[ti] + cfg.lambda * neighbor_mean;
      }
      current.swap(next);
    }
    for (size_t ti = 0; ti < tiles.size(); ++ti) {
      if (tile_valid[ti] == 0u || ti >= scores[fi].size() ||
          !std::isfinite(scores[fi][ti]) || !std::isfinite(current[ti])) {
        continue;
      }
      const float abs_delta = std::fabs(current[ti] - scores[fi][ti]);
      if (abs_delta > 0.0f) {
        ++out.adjusted_entries;
        abs_deltas.push_back(abs_delta);
      }
      scores[fi][ti] = current[ti];
    }
  }

  if (!abs_deltas.empty()) {
    out.mean_abs_q_delta =
        std::accumulate(abs_deltas.begin(), abs_deltas.end(), 0.0f) /
        static_cast<float>(abs_deltas.size());
    std::sort(abs_deltas.begin(), abs_deltas.end());
    out.p95_abs_q_delta = core::percentile_from_sorted(abs_deltas, 95.0f);
  }
  return out;
}

} // namespace tile_compile::reconstruction
