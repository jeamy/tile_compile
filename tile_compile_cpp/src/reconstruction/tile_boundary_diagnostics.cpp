#include "tile_compile/reconstruction/tile_boundary_diagnostics.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace tile_compile::reconstruction {

namespace {

TileBoundaryPairDiagnostic analyze_pair(size_t lhs_index, size_t rhs_index,
                                        const Tile &lhs_tile,
                                        const Tile &rhs_tile,
                                        const Matrix2Df &lhs,
                                        const Matrix2Df &rhs,
                                        const std::vector<uint8_t> &common_valid_mask,
                                        int common_mask_width,
                                        int common_mask_height) {
  TileBoundaryPairDiagnostic out;
  out.lhs = lhs_index;
  out.rhs = rhs_index;

  if (lhs.size() <= 0 || rhs.size() <= 0) {
    return out;
  }

  const int overlap_x0 = std::max(lhs_tile.x, rhs_tile.x);
  const int overlap_y0 = std::max(lhs_tile.y, rhs_tile.y);
  const int overlap_x1 =
      std::min(lhs_tile.x + lhs_tile.width, rhs_tile.x + rhs_tile.width);
  const int overlap_y1 =
      std::min(lhs_tile.y + lhs_tile.height, rhs_tile.y + rhs_tile.height);
  const int overlap_w = overlap_x1 - overlap_x0;
  const int overlap_h = overlap_y1 - overlap_y0;
  if (overlap_w <= 0 || overlap_h <= 0) {
    return out;
  }

  std::vector<float> abs_diffs;
  abs_diffs.reserve(static_cast<size_t>(overlap_w * overlap_h));
  double signed_sum = 0.0;
  for (int oy = 0; oy < overlap_h; ++oy) {
    const int lhs_y = overlap_y0 + oy - lhs_tile.y;
    const int rhs_y = overlap_y0 + oy - rhs_tile.y;
    for (int ox = 0; ox < overlap_w; ++ox) {
      const int lhs_x = overlap_x0 + ox - lhs_tile.x;
      const int rhs_x = overlap_x0 + ox - rhs_tile.x;
      const int global_x = overlap_x0 + ox;
      const int global_y = overlap_y0 + oy;
      if (!common_valid_mask.empty()) {
        if (global_x < 0 || global_x >= common_mask_width || global_y < 0 ||
            global_y >= common_mask_height) {
          continue;
        }
        const size_t mask_idx =
            static_cast<size_t>(global_y) *
                static_cast<size_t>(common_mask_width) +
            static_cast<size_t>(global_x);
        if (mask_idx >= common_valid_mask.size() ||
            common_valid_mask[mask_idx] == 0u) {
          continue;
        }
      }
      const float lhs_v = lhs(lhs_y, lhs_x);
      const float rhs_v = rhs(rhs_y, rhs_x);
      if (!std::isfinite(lhs_v) || !std::isfinite(rhs_v)) {
        continue;
      }
      const float diff = rhs_v - lhs_v;
      signed_sum += static_cast<double>(diff);
      abs_diffs.push_back(std::fabs(diff));
    }
  }

  if (abs_diffs.empty()) {
    return out;
  }

  out.sample_count = abs_diffs.size();
  const double abs_sum = std::accumulate(abs_diffs.begin(), abs_diffs.end(), 0.0);
  out.mean_abs_diff =
      static_cast<float>(abs_sum / static_cast<double>(out.sample_count));
  out.mean_signed_diff =
      static_cast<float>(signed_sum / static_cast<double>(out.sample_count));
  std::sort(abs_diffs.begin(), abs_diffs.end());
  out.p95_abs_diff = core::percentile_from_sorted(abs_diffs, 95.0f);
  out.valid = true;
  return out;
}

} // namespace

TileBoundaryDiagnostics analyze_tile_boundaries(
    const std::vector<Tile> &tiles, const std::vector<Matrix2Df> &images,
    const std::vector<uint8_t> &tile_valid,
    const std::vector<uint8_t> &common_valid_mask, int common_mask_width,
    int common_mask_height) {
  TileBoundaryDiagnostics out;
  if (tiles.empty() || images.size() != tiles.size() ||
      tile_valid.size() != tiles.size()) {
    return out;
  }

  auto pair_key = [](int row, int col) -> long long {
    return (static_cast<long long>(row) << 32) ^
           static_cast<unsigned int>(col);
  };

  std::unordered_map<long long, size_t> tile_index_by_grid;
  tile_index_by_grid.reserve(tiles.size());
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    tile_index_by_grid[pair_key(tiles[ti].row, tiles[ti].col)] = ti;
  }

  std::vector<std::pair<size_t, size_t>> boundary_pairs;
  boundary_pairs.reserve(tiles.size() * 2);
  auto push_pair = [&](size_t lhs, size_t rhs) {
    if (lhs >= tiles.size() || rhs >= tiles.size() || lhs == rhs ||
        tile_valid[lhs] == 0u || tile_valid[rhs] == 0u) {
      return;
    }
    boundary_pairs.push_back({lhs, rhs});
  };
  for (size_t ti = 0; ti < tiles.size(); ++ti) {
    const Tile &tile = tiles[ti];
    auto right_it = tile_index_by_grid.find(pair_key(tile.row, tile.col + 1));
    if (right_it != tile_index_by_grid.end()) {
      push_pair(ti, right_it->second);
    }
    auto down_it = tile_index_by_grid.find(pair_key(tile.row + 1, tile.col));
    if (down_it != tile_index_by_grid.end()) {
      push_pair(ti, down_it->second);
    }
  }
  if (boundary_pairs.empty()) {
    for (size_t lhs = 0; lhs < tiles.size(); ++lhs) {
      if (tile_valid[lhs] == 0u) {
        continue;
      }
      for (size_t rhs = lhs + 1; rhs < tiles.size(); ++rhs) {
        if (tile_valid[rhs] == 0u) {
          continue;
        }
        const Tile &a = tiles[lhs];
        const Tile &b = tiles[rhs];
        const bool overlap = a.x < (b.x + b.width) && b.x < (a.x + a.width) &&
                             a.y < (b.y + b.height) && b.y < (a.y + a.height);
        if (overlap) {
          boundary_pairs.push_back({lhs, rhs});
        }
      }
    }
  }
  out.pair_count = boundary_pairs.size();

  std::vector<float> pair_mean_abs;
  std::vector<float> pair_p95_abs;
  std::vector<float> pair_mean_signed_abs;
  pair_mean_abs.reserve(boundary_pairs.size());
  pair_p95_abs.reserve(boundary_pairs.size());
  pair_mean_signed_abs.reserve(boundary_pairs.size());
  out.pair_diagnostics.reserve(boundary_pairs.size());

  for (const auto &[lhs, rhs] : boundary_pairs) {
    TileBoundaryPairDiagnostic pair =
        analyze_pair(lhs, rhs, tiles[lhs], tiles[rhs], images[lhs], images[rhs],
                     common_valid_mask, common_mask_width, common_mask_height);
    if (!pair.valid) {
      continue;
    }
    ++out.observed_pair_count;
    out.sample_count += pair.sample_count;
    pair_mean_abs.push_back(pair.mean_abs_diff);
    pair_p95_abs.push_back(pair.p95_abs_diff);
    pair_mean_signed_abs.push_back(std::fabs(pair.mean_signed_diff));
    out.pair_diagnostics.push_back(pair);
  }

  if (out.pair_diagnostics.empty()) {
    return out;
  }

  out.pair_mean_abs_diff_mean =
      std::accumulate(pair_mean_abs.begin(), pair_mean_abs.end(), 0.0f) /
      static_cast<float>(pair_mean_abs.size());
  out.pair_p95_abs_diff_mean =
      std::accumulate(pair_p95_abs.begin(), pair_p95_abs.end(), 0.0f) /
      static_cast<float>(pair_p95_abs.size());
  out.pair_mean_signed_diff_mean_abs =
      std::accumulate(pair_mean_signed_abs.begin(), pair_mean_signed_abs.end(),
                      0.0f) /
      static_cast<float>(pair_mean_signed_abs.size());
  std::sort(pair_mean_abs.begin(), pair_mean_abs.end());
  std::sort(pair_p95_abs.begin(), pair_p95_abs.end());
  out.pair_mean_abs_diff_p95 =
      core::percentile_from_sorted(pair_mean_abs, 95.0f);
  out.pair_p95_abs_diff_p95 =
      core::percentile_from_sorted(pair_p95_abs, 95.0f);

  std::sort(out.pair_diagnostics.begin(), out.pair_diagnostics.end(),
            [](const TileBoundaryPairDiagnostic &a,
               const TileBoundaryPairDiagnostic &b) {
              if (a.mean_abs_diff != b.mean_abs_diff) {
                return a.mean_abs_diff > b.mean_abs_diff;
              }
              if (a.p95_abs_diff != b.p95_abs_diff) {
                return a.p95_abs_diff > b.p95_abs_diff;
              }
              return a.sample_count > b.sample_count;
            });
  return out;
}

} // namespace tile_compile::reconstruction
