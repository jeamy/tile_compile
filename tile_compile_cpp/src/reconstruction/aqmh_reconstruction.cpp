#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/aqmh_cherry_pick.hpp"
#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tile_compile::reconstruction {
namespace {

bool canvas_valid(const std::vector<uint8_t> &mask, int width, int height,
                  int x, int y) {
  if (x < 0 || y < 0 || x >= width || y >= height) return false;
  return mask.empty() ||
         (mask.size() == static_cast<size_t>(width * height) &&
          mask[static_cast<size_t>(y * width + x)] != 0u);
}

float global_weight(const VectorXf &weights, size_t fi) {
  if (fi >= static_cast<size_t>(weights.size())) return 0.0f;
  const float value = weights[static_cast<Eigen::Index>(fi)];
  return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

float quantile(std::vector<float> values, float q) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const double pos = std::clamp<double>(q, 0.0, 1.0) * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  const float t = static_cast<float>(pos - lo);
  return values[lo] * (1.0f - t) + values[hi] * t;
}

} // namespace

AqmhUniformControlResult compute_aqmh_uniform_control(
    size_t frame_count, const AqmhFrameLoader &load_frame,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const AqmhMaskLoader &load_frame_valid_mask,
    const AqmhFrameRegionLoader &load_frame_region,
    const AqmhMaskRegionLoader &load_frame_valid_mask_region) {
  AqmhUniformControlResult result;
  result.output = Matrix2Df::Zero(height, width);
  result.valid_mask.assign(static_cast<size_t>(std::max(0, width)) *
                               static_cast<size_t>(std::max(0, height)),
                           0u);
  if (!load_frame || frame_count == 0 || width <= 0 || height <= 0) {
    return result;
  }

  constexpr int control_chunk_rows = 128;
  const int chunk_count = (height + control_chunk_rows - 1) / control_chunk_rows;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    const int y0 = chunk_idx * control_chunk_rows;
    const int rows = std::min(control_chunk_rows, height - y0);
    const size_t pixel_count = static_cast<size_t>(rows) * width;
    std::vector<double> sums(pixel_count, 0.0);
    std::vector<uint32_t> counts(pixel_count, 0u);
    for (size_t fi = 0; fi < frame_count; ++fi) {
      Matrix2Df frame;
      const bool frame_ok = load_frame_region
          ? load_frame_region(fi, y0, rows, frame)
          : load_frame(fi, frame);
      if (!frame_ok || frame.cols() != width ||
          frame.rows() != (load_frame_region ? rows : height)) {
        continue;
      }
      std::vector<uint8_t> frame_mask;
      const bool use_region_mask = static_cast<bool>(load_frame_valid_mask_region);
      const bool mask_ok = use_region_mask
          ? load_frame_valid_mask_region(fi, y0, rows, frame_mask)
          : (!load_frame_valid_mask || load_frame_valid_mask(fi, frame_mask));
      const size_t expected_mask = static_cast<size_t>(width) *
                                   (use_region_mask ? rows : height);
      if ((load_frame_valid_mask || use_region_mask) &&
          (!mask_ok || frame_mask.size() != expected_mask)) {
        continue;
      }
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t local_i = static_cast<size_t>(yy * width + x);
          const size_t mask_i = use_region_mask
              ? local_i : static_cast<size_t>(y * width + x);
          const int source_y = load_frame_region ? yy : y;
          const float value = frame(source_y, x);
          if (!canvas_valid(canvas_mask, width, height, x, y) ||
              (!frame_mask.empty() && frame_mask[mask_i] == 0u) ||
              !std::isfinite(value)) {
            continue;
          }
          sums[local_i] += value;
          ++counts[local_i];
        }
      }
    }
    for (int yy = 0; yy < rows; ++yy) {
      for (int x = 0; x < width; ++x) {
        const size_t local_i = static_cast<size_t>(yy * width + x);
        if (counts[local_i] == 0u) continue;
        const int y = y0 + yy;
        result.output(y, x) = static_cast<float>(
            sums[local_i] / static_cast<double>(counts[local_i]));
        result.valid_mask[static_cast<size_t>(y * width + x)] = 1u;
      }
    }
  }
  return result;
}

AqmhReconstructionResult reconstruct_aqmh_weighted(
    size_t frame_count, const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask,
    const AqmhFrameRegionLoader &load_frame_region,
    const AqmhMaskRegionLoader &load_frame_valid_mask_region,
    const AqmhProgressCallback &progress) {
  AqmhReconstructionResult result;
  result.output = Matrix2Df::Zero(height, width);
  result.weight_sum = Matrix2Df::Zero(height, width);
  if (cfg.compute_uniform_control) {
    result.uniform_control_output = Matrix2Df::Zero(height, width);
    result.uniform_control_valid_mask.assign(
        static_cast<size_t>(std::max(0, width)) *
            static_cast<size_t>(std::max(0, height)),
        0u);
  }
  if (!load_frame || !q_map_cache || frame_count == 0 || width <= 0 || height <= 0)
    return result;

  // Validate each full M_f digest once. The previous slab loop re-read and
  // re-hashed a full mask for every frame and every slab.
  std::vector<uint8_t> frame_mask_compatible(frame_count, 1u);
  if (load_frame_valid_mask) {
    for (size_t fi = 0; fi < frame_count; ++fi) {
      std::vector<uint8_t> full_mask;
      frame_mask_compatible[fi] =
          load_frame_valid_mask(fi, full_mask) &&
          full_mask.size() == static_cast<size_t>(width * height) &&
          q_map_cache->source_mask_hash(fi) ==
              tile_compile::core::sha256_bytes(full_mask);
    }
  }

  bool cherry_enabled = cfg.cherry_pick;
  if (cherry_enabled) {
    std::vector<float> nominal_values;
    nominal_values.reserve(static_cast<size_t>(width * height));
    if (load_frame_region && load_frame_valid_mask_region) {
      constexpr int gate_rows = 128;
      for (int y0 = 0; y0 < height; y0 += gate_rows) {
        const int rows = std::min(gate_rows, height - y0);
        const size_t count = static_cast<size_t>(rows) * width;
        std::vector<uint16_t> rankable(count, 0u);
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 4)
        #endif
        for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(frame_count); ++fi_ptr) {
          const size_t fi = static_cast<size_t>(fi_ptr);
          if (frame_mask_compatible[fi] == 0u ||
              !(global_weight(global_weights, fi) > 0.0f)) continue;
          Matrix2Df q = q_map_cache->read_region(fi, y0, rows);
          std::vector<uint8_t> fm;
          if (q.rows() != rows || q.cols() != width ||
              !load_frame_valid_mask_region(fi, y0, rows, fm) ||
              fm.size() != count) continue;
          for (size_t i = 0; i < count; ++i)
            if (fm[i] != 0u && q.data()[i] > 0.0f) {
              #if defined(_OPENMP)
              #pragma omp atomic
              #endif
              rankable[i] += 1u;
            }
        }
        for (size_t i = 0; i < count; ++i) {
          const size_t full_i = static_cast<size_t>(y0) * width + i;
          if (!canvas_mask.empty() && canvas_mask[full_i] == 0u) continue;
          const int n = rankable[i];
          nominal_values.push_back(cfg.cherry_pick_mode == "auto_reject"
              ? static_cast<float>(n)
              : static_cast<float>(aqmh_k_nominal(
                    n, aqmh_effective_k_frac(n, cfg.cherry_pick_k_frac,
                                             cfg.tiered_k_frac))));
        }
      }
    } else {
      Matrix2Df rankable = Matrix2Df::Zero(height, width);
      for (size_t fi = 0; fi < frame_count; ++fi) {
        if (frame_mask_compatible[fi] == 0u) continue;
        Matrix2Df frame;
        if (!load_frame(fi, frame) || frame.rows() != height ||
            frame.cols() != width) continue;
        Matrix2Df q = q_map_cache->read_cached(fi);
        if (q.rows() != height || q.cols() != width) continue;
        std::vector<uint8_t> fm;
        if (load_frame_valid_mask && !load_frame_valid_mask(fi, fm)) continue;
        const float gw = global_weight(global_weights, fi);
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y * width + x);
            if (canvas_valid(canvas_mask, width, height, x, y) &&
                (fm.empty() || fm[i] != 0u) && std::isfinite(frame(y, x)) &&
                q(y, x) > 0.0f && gw > 0.0f) rankable(y, x) += 1.0f;
          }
      }
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          if (!canvas_valid(canvas_mask, width, height, x, y)) continue;
          const int n = static_cast<int>(rankable(y, x));
          nominal_values.push_back(cfg.cherry_pick_mode == "auto_reject"
              ? static_cast<float>(n)
              : static_cast<float>(aqmh_k_nominal(
                    n, aqmh_effective_k_frac(n, cfg.cherry_pick_k_frac,
                                             cfg.tiered_k_frac))));
        }
      }
    result.k_nominal_median = quantile(std::move(nominal_values), 0.5f);
    if (result.k_nominal_median < cfg.cherry_pick_k_min_required) {
      cherry_enabled = false;
      result.cherry_pick_forced_disabled = true;
    }
  }

  result.cherry_pick_k_map = cherry_enabled ? Matrix2Df::Zero(height, width)
                                             : Matrix2Df();
  std::vector<float> effective_k;
  std::vector<float> margins;
  uint64_t cherry_active_pixels = 0;
  uint64_t canvas_pixels = 0;

  // A larger row slab amortizes file-open and seek overhead. Region loaders
  // keep physical I/O proportional to N*pixels instead of N*full_frame*slabs.
  // Compact pixel-major SoA: one value and one weight per frame slot. Frame
  // index is the slot itself; score equals the non-uniform weight in the main
  // v0.2 path. This avoids one heap allocation and a 24-byte AoS element for
  // every valid pixel/frame pair.
  constexpr size_t bytes_per_sample = sizeof(float) * 2u;
  int chunk_rows;
  if (cfg.chunk_rows > 0) {
    chunk_rows = std::min(height, cfg.chunk_rows);
  } else {
    const size_t target_mb = static_cast<size_t>(std::clamp(
        static_cast<int>(cfg.memory_budget_mb / 2), 128, 1536));
    const size_t target_bytes = target_mb * 1024u * 1024u;
    const size_t denom = std::max<size_t>(1, static_cast<size_t>(width) *
                                               frame_count * bytes_per_sample);
    chunk_rows = std::max(1, std::min(height, static_cast<int>(target_bytes / denom)));
  }
  result.chunk_rows = chunk_rows;
  result.chunk_count = (height + chunk_rows - 1) / chunk_rows;
  result.region_streaming_used = static_cast<bool>(load_frame_region);

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);
    const size_t pixel_count = static_cast<size_t>(rows) * width;
    std::vector<float> sample_values(pixel_count * frame_count,
                                     std::numeric_limits<float>::quiet_NaN());
    std::vector<float> sample_weights(pixel_count * frame_count, 0.0f);
    std::vector<float> sample_scores;
    if (cfg.uniform_weights && cherry_enabled)
      sample_scores.assign(pixel_count * frame_count, 0.0f);
    std::vector<uint32_t> finite_maps(pixel_count, 0u);

    const int num_threads = std::max(1, cfg.parallel_workers);
    std::vector<double> control_sums;
    std::vector<uint32_t> control_counts;
    if (cfg.compute_uniform_control) {
      control_sums.assign(pixel_count, 0.0);
      control_counts.assign(pixel_count, 0u);
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
#endif
    {
      std::vector<double> local_control_sums;
      std::vector<uint32_t> local_control_counts;
      if (cfg.compute_uniform_control) {
        local_control_sums.assign(pixel_count, 0.0);
        local_control_counts.assign(pixel_count, 0u);
      }

#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
      for (std::ptrdiff_t fi_signed = 0;
           fi_signed < static_cast<std::ptrdiff_t>(frame_count);
           ++fi_signed) {
        const size_t fi = static_cast<size_t>(fi_signed);
        if (frame_mask_compatible[fi] == 0u) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
        Matrix2Df frame;
        const bool frame_ok = load_frame_region
            ? load_frame_region(fi, y0, rows, frame)
            : load_frame(fi, frame);
        if (!frame_ok || frame.cols() != width ||
            frame.rows() != (load_frame_region ? rows : height)) continue;
        Matrix2Df q = load_frame_region
            ? q_map_cache->read_region(fi, y0, rows)
            : q_map_cache->read_cached(fi);
        if (q.cols() != width || q.rows() != (load_frame_region ? rows : height)) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
        std::vector<uint8_t> fm;
        const bool use_region_mask = static_cast<bool>(load_frame_valid_mask_region);
        const bool mask_ok = use_region_mask
            ? load_frame_valid_mask_region(fi, y0, rows, fm)
            : (!load_frame_valid_mask || load_frame_valid_mask(fi, fm));
        const size_t expected_mask = static_cast<size_t>(width) *
                                     (use_region_mask ? rows : height);
        if ((load_frame_valid_mask || use_region_mask) &&
            (!mask_ok || fm.size() != expected_mask)) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
        const float gw = global_weight(global_weights, fi);
        const float *frame_ptr = frame.data();
        const float *q_ptr = q.data();
        for (int yy = 0; yy < rows; ++yy) {
          const int y = y0 + yy;
          const size_t row_offset = static_cast<size_t>(yy) * static_cast<size_t>(width);
          const size_t full_row_offset = static_cast<size_t>(y) * static_cast<size_t>(width);
          for (int x = 0; x < width; ++x) {
            const size_t full_i = full_row_offset + static_cast<size_t>(x);
            const size_t local_i = row_offset + static_cast<size_t>(x);
            const size_t mask_i = use_region_mask ? local_i : full_i;
            if (!canvas_valid(canvas_mask, width, height, x, y) ||
                (!fm.empty() && (mask_i >= fm.size() || fm[mask_i] == 0u))) continue;
            const float frame_v = frame_ptr[local_i];
            if (!std::isfinite(frame_v)) continue;
            if (cfg.compute_uniform_control) {
              local_control_sums[local_i] += frame_v;
              ++local_control_counts[local_i];
            }
            const float q_v = q_ptr[local_i];
            if (!std::isfinite(q_v)) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
              ++result.missing_map_samples;
              continue;
            }
#if defined(_OPENMP)
#pragma omp atomic
#endif
            ++finite_maps[local_i];
#if defined(_OPENMP)
#pragma omp atomic
#endif
            ++result.finite_map_samples;
            const float score = gw * std::max(0.0f, q_v);
            const float weight = cfg.uniform_weights && score > 0.0f
                                     ? 1.0f : score;
            if (weight > 0.0f) {
              const size_t sample_i = local_i * frame_count + fi;
              sample_values[sample_i] = frame_v;
              sample_weights[sample_i] = weight;
              if (!sample_scores.empty()) sample_scores[sample_i] = score;
            }
          }
        }
      }

      if (cfg.compute_uniform_control) {
#if defined(_OPENMP)
#pragma omp critical
#endif
        {
          for (size_t i = 0; i < pixel_count; ++i) {
            control_sums[i] += local_control_sums[i];
            control_counts[i] += local_control_counts[i];
          }
        }
      }
    }

    if (cfg.compute_uniform_control) {
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t local_i = static_cast<size_t>(yy * width + x);
          if (control_counts[local_i] > 0u) {
            result.uniform_control_output(y, x) = static_cast<float>(
                control_sums[local_i] / static_cast<double>(control_counts[local_i]));
            result.uniform_control_valid_mask[static_cast<size_t>(y * width + x)] = 1u;
          }
        }
      }
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
#endif
    {
      std::vector<AqmhWeightedSample> samples;
      std::vector<float> local_effective_k;
      std::vector<float> local_margins;
      samples.reserve(frame_count);
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 64)
#endif
      for (std::ptrdiff_t local_pixel = 0;
           local_pixel < static_cast<std::ptrdiff_t>(pixel_count);
           ++local_pixel) {
        const int yy = static_cast<int>(local_pixel / width);
        const int x = static_cast<int>(local_pixel % width);
        const int y = y0 + yy;
        if (!canvas_valid(canvas_mask, width, height, x, y)) continue;
#if defined(_OPENMP)
#pragma omp atomic
#endif
        ++canvas_pixels;
        const size_t li = static_cast<size_t>(local_pixel);
        samples.clear();
        const size_t sample_base = li * frame_count;
        for (size_t fi = 0; fi < frame_count; ++fi) {
          const float weight = sample_weights[sample_base + fi];
          if (weight > 0.0f) {
            samples.push_back({sample_values[sample_base + fi], weight,
                               sample_scores.empty()
                                   ? weight : sample_scores[sample_base + fi],
                               fi});
          }
        }
        if (samples.empty()) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
          ++result.unsupported_pixels;
          if (finite_maps[li] > 0u) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
            ++result.zero_veto_pixels;
          }
          continue;
        }
        if (cherry_enabled) {
          int nominal = 0;
          float margin = -1.0f;
          auto selected = cfg.cherry_pick_mode == "auto_reject"
              ? aqmh_select_auto_reject(
                    samples, cfg.cherry_pick_k_min_required,
                    cfg.cherry_pick_reject_below_best_fraction,
                    cfg.cherry_pick_min_keep_fraction,
                    cfg.cherry_pick_margin_min, &nominal, &margin)
              : aqmh_select_top_k(samples, cfg.cherry_pick_k_min_required,
                                  cfg.cherry_pick_k_frac, cfg.tiered_k_frac,
                                  &nominal, &margin);
          if (!selected.empty()) {
            if (selected.size() < samples.size()) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
              ++cherry_active_pixels;
              local_effective_k.push_back(static_cast<float>(selected.size()));
              if (margin >= 0.0f) local_margins.push_back(margin);
            }
            samples = std::move(selected);
          }
          result.cherry_pick_k_map(y, x) = static_cast<float>(samples.size());
        }
        auto clipped = aqmh_sigma_clip(std::move(samples), cfg.clip_sigma_low,
                                       cfg.clip_sigma_high,
                                       cfg.clip_iterations, cfg.min_fraction,
                                       cfg.min_n_eff);
        if (!clipped.denominator_ok) {
          samples = std::move(clipped.retained);
#if defined(_OPENMP)
#pragma omp atomic
#endif
          ++result.unsupported_pixels;
#if defined(_OPENMP)
#pragma omp atomic
#endif
          ++result.numerical_guard_pixels;
          continue;
        }
        double accum = 0.0;
        for (const auto &s : clipped.retained) accum += s.weight * s.value;
        result.output(y, x) = static_cast<float>(accum / clipped.weight_sum);
        result.weight_sum(y, x) = clipped.weight_sum;
        samples = std::move(clipped.retained);
      }
#if defined(_OPENMP)
#pragma omp critical
#endif
      {
        effective_k.insert(effective_k.end(), local_effective_k.begin(), local_effective_k.end());
        margins.insert(margins.end(), local_margins.begin(), local_margins.end());
      }
    }
    if (progress) progress(y0 + rows, height);
  }

  result.cherry_pick_active = cherry_enabled && cherry_active_pixels > 0;
  result.cherry_pick_per_pixel_mode = cherry_enabled;
  result.cherry_pick_active_frac = canvas_pixels > 0
      ? static_cast<float>(cherry_active_pixels) / canvas_pixels : 0.0f;
  if (!effective_k.empty()) {
    result.k_effective_p10 = quantile(effective_k, 0.10f);
    result.k_effective_p50 = quantile(effective_k, 0.50f);
    result.k_effective_p90 = quantile(effective_k, 0.90f);
    result.cherry_pick_mean_k = std::accumulate(effective_k.begin(), effective_k.end(), 0.0f) /
                                effective_k.size();
    result.cherry_pick_median_k = result.k_effective_p50;
    result.cherry_pick_k_min_observed = static_cast<int>(*std::min_element(effective_k.begin(), effective_k.end()));
    result.cherry_pick_k_max_observed = static_cast<int>(*std::max_element(effective_k.begin(), effective_k.end()));
  }
  if (!margins.empty()) result.low_rank_separation = quantile(margins, 0.5f) < cfg.cherry_pick_margin_min;
  return result;
}

} // namespace tile_compile::reconstruction
