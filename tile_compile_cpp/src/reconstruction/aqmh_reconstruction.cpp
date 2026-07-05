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
  if (cfg.compute_uniform_control)
    result.uniform_control_output = Matrix2Df::Zero(height, width);
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
        for (size_t fi = 0; fi < frame_count; ++fi) {
          if (frame_mask_compatible[fi] == 0u ||
              !(global_weight(global_weights, fi) > 0.0f)) continue;
          Matrix2Df q = q_map_cache->read_region(fi, y0, rows);
          std::vector<uint8_t> fm;
          if (q.rows() != rows || q.cols() != width ||
              !load_frame_valid_mask_region(fi, y0, rows, fm) ||
              fm.size() != count) continue;
          for (size_t i = 0; i < count; ++i)
            if (fm[i] != 0u && q.data()[i] > 0.0f &&
                rankable[i] != std::numeric_limits<uint16_t>::max())
              ++rankable[i];
        }
        for (size_t i = 0; i < count; ++i) {
          const size_t full_i = static_cast<size_t>(y0) * width + i;
          if (!canvas_mask.empty() && canvas_mask[full_i] == 0u) continue;
          const int n = rankable[i];
          nominal_values.push_back(static_cast<float>(aqmh_k_nominal(
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
          nominal_values.push_back(static_cast<float>(aqmh_k_nominal(
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

    for (size_t fi = 0; fi < frame_count; ++fi) {
      if (frame_mask_compatible[fi] == 0u) {
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
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }
      const float gw = global_weight(global_weights, fi);
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t full_i = static_cast<size_t>(y * width + x);
          const size_t local_i = static_cast<size_t>(yy * width + x);
          const int source_y = load_frame_region ? yy : y;
          const size_t mask_i = use_region_mask ? local_i : full_i;
          if (!canvas_valid(canvas_mask, width, height, x, y) ||
              (!fm.empty() && (mask_i >= fm.size() || fm[mask_i] == 0u)) ||
              !std::isfinite(frame(source_y, x))) continue;
          if (!std::isfinite(q(source_y, x))) { ++result.missing_map_samples; continue; }
          ++finite_maps[local_i];
          ++result.finite_map_samples;
          const float score = gw * std::max(0.0f, q(source_y, x));
          const float weight = cfg.uniform_weights && score > 0.0f
                                   ? 1.0f : score;
          if (weight > 0.0f) {
            const size_t sample_i = local_i * frame_count + fi;
            sample_values[sample_i] = frame(source_y, x);
            sample_weights[sample_i] = weight;
            if (!sample_scores.empty()) sample_scores[sample_i] = score;
          }
        }
      }
    }

    const int num_threads = std::max(1, cfg.parallel_workers);
#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
#endif
    {
      std::vector<AqmhWeightedSample> samples;
      std::vector<AqmhWeightedSample> control_samples;
      samples.reserve(frame_count);
      if (cfg.compute_uniform_control) control_samples.reserve(frame_count);
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
          auto selected = aqmh_select_top_k(samples, cfg.cherry_pick_k_min_required,
                                             cfg.cherry_pick_k_frac,
                                             cfg.tiered_k_frac, &nominal, &margin);
          if (!selected.empty()) {
            if (selected.size() < samples.size()) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
              ++cherry_active_pixels;
#if defined(_OPENMP)
#pragma omp critical
#endif
              {
                effective_k.push_back(static_cast<float>(selected.size()));
                if (margin >= 0.0f) margins.push_back(margin);
              }
            }
            samples = std::move(selected);
          }
          result.cherry_pick_k_map(y, x) = static_cast<float>(samples.size());
        }
        bool reuse_control_result = false;
        if (cfg.compute_uniform_control) {
          reuse_control_result = !samples.empty();
          const float first_weight = samples.front().weight;
          for (const auto &sample : samples) {
            if (sample.weight != first_weight) {
              reuse_control_result = false;
              break;
            }
          }
        }
        if (cfg.compute_uniform_control && !reuse_control_result) {
          control_samples.assign(samples.begin(), samples.end());
          for (auto &sample : control_samples) sample.weight = 1.0f;
          auto control = aqmh_sigma_clip(
              std::move(control_samples), cfg.clip_sigma,
              cfg.clip_iterations, cfg.min_fraction, cfg.min_n_eff);
          if (control.denominator_ok) {
            double control_accum = 0.0;
            for (const auto &s : control.retained)
              control_accum += s.weight * s.value;
            result.uniform_control_output(y, x) =
                static_cast<float>(control_accum / control.weight_sum);
          }
          control_samples = std::move(control.retained);
        }
        auto clipped = aqmh_sigma_clip(std::move(samples), cfg.clip_sigma,
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
        if (cfg.compute_uniform_control && reuse_control_result)
          result.uniform_control_output(y, x) = result.output(y, x);
        samples = std::move(clipped.retained);
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
