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

float global_weight(const VectorXf &weights, size_t fi) {
  if (fi >= static_cast<size_t>(weights.size())) return 0.0f;
  const float value = weights[static_cast<Eigen::Index>(fi)];
  return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

float quantile(std::vector<float> values, float q) {
  if (values.empty()) return 0.0f;
  const double pos = std::clamp<double>(q, 0.0, 1.0) * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  // Only ranks lo and hi are needed for the interpolation below, not a full
  // ordering: two nth_element calls place both correctly (the second, on the
  // [lo, end) sub-range nth_element already partitioned, finds hi's value
  // among exactly the elements >= values[lo]) without sorting the rest.
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo),
                   values.end());
  if (hi != lo)
    std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(lo),
                     values.begin() + static_cast<std::ptrdiff_t>(hi),
                     values.end());
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

  // Pre-materialize a flat bool array from canvas_mask for O(1) lookup in the
  // innermost pixel loop below (redundant-reload analysis C2), same pattern
  // as reconstruct_aqmh_weighted's canvas_valid_flat. Read-only once built,
  // so sharing it across the parallel chunk loop is safe.
  const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<uint8_t> canvas_valid_flat(total_pixels, 1u);
  if (!canvas_mask.empty() && canvas_mask.size() == total_pixels) {
    for (size_t i = 0; i < total_pixels; ++i)
      canvas_valid_flat[i] = canvas_mask[i] != 0u ? 1u : 0u;
  }

  // N1 (redundant-reload analysis): iterate frames outermost so every frame
  // and mask is loaded exactly once for the whole image. The previous
  // chunk-outer/frame-inner order reloaded every frame and mask per 128-row
  // chunk. Accumulators are full-image; the inner pixel loop parallelizes
  // over rows, which write disjoint cells, so per-pixel accumulation order
  // and results are identical to the serial accumulation.
  std::vector<double> sums(total_pixels, 0.0);
  std::vector<uint32_t> counts(total_pixels, 0u);
  constexpr int mask_slab_rows = 128;
  for (size_t fi = 0; fi < frame_count; ++fi) {
    Matrix2Df frame;
    bool frame_ok;
    if (load_frame_region) {
      // Same source the old chunk loop used: assemble the full frame
      // band-wise so each row is read exactly once.
      frame = Matrix2Df::Zero(height, width);
      frame_ok = true;
      for (int ry = 0; ry < height && frame_ok; ry += mask_slab_rows) {
        const int rrows = std::min(mask_slab_rows, height - ry);
        Matrix2Df slab;
        if (!load_frame_region(fi, ry, rrows, slab) ||
            slab.cols() != width || slab.rows() != rrows) {
          frame_ok = false;
          break;
        }
        frame.block(ry, 0, rrows, width) = slab;
      }
    } else {
      frame_ok = load_frame(fi, frame);
    }
    if (!frame_ok || frame.cols() != width || frame.rows() != height) {
      continue;
    }
    std::vector<uint8_t> frame_mask;
    if (load_frame_valid_mask_region) {
      // Region loader preferred (matches the old chunk loop): assemble the
      // full mask band-wise, each row read exactly once.
      frame_mask.assign(total_pixels, 0u);
      bool mask_ok = true;
      for (int ry = 0; ry < height; ry += mask_slab_rows) {
        const int rrows = std::min(mask_slab_rows, height - ry);
        std::vector<uint8_t> slab;
        if (!load_frame_valid_mask_region(fi, ry, rrows, slab) ||
            slab.size() != static_cast<size_t>(rrows) * width) {
          mask_ok = false;
          break;
        }
        std::copy(slab.begin(), slab.end(),
                  frame_mask.begin() +
                      static_cast<std::ptrdiff_t>(ry) * width);
      }
      if (!mask_ok) continue;
    } else if (load_frame_valid_mask) {
      if (!load_frame_valid_mask(fi, frame_mask) ||
          frame_mask.size() != total_pixels) {
        continue;
      }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
      const size_t row_base = static_cast<size_t>(y) * width;
      for (int x = 0; x < width; ++x) {
        const size_t full_i = row_base + static_cast<size_t>(x);
        const float value = frame(y, x);
        if (canvas_valid_flat[full_i] == 0u ||
            (!frame_mask.empty() && frame_mask[full_i] == 0u) ||
            !std::isfinite(value)) {
          continue;
        }
        sums[full_i] += value;
        ++counts[full_i];
      }
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t full_i = static_cast<size_t>(y) * width + x;
      if (counts[full_i] == 0u) continue;
      result.output(y, x) =
          static_cast<float>(sums[full_i] / static_cast<double>(counts[full_i]));
      result.valid_mask[full_i] = 1u;
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

  // §8-A: Pre-materialize a flat bool array from canvas_mask for O(1) lookup
  // in the innermost pixel loop, eliminating per-pixel bounds checks.
  const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<uint8_t> canvas_valid_flat(total_pixels, 1u);
  if (!canvas_mask.empty() && canvas_mask.size() == total_pixels) {
    for (size_t i = 0; i < total_pixels; ++i)
      canvas_valid_flat[i] = canvas_mask[i] != 0u ? 1u : 0u;
  }

  // C7 (redundant-reload analysis): global_weight() was re-evaluated for the
  // same fi in the gate pass, the non-region gate and the per-chunk fill.
  // weights is invariant for the whole call --- evaluate once per frame.
  std::vector<float> gw_by_fi(frame_count, 0.0f);
  for (size_t fi = 0; fi < frame_count; ++fi)
    gw_by_fi[fi] = global_weight(global_weights, fi);

  // A4 (fallback path): when no region loaders are wired, the chunk-outer /
  // frame-inner loop reloads every full frame + mask per chunk. When the
  // whole set fits comfortably in the memory budget, keep frames and masks
  // resident and serve synthesized region loaders from them: each frame and
  // mask is then loaded from disk exactly once, whichever pass asks.
  // Sizing: <= 25 % of memory_budget_mb for the stash (the sample buffers
  // already reserve ~budget/2). With region loaders present (production) the
  // stash stays off and everything behaves as before.
  std::vector<Matrix2Df> frame_stash;
  std::vector<std::vector<uint8_t>> mask_stash;
  std::vector<uint8_t> frame_stash_valid, mask_stash_valid;
  AqmhFrameRegionLoader eff_frame_region = load_frame_region;
  AqmhMaskRegionLoader eff_mask_region = load_frame_valid_mask_region;
  const size_t stash_budget_bytes =
      (static_cast<size_t>(cfg.memory_budget_mb) << 20) / 4;
  const size_t stash_bytes_per_frame =
      total_pixels * (sizeof(float) + sizeof(uint8_t));
  const bool stash_active =
      (!eff_frame_region || !eff_mask_region) &&
      frame_count > 0 &&
      stash_bytes_per_frame <= stash_budget_bytes &&
      frame_count <= stash_budget_bytes / stash_bytes_per_frame;
  if (stash_active) {
    frame_stash.resize(frame_count);
    mask_stash.resize(frame_count);
    frame_stash_valid.assign(frame_count, 0u);
    mask_stash_valid.assign(frame_count, 0u);
    if (!eff_frame_region) {
      eff_frame_region = [&](size_t fi, int ry0, int rrows,
                             Matrix2Df &out) -> bool {
        if (fi >= frame_count || rrows <= 0) return false;
        if (!frame_stash_valid[fi]) {
          if (!load_frame(fi, frame_stash[fi])) return false;
          frame_stash_valid[fi] = 1u;
        }
        const Matrix2Df &m = frame_stash[fi];
        if (m.cols() != width || m.rows() != height) return false;
        out = m.block(ry0, 0, rrows, width);
        return true;
      };
    }
    if (!eff_mask_region && load_frame_valid_mask) {
      eff_mask_region = [&](size_t fi, int ry0, int rrows,
                            std::vector<uint8_t> &out) -> bool {
        if (fi >= frame_count || rrows <= 0) return false;
        if (!mask_stash_valid[fi]) {
          if (!load_frame_valid_mask(fi, mask_stash[fi])) return false;
          mask_stash_valid[fi] = 1u;
        }
        const auto &m = mask_stash[fi];
        if (m.size() != total_pixels) return false;
        const size_t first = static_cast<size_t>(ry0) * width;
        const size_t count = static_cast<size_t>(rrows) * width;
        out.assign(m.begin() + static_cast<std::ptrdiff_t>(first),
                   m.begin() + static_cast<std::ptrdiff_t>(first + count));
        return true;
      };
    }
  }

  // Validate each full M_f digest once. The previous slab loop re-read and
  // re-hashed a full mask for every frame and every slab.
  // §8-G: Parallelized — SHA-256 is embarrassingly parallel across frames.
  // A6: when the stash is active the validated mask is kept --- the main
  // pass's mask reads then come from memory instead of a second disk sweep.
  std::vector<uint8_t> frame_mask_compatible(frame_count, 1u);
  if (load_frame_valid_mask) {
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(frame_count); ++fi_ptr) {
      const size_t fi = static_cast<size_t>(fi_ptr);
      std::vector<uint8_t> full_mask;
      const bool ok =
          load_frame_valid_mask(fi, full_mask) &&
          full_mask.size() == static_cast<size_t>(width * height) &&
          q_map_cache->source_mask_hash(fi) ==
              tile_compile::core::sha256_bytes(full_mask);
      frame_mask_compatible[fi] = ok ? 1u : 0u;
      if (ok && stash_active && !mask_stash_valid[fi]) {
        mask_stash[fi] = std::move(full_mask);
        mask_stash_valid[fi] = 1u;
      }
    }
  }

  bool cherry_enabled = cfg.cherry_pick;
  // Hoisted (redundant-reload analysis C1): cfg.cherry_pick_mode is invariant
  // for this whole call; avoid re-comparing the string per pixel.
  const bool is_auto_reject = cfg.cherry_pick_mode == "auto_reject";
  if (cherry_enabled) {
    // A5/A6: single frame-outer gate sweep. Each frame contributes its q map
    // and mask exactly once; read_cached() leaves the q map resident in the
    // LRU so the main pass's region reads decode it at most once whenever the
    // cache can hold the working set, and the mask stash (when active) serves
    // every later read from memory.
    // The semantic mode is fixed by the *caller's* loaders: only when both
    // region loaders were passed do we skip the per-pixel isfinite(frame)
    // check, matching the previous region gate exactly.
    std::vector<float> nominal_values;
    nominal_values.reserve(total_pixels);
    std::vector<uint32_t> rankable(total_pixels, 0u);
    const bool gate_region_mode =
        static_cast<bool>(load_frame_region) &&
        static_cast<bool>(load_frame_valid_mask_region);
    constexpr int gate_rows = 128;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(frame_count); ++fi_ptr) {
      const size_t fi = static_cast<size_t>(fi_ptr);
      if (frame_mask_compatible[fi] == 0u || !(gw_by_fi[fi] > 0.0f)) continue;
      Matrix2Df q = q_map_cache->read_cached(fi);
      if (q.rows() != height || q.cols() != width) continue;
      Matrix2Df frame;
      if (!gate_region_mode) {
        // Preserve the legacy non-region semantics: only the full loaders
        // count here. eff_* is used solely when the stash synthesized it from
        // those same loaders, so the data source is unchanged.
        const bool fok = (stash_active && eff_frame_region)
            ? eff_frame_region(fi, 0, height, frame)
            : load_frame(fi, frame);
        if (!fok || frame.rows() != height || frame.cols() != width) continue;
      }
      if (gate_region_mode) {
        for (int gy = 0; gy < height; gy += gate_rows) {
          const int rows = std::min(gate_rows, height - gy);
          const size_t count = static_cast<size_t>(rows) * width;
          const size_t base = static_cast<size_t>(gy) * width;
          std::vector<uint8_t> fm;
          if (!eff_mask_region(fi, gy, rows, fm) || fm.size() != count) break;
          const float *q_ptr = q.data() + base;
          for (size_t i = 0; i < count; ++i)
            if (fm[i] != 0u && q_ptr[i] > 0.0f) {
              #if defined(_OPENMP)
              #pragma omp atomic
              #endif
              rankable[base + i] += 1u;
            }
        }
      } else {
        std::vector<uint8_t> fm;
        if (stash_active && eff_mask_region) {
          if (!eff_mask_region(fi, 0, height, fm)) continue;
        } else if (load_frame_valid_mask) {
          if (!load_frame_valid_mask(fi, fm)) continue;
        }
        if (!fm.empty() && fm.size() != total_pixels) continue;
        for (size_t i = 0; i < total_pixels; ++i) {
          // Direct flat-array lookup (redundant-reload analysis C2):
          // canvas_valid_flat[i] is the same condition as canvas_valid()
          // without re-deriving mask.empty()/mask.size() every pixel.
          if ((fm.empty() || fm[i] != 0u) && std::isfinite(frame.data()[i]) &&
              q.data()[i] > 0.0f) {
            #if defined(_OPENMP)
            #pragma omp atomic
            #endif
            rankable[i] += 1u;
          }
        }
      }
    }
    for (size_t i = 0; i < total_pixels; ++i) {
      if (canvas_valid_flat[i] == 0u) continue;
      const int n = static_cast<int>(rankable[i]);
      nominal_values.push_back(is_auto_reject
          ? static_cast<float>(n)
          : static_cast<float>(aqmh_k_nominal(
                n, aqmh_effective_k_frac(n, cfg.cherry_pick_k_frac,
                                         cfg.tiered_k_frac))));
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
  result.region_streaming_used = static_cast<bool>(eff_frame_region);

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

    // §8-C: Shared per-thread containers to avoid omp critical merge.
    // Each thread accumulates into its own slot; merge is done serially
    // after the parallel region with no synchronization needed.
#if defined(_OPENMP)
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::vector<uint32_t>> per_thread_finite_maps(max_threads);
    std::vector<std::vector<double>> per_thread_control_sums(max_threads);
    std::vector<std::vector<uint32_t>> per_thread_control_counts(max_threads);

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      per_thread_finite_maps[tid].assign(pixel_count, 0u);
      if (cfg.compute_uniform_control) {
        per_thread_control_sums[tid].assign(pixel_count, 0.0);
        per_thread_control_counts[tid].assign(pixel_count, 0u);
      }
      auto& local_finite_maps = per_thread_finite_maps[tid];
      auto& local_control_sums = per_thread_control_sums[tid];
      auto& local_control_counts = per_thread_control_counts[tid];

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
        const bool frame_ok = eff_frame_region
            ? eff_frame_region(fi, y0, rows, frame)
            : load_frame(fi, frame);
        if (!frame_ok || frame.cols() != width ||
            frame.rows() != (eff_frame_region ? rows : height)) continue;
        Matrix2Df q = eff_frame_region
            ? q_map_cache->read_region(fi, y0, rows)
            : q_map_cache->read_cached(fi);
        if (q.cols() != width || q.rows() != (eff_frame_region ? rows : height)) {
#if defined(_OPENMP)
#pragma omp atomic
#endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
        std::vector<uint8_t> fm;
        const bool use_region_mask = static_cast<bool>(eff_mask_region);
        const bool mask_ok = use_region_mask
            ? eff_mask_region(fi, y0, rows, fm)
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
        const float gw = gw_by_fi[fi];
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
            if ((!canvas_valid_flat.empty() && canvas_valid_flat[full_i] == 0u) ||
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
            ++local_finite_maps[local_i];
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

    }

    // §8-C: Merge per-thread accumulators serially after the parallel region.
    // No omp critical needed — all threads have finished.
    for (int t = 0; t < max_threads; ++t) {
      if (per_thread_finite_maps[t].empty()) continue;
      for (size_t i = 0; i < pixel_count; ++i) {
        finite_maps[i] += per_thread_finite_maps[t][i];
        if (cfg.compute_uniform_control) {
          control_sums[i] += per_thread_control_sums[t][i];
          control_counts[i] += per_thread_control_counts[t][i];
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
      // Heuristic upper-ish bound (redundant-reload analysis D6): only a
      // fraction of this thread's pixels end up cherry-picked, so this is not
      // exact, but reserve() never changes correctness -- it only avoids
      // repeated growth reallocation as the dynamic OpenMP schedule assigns
      // roughly pixel_count/num_threads pixels to each thread.
      local_effective_k.reserve(pixel_count / static_cast<size_t>(num_threads));
      local_margins.reserve(pixel_count / static_cast<size_t>(num_threads));
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 64)
#endif
      for (std::ptrdiff_t local_pixel = 0;
           local_pixel < static_cast<std::ptrdiff_t>(pixel_count);
           ++local_pixel) {
        const int yy = static_cast<int>(local_pixel / width);
        const int x = static_cast<int>(local_pixel % width);
        const int y = y0 + yy;
        if (!canvas_valid_flat.empty() &&
            canvas_valid_flat[static_cast<size_t>(y * width + x)] == 0u) continue;
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
          // NOTE: aqmh_select_* can legitimately return {} (e.g. all scores
          // invalid, or n < k_min_required) while leaving `samples` as the
          // fallback the code below still needs -- so this copy (the ternary
          // takes `samples` by value) cannot be replaced with a move; the
          // original must survive the call for that empty-`selected` case.
          auto selected = is_auto_reject
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
