#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"

#if TILE_COMPILE_WITH_CUDA

#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <future>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tile_compile::reconstruction {

namespace {

constexpr int kMaxFramesCompile = 1024;  // absolute upper bound for CUDA kernels
constexpr int kMinAutoChunkRows = 16;
constexpr size_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr size_t kBytesPerGiB = 1024ULL * kBytesPerMiB;

// Removed next_pow2 and insertion_adaptive_sort.
// A single unified Shellsort-based adaptive_sort handles all array sizes.

// Small epsilon used for device-side comparisons (matches CPU epsilon guard).
__device__ inline float device_eps() {
  return FLT_EPSILON;
}

// Device equivalent of std::isfinite for float.
__device__ inline bool isfinite_f(float v) {
  return !isnan(v) && !isinf(v);
}

// Comparator for bitonic sort: ascending by value, tie-break by index.
struct ValueAsc {
  const float* values;
  __device__ bool operator()(int a, int b) const {
    const float va = values[a];
    const float vb = values[b];
    return va < vb || (va == vb && a < b);
  }
};

// Comparator for bitonic sort: descending by score, tie-break by index.
struct ScoreDesc {
  const float* scores;
  __device__ bool operator()(int a, int b) const {
    const float sa = scores[a];
    const float sb = scores[b];
    return sa > sb || (sa == sb && a < b);
  }
};

// Comparator for bitonic sort: ascending by absolute deviation from center.
struct DeviationAsc {
  const float* values;
  float center;
  __device__ bool operator()(int a, int b) const {
    const float da = fabsf(values[a] - center);
    const float db = fabsf(values[b] - center);
    return da < db || (da == db && a < b);
  }
};

// Comparator for bitonic sort: ascending by normalized distance to center.
struct NormDistanceAsc {
  const float* values;
  float center;
  float sigma;
  __device__ bool operator()(int a, int b) const {
    const float da = fabsf(values[a] - center) / sigma;
    const float db = fabsf(values[b] - center) / sigma;
    return da < db || (da == db && a < b);
  }
};

// Highly optimized adaptive sort using Shellsort with Marcin Ciura's gap sequence.
// Shellsort works on any N seamlessly, avoids padding to next power of 2, 
// has O(N^1.25) worst-case, and scales to O(N) for nearly sorted inputs.
// This dramatically reduces comparisons, instructions, register pressure, and execution time
// for thread-sequential sorting on the GPU compared to Bitonic Sort.
template <int MaxFrames, typename Comp>
__device__ void adaptive_sort(short* indices, int n, Comp comp) {
  for (short i = 0; i < static_cast<short>(n); ++i) indices[i] = i;

  // Empirical gap sequence optimized for N <= 1024
  const int gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
  for (int g = 0; g < 8; ++g) {
    const int gap = gaps[g];
    if (gap >= n || gap >= MaxFrames) continue;
    for (int i = gap; i < n; ++i) {
      const short temp = indices[i];
      int j = i;
      while (j >= gap && comp(temp, indices[j - gap])) {
        indices[j] = indices[j - gap];
        j -= gap;
      }
      indices[j] = temp;
    }
  }
}

// Weighted median of (values, weights).  Sorts by value ascending using bitonic
// sort, then walks cumulative weight to the 50% target.
template <int MaxFrames>
__device__ float weighted_median_value(
    const float* values, const float* weights, int n,
    short* sort_buf) {
  if (n <= 0) return 0.0f;
  adaptive_sort<MaxFrames>(sort_buf, n, ValueAsc{values});

  double total = 0.0;
  for (int i = 0; i < n; ++i) total += static_cast<double>(weights[sort_buf[i]]);
  const double target = total * 0.5;
  double accum = 0.0;
  for (int i = 0; i < n; ++i) {
    accum += static_cast<double>(weights[sort_buf[i]]);
    if (accum >= target) {
      return values[sort_buf[i]];
    }
  }
  return values[sort_buf[n - 1]];
}

// Weighted median of absolute deviations from center.
template <int MaxFrames>
__device__ float weighted_mad_value(
    const float* values, const float* weights, int n,
    float center, short* sort_buf) {
  if (n <= 0) return 0.0f;
  adaptive_sort<MaxFrames>(sort_buf, n, DeviationAsc{values, center});

  double total = 0.0;
  for (int i = 0; i < n; ++i) total += static_cast<double>(weights[sort_buf[i]]);
  const double target = total * 0.5;
  double accum = 0.0;
  for (int i = 0; i < n; ++i) {
    accum += static_cast<double>(weights[sort_buf[i]]);
    if (accum >= target) {
      return fabsf(values[sort_buf[i]] - center);
    }
  }
  return fabsf(values[sort_buf[n - 1]] - center);
}

// Noise floor for the non-small-N path: eps_rel * MAD(values). The caller
// already obtained the unweighted median from the value ordering produced by
// weighted_median_value(), so only the deviation ordering remains.
template <int MaxFrames>
__device__ float noise_floor_from_median(
    const float* values, int n, float median, short* sort_buf) {
  if (n <= 0) return device_eps();
  adaptive_sort<MaxFrames>(sort_buf, n, DeviationAsc{values, median});
  const float mad = (n % 2 == 1)
      ? fabsf(values[sort_buf[n / 2]] - median)
      : 0.5f * (fabsf(values[sort_buf[n / 2 - 1]] - median) +
                fabsf(values[sort_buf[n / 2]] - median));
  const float eps_rel = metrics::aqmh_eps_rel;
  return fmaxf(device_eps(), eps_rel * mad);
}

// Cherry-pick: select top-K by score descending, frame_index ascending.
// Returns new sample count. Operates in-place on the first 'n' entries of
// values/weights/scores arrays.
template <int MaxFrames>
__device__ int cherry_pick_top_k(
    float* values, float* weights, float* scores, int n,
    int k_min_required, float k_frac, short* sort_buf) {
  if (n < k_min_required) return n;

  const int nominal = max(0, static_cast<int>(floorf(k_frac * static_cast<float>(n))));
  const int k = min(n, max(k_min_required, nominal));
  if (k >= n) return n;

  adaptive_sort<MaxFrames>(sort_buf, n, ScoreDesc{scores});

  const float cutoff = scores[sort_buf[k - 1]];
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (scores[i] >= cutoff) {
      values[m] = values[i];
      weights[m] = weights[i];
      scores[m] = scores[i];
      ++m;
    }
  }
  return m;
}

// Conservative cherry-pick mode: keep all locally usable samples except clear
// low-score outliers, while enforcing a minimum retained fraction.
template <int MaxFrames>
__device__ int cherry_pick_auto_reject(
    float* values, float* weights, float* scores, int n,
    int k_min_required, float reject_below_best_fraction,
    float min_keep_fraction, float margin_min, short* sort_buf) {
  if (n < k_min_required) return n;

  adaptive_sort<MaxFrames>(sort_buf, n, ScoreDesc{scores});
  const float best = scores[sort_buf[0]];
  if (!(best > 0.0f) || !isfinite_f(best)) return n;

  const float threshold = best * fminf(1.0f, fmaxf(0.0f, reject_below_best_fraction));
  int keep = 0;
  while (keep < n && scores[sort_buf[keep]] >= threshold) {
    ++keep;
  }

  const int min_keep = min(n, max(k_min_required,
      static_cast<int>(ceilf(fminf(1.0f, fmaxf(0.0f, min_keep_fraction)) *
                             static_cast<float>(n)))));
  keep = max(keep, min_keep);
  if (keep >= n) return n;

  const float margin = (scores[sort_buf[keep - 1]] - scores[sort_buf[keep]]) / best;
  if (margin < margin_min) return n;

  const float cutoff = scores[sort_buf[keep - 1]];
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (scores[i] >= cutoff) {
      values[m] = values[i];
      weights[m] = weights[i];
      scores[m] = scores[i];
      ++m;
    }
  }
  return m;
}

// Device sigma-clip matching the CPU aqmh_sigma_clip logic.
// Operates on the first 'n' entries of values/weights. Returns retained count
// and writes weighted sum / effective N into out_*.
// sort_buf is caller-provided scratch storage shared with cherry_pick_top_k.
template <int MaxFrames>
__device__ int sigma_clip(
    float* values, float* weights, int n,
    float clip_sigma_low, float clip_sigma_high, int iterations,
    float min_fraction, float min_n_eff,
    float* out_weight_sum, float* out_effective_n, short* sort_buf) {
  if (n <= 0) {
    *out_weight_sum = 0.0f;
    *out_effective_n = 0.0f;
    return 0;
  }

  // Remove non-finite / non-positive weights in-place.
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (isfinite_f(values[i]) && isfinite_f(weights[i]) && weights[i] > 0.0f) {
      values[m] = values[i];
      weights[m] = weights[i];
      ++m;
    }
  }
  n = m;
  if (n <= 0) {
    *out_weight_sum = 0.0f;
    *out_effective_n = 0.0f;
    return 0;
  }

  const int keep_floor = max(1, static_cast<int>(ceilf(min_fraction * static_cast<float>(n))));

  // Sentinel-pad the unused tail so bitonic sorts over the fixed compile-time
  // size do not touch uninitialized memory.
  for (int i = n; i < MaxFrames; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    const float center = weighted_median_value<MaxFrames>(values, weights, n, sort_buf);
    // The sort buffer is still value-ordered here. Reuse it before the
    // weighted-MAD step overwrites the ordering.
    const float value_median = (n % 2 == 1)
        ? values[sort_buf[n / 2]]
        : 0.5f * (values[sort_buf[n / 2 - 1]] +
                  values[sort_buf[n / 2]]);

    const float mad = weighted_mad_value<MaxFrames>(values, weights, n, center, sort_buf);
    const float floor_val = noise_floor_from_median<MaxFrames>(values, n, value_median, sort_buf);

    const float eps_center = device_eps() * fmaxf(fabsf(center), 1.0f);
    if (mad <= floor_val) {
      // All equal (within epsilon guard): keep only samples within eps_center.
      int keep_count = 0;
      for (int i = 0; i < n; ++i) {
        if (fabsf(values[i] - center) <= eps_center) ++keep_count;
      }
      if (keep_count == n) break;
      if (keep_count < keep_floor) {
        // Sort by normalized distance and keep floor closest to center.
        adaptive_sort<MaxFrames>(
            sort_buf, n,
            NormDistanceAsc{values, center, floor_val});
        // In-place gather via cycle-following: avoids tmp_v/tmp_w float arrays.
        // Marks visited positions in sort_buf with bitwise NOT (gives negative).
        for (int ci = 0; ci < keep_floor; ++ci) {
          if (sort_buf[ci] < 0) continue;
          const float sv = values[ci];
          const float sw = weights[ci];
          int j = ci;
          for (;;) {
            const short src = sort_buf[j];
            sort_buf[j] = static_cast<short>(~src);
            if (src == static_cast<short>(ci)) { values[j] = sv; weights[j] = sw; break; }
            values[j] = values[src];
            weights[j] = weights[src];
            j = src;
          }
        }
        for (int ci = 0; ci < keep_floor; ++ci)
          if (sort_buf[ci] < 0) sort_buf[ci] = static_cast<short>(~sort_buf[ci]);
        n = keep_floor;
      } else {
        // Keep only samples within eps_center.
        int m = 0;
        for (int i = 0; i < n; ++i) {
          if (fabsf(values[i] - center) <= eps_center) {
            values[m] = values[i];
            weights[m] = weights[i];
            ++m;
          }
        }
        n = m;
      }
      break;
    }

    const float sigma = 1.4826f * mad;
    int keep_count = 0;
    for (int i = 0; i < n; ++i) {
      if ((values[i] >= center &&
           values[i] - center <= clip_sigma_high * sigma) ||
          (values[i] < center &&
           center - values[i] <= clip_sigma_low * sigma)) ++keep_count;
    }

    if (keep_count == n) {
      break;
    }
    if (keep_count < keep_floor) {
      // Sort by normalized distance and keep floor.
      adaptive_sort<MaxFrames>(sort_buf, n, NormDistanceAsc{values, center, sigma});
      // In-place gather via cycle-following: avoids tmp_v/tmp_w float arrays.
      for (int ci = 0; ci < keep_floor; ++ci) {
        if (sort_buf[ci] < 0) continue;
        const float sv = values[ci];
        const float sw = weights[ci];
        int j = ci;
        for (;;) {
          const short src = sort_buf[j];
          sort_buf[j] = static_cast<short>(~src);
          if (src == static_cast<short>(ci)) { values[j] = sv; weights[j] = sw; break; }
          values[j] = values[src];
          weights[j] = weights[src];
          j = src;
        }
      }
      for (int ci = 0; ci < keep_floor; ++ci)
        if (sort_buf[ci] < 0) sort_buf[ci] = static_cast<short>(~sort_buf[ci]);
      n = keep_floor;
      break;
    }

    // Keep samples within clip band.
    m = 0;
    for (int i = 0; i < n; ++i) {
      if ((values[i] >= center &&
           values[i] - center <= clip_sigma_high * sigma) ||
          (values[i] < center &&
           center - values[i] <= clip_sigma_low * sigma)) {
        values[m] = values[i];
        weights[m] = weights[i];
        ++m;
      }
    }
    n = m;
    // Pad the new tail before the next iteration's sorts.
    for (int i = n; i < MaxFrames; ++i) {
      values[i] = INFINITY;
      weights[i] = 0.0f;
    }
  }

  double d = 0.0;
  double d2 = 0.0;
  float wmax = 0.0f;
  for (int i = 0; i < n; ++i) {
    d += static_cast<double>(weights[i]);
    d2 += static_cast<double>(weights[i]) * weights[i];
    wmax = fmaxf(wmax, weights[i]);
  }
  *out_weight_sum = static_cast<float>(d);
  *out_effective_n = (d2 > 0.0) ? static_cast<float>(d * d / d2) : 0.0f;

  const float guard = static_cast<float>(n) * device_eps() * wmax;
  if (d <= guard || *out_effective_n < min_n_eff) {
    *out_weight_sum = 0.0f;
    *out_effective_n = 0.0f;
    return 0;
  }
  return n;
}

template <bool CherryPickEnabled, int MaxFrames>
__global__ __launch_bounds__(256, 2)
void aqmh_reconstruction_kernel(
    const float* __restrict__ d_frames,
    const float* __restrict__ d_q_maps,
    const uint8_t* __restrict__ d_canvas_mask,
    const uint8_t* __restrict__ d_frame_masks,
    const float* __restrict__ d_global_weights,
    float* __restrict__ d_output,
    float* __restrict__ d_weight_sum,
    float* __restrict__ d_uniform_control,
    uint8_t* __restrict__ d_uniform_control_valid,
    float* __restrict__ d_cherry_k_map,
    unsigned long long* __restrict__ d_unsupported_pixels,
    unsigned long long* __restrict__ d_zero_veto_pixels,
    unsigned long long* __restrict__ d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height, int frame_count,
    float clip_sigma_low, float clip_sigma_high, int clip_iterations,
    float min_fraction, float min_n_eff,
    float cherry_pick_k_frac,
    int cherry_pick_k_min_required,
    int cherry_pick_mode,
    float cherry_pick_reject_below_best_fraction,
    float cherry_pick_min_keep_fraction,
    float cherry_pick_margin_min) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= chunk_rows) return;

  const int y = y0 + yy;
  if (y >= height) return;

  const int canvas_idx = yy * width + x;
  if (d_canvas_mask[canvas_idx] == 0u) {
    if (d_uniform_control != nullptr) {
      d_uniform_control[canvas_idx] = 0.0f;
      d_uniform_control_valid[canvas_idx] = 0u;
    }
    return;
  }

  // The index buffer is shared between cherry-pick and sigma-clip. Keeping
  // medians index-based avoids two additional MaxFrames-sized local arrays.
  float values[MaxFrames];
  float weights[MaxFrames];
  float scores[CherryPickEnabled ? MaxFrames : 1];
  short scratch_buf[MaxFrames];
  int n_samples = 0;
  bool has_finite_q = false;
  double uniform_sum = 0.0;
  int uniform_count = 0;

  // Pixel-major layout: all frames for one pixel are contiguous, so each
  // thread's frame loop reads with stride 1.
  const int pixel_base = canvas_idx * frame_count;
  for (int fi = 0; fi < frame_count; ++fi) {
    const int idx = pixel_base + fi;
    const uint8_t mask_val = d_frame_masks[idx];
    if (mask_val == 0u) continue;

    const float v = d_frames[idx];
    if (!isfinite_f(v)) continue;
    if (d_uniform_control != nullptr) {
      uniform_sum += static_cast<double>(v);
      ++uniform_count;
    }
    const float q = d_q_maps[idx];
    if (!isfinite_f(q)) continue;
    has_finite_q = true;

    const float gw = d_global_weights[fi];
    const float score = gw * fmaxf(0.0f, q);

    if (score > 0.0f && n_samples < MaxFrames) {
      values[n_samples] = v;
      weights[n_samples] = score;
      if constexpr (CherryPickEnabled) {
        scores[n_samples] = score;
      }
      ++n_samples;
    }
  }

  if (d_uniform_control != nullptr) {
    d_uniform_control[canvas_idx] =
        uniform_count > 0
            ? static_cast<float>(uniform_sum / static_cast<double>(uniform_count))
            : 0.0f;
    d_uniform_control_valid[canvas_idx] = uniform_count > 0 ? 1u : 0u;
  }

  // Sentinel-pad the unused tail so all bitonic sorts over the fixed 1024-wide
  // arrays see deterministic, harmless values.
  for (int i = n_samples; i < MaxFrames; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
    if constexpr (CherryPickEnabled) {
      scores[i] = -INFINITY;
    }
  }

  if (n_samples == 0) {
    d_output[canvas_idx] = 0.0f;
    d_weight_sum[canvas_idx] = 0.0f;
    if constexpr (CherryPickEnabled) d_cherry_k_map[canvas_idx] = 0.0f;
    atomicAdd(d_unsupported_pixels, 1ULL);
    if (has_finite_q) atomicAdd(d_zero_veto_pixels, 1ULL);
    return;
  }

  int k_effective = n_samples;
  if constexpr (CherryPickEnabled) {
    if (cherry_pick_mode == 1) {
      k_effective = cherry_pick_auto_reject<MaxFrames>(
          values, weights, scores, n_samples,
          cherry_pick_k_min_required,
          cherry_pick_reject_below_best_fraction,
          cherry_pick_min_keep_fraction,
          cherry_pick_margin_min, scratch_buf);
    } else {
      k_effective = cherry_pick_top_k<MaxFrames>(
          values, weights, scores, n_samples,
          cherry_pick_k_min_required, cherry_pick_k_frac, scratch_buf);
    }
    d_cherry_k_map[canvas_idx] = static_cast<float>(k_effective);
  }

  float weight_sum = 0.0f;
  float effective_n = 0.0f;
  const int retained = sigma_clip<MaxFrames>(
      values, weights, k_effective,
      clip_sigma_low, clip_sigma_high, clip_iterations, min_fraction, min_n_eff,
      &weight_sum, &effective_n, scratch_buf);

  if (retained <= 0 || weight_sum <= 0.0f) {
    d_output[canvas_idx] = 0.0f;
    d_weight_sum[canvas_idx] = 0.0f;
    atomicAdd(d_unsupported_pixels, 1ULL);
    atomicAdd(d_numerical_guard_pixels, 1ULL);
    return;
  }

  double accum = 0.0;
  for (int i = 0; i < retained; ++i) {
    accum += static_cast<double>(weights[i]) * values[i];
  }
  d_output[canvas_idx] = static_cast<float>(accum / weight_sum);
  d_weight_sum[canvas_idx] = weight_sum;
}

// ---------------------------------------------------------------------------
// WP-E: Dequantize kernel — converts fp16 Q-Maps and/or bit-packed masks
// to full-precision float/byte in-place on the GPU.  Launched once per chunk
// before the main reconstruction kernel when the corresponding config flags
// are set (default: both false → no-op / not launched).
// ---------------------------------------------------------------------------
__global__ void aqmh_dequantize_kernel(
    const uint16_t* __restrict__ d_q_maps_half,  // null if !half_qmaps
    float* __restrict__ d_q_maps,                // float Q-Map to fill
    const uint8_t* __restrict__ d_masks_packed,  // null if !packed_masks
    uint8_t* __restrict__ d_masks,               // byte masks to fill
    bool half_qmaps, bool packed_masks, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (half_qmaps)
    d_q_maps[i] = __half2float(*reinterpret_cast<const __half*>(&d_q_maps_half[i]));
  if (packed_masks)
    d_masks[i] = (d_masks_packed[i >> 3] >> (i & 7)) & 1u;
}

// ---------------------------------------------------------------------------
// WP-B: aqmh_select_kernel — channel-independent selection phase.
// Runs once per chunk for all planes.  Filters valid frames by mask +
// Q-Map + global weight, then applies cherry-pick selection.
// Outputs sorted valid frame indices and count per pixel.
// ---------------------------------------------------------------------------
template <bool CherryPickEnabled, int MaxFrames>
__global__ __launch_bounds__(256, 2)
void aqmh_select_kernel(
    const float* __restrict__ d_q_maps,
    const uint8_t* __restrict__ d_canvas_mask,
    const uint8_t* __restrict__ d_frame_masks,
    const float* __restrict__ d_global_weights,
    int16_t* __restrict__ d_sel_indices,   // [chunk_pixels * frame_count]
    uint16_t* __restrict__ d_sel_k,        // [chunk_pixels]
    float* __restrict__ d_cherry_k_map,    // nullable
    int width, int chunk_rows, int y0, int height, int frame_count,
    float cherry_pick_k_frac, int cherry_pick_k_min_required, int cherry_pick_mode,
    float cherry_pick_reject_below_best_fraction,
    float cherry_pick_min_keep_fraction, float cherry_pick_margin_min) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= chunk_rows || y0 + yy >= height) return;

  const int canvas_idx = yy * width + x;
  d_sel_k[canvas_idx] = 0;
  if (d_cherry_k_map) d_cherry_k_map[canvas_idx] = 0.0f;
  if (d_canvas_mask[canvas_idx] == 0u) return;

  // Local arrays: scores for cherry-pick sorting, idx_buf for frame indices.
  float scores[CherryPickEnabled ? MaxFrames : 1];
  float wt_buf[MaxFrames];   // scratch (weights / sorting proxy)
  short sort_buf[MaxFrames];
  int n = 0;

  const int pixel_base = canvas_idx * frame_count;
  const int16_t* out_base = d_sel_indices + pixel_base;

  for (int fi = 0; fi < frame_count && n < MaxFrames; ++fi) {
    const int idx = pixel_base + fi;
    if (d_frame_masks[idx] == 0u) continue;
    const float q = d_q_maps[idx];
    if (!isfinite_f(q)) continue;
    const float gw = d_global_weights[fi];
    const float score = gw * fmaxf(0.0f, q);
    if (score <= 0.0f) continue;
    if constexpr (CherryPickEnabled) scores[n] = score;
    wt_buf[n] = static_cast<float>(fi);  // store fi in wt_buf as float (exact for fi < 2^24)
    sort_buf[n] = static_cast<short>(n);
    ++n;
  }

  int k = n;
  if constexpr (CherryPickEnabled) {
    if (n >= cherry_pick_k_min_required) {
      for (int i = n; i < MaxFrames; ++i) scores[i] = -INFINITY;
      adaptive_sort<MaxFrames>(sort_buf, n, ScoreDesc{scores});

      const float best = scores[sort_buf[0]];
      if (cherry_pick_mode == 1 && best > 0.0f && isfinite_f(best)) {
        // auto_reject
        const float threshold = best * fminf(1.0f, fmaxf(0.0f, cherry_pick_reject_below_best_fraction));
        int keep = 0;
        while (keep < n && scores[sort_buf[keep]] >= threshold) ++keep;
        const int min_keep = min(n, max(cherry_pick_k_min_required,
            static_cast<int>(ceilf(fminf(1.0f, fmaxf(0.0f, cherry_pick_min_keep_fraction))
                                   * static_cast<float>(n)))));
        keep = max(keep, min_keep);
        if (keep < n) {
          const float margin = (scores[sort_buf[keep - 1]] - scores[sort_buf[keep]]) / best;
          k = (margin >= cherry_pick_margin_min) ? keep : n;
        } else {
          k = n;
        }
      } else {
        // top_k
        const int nominal = max(0, static_cast<int>(floorf(cherry_pick_k_frac * static_cast<float>(n))));
        k = min(n, max(cherry_pick_k_min_required, nominal));
      }

      if (k < n) {
        // Reorder wt_buf (which holds fi values) to sort_buf[0..k-1] order.
        // We use scores[] as a float temp for the reordering.
        for (int i = 0; i < k; ++i) scores[i] = wt_buf[sort_buf[i]];
        for (int i = 0; i < k; ++i) wt_buf[i] = scores[i];
        n = k;
      }
    }
  }

  // Write selected frame indices to output buffer.
  for (int i = 0; i < n; ++i)
    const_cast<int16_t*>(out_base)[i] = static_cast<int16_t>(wt_buf[i]);
  d_sel_k[canvas_idx] = static_cast<uint16_t>(n);
  if (d_cherry_k_map) d_cherry_k_map[canvas_idx] = static_cast<float>(n);
}

// ---------------------------------------------------------------------------
// WP-B: aqmh_reduce_kernel — channel-specific sigma-clip + weighted mean.
// Runs once per plane per chunk.  Reads pre-selected frame indices from
// aqmh_select_kernel, loads channel values from d_frames, then applies
// sigma-clip and computes the weighted output.
// ---------------------------------------------------------------------------
template <int MaxFrames>
__global__ __launch_bounds__(256, 2)
void aqmh_reduce_kernel(
    const int16_t* __restrict__ d_sel_indices,  // [chunk_pixels * frame_count]
    const uint16_t* __restrict__ d_sel_k,       // [chunk_pixels]
    const float* __restrict__ d_frames,         // [chunk_pixels * frame_count]
    const float* __restrict__ d_q_maps,         // to re-derive weights
    const float* __restrict__ d_global_weights,
    const uint8_t* __restrict__ d_canvas_mask,
    float* __restrict__ d_output,
    float* __restrict__ d_weight_sum,
    unsigned long long* __restrict__ d_unsupported_pixels,
    unsigned long long* __restrict__ d_zero_veto_pixels,
    unsigned long long* __restrict__ d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height, int frame_count,
    float clip_sigma_low, float clip_sigma_high, int clip_iterations,
    float min_fraction, float min_n_eff) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= chunk_rows || y0 + yy >= height) return;

  const int canvas_idx = yy * width + x;
  d_output[canvas_idx] = 0.0f;
  d_weight_sum[canvas_idx] = 0.0f;
  if (d_canvas_mask[canvas_idx] == 0u) return;

  const int k_sel = static_cast<int>(d_sel_k[canvas_idx]);
  if (k_sel == 0) {
    atomicAdd(d_unsupported_pixels, 1ULL);
    return;
  }

  float values[MaxFrames];
  float weights[MaxFrames];
  short sort_buf[MaxFrames];
  int n = 0;
  bool has_finite_q = false;

  const int pixel_base = canvas_idx * frame_count;
  const int16_t* sel = d_sel_indices + pixel_base;

  for (int si = 0; si < k_sel && n < MaxFrames; ++si) {
    const int fi = static_cast<int>(sel[si]);
    const int idx = pixel_base + fi;
    const float v = d_frames[idx];
    if (!isfinite_f(v)) continue;
    const float q = d_q_maps[idx];
    if (!isfinite_f(q)) continue;
    has_finite_q = true;
    const float gw = d_global_weights[fi];
    const float score = gw * fmaxf(0.0f, q);
    if (score <= 0.0f) continue;
    values[n] = v;
    weights[n] = score;
    sort_buf[n] = static_cast<short>(n);
    ++n;
  }

  if (n == 0) {
    atomicAdd(d_unsupported_pixels, 1ULL);
    if (has_finite_q) atomicAdd(d_zero_veto_pixels, 1ULL);
    return;
  }

  for (int i = n; i < MaxFrames; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
  }

  float weight_sum = 0.0f;
  float effective_n = 0.0f;
  const int retained = sigma_clip<MaxFrames>(
      values, weights, n,
      clip_sigma_low, clip_sigma_high, clip_iterations, min_fraction, min_n_eff,
      &weight_sum, &effective_n, sort_buf);

  if (retained <= 0 || weight_sum <= 0.0f) {
    atomicAdd(d_unsupported_pixels, 1ULL);
    atomicAdd(d_numerical_guard_pixels, 1ULL);
    return;
  }

  double accum = 0.0;
  for (int i = 0; i < retained; ++i)
    accum += static_cast<double>(weights[i]) * values[i];
  d_output[canvas_idx] = static_cast<float>(accum / weight_sum);
  d_weight_sum[canvas_idx] = weight_sum;
}

// CUDA error check helper.
#define CUDA_CHECK(expr)                                                     \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "      \
                << cudaGetErrorString(err) << std::endl;                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct GpuBuffers {
  float* frames = nullptr;
  float* q_maps = nullptr;
  uint8_t* frame_masks = nullptr;
  uint8_t* canvas_mask = nullptr;
  float* output = nullptr;
  float* weight_sum = nullptr;
  float* uniform_control = nullptr;
  uint8_t* uniform_control_valid = nullptr;
  float* cherry_k_map = nullptr;
  // WP-E: optional half-precision / bit-packed staging buffers
  uint16_t* q_maps_half = nullptr;       // fp16 Q-Map staging (half the bytes)
  uint8_t*  frame_masks_packed = nullptr; // bit-packed masks (1 bit/pixel)
};

bool allocate_chunk_buffers(
    GpuBuffers& bufs, int width, int chunk_rows, int frame_count,
    bool compute_uniform_control, bool cherry_pick_enabled,
    bool half_qmaps = false, bool packed_masks = false) {
  const size_t chunk_pixels = static_cast<size_t>(chunk_rows) * width;
  const size_t all_chunk_pixels = static_cast<size_t>(frame_count) * chunk_pixels;
  CUDA_CHECK(cudaMalloc(&bufs.frames, all_chunk_pixels * sizeof(float)));
  // Float Q-Map buffer is always allocated (dequantize target when half_qmaps).
  CUDA_CHECK(cudaMalloc(&bufs.q_maps, all_chunk_pixels * sizeof(float)));
  if (half_qmaps)
    CUDA_CHECK(cudaMalloc(&bufs.q_maps_half, all_chunk_pixels * sizeof(uint16_t)));
  // Byte mask buffer is always allocated (dequantize target when packed_masks).
  CUDA_CHECK(cudaMalloc(&bufs.frame_masks, all_chunk_pixels * sizeof(uint8_t)));
  if (packed_masks)
    CUDA_CHECK(cudaMalloc(&bufs.frame_masks_packed,
                          (all_chunk_pixels + 7) / 8 * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&bufs.canvas_mask, chunk_pixels * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&bufs.output, chunk_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bufs.weight_sum, chunk_pixels * sizeof(float)));
  if (compute_uniform_control) {
    CUDA_CHECK(cudaMalloc(&bufs.uniform_control, chunk_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bufs.uniform_control_valid,
                          chunk_pixels * sizeof(uint8_t)));
  }
  if (cherry_pick_enabled) {
    CUDA_CHECK(cudaMalloc(&bufs.cherry_k_map, chunk_pixels * sizeof(float)));
  }
  return true;
}

void free_chunk_buffers(GpuBuffers& bufs) {
  if (bufs.frames) cudaFree(bufs.frames);
  if (bufs.q_maps) cudaFree(bufs.q_maps);
  if (bufs.q_maps_half) cudaFree(bufs.q_maps_half);
  if (bufs.frame_masks) cudaFree(bufs.frame_masks);
  if (bufs.frame_masks_packed) cudaFree(bufs.frame_masks_packed);
  if (bufs.canvas_mask) cudaFree(bufs.canvas_mask);
  if (bufs.output) cudaFree(bufs.output);
  if (bufs.weight_sum) cudaFree(bufs.weight_sum);
  if (bufs.uniform_control) cudaFree(bufs.uniform_control);
  if (bufs.uniform_control_valid) cudaFree(bufs.uniform_control_valid);
  if (bufs.cherry_k_map) cudaFree(bufs.cherry_k_map);
  bufs = GpuBuffers{};
}

template <int MaxFrames>
void launch_reconstruction_kernel(
    bool cherry_enabled,
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs,
    const float* d_global_weights,
    unsigned long long* d_unsupported_pixels,
    unsigned long long* d_zero_veto_pixels,
    unsigned long long* d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height, int frame_count,
    const AqmhReconstructionConfig& cfg) {
  const int cherry_pick_mode = cfg.cherry_pick_mode == "auto_reject" ? 1 : 0;
  if (cherry_enabled) {
    aqmh_reconstruction_kernel<true, MaxFrames><<<grid, block, 0, stream>>>(
        bufs.frames, bufs.q_maps, bufs.canvas_mask, bufs.frame_masks,
        d_global_weights, bufs.output, bufs.weight_sum,
        bufs.uniform_control, bufs.uniform_control_valid,
        bufs.cherry_k_map,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count,
        cfg.clip_sigma_low, cfg.clip_sigma_high, cfg.clip_iterations,
        cfg.min_fraction, cfg.min_n_eff,
        cfg.cherry_pick_k_frac,
        cfg.cherry_pick_k_min_required,
        cherry_pick_mode,
        cfg.cherry_pick_reject_below_best_fraction,
        cfg.cherry_pick_min_keep_fraction,
        cfg.cherry_pick_margin_min);
  } else {
    aqmh_reconstruction_kernel<false, MaxFrames><<<grid, block, 0, stream>>>(
        bufs.frames, bufs.q_maps, bufs.canvas_mask, bufs.frame_masks,
        d_global_weights, bufs.output, bufs.weight_sum,
        bufs.uniform_control, bufs.uniform_control_valid,
        bufs.cherry_k_map,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count,
        cfg.clip_sigma_low, cfg.clip_sigma_high, cfg.clip_iterations,
        cfg.min_fraction, cfg.min_n_eff,
        cfg.cherry_pick_k_frac,
        cfg.cherry_pick_k_min_required,
        cherry_pick_mode,
        cfg.cherry_pick_reject_below_best_fraction,
        cfg.cherry_pick_min_keep_fraction,
        cfg.cherry_pick_margin_min);
  }
}

void launch_reconstruction_kernel_for_frame_count(
    int frame_count,
    bool cherry_enabled,
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs,
    const float* d_global_weights,
    unsigned long long* d_unsupported_pixels,
    unsigned long long* d_zero_veto_pixels,
    unsigned long long* d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height,
    const AqmhReconstructionConfig& cfg) {
  if (frame_count <= 32) {
    launch_reconstruction_kernel<32>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 64) {
    launch_reconstruction_kernel<64>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 128) {
    launch_reconstruction_kernel<128>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 256) {
    launch_reconstruction_kernel<256>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 512) {
    launch_reconstruction_kernel<512>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 640) {
    // §8-F: Tighter tier for 513-640 frames — 37% less local memory than 1024.
    launch_reconstruction_kernel<640>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else if (frame_count <= 768) {
    // §8-F: Tighter tier for 641-768 frames — 25% less local memory than 1024.
    launch_reconstruction_kernel<768>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  } else {
    launch_reconstruction_kernel<1024>(
        cherry_enabled, grid, block, stream, bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, frame_count, cfg);
  }
}

// ---------------------------------------------------------------------------
// WP-E: launch the dequantize kernel for a chunk.
// n_elements = frame_count * chunk_pixels (one entry per (pixel,frame) pair).
// ---------------------------------------------------------------------------
void launch_dequantize(
    const GpuBuffers& bufs, cudaStream_t stream,
    bool half_qmaps, bool packed_masks, int n_elements) {
  if (!half_qmaps && !packed_masks) return;
  const int threads = 256;
  const int blocks = (n_elements + threads - 1) / threads;
  aqmh_dequantize_kernel<<<blocks, threads, 0, stream>>>(
      half_qmaps ? bufs.q_maps_half : nullptr, bufs.q_maps,
      packed_masks ? bufs.frame_masks_packed : nullptr, bufs.frame_masks,
      half_qmaps, packed_masks, n_elements);
}

// ---------------------------------------------------------------------------
// WP-B: dispatch aqmh_select_kernel for the given frame_count tier.
// ---------------------------------------------------------------------------
template <int MaxFrames>
void launch_select_kernel(
    bool cherry_enabled,
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs,
    const float* d_global_weights,
    int16_t* d_sel_indices, uint16_t* d_sel_k,
    int width, int chunk_rows, int y0, int height, int frame_count,
    const AqmhReconstructionConfig& cfg) {
  const int cherry_pick_mode = cfg.cherry_pick_mode == "auto_reject" ? 1 : 0;
  if (cherry_enabled) {
    aqmh_select_kernel<true, MaxFrames><<<grid, block, 0, stream>>>(
        bufs.q_maps, bufs.canvas_mask, bufs.frame_masks, d_global_weights,
        d_sel_indices, d_sel_k, bufs.cherry_k_map,
        width, chunk_rows, y0, height, frame_count,
        cfg.cherry_pick_k_frac, cfg.cherry_pick_k_min_required, cherry_pick_mode,
        cfg.cherry_pick_reject_below_best_fraction,
        cfg.cherry_pick_min_keep_fraction, cfg.cherry_pick_margin_min);
  } else {
    aqmh_select_kernel<false, MaxFrames><<<grid, block, 0, stream>>>(
        bufs.q_maps, bufs.canvas_mask, bufs.frame_masks, d_global_weights,
        d_sel_indices, d_sel_k, nullptr,
        width, chunk_rows, y0, height, frame_count,
        cfg.cherry_pick_k_frac, cfg.cherry_pick_k_min_required, cherry_pick_mode,
        cfg.cherry_pick_reject_below_best_fraction,
        cfg.cherry_pick_min_keep_fraction, cfg.cherry_pick_margin_min);
  }
}

template <int MaxFrames>
void launch_reduce_kernel(
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs,
    const float* d_global_weights,
    const int16_t* d_sel_indices, const uint16_t* d_sel_k,
    unsigned long long* d_unsupported, unsigned long long* d_zero_veto,
    unsigned long long* d_numerical_guard,
    int width, int chunk_rows, int y0, int height, int frame_count,
    const AqmhReconstructionConfig& cfg) {
  aqmh_reduce_kernel<MaxFrames><<<grid, block, 0, stream>>>(
      d_sel_indices, d_sel_k,
      bufs.frames, bufs.q_maps, d_global_weights, bufs.canvas_mask,
      bufs.output, bufs.weight_sum,
      d_unsupported, d_zero_veto, d_numerical_guard,
      width, chunk_rows, y0, height, frame_count,
      cfg.clip_sigma_low, cfg.clip_sigma_high, cfg.clip_iterations,
      cfg.min_fraction, cfg.min_n_eff);
}

void launch_select_kernel_for_frame_count(
    int frame_count, bool cherry_enabled,
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs, const float* d_global_weights,
    int16_t* d_sel_indices, uint16_t* d_sel_k,
    int width, int chunk_rows, int y0, int height,
    const AqmhReconstructionConfig& cfg) {
  if (frame_count <= 32)
    launch_select_kernel<32>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 64)
    launch_select_kernel<64>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 128)
    launch_select_kernel<128>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 256)
    launch_select_kernel<256>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 512)
    launch_select_kernel<512>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 640)
    launch_select_kernel<640>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 768)
    launch_select_kernel<768>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
  else
    launch_select_kernel<1024>(cherry_enabled, grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, width, chunk_rows, y0, height, frame_count, cfg);
}

void launch_reduce_kernel_for_frame_count(
    int frame_count,
    dim3 grid, dim3 block, cudaStream_t stream,
    const GpuBuffers& bufs, const float* d_global_weights,
    const int16_t* d_sel_indices, const uint16_t* d_sel_k,
    unsigned long long* d_unsupported, unsigned long long* d_zero_veto,
    unsigned long long* d_numerical_guard,
    int width, int chunk_rows, int y0, int height,
    const AqmhReconstructionConfig& cfg) {
  if (frame_count <= 32)
    launch_reduce_kernel<32>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 64)
    launch_reduce_kernel<64>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 128)
    launch_reduce_kernel<128>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 256)
    launch_reduce_kernel<256>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 512)
    launch_reduce_kernel<512>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 640)
    launch_reduce_kernel<640>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else if (frame_count <= 768)
    launch_reduce_kernel<768>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
  else
    launch_reduce_kernel<1024>(grid, block, stream, bufs, d_global_weights, d_sel_indices, d_sel_k, d_unsupported, d_zero_veto, d_numerical_guard, width, chunk_rows, y0, height, frame_count, cfg);
}

template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() : ptr_(nullptr), size_(0), is_pinned_(false) {}
  explicit PinnedBuffer(size_t n, T init_val = T()) : ptr_(nullptr), size_(n), is_pinned_(false) {
    if (n > 0) {
      cudaError_t err = cudaHostAlloc(&ptr_, n * sizeof(T), cudaHostAllocPortable);
      if (err != cudaSuccess || !ptr_) {
        ptr_ = static_cast<T*>(std::malloc(n * sizeof(T)));
        is_pinned_ = false;
      } else {
        is_pinned_ = true;
      }
      if (ptr_) {
        std::fill(ptr_, ptr_ + n, init_val);
      }
    }
  }
  ~PinnedBuffer() {
    if (ptr_) {
      if (is_pinned_) {
        cudaFreeHost(ptr_);
      } else {
        std::free(ptr_);
      }
      ptr_ = nullptr;
    }
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_), is_pinned_(o.is_pinned_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
    o.is_pinned_ = false;
  }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_) {
        if (is_pinned_) cudaFreeHost(ptr_);
        else std::free(ptr_);
      }
      ptr_ = o.ptr_;
      size_ = o.size_;
      is_pinned_ = o.is_pinned_;
      o.ptr_ = nullptr;
      o.size_ = 0;
      o.is_pinned_ = false;
    }
    return *this;
  }
  void assign(size_t n, T val) {
    if (n != size_ || !ptr_) {
      if (ptr_) {
        if (is_pinned_) cudaFreeHost(ptr_);
        else std::free(ptr_);
        ptr_ = nullptr;
      }
      size_ = n;
      if (n > 0) {
        cudaError_t err = cudaHostAlloc(&ptr_, n * sizeof(T), cudaHostAllocPortable);
        if (err != cudaSuccess || !ptr_) {
          ptr_ = static_cast<T*>(std::malloc(n * sizeof(T)));
          is_pinned_ = false;
        } else {
          is_pinned_ = true;
        }
      }
    }
    if (ptr_ && size_ > 0) {
      std::fill(ptr_, ptr_ + size_, val);
    }
  }
  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0 || ptr_ == nullptr; }
  T& operator[](size_t i) noexcept { return ptr_[i]; }
  const T& operator[](size_t i) const noexcept { return ptr_[i]; }
  T* begin() noexcept { return ptr_; }
  T* end() noexcept { return ptr_ + size_; }
  const T* begin() const noexcept { return ptr_; }
  const T* end() const noexcept { return ptr_ + size_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
  bool is_pinned_ = false;
};

#undef CUDA_CHECK

} // namespace

AqmhReconstructionResult reconstruct_aqmh_weighted_cuda(
    size_t frame_count,
    const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache,
    const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask,
    int width, int height,
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
  if (cfg.compute_uniform_control)
    result.uniform_control_valid_mask.assign(
        static_cast<size_t>(height) * static_cast<size_t>(width), 0u);

  if (frame_count == 0 || !q_map_cache || width <= 0 || height <= 0) {
    result.acceleration_fallback = true;
    return result;
  }

  // Validate frame_count against compile-time limit.
  if (static_cast<int>(frame_count) > kMaxFramesCompile) {
    std::cerr << "[CUDA] frame_count " << frame_count
              << " exceeds compile-time limit " << kMaxFramesCompile
              << "; falling back to CPU" << std::endl;
    result = reconstruct_aqmh_weighted(
        frame_count, load_frame, q_map_cache, global_weights, canvas_mask,
        width, height, cfg, load_frame_valid_mask, load_frame_region,
        load_frame_valid_mask_region, progress);
    result.acceleration_fallback = true;
    return result;
  }

  // Simple global cherry-pick enable check (conservative vs CPU median heuristic).
  const bool cherry_enabled = cfg.cherry_pick &&
      static_cast<int>(frame_count) >= cfg.cherry_pick_k_min_required;
  if (cherry_enabled) {
    result.cherry_pick_k_map = Matrix2Df::Zero(height, width);
    result.cherry_pick_per_pixel_mode = true;
  }
  result.k_nominal_median = static_cast<float>(
      cherry_enabled
          ? (cfg.cherry_pick_mode == "auto_reject"
                 ? static_cast<int>(frame_count)
                 : cfg.cherry_pick_k_min_required)
          : 0);

  // Determine GPU chunk size from available memory.
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (mem_err != cudaSuccess) {
    std::cerr << "[CUDA] cudaMemGetInfo failed: " << cudaGetErrorString(mem_err)
              << std::endl;
    result.acceleration_fallback = true;
    return result;
  }
  const size_t device_budget = std::min<size_t>(
      static_cast<size_t>(0.60 * static_cast<double>(free_bytes)),
      4ULL * kBytesPerGiB);
  result.cuda_free_bytes = static_cast<uint64_t>(free_bytes);
  result.cuda_total_bytes = static_cast<uint64_t>(total_bytes);
  result.cuda_device_budget_bytes = static_cast<uint64_t>(device_budget);

  // Per-row estimate: frames + q_maps + masks + output scratch (3 floats + 1 mask)
  const size_t bytes_per_row =
      static_cast<size_t>(frame_count) * width * sizeof(float) * 2 +
      static_cast<size_t>(frame_count) * width * sizeof(uint8_t) +
      static_cast<size_t>(width) * sizeof(float) * 4 +
      static_cast<size_t>(width) * sizeof(uint8_t) *
          (cfg.compute_uniform_control ? 2u : 1u);
  result.cuda_bytes_per_row = static_cast<uint64_t>(bytes_per_row);
  int chunk_rows;
  if (cfg.chunk_rows > 0) {
    chunk_rows = std::min(height, cfg.chunk_rows);
  } else {
    const int budget_rows = static_cast<int>(
        device_budget / std::max<size_t>(1, bytes_per_row));
    chunk_rows = std::min(height, std::max(kMinAutoChunkRows, budget_rows));
  }
  result.cuda_auto_chunk_rows_initial = chunk_rows;
  result.chunk_rows = chunk_rows;
  result.chunk_count = (height + chunk_rows - 1) / chunk_rows;
  result.region_streaming_used = true;

  // Global weights upload.
  std::vector<float> h_global_weights(frame_count, 1.0f);
  for (Eigen::Index fi = 0; fi < global_weights.size() && fi < static_cast<Eigen::Index>(frame_count); ++fi) {
    const float w = global_weights[fi];
    h_global_weights[fi] = std::isfinite(w) && w > 0.0f ? w : 0.0f;
  }
  float* d_global_weights = nullptr;
#define CUDA_CHECK(expr)                                                     \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "      \
                << cudaGetErrorString(err) << std::endl;                       \
      result.acceleration_fallback = true;                                     \
      return result;                                                           \
    }                                                                          \
  } while (0)
  CUDA_CHECK(cudaMalloc(&d_global_weights, frame_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_global_weights, h_global_weights.data(),
                        frame_count * sizeof(float), cudaMemcpyHostToDevice));

  // P3: Adaptive double-buffering — allocate one buffer set at full budget,
  // then try to allocate a second set for stream overlap. If VRAM is
  // insufficient for two sets, fall back to single-buffer single-stream mode.
  GpuBuffers bufs;
  while (!allocate_chunk_buffers(bufs, width, chunk_rows, static_cast<int>(frame_count),
                                 cfg.compute_uniform_control, cherry_enabled)) {
    free_chunk_buffers(bufs);
    if (cfg.chunk_rows > 0 || chunk_rows <= 1) {
      cudaFree(d_global_weights);
      result.acceleration_fallback = true;
      return result;
    }
    chunk_rows = std::max(1, chunk_rows / 2);
    ++result.cuda_allocation_retries;
    result.chunk_rows = chunk_rows;
    result.chunk_count = (height + chunk_rows - 1) / chunk_rows;
  }

  // Try to allocate a second buffer set for double-buffering.
  GpuBuffers bufs2;
  const bool use_double_buffer = allocate_chunk_buffers(
      bufs2, width, chunk_rows, static_cast<int>(frame_count),
      cfg.compute_uniform_control, cherry_enabled);
  if (!use_double_buffer) {
    free_chunk_buffers(bufs2);
  }

  // Pinned host staging buffers.
  PinnedBuffer<float> h_frames(
      static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
  PinnedBuffer<float> h_q_maps(
      static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
  PinnedBuffer<uint8_t> h_masks(
      static_cast<size_t>(frame_count) * chunk_rows * width, 0u);
  PinnedBuffer<uint8_t> h_canvas_mask(
      static_cast<size_t>(chunk_rows) * width, 0u);
  PinnedBuffer<float> h_output(
      static_cast<size_t>(chunk_rows) * width, 0.0f);
  PinnedBuffer<float> h_weight_sum(
      static_cast<size_t>(chunk_rows) * width, 0.0f);
  PinnedBuffer<float> h_uniform_control;
  PinnedBuffer<uint8_t> h_uniform_control_valid;
  if (cfg.compute_uniform_control) {
    h_uniform_control.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
    h_uniform_control_valid.assign(static_cast<size_t>(chunk_rows) * width, 0u);
  }
  PinnedBuffer<float> h_cherry_k_map;
  if (cherry_enabled) {
    h_cherry_k_map.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
  }

  // Second set of pinned buffers for double-buffering.
  PinnedBuffer<float> h_frames2, h_q_maps2, h_output2, h_weight_sum2;
  PinnedBuffer<uint8_t> h_masks2, h_canvas_mask2;
  PinnedBuffer<float> h_uniform_control2, h_cherry_k_map2;
  PinnedBuffer<uint8_t> h_uniform_control_valid2;
  if (use_double_buffer) {
    h_frames2 = PinnedBuffer<float>(
        static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
    h_q_maps2 = PinnedBuffer<float>(
        static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
    h_masks2 = PinnedBuffer<uint8_t>(
        static_cast<size_t>(frame_count) * chunk_rows * width, 0u);
    h_canvas_mask2 = PinnedBuffer<uint8_t>(
        static_cast<size_t>(chunk_rows) * width, 0u);
    h_output2 = PinnedBuffer<float>(
        static_cast<size_t>(chunk_rows) * width, 0.0f);
    h_weight_sum2 = PinnedBuffer<float>(
        static_cast<size_t>(chunk_rows) * width, 0.0f);
    if (cfg.compute_uniform_control) {
      h_uniform_control2.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
      h_uniform_control_valid2.assign(static_cast<size_t>(chunk_rows) * width, 0u);
    }
    if (cherry_enabled) {
      h_cherry_k_map2.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
    }
  }

  unsigned long long* d_unsupported_pixels = nullptr;
  unsigned long long* d_zero_veto_pixels = nullptr;
  unsigned long long* d_numerical_guard_pixels = nullptr;
  cudaStream_t stream = nullptr;
  cudaStream_t stream2 = nullptr;
  cudaEvent_t h2d_start = nullptr, kernel_start = nullptr;
  cudaEvent_t kernel_end = nullptr, d2h_end = nullptr;
  cudaEvent_t h2d_start2 = nullptr, kernel_start2 = nullptr;
  cudaEvent_t kernel_end2 = nullptr, d2h_end2 = nullptr;

  auto cleanup_on_error = [&]() {
    cudaFree(d_global_weights);
    free_chunk_buffers(bufs);
    if (use_double_buffer) free_chunk_buffers(bufs2);
    cudaFree(d_unsupported_pixels);
    cudaFree(d_zero_veto_pixels);
    cudaFree(d_numerical_guard_pixels);
    if (stream) cudaStreamDestroy(stream);
    if (stream2) cudaStreamDestroy(stream2);
    if (h2d_start) cudaEventDestroy(h2d_start);
    if (kernel_start) cudaEventDestroy(kernel_start);
    if (kernel_end) cudaEventDestroy(kernel_end);
    if (d2h_end) cudaEventDestroy(d2h_end);
    if (h2d_start2) cudaEventDestroy(h2d_start2);
    if (kernel_start2) cudaEventDestroy(kernel_start2);
    if (kernel_end2) cudaEventDestroy(kernel_end2);
    if (d2h_end2) cudaEventDestroy(d2h_end2);
  };
#define CUDA_CHECK_ALLOC(expr)                                                \
  do {                                                                        \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "      \
                << cudaGetErrorString(err) << std::endl;                       \
      cleanup_on_error();                                                      \
      result.acceleration_fallback = true;                                     \
      return result;                                                           \
    }                                                                          \
  } while (0)

  CUDA_CHECK_ALLOC(cudaMalloc(&d_unsupported_pixels, sizeof(unsigned long long)));
  CUDA_CHECK_ALLOC(cudaMalloc(&d_zero_veto_pixels, sizeof(unsigned long long)));
  CUDA_CHECK_ALLOC(cudaMalloc(&d_numerical_guard_pixels, sizeof(unsigned long long)));
  CUDA_CHECK_ALLOC(cudaMemsetAsync(d_unsupported_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK_ALLOC(cudaMemsetAsync(d_zero_veto_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK_ALLOC(cudaMemsetAsync(d_numerical_guard_pixels, 0, sizeof(unsigned long long), 0));

  const dim3 block(32, 8);
  CUDA_CHECK_ALLOC(cudaStreamCreate(&stream));
  if (use_double_buffer) {
    CUDA_CHECK_ALLOC(cudaStreamCreate(&stream2));
  }
  CUDA_CHECK_ALLOC(cudaEventCreate(&h2d_start));
  CUDA_CHECK_ALLOC(cudaEventCreate(&kernel_start));
  CUDA_CHECK_ALLOC(cudaEventCreate(&kernel_end));
  CUDA_CHECK_ALLOC(cudaEventCreate(&d2h_end));
  if (use_double_buffer) {
    CUDA_CHECK_ALLOC(cudaEventCreate(&h2d_start2));
    CUDA_CHECK_ALLOC(cudaEventCreate(&kernel_start2));
    CUDA_CHECK_ALLOC(cudaEventCreate(&kernel_end2));
    CUDA_CHECK_ALLOC(cudaEventCreate(&d2h_end2));
  }
#undef CUDA_CHECK_ALLOC

  // Prepare the next frame regions while the current chunk is executing on
  // the device.  The loader owns the returned matrices, so the current chunk
  // remains immutable until its host packing has completed.  This is a
  // conservative first overlap step: Q-map/mask loading stays synchronous,
  // preserving their existing cache and error semantics.
  std::future<std::vector<Matrix2Df>> next_frame_prefetch;
  bool have_prefetched_frames = false;
  std::vector<Matrix2Df> prefetched_frames;
  const bool can_prefetch_frames = static_cast<bool>(load_frame_region);
  auto launch_frame_prefetch = [&](int next_y0, int next_rows) {
    return std::async(std::launch::async, [&, next_y0, next_rows]() {
      std::vector<Matrix2Df> frames(frame_count);
      for (size_t fi = 0; fi < frame_count; ++fi) {
        load_frame_region(fi, next_y0, next_rows, frames[fi]);
      }
      return frames;
    });
  };

  // Main chunk processing loop.
  // Single-buffer mode: sequential H2D → kernel → D2H → sync → commit.
  // Double-buffer mode: ping-pong with two streams, H2D[k+1] overlaps Kernel[k].
  struct PendingChunk { int y0 = 0; int rows = 0; bool valid = false; };
  PendingChunk pending_slot0, pending_slot1;
  int chunk_idx = 0;

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);

    // Select buffer set and stream for this chunk.
    const bool use_slot2 = use_double_buffer && (chunk_idx % 2 == 1);
    cudaStream_t& cur_stream = use_slot2 ? stream2 : stream;
    GpuBuffers& cur_bufs = use_slot2 ? bufs2 : bufs;
    PinnedBuffer<float>& cur_h_frames = use_slot2 ? h_frames2 : h_frames;
    PinnedBuffer<float>& cur_h_q_maps = use_slot2 ? h_q_maps2 : h_q_maps;
    PinnedBuffer<uint8_t>& cur_h_masks = use_slot2 ? h_masks2 : h_masks;
    PinnedBuffer<uint8_t>& cur_h_canvas_mask = use_slot2 ? h_canvas_mask2 : h_canvas_mask;
    PinnedBuffer<float>& cur_h_output = use_slot2 ? h_output2 : h_output;
    PinnedBuffer<float>& cur_h_weight_sum = use_slot2 ? h_weight_sum2 : h_weight_sum;
    PinnedBuffer<float>& cur_h_uniform_control = use_slot2 ? h_uniform_control2 : h_uniform_control;
    PinnedBuffer<uint8_t>& cur_h_uniform_control_valid = use_slot2 ? h_uniform_control_valid2 : h_uniform_control_valid;
    PinnedBuffer<float>& cur_h_cherry_k_map = use_slot2 ? h_cherry_k_map2 : h_cherry_k_map;
    cudaEvent_t& cur_h2d_start = use_slot2 ? h2d_start2 : h2d_start;
    cudaEvent_t& cur_kernel_start = use_slot2 ? kernel_start2 : kernel_start;
    cudaEvent_t& cur_kernel_end = use_slot2 ? kernel_end2 : kernel_end;
    cudaEvent_t& cur_d2h_end = use_slot2 ? d2h_end2 : d2h_end;

    // In double-buffer mode, sync and commit the previous chunk on this slot.
    if (use_double_buffer) {
      PendingChunk& prev = use_slot2 ? pending_slot1 : pending_slot0;
      if (prev.valid) {
        CUDA_CHECK(cudaStreamSynchronize(cur_stream));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_h2d_start, cur_kernel_start));
        result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_kernel_start, cur_kernel_end));
        result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_kernel_end, cur_d2h_end));
        result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

        const auto result_commit_start = std::chrono::steady_clock::now();
        const int p_y0 = prev.y0;
        const int p_rows = prev.rows;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(p_rows > 4)
        #endif
        for (int yy = 0; yy < p_rows; ++yy) {
          const int y = p_y0 + yy;
          const size_t src_off = static_cast<size_t>(yy) * width;
          std::memcpy(&result.output(y, 0), cur_h_output.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          std::memcpy(&result.weight_sum(y, 0), cur_h_weight_sum.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          if (cfg.compute_uniform_control) {
            std::memcpy(&result.uniform_control_output(y, 0),
                        cur_h_uniform_control.data() + src_off,
                        static_cast<size_t>(width) * sizeof(float));
            std::memcpy(result.uniform_control_valid_mask.data() +
                            static_cast<size_t>(y) * width,
                        cur_h_uniform_control_valid.data() + src_off,
                        static_cast<size_t>(width));
          }
          if (cherry_enabled)
            std::memcpy(&result.cherry_pick_k_map(y, 0),
                        cur_h_cherry_k_map.data() + src_off,
                        static_cast<size_t>(width) * sizeof(float));
        }
        result.cuda_result_commit_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - result_commit_start)
                .count();
        if (progress) progress(p_y0 + p_rows, height);
        prev.valid = false;
      }
    }

    const auto host_prepare_start = std::chrono::steady_clock::now();

    if (have_prefetched_frames) {
      prefetched_frames = next_frame_prefetch.get();
      have_prefetched_frames = false;
    }

    // Masks must be zeroed each chunk so frames that fail to load are skipped.
    std::fill(cur_h_masks.begin(), cur_h_masks.end(), 0u);

    // Prepare canvas mask slice.
    for (int yy = 0; yy < rows; ++yy) {
      const int y = y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t full_i = static_cast<size_t>(y) * width + x;
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        cur_h_canvas_mask[local_i] =
            (canvas_mask.empty() || full_i >= canvas_mask.size()) ? 1u
                                                                  : canvas_mask[full_i];
      }
    }
    result.cuda_host_chunk_setup_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - host_prepare_start)
            .count();

    // Load all frames / q-maps / masks for this chunk using region loaders
    // when available (avoids loading full W×H frames per chunk — only rows
    // rows are needed, cutting I/O by ~height/chunk_rows×).
    // Parallelized across frames with OpenMP for I/O overlap.
    const bool use_region = static_cast<bool>(load_frame_region);
    const int num_host_threads = std::min(
        static_cast<int>(frame_count),
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
    double frame_read_worker_seconds = 0.0;
    double q_map_read_worker_seconds = 0.0;
    double mask_read_worker_seconds = 0.0;
    double pack_worker_seconds = 0.0;
    #if defined(_OPENMP)
    #pragma omp parallel for num_threads(num_host_threads) schedule(dynamic, 4) \
        reduction(+:frame_read_worker_seconds,q_map_read_worker_seconds, \
                    mask_read_worker_seconds,pack_worker_seconds)
    #endif
    for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(frame_count); ++fi_ptr) {
      const size_t fi = static_cast<size_t>(fi_ptr);
      Matrix2Df frame_region;
      const auto frame_read_start = std::chrono::steady_clock::now();
      bool frame_ok = false;
      if (use_region && !prefetched_frames.empty() &&
          fi < prefetched_frames.size()) {
        frame_region = std::move(prefetched_frames[fi]);
        frame_ok = frame_region.size() > 0;
      } else {
        frame_ok = use_region
            ? load_frame_region(fi, y0, rows, frame_region)
            : load_frame(fi, frame_region);
      }
      frame_read_worker_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - frame_read_start)
              .count();
      // For full-frame loads (no region loader) the frame has height rows;
      // for region loads it has exactly 'rows' rows.
      const int expected_rows = use_region ? rows : height;
      if (!frame_ok || frame_region.rows() != expected_rows ||
          frame_region.cols() != width) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }
      const auto q_map_read_start = std::chrono::steady_clock::now();
      Matrix2Df q = use_region
          ? q_map_cache->read_region(fi, y0, rows)
          : q_map_cache->read_cached(fi);
      q_map_read_worker_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - q_map_read_start)
              .count();
      const int expected_q_rows = use_region ? rows : height;
      const bool q_map_ok =
          q.rows() == expected_q_rows && q.cols() == width;
      if (!q_map_ok) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
      }

      std::vector<uint8_t> fm;
      bool mask_ok = true;
      const auto mask_read_start = std::chrono::steady_clock::now();
      if (load_frame_valid_mask_region && use_region) {
        mask_ok = load_frame_valid_mask_region(fi, y0, rows, fm);
      } else if (load_frame_valid_mask) {
        mask_ok = load_frame_valid_mask(fi, fm);
      }
      mask_read_worker_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - mask_read_start)
              .count();
      const size_t expected_mask_size =
          (load_frame_valid_mask_region && use_region)
              ? static_cast<size_t>(width) * rows
              : (load_frame_valid_mask
                     ? static_cast<size_t>(width) * height
                     : 0u);
      if (!mask_ok ||
          (expected_mask_size > 0 && fm.size() != expected_mask_size)) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }

      uint64_t local_missing = 0;
      uint64_t local_finite = 0;
      const auto pack_start = std::chrono::steady_clock::now();
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t full_i = static_cast<size_t>(y) * width + x;
          const size_t local_i = static_cast<size_t>(yy) * width + x;
          const size_t idx =
              static_cast<size_t>(local_i) * frame_count + fi;
          // For full-frame (non-region) loads, offset by y0 within the matrix.
          const int fr_row = use_region ? yy : (y0 + yy);
          cur_h_frames[idx] = frame_region(fr_row, x);
          cur_h_q_maps[idx] =
              q_map_ok ? q(fr_row, x)
                       : std::numeric_limits<float>::quiet_NaN();
          if (fm.empty()) {
            cur_h_masks[idx] = 1u;
          } else {
            const size_t mask_i = use_region ? local_i : full_i;
            cur_h_masks[idx] = fm[mask_i];
            if (fm[mask_i] == 0u) continue;
          }
          if (cur_h_canvas_mask[local_i] == 0u) continue;
          if (!q_map_ok)
            continue;
          if (!std::isfinite(q(fr_row, x))) {
            ++local_missing;
          } else {
            ++local_finite;
          }
        }
      }
      pack_worker_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - pack_start)
              .count();
      if (local_missing > 0) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += local_missing;
      }
      if (local_finite > 0) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.finite_map_samples += local_finite;
      }
    }
    result.cuda_host_frame_read_worker_seconds += frame_read_worker_seconds;
    result.cuda_host_q_map_read_worker_seconds += q_map_read_worker_seconds;
    result.cuda_host_mask_read_worker_seconds += mask_read_worker_seconds;
    result.cuda_host_pack_worker_seconds += pack_worker_seconds;
    result.cuda_host_prepare_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - host_prepare_start)
            .count();

    if (can_prefetch_frames && y0 + rows < height) {
      const int next_y0 = y0 + rows;
      const int next_rows = std::min(chunk_rows, height - next_y0);
      next_frame_prefetch = launch_frame_prefetch(next_y0, next_rows);
      have_prefetched_frames = true;
    }

    // Upload frame/q-map/mask chunk on current stream.
    const size_t used_all_pixels = static_cast<size_t>(frame_count) * rows * width;
    CUDA_CHECK(cudaEventRecord(cur_h2d_start, cur_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cur_bufs.canvas_mask, cur_h_canvas_mask.data(),
        static_cast<size_t>(rows) * width * sizeof(uint8_t),
        cudaMemcpyHostToDevice, cur_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cur_bufs.frames, cur_h_frames.data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, cur_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cur_bufs.q_maps, cur_h_q_maps.data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, cur_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cur_bufs.frame_masks, cur_h_masks.data(),
        used_all_pixels * sizeof(uint8_t),
        cudaMemcpyHostToDevice, cur_stream));
    CUDA_CHECK(cudaEventRecord(cur_kernel_start, cur_stream));

    const dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    launch_reconstruction_kernel_for_frame_count(
        static_cast<int>(frame_count), cherry_enabled,
        grid, block, cur_stream, cur_bufs, d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, cfg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(cur_kernel_end, cur_stream));

    // Download outputs on current stream.
    const size_t used_chunk_pixels = static_cast<size_t>(rows) * width;
    CUDA_CHECK(cudaMemcpyAsync(
        cur_h_output.data(), cur_bufs.output,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, cur_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cur_h_weight_sum.data(), cur_bufs.weight_sum,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, cur_stream));
    if (cfg.compute_uniform_control) {
      CUDA_CHECK(cudaMemcpyAsync(
          cur_h_uniform_control.data(), cur_bufs.uniform_control,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, cur_stream));
      CUDA_CHECK(cudaMemcpyAsync(
          cur_h_uniform_control_valid.data(), cur_bufs.uniform_control_valid,
          used_chunk_pixels * sizeof(uint8_t),
          cudaMemcpyDeviceToHost, cur_stream));
    }
    if (cherry_enabled) {
      CUDA_CHECK(cudaMemcpyAsync(
          cur_h_cherry_k_map.data(), cur_bufs.cherry_k_map,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, cur_stream));
    }
    CUDA_CHECK(cudaEventRecord(cur_d2h_end, cur_stream));

    if (use_double_buffer) {
      // Mark this slot as pending for later commit.
      PendingChunk& cur_pending = use_slot2 ? pending_slot1 : pending_slot0;
      cur_pending = {y0, rows, true};
    } else {
      // Single-buffer mode: sync and commit immediately.
      CUDA_CHECK(cudaStreamSynchronize(cur_stream));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_h2d_start, cur_kernel_start));
      result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_kernel_start, cur_kernel_end));
      result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, cur_kernel_end, cur_d2h_end));
      result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

      const auto result_commit_start = std::chrono::steady_clock::now();
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static) if(rows > 4)
      #endif
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        const size_t src_off = static_cast<size_t>(yy) * width;
        std::memcpy(&result.output(y, 0), cur_h_output.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        std::memcpy(&result.weight_sum(y, 0), cur_h_weight_sum.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        if (cfg.compute_uniform_control) {
          std::memcpy(&result.uniform_control_output(y, 0),
                      cur_h_uniform_control.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          std::memcpy(result.uniform_control_valid_mask.data() +
                          static_cast<size_t>(y) * width,
                      cur_h_uniform_control_valid.data() + src_off,
                      static_cast<size_t>(width));
        }
        if (cherry_enabled)
          std::memcpy(&result.cherry_pick_k_map(y, 0),
                      cur_h_cherry_k_map.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
      }
      result.cuda_result_commit_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - result_commit_start)
              .count();
      if (progress) progress(y0 + rows, height);
    }
    ++chunk_idx;
  }

  // Commit remaining pending chunks from double-buffer mode.
  if (use_double_buffer) {
    if (pending_slot0.valid) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_start, kernel_start));
      result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_end));
      result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_end, d2h_end));
      result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

      const auto result_commit_start = std::chrono::steady_clock::now();
      const int p_y0 = pending_slot0.y0;
      const int p_rows = pending_slot0.rows;
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static) if(p_rows > 4)
      #endif
      for (int yy = 0; yy < p_rows; ++yy) {
        const int y = p_y0 + yy;
        const size_t src_off = static_cast<size_t>(yy) * width;
        std::memcpy(&result.output(y, 0), h_output.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        std::memcpy(&result.weight_sum(y, 0), h_weight_sum.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        if (cfg.compute_uniform_control) {
          std::memcpy(&result.uniform_control_output(y, 0),
                      h_uniform_control.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          std::memcpy(result.uniform_control_valid_mask.data() +
                          static_cast<size_t>(y) * width,
                      h_uniform_control_valid.data() + src_off,
                      static_cast<size_t>(width));
        }
        if (cherry_enabled)
          std::memcpy(&result.cherry_pick_k_map(y, 0),
                      h_cherry_k_map.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
      }
      result.cuda_result_commit_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - result_commit_start)
              .count();
      if (progress) progress(p_y0 + p_rows, height);
    }
    if (pending_slot1.valid) {
      CUDA_CHECK(cudaStreamSynchronize(stream2));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_start2, kernel_start2));
      result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_start2, kernel_end2));
      result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_end2, d2h_end2));
      result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

      const auto result_commit_start = std::chrono::steady_clock::now();
      const int p_y0 = pending_slot1.y0;
      const int p_rows = pending_slot1.rows;
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static) if(p_rows > 4)
      #endif
      for (int yy = 0; yy < p_rows; ++yy) {
        const int y = p_y0 + yy;
        const size_t src_off = static_cast<size_t>(yy) * width;
        std::memcpy(&result.output(y, 0), h_output2.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        std::memcpy(&result.weight_sum(y, 0), h_weight_sum2.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        if (cfg.compute_uniform_control) {
          std::memcpy(&result.uniform_control_output(y, 0),
                      h_uniform_control2.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          std::memcpy(result.uniform_control_valid_mask.data() +
                          static_cast<size_t>(y) * width,
                      h_uniform_control_valid2.data() + src_off,
                      static_cast<size_t>(width));
        }
        if (cherry_enabled)
          std::memcpy(&result.cherry_pick_k_map(y, 0),
                      h_cherry_k_map2.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
      }
      result.cuda_result_commit_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - result_commit_start)
              .count();
      if (progress) progress(p_y0 + p_rows, height);
    }
  }

  cudaEventDestroy(h2d_start);
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_end);
  cudaEventDestroy(d2h_end);
  if (use_double_buffer) {
    cudaEventDestroy(h2d_start2);
    cudaEventDestroy(kernel_start2);
    cudaEventDestroy(kernel_end2);
    cudaEventDestroy(d2h_end2);
  }

  // Download aggregate pixel counters from device.
  {
    unsigned long long h_unsupported = 0, h_zero_veto = 0, h_numerical_guard = 0;
    CUDA_CHECK(cudaMemcpy(&h_unsupported, d_unsupported_pixels,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_zero_veto, d_zero_veto_pixels,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_numerical_guard, d_numerical_guard_pixels,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    result.unsupported_pixels += static_cast<uint64_t>(h_unsupported);
    result.zero_veto_pixels += static_cast<uint64_t>(h_zero_veto);
    result.numerical_guard_pixels += static_cast<uint64_t>(h_numerical_guard);
  }

  // Post-process cherry-pick diagnostics (match CPU result fields).
  if (cherry_enabled) {
    uint64_t cherry_active_pixels = 0;
    uint64_t canvas_pixels = 0;
    std::vector<float> effective_k;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (canvas_mask.empty() ||
            static_cast<size_t>(y * width + x) >= canvas_mask.size() ||
            canvas_mask[y * width + x] == 0u)
          continue;
        ++canvas_pixels;
        const float k = result.cherry_pick_k_map(y, x);
        if (k > 0.0f && k < static_cast<float>(frame_count)) {
          ++cherry_active_pixels;
          effective_k.push_back(k);
        }
      }
    }
    result.cherry_pick_active = cherry_active_pixels > 0;
    result.cherry_pick_active_frac = canvas_pixels > 0
        ? static_cast<float>(cherry_active_pixels) / static_cast<float>(canvas_pixels)
        : 0.0f;
    if (!effective_k.empty()) {
      std::sort(effective_k.begin(), effective_k.end());
      result.cherry_pick_mean_k = std::accumulate(
          effective_k.begin(), effective_k.end(), 0.0f) / effective_k.size();
      result.cherry_pick_median_k = effective_k[effective_k.size() / 2];
      result.cherry_pick_k_min_observed = static_cast<int>(effective_k.front());
      result.cherry_pick_k_max_observed = static_cast<int>(effective_k.back());
      result.k_effective_p10 = effective_k[static_cast<size_t>(0.10f * effective_k.size())];
      result.k_effective_p50 = result.cherry_pick_median_k;
      result.k_effective_p90 = effective_k[static_cast<size_t>(0.90f * effective_k.size())];
    }
  }

  result.acceleration_used = true;
  result.acceleration_fallback = false;

  cudaStreamDestroy(stream);
  if (use_double_buffer) {
    cudaStreamDestroy(stream2);
  }
  free_chunk_buffers(bufs);
  if (use_double_buffer) {
    free_chunk_buffers(bufs2);
  }
  cudaFree(d_global_weights);
  cudaFree(d_unsupported_pixels);
  cudaFree(d_zero_veto_pixels);
  cudaFree(d_numerical_guard_pixels);
#undef CUDA_CHECK
  return result;
}

// ---------------------------------------------------------------------------
// AqmhCudaReconstructionSession — R1 multi-plane session (chunk-outside,
// plane-inside). Q-Maps and Frame-Masks are uploaded to the GPU once per
// chunk; only d_frames differs per plane. This eliminates 3/4 of Q-Map
// cache reads and H2D transfers in the debayer-first-RGB 4-pass case.
// ---------------------------------------------------------------------------

struct AqmhCudaReconstructionSession::Impl {
  // Config captured in init()
  size_t frame_count = 0;
  int width = 0, height = 0, chunk_rows = 0;
  bool cherry_enabled = false;
  AqmhReconstructionConfig cfg;

  // Non-owning references to shared inputs (valid for session lifetime)
  metrics::QualityMapCache* q_map_cache = nullptr;
  AqmhMaskLoader load_mask;
  AqmhMaskRegionLoader load_mask_region;
  const std::vector<uint8_t>* canvas_mask = nullptr;
  VectorXf global_weights_stored;  // stored for run_plane() forwarding

  // GPU device pointers
  float*               d_global_weights     = nullptr;
  unsigned long long*  d_unsupported_pixels = nullptr;
  unsigned long long*  d_zero_veto_pixels   = nullptr;
  unsigned long long*  d_numerical_guard_pixels = nullptr;
  cudaStream_t         stream = nullptr;
  cudaEvent_t          h2d_start = nullptr, kernel_start = nullptr,
                       kernel_end = nullptr, d2h_end = nullptr;
  GpuBuffers           bufs;

  // Pinned host buffers (one chunk at a time)
  PinnedBuffer<float>    h_frames;
  PinnedBuffer<float>    h_q_maps;
  PinnedBuffer<uint8_t>  h_masks;
  PinnedBuffer<uint8_t>  h_canvas_mask;
  PinnedBuffer<float>    h_output;
  PinnedBuffer<float>    h_weight_sum;
  PinnedBuffer<float>    h_uniform_control;
  PinnedBuffer<uint8_t>  h_uniform_control_valid;
  PinnedBuffer<float>    h_cherry_k_map;
  // WP-E: optional half-precision / bit-packed host staging buffers
  PinnedBuffer<uint16_t> h_q_maps_half;      // fp16 Q-Map host staging
  PinnedBuffer<uint8_t>  h_masks_packed;     // bit-packed mask host staging
  // WP-B: two-stage buffers on device (allocated in init when n_planes > 1)
  int16_t*  d_sel_indices = nullptr;  // [chunk_pixels * frame_count]
  uint16_t* d_sel_k       = nullptr;  // [chunk_pixels]
  PinnedBuffer<float> h_cherry_k_map_shared; // D2H for cherry_k_map after select kernel

  bool initialized = false;

  ~Impl() {
    if (!initialized) return;
    if (stream)               cudaStreamDestroy(stream);
    if (h2d_start)            cudaEventDestroy(h2d_start);
    if (kernel_start)         cudaEventDestroy(kernel_start);
    if (kernel_end)           cudaEventDestroy(kernel_end);
    if (d2h_end)              cudaEventDestroy(d2h_end);
    free_chunk_buffers(bufs);
    if (d_global_weights)         cudaFree(d_global_weights);
    if (d_unsupported_pixels)     cudaFree(d_unsupported_pixels);
    if (d_zero_veto_pixels)       cudaFree(d_zero_veto_pixels);
    if (d_numerical_guard_pixels) cudaFree(d_numerical_guard_pixels);
    if (d_sel_indices)            cudaFree(d_sel_indices);
    if (d_sel_k)                  cudaFree(d_sel_k);
  }
};

AqmhCudaReconstructionSession::AqmhCudaReconstructionSession()
    : impl_(std::make_unique<Impl>()) {}

AqmhCudaReconstructionSession::~AqmhCudaReconstructionSession() = default;

bool AqmhCudaReconstructionSession::init(
    size_t frame_count,
    metrics::QualityMapCache* q_map_cache,
    const VectorXf& global_weights,
    const std::vector<uint8_t>& canvas_mask,
    int width, int height,
    const AqmhReconstructionConfig& cfg,
    const AqmhMaskLoader& load_mask,
    const AqmhMaskRegionLoader& load_mask_region) {
  Impl& I = *impl_;

#define SESS_CHECK(expr)                                          \
  do {                                                            \
    cudaError_t _e = (expr);                                      \
    if (_e != cudaSuccess) {                                      \
      std::cerr << "[CUDA Session] " << cudaGetErrorString(_e)   \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";\
      return false;                                               \
    }                                                             \
  } while (0)

  if (frame_count == 0 || !q_map_cache || width <= 0 || height <= 0)
    return false;
  if (static_cast<int>(frame_count) > kMaxFramesCompile) return false;

  I.frame_count    = frame_count;
  I.width          = width;
  I.height         = height;
  I.cfg            = cfg;
  I.q_map_cache    = q_map_cache;
  I.load_mask      = load_mask;
  I.load_mask_region = load_mask_region;
  I.canvas_mask    = &canvas_mask;
  I.global_weights_stored = global_weights;

  I.cherry_enabled = cfg.cherry_pick &&
      static_cast<int>(frame_count) >= cfg.cherry_pick_k_min_required;

  // Determine VRAM budget and chunk_rows.
  size_t free_bytes = 0, total_bytes = 0;
  SESS_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t device_budget = std::min<size_t>(
      static_cast<size_t>(0.60 * static_cast<double>(free_bytes)),
      4ULL * kBytesPerGiB);
  // Per-row bytes: frames + q_maps + masks (float*2 + uint8) + output scratch
  const size_t bytes_per_row =
      static_cast<size_t>(frame_count) * width * sizeof(float) * 2 +
      static_cast<size_t>(frame_count) * width * sizeof(uint8_t) +
      static_cast<size_t>(width) * sizeof(float) * 4 +
      static_cast<size_t>(width) * sizeof(uint8_t) *
          (cfg.compute_uniform_control ? 2u : 1u);
  int chunk_rows;
  if (cfg.chunk_rows > 0) {
    chunk_rows = std::min(height, cfg.chunk_rows);
  } else {
    const int budget_rows = static_cast<int>(
        device_budget / std::max<size_t>(1, bytes_per_row));
    chunk_rows = std::min(height, std::max(kMinAutoChunkRows, budget_rows));
  }
  I.chunk_rows = chunk_rows;

  // Upload global weights.
  std::vector<float> h_gw(frame_count, 1.0f);
  for (Eigen::Index fi = 0; fi < global_weights.size() &&
       fi < static_cast<Eigen::Index>(frame_count); ++fi) {
    const float w = global_weights[fi];
    h_gw[fi] = std::isfinite(w) && w > 0.0f ? w : 0.0f;
  }
  SESS_CHECK(cudaMalloc(&I.d_global_weights, frame_count * sizeof(float)));
  SESS_CHECK(cudaMemcpy(I.d_global_weights, h_gw.data(),
                        frame_count * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate GPU chunk buffers.
  while (!allocate_chunk_buffers(I.bufs, width, chunk_rows,
                                 static_cast<int>(frame_count),
                                 cfg.compute_uniform_control,
                                 I.cherry_enabled,
                                 cfg.gpu_half_qmaps,
                                 cfg.gpu_packed_masks)) {
    free_chunk_buffers(I.bufs);
    if (cfg.chunk_rows > 0 || chunk_rows <= 1) {
      cudaFree(I.d_global_weights); I.d_global_weights = nullptr;
      return false;
    }
    chunk_rows = std::max(1, chunk_rows / 2);
    I.chunk_rows = chunk_rows;
  }

  // Allocate pixel counters.
  SESS_CHECK(cudaMalloc(&I.d_unsupported_pixels,     sizeof(unsigned long long)));
  SESS_CHECK(cudaMalloc(&I.d_zero_veto_pixels,       sizeof(unsigned long long)));
  SESS_CHECK(cudaMalloc(&I.d_numerical_guard_pixels, sizeof(unsigned long long)));

  // Stream + events.
  SESS_CHECK(cudaStreamCreate(&I.stream));
  SESS_CHECK(cudaEventCreate(&I.h2d_start));
  SESS_CHECK(cudaEventCreate(&I.kernel_start));
  SESS_CHECK(cudaEventCreate(&I.kernel_end));
  SESS_CHECK(cudaEventCreate(&I.d2h_end));

  // Pinned host buffers (sized for one chunk).
  const size_t chunk_pixels = static_cast<size_t>(chunk_rows) * width;
  const size_t all_chunk   = static_cast<size_t>(frame_count) * chunk_pixels;
  I.h_frames.assign(all_chunk, 0.0f);
  I.h_q_maps.assign(all_chunk, 0.0f);
  I.h_masks.assign(all_chunk, 0u);
  I.h_canvas_mask.assign(chunk_pixels, 0u);
  I.h_output.assign(chunk_pixels, 0.0f);
  I.h_weight_sum.assign(chunk_pixels, 0.0f);
  if (cfg.compute_uniform_control) {
    I.h_uniform_control.assign(chunk_pixels, 0.0f);
    I.h_uniform_control_valid.assign(chunk_pixels, 0u);
  }
  if (I.cherry_enabled) {
    I.h_cherry_k_map.assign(chunk_pixels, 0.0f);
    I.h_cherry_k_map_shared.assign(chunk_pixels, 0.0f);
  }
  // WP-E: half-precision Q-Map and bit-packed mask host staging buffers.
  if (cfg.gpu_half_qmaps)
    I.h_q_maps_half.assign(all_chunk, uint16_t(0));
  if (cfg.gpu_packed_masks)
    I.h_masks_packed.assign((all_chunk + 7) / 8, uint8_t(0));

  // WP-B: per-pixel selection device buffers for two-stage path.
  // Allocated unconditionally so run_planes_rgb can use them when n_planes > 1.
  const cudaError_t sel_err1 = cudaMalloc(
      &I.d_sel_indices, chunk_pixels * frame_count * sizeof(int16_t));
  const cudaError_t sel_err2 = cudaMalloc(
      &I.d_sel_k, chunk_pixels * sizeof(uint16_t));
  // Non-fatal if allocation fails — two-stage path falls back to single-stage.
  if (sel_err1 != cudaSuccess || sel_err2 != cudaSuccess) {
    if (I.d_sel_indices) { cudaFree(I.d_sel_indices); I.d_sel_indices = nullptr; }
    if (I.d_sel_k)       { cudaFree(I.d_sel_k);       I.d_sel_k       = nullptr; }
  }

  I.initialized = true;
#undef SESS_CHECK
  return true;
}

// Helper: post-process cherry-pick stats into a result (same logic as the
// single-plane function).
static void session_postprocess_cherry(
    AqmhReconstructionResult& r, int height, int width, size_t frame_count,
    const std::vector<uint8_t>& canvas_mask) {
  uint64_t cherry_active = 0, canvas_pixels = 0;
  std::vector<float> effective_k;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (canvas_mask.empty() ||
          static_cast<size_t>(y * width + x) >= canvas_mask.size() ||
          canvas_mask[y * width + x] == 0u)
        continue;
      ++canvas_pixels;
      const float k = r.cherry_pick_k_map(y, x);
      if (k > 0.0f && k < static_cast<float>(frame_count)) {
        ++cherry_active;
        effective_k.push_back(k);
      }
    }
  }
  r.cherry_pick_active = cherry_active > 0;
  r.cherry_pick_active_frac = canvas_pixels > 0
      ? static_cast<float>(cherry_active) / static_cast<float>(canvas_pixels)
      : 0.0f;
  if (!effective_k.empty()) {
    std::sort(effective_k.begin(), effective_k.end());
    r.cherry_pick_mean_k = std::accumulate(
        effective_k.begin(), effective_k.end(), 0.0f) / effective_k.size();
    r.cherry_pick_median_k = effective_k[effective_k.size() / 2];
    r.cherry_pick_k_min_observed = static_cast<int>(effective_k.front());
    r.cherry_pick_k_max_observed = static_cast<int>(effective_k.back());
    r.k_effective_p10 = effective_k[static_cast<size_t>(0.10f * effective_k.size())];
    r.k_effective_p50 = r.cherry_pick_median_k;
    r.k_effective_p90 = effective_k[static_cast<size_t>(0.90f * effective_k.size())];
  }
}

std::vector<AqmhReconstructionResult> AqmhCudaReconstructionSession::run_planes_rgb(
    const std::vector<AqmhFrameRegionLoader>& frame_region_loaders,
    const std::vector<bool>& compute_uniform_control,
    const AqmhProgressCallback& progress) {
  Impl& I = *impl_;
  const size_t n_planes = frame_region_loaders.size();
  std::vector<AqmhReconstructionResult> results(n_planes);
  if (!I.initialized || n_planes == 0) return results;

  // Initialize result matrices.
  for (size_t pi = 0; pi < n_planes; ++pi) {
    auto& r = results[pi];
    r.output = Matrix2Df::Zero(I.height, I.width);
    r.weight_sum = Matrix2Df::Zero(I.height, I.width);
    const bool uc = pi < compute_uniform_control.size() && compute_uniform_control[pi];
    if (uc) {
      r.uniform_control_output = Matrix2Df::Zero(I.height, I.width);
      r.uniform_control_valid_mask.assign(
          static_cast<size_t>(I.height) * I.width, 0u);
    }
    if (I.cherry_enabled)
      r.cherry_pick_k_map = Matrix2Df::Zero(I.height, I.width);
    r.chunk_rows = I.chunk_rows;
    r.chunk_count = (I.height + I.chunk_rows - 1) / I.chunk_rows;
    r.region_streaming_used = true;
  }

  const int frame_count = static_cast<int>(I.frame_count);
  const int width = I.width, height = I.height, chunk_rows = I.chunk_rows;
  const size_t chunk_pixels = static_cast<size_t>(chunk_rows) * width;
  const dim3 block(32, 8);
  const dim3 grid((width + 31) / 32, (chunk_rows + 7) / 8);

#define SESS_RUN_CHECK(expr)                                        \
  do {                                                              \
    cudaError_t _e = (expr);                                        \
    if (_e != cudaSuccess) {                                        \
      std::cerr << "[CUDA Session run] " << cudaGetErrorString(_e) \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
      for (auto& r : results) r.acceleration_fallback = true;      \
      return results;                                               \
    }                                                               \
  } while (0)

  // Pixel counters are reset per-plane (inside the plane loop) so each plane
  // gets its own counts. No global reset here.

  const int num_host_threads = std::min(
      frame_count,
      std::max(1, static_cast<int>(std::thread::hardware_concurrency())));

  // -----------------------------------------------------------------------
  // R1 main loop: chunk-outside, plane-inside.
  // -----------------------------------------------------------------------
  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);
    const size_t used_chunk_pixels = static_cast<size_t>(rows) * width;
    const size_t used_all_pixels = static_cast<size_t>(frame_count) * used_chunk_pixels;

    // --- Canvas mask slice ---
    std::fill(I.h_canvas_mask.begin(), I.h_canvas_mask.end(), 0u);
    for (int yy = 0; yy < rows; ++yy) {
      const int y = y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t full_i = static_cast<size_t>(y) * width + x;
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        I.h_canvas_mask[local_i] = (I.canvas_mask->empty() ||
            full_i >= I.canvas_mask->size()) ? 1u : (*I.canvas_mask)[full_i];
      }
    }

    // --- Pack Q-Maps + Masks for this chunk (OMP, ONCE per chunk) ---
    std::fill(I.h_masks.begin(), I.h_masks.end(), 0u);
    double q_map_read_s = 0.0, mask_read_s = 0.0;
    #if defined(_OPENMP)
    #pragma omp parallel for num_threads(num_host_threads) schedule(dynamic,4) \
        reduction(+:q_map_read_s, mask_read_s)
    #endif
    for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(I.frame_count); ++fi_ptr) {
      const size_t fi = static_cast<size_t>(fi_ptr);
      // Q-Map
      const auto tq = std::chrono::steady_clock::now();
      Matrix2Df q = I.q_map_cache->read_region(fi, y0, rows);
      q_map_read_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - tq).count();
      const bool q_ok = q.rows() == rows && q.cols() == width;

      // Mask
      const auto tm = std::chrono::steady_clock::now();
      std::vector<uint8_t> fm;
      if (I.load_mask_region)
        I.load_mask_region(fi, y0, rows, fm);
      else if (I.load_mask)
        I.load_mask(fi, fm);
      mask_read_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - tm).count();

      for (int yy = 0; yy < rows; ++yy) {
        for (int x = 0; x < width; ++x) {
          const size_t local_i = static_cast<size_t>(yy) * width + x;
          const size_t idx = local_i * I.frame_count + fi;
          I.h_q_maps[idx] = q_ok ? q(yy, x)
                                  : std::numeric_limits<float>::quiet_NaN();
          if (fm.empty()) {
            I.h_masks[idx] = 1u;
          } else {
            const size_t mask_i = I.load_mask_region
                ? local_i
                : (static_cast<size_t>(y0 + yy) * width + x);
            I.h_masks[idx] = (mask_i < fm.size()) ? fm[mask_i] : 0u;
          }
        }
      }
    }
    for (size_t pi = 0; pi < n_planes; ++pi) {
      results[pi].cuda_host_q_map_read_worker_seconds += q_map_read_s;
      results[pi].cuda_host_mask_read_worker_seconds  += mask_read_s;
    }

    // WP-E: Convert float Q-Maps → fp16 and byte masks → bit-packed if requested.
    if (I.cfg.gpu_half_qmaps) {
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static) num_threads(num_host_threads)
      #endif
      for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(used_all_pixels); ++idx) {
        const __half h = __float2half_rn(I.h_q_maps[static_cast<size_t>(idx)]);
        I.h_q_maps_half[static_cast<size_t>(idx)] =
            *reinterpret_cast<const uint16_t*>(&h);
      }
    }
    if (I.cfg.gpu_packed_masks) {
      std::fill(I.h_masks_packed.begin(), I.h_masks_packed.end(), uint8_t(0));
      for (size_t idx = 0; idx < used_all_pixels; ++idx) {
        if (I.h_masks[idx])
          I.h_masks_packed[idx >> 3] |= uint8_t(1u << (idx & 7));
      }
    }

    // H2D: canvas_mask + q_maps (or fp16 staging) + masks (or packed staging)
    // (once per chunk for all planes)
    SESS_RUN_CHECK(cudaMemcpyAsync(
        I.bufs.canvas_mask, I.h_canvas_mask.data(),
        used_chunk_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice, I.stream));
    if (I.cfg.gpu_half_qmaps) {
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.bufs.q_maps_half, I.h_q_maps_half.data(),
          used_all_pixels * sizeof(uint16_t), cudaMemcpyHostToDevice, I.stream));
    } else {
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.bufs.q_maps, I.h_q_maps.data(),
          used_all_pixels * sizeof(float), cudaMemcpyHostToDevice, I.stream));
    }
    if (I.cfg.gpu_packed_masks) {
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.bufs.frame_masks_packed, I.h_masks_packed.data(),
          (used_all_pixels + 7) / 8 * sizeof(uint8_t),
          cudaMemcpyHostToDevice, I.stream));
    } else {
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.bufs.frame_masks, I.h_masks.data(),
          used_all_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice, I.stream));
    }
    // WP-E: Dequantize fp16 / bit-packed data to float/byte in-place on GPU.
    launch_dequantize(I.bufs, I.stream,
                      I.cfg.gpu_half_qmaps, I.cfg.gpu_packed_masks,
                      static_cast<int>(used_all_pixels));

    // WP-B: Two-stage path — launch aqmh_select_kernel once for all planes,
    // then aqmh_reduce_kernel per plane.  Falls back to single-stage if the
    // selection buffers were not allocated (VRAM tight) or n_planes == 1.
    const bool use_two_stage = n_planes > 1
        && I.d_sel_indices != nullptr && I.d_sel_k != nullptr;

    if (use_two_stage) {
      // Zero selection counts so pixels outside the canvas are clearly empty.
      SESS_RUN_CHECK(cudaMemsetAsync(I.d_sel_k, 0,
          used_chunk_pixels * sizeof(uint16_t), I.stream));
      launch_select_kernel_for_frame_count(
          frame_count, I.cherry_enabled, grid, block, I.stream, I.bufs,
          I.d_global_weights, I.d_sel_indices, I.d_sel_k,
          width, rows, y0, height, I.cfg);
      // D2H cherry_k_map once (shared across all planes).
      if (I.cherry_enabled) {
        SESS_RUN_CHECK(cudaMemcpyAsync(
            I.h_cherry_k_map_shared.data(), I.bufs.cherry_k_map,
            used_chunk_pixels * sizeof(float), cudaMemcpyDeviceToHost, I.stream));
        SESS_RUN_CHECK(cudaStreamSynchronize(I.stream));
      }
    }

    // --- Per-plane inner loop ---
    for (size_t pi = 0; pi < n_planes; ++pi) {
      auto& r = results[pi];
      const bool uc = pi < compute_uniform_control.size() && compute_uniform_control[pi];
      const auto& frame_loader = frame_region_loaders[pi];

      // Pack frames for this plane/chunk (OMP parallel)
      std::fill(I.h_frames.begin(), I.h_frames.end(), 0.0f);
      double frame_read_s = 0.0, pack_s = 0.0;
      #if defined(_OPENMP)
      #pragma omp parallel for num_threads(num_host_threads) schedule(dynamic,4) \
          reduction(+:frame_read_s, pack_s)
      #endif
      for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(I.frame_count); ++fi_ptr) {
        const size_t fi = static_cast<size_t>(fi_ptr);
        const auto tf = std::chrono::steady_clock::now();
        Matrix2Df frame_region;
        const bool frame_ok = frame_loader
            ? frame_loader(fi, y0, rows, frame_region)
            : false;
        frame_read_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tf).count();
        if (!frame_ok || frame_region.rows() != rows ||
            frame_region.cols() != width) {
          #if defined(_OPENMP)
          #pragma omp atomic
          #endif
          r.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
        const auto tp = std::chrono::steady_clock::now();
        for (int yy = 0; yy < rows; ++yy) {
          for (int x = 0; x < width; ++x) {
            const size_t local_i = static_cast<size_t>(yy) * width + x;
            const size_t idx = local_i * I.frame_count + fi;
            I.h_frames[idx] = frame_region(yy, x);
          }
        }
        pack_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tp).count();
      }
      r.cuda_host_frame_read_worker_seconds += frame_read_s;
      r.cuda_host_pack_worker_seconds       += pack_s;

      // Reset per-plane pixel counters before this plane's kernels.
      SESS_RUN_CHECK(cudaMemset(I.d_unsupported_pixels,     0, sizeof(unsigned long long)));
      SESS_RUN_CHECK(cudaMemset(I.d_zero_veto_pixels,       0, sizeof(unsigned long long)));
      SESS_RUN_CHECK(cudaMemset(I.d_numerical_guard_pixels, 0, sizeof(unsigned long long)));

      // H2D: frames
      SESS_RUN_CHECK(cudaEventRecord(I.h2d_start, I.stream));
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.bufs.frames, I.h_frames.data(),
          used_all_pixels * sizeof(float), cudaMemcpyHostToDevice, I.stream));

      SESS_RUN_CHECK(cudaEventRecord(I.kernel_start, I.stream));
      if (use_two_stage) {
        // WP-B: reduce kernel — reads pre-selected indices from select kernel.
        launch_reduce_kernel_for_frame_count(
            frame_count, grid, block, I.stream, I.bufs, I.d_global_weights,
            I.d_sel_indices, I.d_sel_k,
            I.d_unsupported_pixels, I.d_zero_veto_pixels,
            I.d_numerical_guard_pixels,
            width, rows, y0, height, I.cfg);
      } else {
        // Single-stage reconstruction kernel (original path).
        AqmhReconstructionConfig plane_cfg = I.cfg;
        plane_cfg.compute_uniform_control = uc;
        launch_reconstruction_kernel_for_frame_count(
            frame_count, I.cherry_enabled, grid, block, I.stream, I.bufs,
            I.d_global_weights, I.d_unsupported_pixels, I.d_zero_veto_pixels,
            I.d_numerical_guard_pixels, width, rows, y0, height, plane_cfg);
      }

      // D2H: output + weight_sum
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.h_output.data(), I.bufs.output,
          used_chunk_pixels * sizeof(float), cudaMemcpyDeviceToHost, I.stream));
      SESS_RUN_CHECK(cudaMemcpyAsync(
          I.h_weight_sum.data(), I.bufs.weight_sum,
          used_chunk_pixels * sizeof(float), cudaMemcpyDeviceToHost, I.stream));
      if (!use_two_stage) {
        if (uc) {
          SESS_RUN_CHECK(cudaMemcpyAsync(
              I.h_uniform_control.data(), I.bufs.uniform_control,
              used_chunk_pixels * sizeof(float), cudaMemcpyDeviceToHost, I.stream));
          SESS_RUN_CHECK(cudaMemcpyAsync(
              I.h_uniform_control_valid.data(), I.bufs.uniform_control_valid,
              used_chunk_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost, I.stream));
        }
        if (I.cherry_enabled) {
          SESS_RUN_CHECK(cudaMemcpyAsync(
              I.h_cherry_k_map.data(), I.bufs.cherry_k_map,
              used_chunk_pixels * sizeof(float), cudaMemcpyDeviceToHost, I.stream));
        }
      }
      SESS_RUN_CHECK(cudaEventRecord(I.kernel_end, I.stream));
      SESS_RUN_CHECK(cudaEventRecord(I.d2h_end, I.stream));
      SESS_RUN_CHECK(cudaStreamSynchronize(I.stream));

      // Accumulate timings
      float elapsed_ms = 0.0f;
      cudaEventElapsedTime(&elapsed_ms, I.h2d_start, I.kernel_start);
      r.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      cudaEventElapsedTime(&elapsed_ms, I.kernel_start, I.kernel_end);
      r.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      cudaEventElapsedTime(&elapsed_ms, I.kernel_end, I.d2h_end);
      r.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

      // Commit this plane's chunk result
      const auto commit_start = std::chrono::steady_clock::now();
      const float* cherry_src = use_two_stage
          ? I.h_cherry_k_map_shared.data()
          : I.h_cherry_k_map.data();
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static) if(rows > 4)
      #endif
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        const size_t src_off = static_cast<size_t>(yy) * width;
        std::memcpy(&r.output(y, 0), I.h_output.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        std::memcpy(&r.weight_sum(y, 0), I.h_weight_sum.data() + src_off,
                    static_cast<size_t>(width) * sizeof(float));
        if (!use_two_stage && uc) {
          std::memcpy(&r.uniform_control_output(y, 0),
                      I.h_uniform_control.data() + src_off,
                      static_cast<size_t>(width) * sizeof(float));
          std::memcpy(r.uniform_control_valid_mask.data() +
                          static_cast<size_t>(y) * width,
                      I.h_uniform_control_valid.data() + src_off,
                      static_cast<size_t>(width));
        }
        if (I.cherry_enabled)
          std::memcpy(&r.cherry_pick_k_map(y, 0),
                      cherry_src + src_off,
                      static_cast<size_t>(width) * sizeof(float));
      }
      // Download per-plane pixel counters immediately after sync.
      {
        unsigned long long h_u = 0, h_z = 0, h_g = 0;
        cudaMemcpy(&h_u, I.d_unsupported_pixels,
                   sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_z, I.d_zero_veto_pixels,
                   sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_g, I.d_numerical_guard_pixels,
                   sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        r.unsupported_pixels     += static_cast<uint64_t>(h_u);
        r.zero_veto_pixels       += static_cast<uint64_t>(h_z);
        r.numerical_guard_pixels += static_cast<uint64_t>(h_g);
      }

      r.cuda_result_commit_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - commit_start).count();
    } // end plane loop

    if (progress) progress(y0 + rows, height);
  } // end chunk loop

  if (I.cherry_enabled) {
    for (auto& r : results)
      session_postprocess_cherry(r, height, I.width, I.frame_count, *I.canvas_mask);
  }

  for (auto& r : results) {
    r.acceleration_used     = true;
    r.acceleration_fallback = false;
  }
#undef SESS_RUN_CHECK
  return results;
}

AqmhReconstructionResult AqmhCudaReconstructionSession::run_plane(
    const AqmhFrameLoader& load_frame,
    const AqmhFrameRegionLoader& load_frame_region,
    bool compute_uniform_control_plane,
    const AqmhProgressCallback& progress) {
  if (!impl_->initialized) {
    AqmhReconstructionResult r;
    r.acceleration_fallback = true;
    return r;
  }
  // Wrap load_frame as a region loader if no region loader provided.
  AqmhFrameRegionLoader effective_region_loader = load_frame_region;
  if (!effective_region_loader && load_frame) {
    effective_region_loader = [lf = load_frame](size_t fi, int y0, int rows,
                                                Matrix2Df& out) -> bool {
      Matrix2Df full;
      if (!lf(fi, full)) return false;
      const int h = static_cast<int>(full.rows());
      const int w = static_cast<int>(full.cols());
      if (y0 < 0 || y0 + rows > h || rows <= 0) return false;
      out = full.middleRows(y0, rows);
      (void)w;
      return true;
    };
  }
  auto results = run_planes_rgb({effective_region_loader},
                                {compute_uniform_control_plane}, progress);
  if (results.empty()) {
    AqmhReconstructionResult r;
    r.acceleration_fallback = true;
    return r;
  }
  return std::move(results[0]);
}

} // namespace tile_compile::reconstruction

#endif
