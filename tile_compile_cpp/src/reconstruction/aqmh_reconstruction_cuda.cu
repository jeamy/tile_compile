#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"

#if TILE_COMPILE_WITH_CUDA

#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include <cuda_runtime.h>
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
        float tmp_v[MaxFrames];
        float tmp_w[MaxFrames];
        for (int i = 0; i < keep_floor; ++i) {
          const short src = sort_buf[i];
          tmp_v[i] = values[src];
          tmp_w[i] = weights[src];
        }
        for (int i = 0; i < keep_floor; ++i) {
          values[i] = tmp_v[i];
          weights[i] = tmp_w[i];
        }
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
      float tmp_v[MaxFrames];
      float tmp_w[MaxFrames];
      for (int i = 0; i < keep_floor; ++i) {
        const int src = sort_buf[i];
        tmp_v[i] = values[src];
        tmp_w[i] = weights[src];
      }
      for (int i = 0; i < keep_floor; ++i) {
        values[i] = tmp_v[i];
        weights[i] = tmp_w[i];
      }
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
__global__ void aqmh_reconstruction_kernel(
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
};

bool allocate_chunk_buffers(
    GpuBuffers& bufs, int width, int chunk_rows, int frame_count,
    bool compute_uniform_control, bool cherry_pick_enabled) {
  const size_t chunk_pixels = static_cast<size_t>(chunk_rows) * width;
  const size_t all_chunk_pixels = static_cast<size_t>(frame_count) * chunk_pixels;
  CUDA_CHECK(cudaMalloc(&bufs.frames, all_chunk_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bufs.q_maps, all_chunk_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bufs.frame_masks, all_chunk_pixels * sizeof(uint8_t)));
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
  if (bufs.frame_masks) cudaFree(bufs.frame_masks);
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
      static_cast<size_t>(0.30 * static_cast<double>(free_bytes)),
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

  // P3: Double-buffering — allocate two sets of device buffers for stream overlap.
  GpuBuffers bufs[2];
  for (int b = 0; b < 2; ++b) {
    while (!allocate_chunk_buffers(bufs[b], width, chunk_rows, static_cast<int>(frame_count),
                                   cfg.compute_uniform_control, cherry_enabled)) {
      free_chunk_buffers(bufs[b]);
      if (b == 1) free_chunk_buffers(bufs[0]);
      if (cfg.chunk_rows > 0 || chunk_rows <= 1) {
        cudaFree(d_global_weights);
        result.acceleration_fallback = true;
        return result;
      }
      chunk_rows = std::max(1, chunk_rows / 2);
      ++result.cuda_allocation_retries;
      result.chunk_rows = chunk_rows;
      result.chunk_count = (height + chunk_rows - 1) / chunk_rows;
      b = -1; // Restart allocation from scratch with new chunk_rows
      break;
    }
  }

  // P3: Double-buffered pinned host staging buffers — two sets for stream overlap.
  PinnedBuffer<float> h_frames[2] = {
    PinnedBuffer<float>(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f),
    PinnedBuffer<float>(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f)
  };
  PinnedBuffer<float> h_q_maps[2] = {
    PinnedBuffer<float>(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f),
    PinnedBuffer<float>(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f)
  };
  PinnedBuffer<uint8_t> h_masks[2] = {
    PinnedBuffer<uint8_t>(static_cast<size_t>(frame_count) * chunk_rows * width, 0u),
    PinnedBuffer<uint8_t>(static_cast<size_t>(frame_count) * chunk_rows * width, 0u)
  };
  PinnedBuffer<uint8_t> h_canvas_mask[2] = {
    PinnedBuffer<uint8_t>(static_cast<size_t>(chunk_rows) * width, 0u),
    PinnedBuffer<uint8_t>(static_cast<size_t>(chunk_rows) * width, 0u)
  };
  PinnedBuffer<float> h_output[2] = {
    PinnedBuffer<float>(static_cast<size_t>(chunk_rows) * width, 0.0f),
    PinnedBuffer<float>(static_cast<size_t>(chunk_rows) * width, 0.0f)
  };
  PinnedBuffer<float> h_weight_sum[2] = {
    PinnedBuffer<float>(static_cast<size_t>(chunk_rows) * width, 0.0f),
    PinnedBuffer<float>(static_cast<size_t>(chunk_rows) * width, 0.0f)
  };
  PinnedBuffer<float> h_uniform_control[2];
  PinnedBuffer<uint8_t> h_uniform_control_valid[2];
  if (cfg.compute_uniform_control) {
    h_uniform_control[0].assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
    h_uniform_control[1].assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
    h_uniform_control_valid[0].assign(static_cast<size_t>(chunk_rows) * width, 0u);
    h_uniform_control_valid[1].assign(static_cast<size_t>(chunk_rows) * width, 0u);
  }
  PinnedBuffer<float> h_cherry_k_map[2];
  if (cherry_enabled) {
    h_cherry_k_map[0].assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
    h_cherry_k_map[1].assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
  }

  unsigned long long* d_unsupported_pixels = nullptr;
  unsigned long long* d_zero_veto_pixels = nullptr;
  unsigned long long* d_numerical_guard_pixels = nullptr;
  CUDA_CHECK(cudaMalloc(&d_unsupported_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_zero_veto_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_numerical_guard_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemsetAsync(d_unsupported_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK(cudaMemsetAsync(d_zero_veto_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK(cudaMemsetAsync(d_numerical_guard_pixels, 0, sizeof(unsigned long long), 0));

  const dim3 block(32, 8);
  // P3: Two streams for ping-pong double-buffering.
  cudaStream_t streams[2] = {nullptr, nullptr};
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));
  cudaEvent_t h2d_start[2] = {nullptr, nullptr};
  cudaEvent_t kernel_start[2] = {nullptr, nullptr};
  cudaEvent_t kernel_end[2] = {nullptr, nullptr};
  cudaEvent_t d2h_end[2] = {nullptr, nullptr};
  for (int s = 0; s < 2; ++s) {
    CUDA_CHECK(cudaEventCreate(&h2d_start[s]));
    CUDA_CHECK(cudaEventCreate(&kernel_start[s]));
    CUDA_CHECK(cudaEventCreate(&kernel_end[s]));
    CUDA_CHECK(cudaEventCreate(&d2h_end[s]));
  }

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

  // P3: Double-buffered main loop — H2D[k+1] overlaps with Kernel[k],
  // D2H[k] overlaps with Kernel[k+1]. Two streams and two buffer sets.
  struct PendingChunk { int y0 = 0; int rows = 0; bool valid = false; };
  PendingChunk pending[2];
  int chunk_idx = 0;

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);
    const int slot = chunk_idx % 2;

    // If this slot has pending results from chunk_idx-2, sync and commit.
    if (pending[slot].valid) {
      CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_start[slot], kernel_start[slot]));
      result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_start[slot], kernel_end[slot]));
      result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_end[slot], d2h_end[slot]));
      result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

      const auto result_commit_start = std::chrono::steady_clock::now();
      const int p_y0 = pending[slot].y0;
      const int p_rows = pending[slot].rows;
      for (int yy = 0; yy < p_rows; ++yy) {
        const int y = p_y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t local_i = static_cast<size_t>(yy) * width + x;
          result.output(y, x) = h_output[slot][local_i];
          result.weight_sum(y, x) = h_weight_sum[slot][local_i];
          if (cfg.compute_uniform_control) {
            result.uniform_control_output(y, x) = h_uniform_control[slot][local_i];
            result.uniform_control_valid_mask[
                static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x)] = h_uniform_control_valid[slot][local_i];
          }
          if (cherry_enabled)
            result.cherry_pick_k_map(y, x) = h_cherry_k_map[slot][local_i];
        }
      }
      result.cuda_result_commit_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - result_commit_start)
              .count();
      if (progress) progress(p_y0 + p_rows, height);
      pending[slot].valid = false;
    }

    const auto host_prepare_start = std::chrono::steady_clock::now();

    if (have_prefetched_frames) {
      prefetched_frames = next_frame_prefetch.get();
      have_prefetched_frames = false;
    }

    // Masks must be zeroed each chunk so frames that fail to load are skipped.
    std::fill(h_masks[slot].begin(), h_masks[slot].end(), 0u);

    // Prepare canvas mask slice.
    for (int yy = 0; yy < rows; ++yy) {
      const int y = y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t full_i = static_cast<size_t>(y) * width + x;
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        h_canvas_mask[slot][local_i] =
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
          h_frames[slot][idx] = frame_region(fr_row, x);
          h_q_maps[slot][idx] =
              q_map_ok ? q(fr_row, x)
                       : std::numeric_limits<float>::quiet_NaN();
          if (fm.empty()) {
            h_masks[slot][idx] = 1u;
          } else {
            const size_t mask_i = use_region ? local_i : full_i;
            h_masks[slot][idx] = fm[mask_i];
            if (fm[mask_i] == 0u) continue;
          }
          if (h_canvas_mask[slot][local_i] == 0u) continue;
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

    // Upload frame/q-map/mask chunk on streams[slot].
    const size_t used_all_pixels = static_cast<size_t>(frame_count) * rows * width;
    CUDA_CHECK(cudaEventRecord(h2d_start[slot], streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs[slot].canvas_mask, h_canvas_mask[slot].data(),
        static_cast<size_t>(rows) * width * sizeof(uint8_t),
        cudaMemcpyHostToDevice, streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs[slot].frames, h_frames[slot].data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs[slot].q_maps, h_q_maps[slot].data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs[slot].frame_masks, h_masks[slot].data(),
        used_all_pixels * sizeof(uint8_t),
        cudaMemcpyHostToDevice, streams[slot]));
    CUDA_CHECK(cudaEventRecord(kernel_start[slot], streams[slot]));

    const dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    launch_reconstruction_kernel_for_frame_count(
        static_cast<int>(frame_count), cherry_enabled,
        grid, block, streams[slot], bufs[slot], d_global_weights,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, cfg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_end[slot], streams[slot]));

    // Download outputs on streams[slot].
    const size_t used_chunk_pixels = static_cast<size_t>(rows) * width;
    CUDA_CHECK(cudaMemcpyAsync(
        h_output[slot].data(), bufs[slot].output,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        h_weight_sum[slot].data(), bufs[slot].weight_sum,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, streams[slot]));
    if (cfg.compute_uniform_control) {
      CUDA_CHECK(cudaMemcpyAsync(
          h_uniform_control[slot].data(), bufs[slot].uniform_control,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, streams[slot]));
      CUDA_CHECK(cudaMemcpyAsync(
          h_uniform_control_valid[slot].data(), bufs[slot].uniform_control_valid,
          used_chunk_pixels * sizeof(uint8_t),
          cudaMemcpyDeviceToHost, streams[slot]));
    }
    if (cherry_enabled) {
      CUDA_CHECK(cudaMemcpyAsync(
          h_cherry_k_map[slot].data(), bufs[slot].cherry_k_map,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, streams[slot]));
    }
    CUDA_CHECK(cudaEventRecord(d2h_end[slot], streams[slot]));

    // Mark this slot as pending for later commit.
    pending[slot] = {y0, rows, true};
    ++chunk_idx;
  }

  // P3: Commit remaining pending chunks after the loop.
  for (int s = 0; s < 2; ++s) {
    if (!pending[s].valid) continue;
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_start[s], kernel_start[s]));
    result.cuda_h2d_seconds += static_cast<double>(elapsed_ms) / 1000.0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_start[s], kernel_end[s]));
    result.cuda_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, kernel_end[s], d2h_end[s]));
    result.cuda_d2h_seconds += static_cast<double>(elapsed_ms) / 1000.0;

    const auto result_commit_start = std::chrono::steady_clock::now();
    const int p_y0 = pending[s].y0;
    const int p_rows = pending[s].rows;
    for (int yy = 0; yy < p_rows; ++yy) {
      const int y = p_y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        result.output(y, x) = h_output[s][local_i];
        result.weight_sum(y, x) = h_weight_sum[s][local_i];
        if (cfg.compute_uniform_control) {
          result.uniform_control_output(y, x) = h_uniform_control[s][local_i];
          result.uniform_control_valid_mask[
              static_cast<size_t>(y) * static_cast<size_t>(width) +
              static_cast<size_t>(x)] = h_uniform_control_valid[s][local_i];
        }
        if (cherry_enabled)
          result.cherry_pick_k_map(y, x) = h_cherry_k_map[s][local_i];
      }
    }
    result.cuda_result_commit_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - result_commit_start)
            .count();
    if (progress) progress(p_y0 + p_rows, height);
  }

  for (int s = 0; s < 2; ++s) {
    cudaEventDestroy(h2d_start[s]);
    cudaEventDestroy(kernel_start[s]);
    cudaEventDestroy(kernel_end[s]);
    cudaEventDestroy(d2h_end[s]);
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

  for (int s = 0; s < 2; ++s)
    cudaStreamDestroy(streams[s]);
  for (int b = 0; b < 2; ++b)
    free_chunk_buffers(bufs[b]);
  cudaFree(d_global_weights);
  cudaFree(d_unsupported_pixels);
  cudaFree(d_zero_veto_pixels);
  cudaFree(d_numerical_guard_pixels);
#undef CUDA_CHECK
  return result;
}

} // namespace tile_compile::reconstruction

#endif
