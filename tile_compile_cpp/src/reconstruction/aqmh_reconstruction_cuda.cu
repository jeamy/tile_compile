#include "tile_compile/reconstruction/aqmh_reconstruction_cuda.hpp"

#if TILE_COMPILE_WITH_CUDA

#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tile_compile::reconstruction {

namespace {

constexpr int kMaxFramesCompile = 1024;  // upper bound for local arrays

// Next power of two >= n, capped at kMaxFramesCompile.
__device__ inline int next_pow2(int n) {
  int p = 1;
  while (p < n && p < kMaxFramesCompile) p *= 2;
  return p;
}

// Insertion sort for small n (<= 64).  O(N²) but with low constant factor;
// for N=64 that's ~2016 comparisons vs ~3000+ for bitonic sort over 64.
// Sorts indices [0, n) using comp for ordering.
template <typename Comp>
__device__ void insertion_adaptive_sort(int* indices, int n, Comp comp) {
  for (int i = 1; i < n; ++i) {
    int key = indices[i];
    int j = i - 1;
    while (j >= 0 && comp(key, indices[j])) {
      indices[j + 1] = indices[j];
      --j;
    }
    indices[j + 1] = key;
  }
}

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

// Adaptive sort: insertion sort for small n, bitonic sort for larger n.
// Bitonic sort operates on next_pow2(n) elements instead of always 1024,
// dramatically reducing work when n << kMaxFramesCompile.
template <typename Comp>
__device__ void adaptive_sort(int* indices, int n, Comp comp) {
  if (n <= 64) {
    for (int i = 0; i < n; ++i) indices[i] = i;
    insertion_adaptive_sort(indices, n, comp);
    return;
  }
  // Bitonic sort over next_pow2(n) elements.
  const int sort_size = next_pow2(n);
  for (int i = n; i < sort_size; ++i) indices[i] = i;
  for (int k = 2; k <= sort_size; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = 0; i < sort_size; ++i) {
        const int l = i ^ j;
        if (l > i) {
          const bool up = (i & k) == 0;
          if (up && comp(indices[l], indices[i])) {
            const int tmp = indices[i];
            indices[i] = indices[l];
            indices[l] = tmp;
          } else if (!up && comp(indices[i], indices[l])) {
            const int tmp = indices[i];
            indices[i] = indices[l];
            indices[l] = tmp;
          }
        }
      }
    }
  }
}

// Weighted median of (values, weights).  Sorts by value ascending using bitonic
// sort, then walks cumulative weight to the 50% target.
__device__ float weighted_median_value(
    const float* values, const float* weights, int n,
    int* sort_buf) {
  if (n <= 0) return 0.0f;
  for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
  adaptive_sort(sort_buf, n, ValueAsc{values});

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
__device__ float weighted_mad_value(
    const float* values, const float* weights, int n,
    float center, int* sort_buf) {
  if (n <= 0) return 0.0f;
  for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
  adaptive_sort(sort_buf, n, DeviationAsc{values, center});

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

// Noise floor for the non-small-N path: eps_rel * MAD(values).
// Sorts by value, computes median of values, then sorts by deviation and
// computes median of deviations.
__device__ float noise_floor_value(
    const float* values, int n, int* sort_buf) {
  if (n <= 0) return device_eps();

  for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
  adaptive_sort(sort_buf, n, ValueAsc{values});
  const float med = (n % 2 == 1)
      ? values[sort_buf[n / 2]]
      : 0.5f * (values[sort_buf[n / 2 - 1]] + values[sort_buf[n / 2]]);

  for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
  adaptive_sort(sort_buf, n, DeviationAsc{values, med});
  const float mad = (n % 2 == 1)
      ? fabsf(values[sort_buf[n / 2]] - med)
      : 0.5f * (fabsf(values[sort_buf[n / 2 - 1]] - med) +
                fabsf(values[sort_buf[n / 2]] - med));

  const float eps_rel = metrics::aqmh_eps_rel;
  return fmaxf(device_eps(), eps_rel * mad);
}

// Cherry-pick: select top-K by score descending, frame_index ascending.
// Returns new sample count. Operates in-place on the first 'n' entries of
// values/weights/scores arrays.
__device__ int cherry_pick_top_k(
    float* values, float* weights, float* scores, int n,
    int k_min_required, float k_frac, int* sort_buf) {
  if (n < k_min_required) return n;

  const int nominal = max(0, static_cast<int>(floorf(k_frac * static_cast<float>(n))));
  const int k = min(n, max(k_min_required, nominal));
  if (k >= n) return n;

  for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
  adaptive_sort(sort_buf, n, ScoreDesc{scores});

  for (int i = 0; i < k; ++i) {
    const int src = sort_buf[i];
    if (src != i) {
      values[i] = values[src];
      weights[i] = weights[src];
      scores[i] = scores[src];
    }
  }
  return k;
}

// Device sigma-clip matching the CPU aqmh_sigma_clip logic.
// Operates on the first 'n' entries of values/weights. Returns retained count
// and writes weighted sum / effective N into out_*.
__device__ int sigma_clip(
    float* values, float* weights, int n,
    float clip_sigma, int iterations,
    float min_fraction, float min_n_eff,
    float* out_weight_sum, float* out_effective_n) {
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
  for (int i = n; i < kMaxFramesCompile; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
  }

  // Local scratch array reused for median/MAD computations.
  int sort_buf[kMaxFramesCompile];
  float deviations[kMaxFramesCompile];
  float sorted_values[kMaxFramesCompile];

  for (int iter = 0; iter < iterations; ++iter) {
    const float center = weighted_median_value(values, weights, n, sort_buf);

    float mad;
    float floor_val;
    if (n <= 8) {
      // Small-N path identical to CPU.
      for (int i = 0; i < n; ++i) deviations[i] = fabsf(values[i] - center);
      for (int i = 1; i < n; ++i) {
        float key = deviations[i];
        int j = i - 1;
        while (j >= 0 && deviations[j] > key) {
          deviations[j + 1] = deviations[j];
          --j;
        }
        deviations[j + 1] = key;
      }
      const int mid = n / 2;
      mad = (n % 2 == 1)
          ? deviations[mid]
          : 0.5f * (deviations[mid - 1] + deviations[mid]);

      for (int i = 0; i < n; ++i) sorted_values[i] = values[i];
      for (int i = 1; i < n; ++i) {
        float key = sorted_values[i];
        int j = i - 1;
        while (j >= 0 && sorted_values[j] > key) {
          sorted_values[j + 1] = sorted_values[j];
          --j;
        }
        sorted_values[j + 1] = key;
      }
      const float val_med = (n % 2 == 1)
          ? sorted_values[n / 2]
          : 0.5f * (sorted_values[n / 2 - 1] + sorted_values[n / 2]);
      for (int i = 0; i < n; ++i) sorted_values[i] = fabsf(sorted_values[i] - val_med);
      for (int i = 1; i < n; ++i) {
        float key = sorted_values[i];
        int j = i - 1;
        while (j >= 0 && sorted_values[j] > key) {
          sorted_values[j + 1] = sorted_values[j];
          --j;
        }
        sorted_values[j + 1] = key;
      }
      const float noise_mad = (n % 2 == 1)
          ? sorted_values[n / 2]
          : 0.5f * (sorted_values[n / 2 - 1] + sorted_values[n / 2]);
      const float eps_rel = metrics::aqmh_eps_rel;
      floor_val = fmaxf(device_eps(), eps_rel * noise_mad);
    } else {
      mad = weighted_mad_value(values, weights, n, center, sort_buf);
      floor_val = noise_floor_value(values, n, sort_buf);
    }

    if (mad <= floor_val) {
      // All equal (within epsilon guard): keep only samples equal to center.
      int keep_count = 0;
      for (int i = 0; i < n; ++i) {
        if (values[i] == center) ++keep_count;
      }
      if (keep_count == n) break;
      if (keep_count < keep_floor) {
        // Sort by normalized distance and keep floor closest to center.
        for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
        adaptive_sort(
            sort_buf, n,
            NormDistanceAsc{values, center, fmaxf(1.4826f * mad, floor_val)});
        for (int i = 0; i < keep_floor; ++i) {
          const int src = sort_buf[i];
          if (src != i) {
            values[i] = values[src];
            weights[i] = weights[src];
          }
        }
        n = keep_floor;
      } else {
        // Keep only samples equal to center.
        int m = 0;
        for (int i = 0; i < n; ++i) {
          if (values[i] == center) {
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
      if (fabsf(values[i] - center) <= clip_sigma * sigma) ++keep_count;
    }

    if (keep_count == n) {
      break;
    }
    if (keep_count < keep_floor) {
      // Sort by normalized distance and keep floor.
      for (int i = 0; i < kMaxFramesCompile; ++i) sort_buf[i] = i;
      adaptive_sort(sort_buf, n, NormDistanceAsc{values, center, sigma});
      for (int i = 0; i < keep_floor; ++i) {
        const int src = sort_buf[i];
        if (src != i) {
          values[i] = values[src];
          weights[i] = weights[src];
        }
      }
      n = keep_floor;
      break;
    }

    // Keep samples within clip band.
    m = 0;
    for (int i = 0; i < n; ++i) {
      if (fabsf(values[i] - center) <= clip_sigma * sigma) {
        values[m] = values[i];
        weights[m] = weights[i];
        ++m;
      }
    }
    n = m;
    // Pad the new tail before the next iteration's sorts.
    for (int i = n; i < kMaxFramesCompile; ++i) {
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

// Uniform-control kernel: separate from the weighted kernel so the weighted
// kernel does not need to duplicate 2*kMaxFrames floats of local memory.
// Each thread processes one pixel with uniform weights = 1.0f.
__global__ void aqmh_uniform_control_kernel(
    const float* __restrict__ d_frames,
    const uint8_t* __restrict__ d_canvas_mask,
    const uint8_t* __restrict__ d_frame_masks,
    float* __restrict__ d_uniform_control,
    unsigned long long* __restrict__ d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height, int frame_count,
    float clip_sigma, int clip_iterations,
    float min_fraction, float min_n_eff) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= chunk_rows) return;

  const int y = y0 + yy;
  if (y >= height) return;

  const int canvas_idx = yy * width + x;
  if (d_canvas_mask[canvas_idx] == 0u) {
    d_uniform_control[canvas_idx] = 0.0f;
    return;
  }

  float values[kMaxFramesCompile];
  float weights[kMaxFramesCompile];
  int n_samples = 0;

  const int pixel_base = canvas_idx * frame_count;
  for (int fi = 0; fi < frame_count; ++fi) {
    const int idx = pixel_base + fi;
    if (d_frame_masks[idx] == 0u) continue;
    const float v = d_frames[idx];
    if (!isfinite_f(v)) continue;
    values[n_samples] = v;
    weights[n_samples] = 1.0f;
    ++n_samples;
  }

  for (int i = n_samples; i < kMaxFramesCompile; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
  }

  if (n_samples <= 0) {
    d_uniform_control[canvas_idx] = 0.0f;
    atomicAdd(d_numerical_guard_pixels, 1ULL);
    return;
  }

  float weight_sum = 0.0f;
  float effective_n = 0.0f;
  const int retained = sigma_clip(
      values, weights, n_samples,
      clip_sigma, clip_iterations, min_fraction, min_n_eff,
      &weight_sum, &effective_n);

  if (retained <= 0 || weight_sum <= 0.0f) {
    d_uniform_control[canvas_idx] = 0.0f;
    atomicAdd(d_numerical_guard_pixels, 1ULL);
    return;
  }

  double accum = 0.0;
  for (int i = 0; i < retained; ++i) {
    accum += static_cast<double>(weights[i]) * values[i];
  }
  d_uniform_control[canvas_idx] = static_cast<float>(accum / weight_sum);
}

__global__ void aqmh_reconstruction_kernel(
    const float* __restrict__ d_frames,
    const float* __restrict__ d_q_maps,
    const uint8_t* __restrict__ d_canvas_mask,
    const uint8_t* __restrict__ d_frame_masks,
    const float* __restrict__ d_global_weights,
    float* __restrict__ d_output,
    float* __restrict__ d_weight_sum,
    float* __restrict__ d_cherry_k_map,
    unsigned long long* __restrict__ d_unsupported_pixels,
    unsigned long long* __restrict__ d_zero_veto_pixels,
    unsigned long long* __restrict__ d_numerical_guard_pixels,
    int width, int chunk_rows, int y0, int height, int frame_count,
    float clip_sigma, int clip_iterations,
    float min_fraction, float min_n_eff,
    bool cherry_pick_enabled, float cherry_pick_k_frac,
    int cherry_pick_k_min_required) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= chunk_rows) return;

  const int y = y0 + yy;
  if (y >= height) return;

  const int canvas_idx = yy * width + x;
  if (d_canvas_mask[canvas_idx] == 0u) return;

  // Per-thread local arrays (spill to L2-cached local memory for large
  // frame_count). Index 0..n_samples-1 holds gathered samples.
  float values[kMaxFramesCompile];
  float weights[kMaxFramesCompile];
  float scores[kMaxFramesCompile];
  int n_samples = 0;
  bool has_finite_q = false;

  // Pixel-major layout: all frames for one pixel are contiguous, so the
  // inner loop reads with stride 1 and neighboring threads read coalesced.
  const int pixel_base = canvas_idx * frame_count;
  for (int fi = 0; fi < frame_count; ++fi) {
    const int idx = pixel_base + fi;
    const uint8_t mask_val = d_frame_masks[idx];
    if (mask_val == 0u) continue;

    const float v = d_frames[idx];
    const float q = d_q_maps[idx];
    if (!isfinite_f(v) || !isfinite_f(q)) continue;
    has_finite_q = true;

    const float gw = d_global_weights[fi];
    const float score = gw * fmaxf(0.0f, q);

    if (score > 0.0f && n_samples < kMaxFramesCompile) {
      values[n_samples] = v;
      weights[n_samples] = score;
      scores[n_samples] = score;
      ++n_samples;
    }
  }

  // Sentinel-pad the unused tail so all bitonic sorts over the fixed 1024-wide
  // arrays see deterministic, harmless values.
  for (int i = n_samples; i < kMaxFramesCompile; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
    scores[i] = -INFINITY;
  }

  if (n_samples == 0) {
    d_output[canvas_idx] = 0.0f;
    d_weight_sum[canvas_idx] = 0.0f;
    if (cherry_pick_enabled) d_cherry_k_map[canvas_idx] = 0.0f;
    atomicAdd(d_unsupported_pixels, 1ULL);
    if (has_finite_q) atomicAdd(d_zero_veto_pixels, 1ULL);
    return;
  }

  int k_effective = n_samples;
  if (cherry_pick_enabled) {
    int sort_buf[kMaxFramesCompile];
    k_effective = cherry_pick_top_k(
        values, weights, scores, n_samples,
        cherry_pick_k_min_required, cherry_pick_k_frac, sort_buf);
    d_cherry_k_map[canvas_idx] = static_cast<float>(k_effective);
  }

  float weight_sum = 0.0f;
  float effective_n = 0.0f;
  const int retained = sigma_clip(
      values, weights, k_effective,
      clip_sigma, clip_iterations, min_fraction, min_n_eff,
      &weight_sum, &effective_n);

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
  if (bufs.cherry_k_map) cudaFree(bufs.cherry_k_map);
  bufs = GpuBuffers{};
}

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
      cherry_enabled ? cfg.cherry_pick_k_min_required : 0);

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
      4u * 1024u * 1024u * 1024u);

  // Per-row estimate: frames + q_maps + masks + output scratch (3 floats + 1 mask)
  const size_t bytes_per_row =
      static_cast<size_t>(frame_count) * width * sizeof(float) * 2 +
      static_cast<size_t>(frame_count) * width * sizeof(uint8_t) +
      static_cast<size_t>(width) * sizeof(float) * 4 +
      static_cast<size_t>(width) * sizeof(uint8_t);
  int chunk_rows;
  if (cfg.chunk_rows > 0) {
    chunk_rows = std::min(height, cfg.chunk_rows);
  } else {
    chunk_rows = std::max(1, std::min(height,
        static_cast<int>(device_budget / std::max<size_t>(1, bytes_per_row))));
  }
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

  GpuBuffers bufs;
  if (!allocate_chunk_buffers(bufs, width, chunk_rows, static_cast<int>(frame_count),
                              cfg.compute_uniform_control, cherry_enabled)) {
    free_chunk_buffers(bufs);
    cudaFree(d_global_weights);
    result.acceleration_fallback = true;
    return result;
  }

  // Host staging buffers.
  std::vector<float> h_frames(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
  std::vector<float> h_q_maps(static_cast<size_t>(frame_count) * chunk_rows * width, 0.0f);
  std::vector<uint8_t> h_masks(static_cast<size_t>(frame_count) * chunk_rows * width, 0u);
  std::vector<uint8_t> h_canvas_mask(static_cast<size_t>(chunk_rows) * width, 0u);
  std::vector<float> h_output(static_cast<size_t>(chunk_rows) * width, 0.0f);
  std::vector<float> h_weight_sum(static_cast<size_t>(chunk_rows) * width, 0.0f);
  std::vector<float> h_uniform_control;
  if (cfg.compute_uniform_control)
    h_uniform_control.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);
  std::vector<float> h_cherry_k_map;
  if (cherry_enabled)
    h_cherry_k_map.assign(static_cast<size_t>(chunk_rows) * width, 0.0f);

  // Track per-frame compatibility and missing samples.
  std::vector<uint8_t> frame_compatible(frame_count, 0u);

  unsigned long long* d_unsupported_pixels = nullptr;
  unsigned long long* d_zero_veto_pixels = nullptr;
  unsigned long long* d_numerical_guard_pixels = nullptr;
  CUDA_CHECK(cudaMalloc(&d_unsupported_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_zero_veto_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_numerical_guard_pixels, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemsetAsync(d_unsupported_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK(cudaMemsetAsync(d_zero_veto_pixels, 0, sizeof(unsigned long long), 0));
  CUDA_CHECK(cudaMemsetAsync(d_numerical_guard_pixels, 0, sizeof(unsigned long long), 0));

  const dim3 block(8, 8);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);

    // Masks must be zeroed each chunk so frames that fail to load are skipped.
    std::fill(h_masks.begin(), h_masks.end(), 0u);

    // Prepare canvas mask slice.
    for (int yy = 0; yy < rows; ++yy) {
      const int y = y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t full_i = static_cast<size_t>(y) * width + x;
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        h_canvas_mask[local_i] =
            (canvas_mask.empty() || full_i >= canvas_mask.size()) ? 1u
                                                                  : canvas_mask[full_i];
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        bufs.canvas_mask, h_canvas_mask.data(),
        static_cast<size_t>(rows) * width * sizeof(uint8_t),
        cudaMemcpyHostToDevice, stream));

    // Load all frames / q-maps / masks for this chunk using region loaders
    // when available (avoids loading full W×H frames per chunk — only rows
    // rows are needed, cutting I/O by ~height/chunk_rows×).
    // Parallelized across frames with OpenMP for I/O overlap.
    const bool use_region = static_cast<bool>(load_frame_region);
    const int num_host_threads = std::min(
        static_cast<int>(frame_count),
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
    #if defined(_OPENMP)
    #pragma omp parallel for num_threads(num_host_threads) schedule(dynamic, 4)
    #endif
    for (ptrdiff_t fi_ptr = 0; fi_ptr < static_cast<ptrdiff_t>(frame_count); ++fi_ptr) {
      const size_t fi = static_cast<size_t>(fi_ptr);
      Matrix2Df frame_region;
      const bool frame_ok = use_region
          ? load_frame_region(fi, y0, rows, frame_region)
          : load_frame(fi, frame_region);
      if (!frame_ok || frame_region.rows() != rows ||
          frame_region.cols() != width) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }
      frame_compatible[fi] = 1u;

      Matrix2Df q = use_region
          ? q_map_cache->read_region(fi, y0, rows)
          : q_map_cache->read_cached(fi);
      if (q.rows() != rows || q.cols() != width) {
        #if defined(_OPENMP)
        #pragma omp atomic
        #endif
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        frame_compatible[fi] = 0u;
        continue;
      }

      std::vector<uint8_t> fm;
      bool mask_ok = true;
      if (load_frame_valid_mask_region && use_region) {
        mask_ok = load_frame_valid_mask_region(fi, y0, rows, fm);
        if (!mask_ok || fm.size() != static_cast<size_t>(width * rows)) {
          #if defined(_OPENMP)
          #pragma omp atomic
          #endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          frame_compatible[fi] = 0u;
          continue;
        }
      } else if (load_frame_valid_mask) {
        mask_ok = load_frame_valid_mask(fi, fm);
        if (!mask_ok || fm.size() != static_cast<size_t>(width * height)) {
          #if defined(_OPENMP)
          #pragma omp atomic
          #endif
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          frame_compatible[fi] = 0u;
          continue;
        }
      }

      uint64_t local_missing = 0;
      uint64_t local_finite = 0;
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t full_i = static_cast<size_t>(y) * width + x;
          const size_t local_i = static_cast<size_t>(yy) * width + x;
          // Pixel-major layout: index = local_i * frame_count + fi. This makes the
          // GPU kernel's inner frame loop coalesced (neighboring threads read the
          // same frame index for their respective pixel), replacing the previous
          // frame-major stride of chunk_rows * width.
          const size_t idx = static_cast<size_t>(local_i) * frame_count + fi;
          h_frames[idx] = frame_region(yy, x);
          h_q_maps[idx] = q(yy, x);
          if (fm.empty()) {
            h_masks[idx] = 1u;
          } else {
            const size_t mask_i = use_region ? local_i : full_i;
            h_masks[idx] = fm[mask_i];
            if (fm[mask_i] == 0u) continue;
          }
          if (h_canvas_mask[local_i] == 0u) continue;
          if (!std::isfinite(q(yy, x))) {
            ++local_missing;
          } else {
            ++local_finite;
          }
        }
      }
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

    // Upload frame/q-map/mask chunk.
    const size_t used_all_pixels = static_cast<size_t>(frame_count) * rows * width;
    CUDA_CHECK(cudaMemcpyAsync(
        bufs.frames, h_frames.data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs.q_maps, h_q_maps.data(),
        used_all_pixels * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        bufs.frame_masks, h_masks.data(),
        used_all_pixels * sizeof(uint8_t),
        cudaMemcpyHostToDevice, stream));

    const dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    aqmh_reconstruction_kernel<<<grid, block, 0, stream>>>(
        bufs.frames, bufs.q_maps, bufs.canvas_mask, bufs.frame_masks,
        d_global_weights, bufs.output, bufs.weight_sum,
        bufs.cherry_k_map,
        d_unsupported_pixels, d_zero_veto_pixels, d_numerical_guard_pixels,
        width, chunk_rows, y0, height, static_cast<int>(frame_count),
        cfg.clip_sigma, cfg.clip_iterations,
        cfg.min_fraction, cfg.min_n_eff,
        cherry_enabled, cfg.cherry_pick_k_frac,
        cfg.cherry_pick_k_min_required);
    CUDA_CHECK(cudaGetLastError());

    if (cfg.compute_uniform_control) {
      aqmh_uniform_control_kernel<<<grid, block, 0, stream>>>(
          bufs.frames, bufs.canvas_mask, bufs.frame_masks,
          bufs.uniform_control, d_numerical_guard_pixels,
          width, chunk_rows, y0, height, static_cast<int>(frame_count),
          cfg.clip_sigma, cfg.clip_iterations,
          cfg.min_fraction, cfg.min_n_eff);
      CUDA_CHECK(cudaGetLastError());
    }

    // Download outputs.
    const size_t used_chunk_pixels = static_cast<size_t>(rows) * width;
    CUDA_CHECK(cudaMemcpyAsync(
        h_output.data(), bufs.output,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        h_weight_sum.data(), bufs.weight_sum,
        used_chunk_pixels * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    if (cfg.compute_uniform_control) {
      CUDA_CHECK(cudaMemcpyAsync(
          h_uniform_control.data(), bufs.uniform_control,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, stream));
    }
    if (cherry_enabled) {
      CUDA_CHECK(cudaMemcpyAsync(
          h_cherry_k_map.data(), bufs.cherry_k_map,
          used_chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy to result matrices.
    for (int yy = 0; yy < rows; ++yy) {
      const int y = y0 + yy;
      for (int x = 0; x < width; ++x) {
        const size_t local_i = static_cast<size_t>(yy) * width + x;
        result.output(y, x) = h_output[local_i];
        result.weight_sum(y, x) = h_weight_sum[local_i];
        if (cfg.compute_uniform_control)
          result.uniform_control_output(y, x) = h_uniform_control[local_i];
        if (cherry_enabled)
          result.cherry_pick_k_map(y, x) = h_cherry_k_map[local_i];
      }
    }

    if (progress) progress(y0 + rows, height);
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
  free_chunk_buffers(bufs);
  cudaFree(d_global_weights);
  cudaFree(d_unsupported_pixels);
  cudaFree(d_zero_veto_pixels);
  cudaFree(d_numerical_guard_pixels);
#undef CUDA_CHECK
  return result;
}

} // namespace tile_compile::reconstruction

#endif
