#include "tile_compile/reconstruction/aqmh_reconstruction_opencl.hpp"

#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

constexpr int kMaxFramesCompile = 1024;

// OpenCL C kernel source. Mirrors the CUDA kernel logic.
const char* kAqmhReconstructionKernelSrc = R"ocl(
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.1920929e-7f
#endif

#define MAX_FRAMES 1024

// Bitonic sort helpers for OpenCL C.  Separate functions are required because
// OpenCL C has no function overloading/templates.

void bitonic_sort_by_value_asc(int* indices, int n, const float* values) {
  for (int i = n; i < MAX_FRAMES; ++i) indices[i] = i;
  for (int k = 2; k <= MAX_FRAMES; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = 0; i < MAX_FRAMES; ++i) {
        int l = i ^ j;
        if (l > i) {
          int up = (i & k) == 0;
          int a = indices[i];
          int b = indices[l];
          int swap = 0;
          if (up) {
            if (values[b] < values[a] || (values[b] == values[a] && b < a)) swap = 1;
          } else {
            if (values[a] < values[b] || (values[a] == values[b] && a < b)) swap = 1;
          }
          if (swap) {
            indices[i] = b;
            indices[l] = a;
          }
        }
      }
    }
  }
}

void bitonic_sort_by_score_desc(int* indices, int n, const float* scores) {
  for (int i = n; i < MAX_FRAMES; ++i) indices[i] = i;
  for (int k = 2; k <= MAX_FRAMES; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = 0; i < MAX_FRAMES; ++i) {
        int l = i ^ j;
        if (l > i) {
          int up = (i & k) == 0;
          int a = indices[i];
          int b = indices[l];
          int swap = 0;
          if (up) {
            // descending: higher score should be at i in up part
            if (scores[b] > scores[a] || (scores[b] == scores[a] && b < a)) swap = 1;
          } else {
            if (scores[a] > scores[b] || (scores[a] == scores[b] && a < b)) swap = 1;
          }
          if (swap) {
            indices[i] = b;
            indices[l] = a;
          }
        }
      }
    }
  }
}

void bitonic_sort_by_deviation_asc(int* indices, int n, const float* values, float center) {
  for (int i = n; i < MAX_FRAMES; ++i) indices[i] = i;
  for (int k = 2; k <= MAX_FRAMES; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = 0; i < MAX_FRAMES; ++i) {
        int l = i ^ j;
        if (l > i) {
          int up = (i & k) == 0;
          int a = indices[i];
          int b = indices[l];
          float da = fabs(values[a] - center);
          float db = fabs(values[b] - center);
          int swap = 0;
          if (up) {
            if (db < da || (db == da && b < a)) swap = 1;
          } else {
            if (da < db || (da == db && a < b)) swap = 1;
          }
          if (swap) {
            indices[i] = b;
            indices[l] = a;
          }
        }
      }
    }
  }
}

void bitonic_sort_by_norm_distance_asc(int* indices, int n, const float* values, float center, float sigma) {
  for (int i = n; i < MAX_FRAMES; ++i) indices[i] = i;
  for (int k = 2; k <= MAX_FRAMES; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = 0; i < MAX_FRAMES; ++i) {
        int l = i ^ j;
        if (l > i) {
          int up = (i & k) == 0;
          int a = indices[i];
          int b = indices[l];
          float da = fabs(values[a] - center) / sigma;
          float db = fabs(values[b] - center) / sigma;
          int swap = 0;
          if (up) {
            if (db < da || (db == da && b < a)) swap = 1;
          } else {
            if (da < db || (da == db && a < b)) swap = 1;
          }
          if (swap) {
            indices[i] = b;
            indices[l] = a;
          }
        }
      }
    }
  }
}

float device_weighted_median(
     float* values,  float* weights, int n,
     int* sort_indices) {
  if (n <= 0) return 0.0f;
  for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
  bitonic_sort_by_value_asc(sort_indices, n, values);
  float total = 0.0f;
  for (int i = 0; i < n; ++i) total += weights[sort_indices[i]];
  float target = total * 0.5f;
  float accum = 0.0f;
  for (int i = 0; i < n; ++i) {
    accum += weights[sort_indices[i]];
    if (accum >= target) return values[sort_indices[i]];
  }
  return values[sort_indices[n - 1]];
}

float weighted_mad_value(
     float* values,  float* weights, int n,
    float center,  int* sort_indices) {
  if (n <= 0) return 0.0f;
  for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
  bitonic_sort_by_deviation_asc(sort_indices, n, values, center);
  float total = 0.0f;
  for (int i = 0; i < n; ++i) total += weights[sort_indices[i]];
  float target = total * 0.5f;
  float accum = 0.0f;
  for (int i = 0; i < n; ++i) {
    accum += weights[sort_indices[i]];
    if (accum >= target) return fabs(values[sort_indices[i]] - center);
  }
  return fabs(values[sort_indices[n - 1]] - center);
}

float noise_floor_value(float* values, int n, int* sort_indices) {
  if (n <= 0) return FLT_EPSILON;
  for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
  bitonic_sort_by_value_asc(sort_indices, n, values);
  float med = (n % 2 == 1)
      ? values[sort_indices[n / 2]]
      : 0.5f * (values[sort_indices[n / 2 - 1]] + values[sort_indices[n / 2]]);
  for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
  bitonic_sort_by_deviation_asc(sort_indices, n, values, med);
  float mad = (n % 2 == 1)
      ? fabs(values[sort_indices[n / 2]] - med)
      : 0.5f * (fabs(values[sort_indices[n / 2 - 1]] - med) +
                fabs(values[sort_indices[n / 2]] - med));
  const float eps_rel = 1.0e-6f;
  return fmax(FLT_EPSILON, eps_rel * mad);
}

int cherry_pick_top_k(
     float* values,  float* weights,  float* scores, int n,
    int k_min_required, float k_frac,  int* sort_indices) {
  if (n < k_min_required) return n;
  int nominal = (int)(floor(k_frac * (float)n));
  if (nominal < 0) nominal = 0;
  int k = min(n, max(k_min_required, nominal));
  if (k >= n) return n;
  for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
  bitonic_sort_by_score_desc(sort_indices, n, scores);
  for (int i = 0; i < k; ++i) {
    int src = sort_indices[i];
    if (src != i) {
      values[i] = values[src];
      weights[i] = weights[src];
      scores[i] = scores[src];
    }
  }
  return k;
}

int sigma_clip(
     float* values,  float* weights, int n,
    float clip_sigma, int iterations,
    float min_fraction, float min_n_eff,
    float* out_weight_sum, float* out_effective_n) {
  if (n <= 0) {
    *out_weight_sum = 0.0f;
    *out_effective_n = 0.0f;
    return 0;
  }
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (!isnan(values[i]) && !isinf(values[i]) &&
        !isnan(weights[i]) && !isinf(weights[i]) && weights[i] > 0.0f) {
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
  int keep_floor = max(1, (int)(ceil(min_fraction * (float)n)));

  // Sentinel-pad the unused tail so bitonic sorts over the fixed compile-time
  // size do not touch uninitialized memory.
  for (int i = n; i < MAX_FRAMES; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
  }

   int sort_indices[MAX_FRAMES];
   float deviations[MAX_FRAMES];
   float sorted_values[MAX_FRAMES];

  for (int iter = 0; iter < iterations; ++iter) {
    float center = device_weighted_median(values, weights, n, sort_indices);
    float mad;
    float floor_val;
    if (n <= 8) {
      for (int i = 0; i < n; ++i) deviations[i] = fabs(values[i] - center);
      for (int i = 1; i < n; ++i) {
        float key = deviations[i];
        int j = i - 1;
        while (j >= 0 && deviations[j] > key) {
          deviations[j + 1] = deviations[j];
          --j;
        }
        deviations[j + 1] = key;
      }
      int mid = n / 2;
      mad = (n % 2 == 1) ? deviations[mid] : 0.5f * (deviations[mid - 1] + deviations[mid]);

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
      float val_med = (n % 2 == 1)
          ? sorted_values[n / 2]
          : 0.5f * (sorted_values[n / 2 - 1] + sorted_values[n / 2]);
      for (int i = 0; i < n; ++i) sorted_values[i] = fabs(sorted_values[i] - val_med);
      for (int i = 1; i < n; ++i) {
        float key = sorted_values[i];
        int j = i - 1;
        while (j >= 0 && sorted_values[j] > key) {
          sorted_values[j + 1] = sorted_values[j];
          --j;
        }
        sorted_values[j + 1] = key;
      }
      float noise_mad = (n % 2 == 1)
          ? sorted_values[n / 2]
          : 0.5f * (sorted_values[n / 2 - 1] + sorted_values[n / 2]);
      const float eps_rel = 1.0e-6f;
      floor_val = fmax(FLT_EPSILON, eps_rel * noise_mad);
    } else {
      mad = weighted_mad_value(values, weights, n, center, sort_indices);
      floor_val = noise_floor_value(values, n, sort_indices);
    }

    if (mad <= floor_val) {
      int keep_count = 0;
      for (int i = 0; i < n; ++i) if (values[i] == center) ++keep_count;
      if (keep_count == n) break;
      if (keep_count < keep_floor) {
        for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
        bitonic_sort_by_norm_distance_asc(
            sort_indices, n, values, center, fmax(1.4826f * mad, floor_val));
        for (int i = 0; i < keep_floor; ++i) {
          int src = sort_indices[i];
          if (src != i) {
            values[i] = values[src];
            weights[i] = weights[src];
          }
        }
        n = keep_floor;
      } else {
        int m2 = 0;
        for (int i = 0; i < n; ++i) {
          if (values[i] == center) {
            values[m2] = values[i];
            weights[m2] = weights[i];
            ++m2;
          }
        }
        n = m2;
      }
      break;
    }

    float sigma = 1.4826f * mad;
    int keep_count = 0;
    for (int i = 0; i < n; ++i) if (fabs(values[i] - center) <= clip_sigma * sigma) ++keep_count;
    if (keep_count == n) break;
    if (keep_count < keep_floor) {
      for (int i = 0; i < MAX_FRAMES; ++i) sort_indices[i] = i;
      bitonic_sort_by_norm_distance_asc(sort_indices, n, values, center, sigma);
      for (int i = 0; i < keep_floor; ++i) {
        int src = sort_indices[i];
        if (src != i) {
          values[i] = values[src];
          weights[i] = weights[src];
        }
      }
      n = keep_floor;
      break;
    }

    int m2 = 0;
    for (int i = 0; i < n; ++i) {
      if (fabs(values[i] - center) <= clip_sigma * sigma) {
        values[m2] = values[i];
        weights[m2] = weights[i];
        ++m2;
      }
    }
    n = m2;
    // Pad the new tail before the next iteration's sorts.
    for (int i = n; i < MAX_FRAMES; ++i) {
      values[i] = INFINITY;
      weights[i] = 0.0f;
    }
  }

  float d = 0.0f;
  float d2 = 0.0f;
  float wmax = 0.0f;
  for (int i = 0; i < n; ++i) {
    d += weights[i];
    d2 += weights[i] * weights[i];
    wmax = fmax(wmax, weights[i]);
  }
  *out_weight_sum = d;
  *out_effective_n = (d2 > 0.0f) ? (d * d / d2) : 0.0f;
  float guard = (float)n * FLT_EPSILON * wmax;
  if (d <= guard || *out_effective_n < min_n_eff) {
    *out_weight_sum = 0.0f;
    *out_effective_n = 0.0f;
    return 0;
  }
  return n;
}

__kernel void aqmh_reconstruction_kernel(
    __global const float* frames,
    __global const float* q_maps,
    __global const uchar* canvas_mask,
    __global const uchar* frame_masks,
    __global const float* global_weights,
    __global float* output,
    __global float* weight_sum,
    __global float* uniform_control,
    __global float* cherry_k_map,
    int width, int chunk_rows, int y0, int height, int frame_count,
    float clip_sigma, int clip_iterations,
    float min_fraction, float min_n_eff,
    int cherry_pick_enabled, float cherry_pick_k_frac,
    int cherry_pick_k_min_required,
    int compute_uniform_control, int uniform_weights) {
  int x = get_global_id(0);
  int yy = get_global_id(1);
  if (x >= width || yy >= chunk_rows) return;
  int y = y0 + yy;
  if (y >= height) return;

  int canvas_idx = yy * width + x;
  if (canvas_mask[canvas_idx] == 0u) return;

   float values[MAX_FRAMES];
   float weights[MAX_FRAMES];
   float scores[MAX_FRAMES];
  int n_samples = 0;

  for (int fi = 0; fi < frame_count; ++fi) {
    int idx = fi * chunk_rows * width + canvas_idx;
    uchar mask_val = frame_masks[idx];
    if (mask_val == 0u) continue;

    float v = frames[idx];
    float q = q_maps[idx];
    if (isnan(v) || isinf(v) || isnan(q) || isinf(q)) continue;

    float gw = global_weights[fi];
    float score = gw * fmax(0.0f, q);
    float weight = uniform_weights ? 1.0f : score;

    if (score > 0.0f && weight > 0.0f && n_samples < MAX_FRAMES) {
      values[n_samples] = v;
      weights[n_samples] = weight;
      scores[n_samples] = score;
      ++n_samples;
    }
  }

  // Sentinel-pad the unused tail so all bitonic sorts see deterministic values.
  for (int i = n_samples; i < MAX_FRAMES; ++i) {
    values[i] = INFINITY;
    weights[i] = 0.0f;
    scores[i] = -INFINITY;
  }

  if (n_samples == 0) {
    output[canvas_idx] = 0.0f;
    weight_sum[canvas_idx] = 0.0f;
    if (compute_uniform_control) uniform_control[canvas_idx] = 0.0f;
    if (cherry_pick_enabled) cherry_k_map[canvas_idx] = 0.0f;
    return;
  }

  int k_effective = n_samples;
  if (cherry_pick_enabled) {
     int sort_indices[MAX_FRAMES];
    k_effective = cherry_pick_top_k(
        values, weights, scores, n_samples,
        cherry_pick_k_min_required, cherry_pick_k_frac, sort_indices);
    cherry_k_map[canvas_idx] = (float)k_effective;
  }

  int reuse_control = 0;
  if (compute_uniform_control) {
    reuse_control = 1;
    float first_weight = weights[0];
    for (int i = 1; i < k_effective; ++i) {
      if (weights[i] != first_weight) {
        reuse_control = 0;
        break;
      }
    }
  }

  if (compute_uniform_control && !reuse_control) {
     float control_values[MAX_FRAMES];
     float control_weights[MAX_FRAMES];
    for (int i = 0; i < k_effective; ++i) {
      control_values[i] = values[i];
      control_weights[i] = 1.0f;
    }
    for (int i = k_effective; i < MAX_FRAMES; ++i) {
      control_values[i] = INFINITY;
      control_weights[i] = 0.0f;
    }
    float ctrl_weight_sum = 0.0f;
    float ctrl_effective_n = 0.0f;
    int ctrl_n = sigma_clip(
        control_values, control_weights, k_effective,
        clip_sigma, clip_iterations, min_fraction, min_n_eff,
        &ctrl_weight_sum, &ctrl_effective_n);
    if (ctrl_n > 0 && ctrl_weight_sum > 0.0f) {
      float accum = 0.0f;
      for (int i = 0; i < ctrl_n; ++i) accum += control_weights[i] * control_values[i];
      uniform_control[canvas_idx] = accum / ctrl_weight_sum;
    } else {
      uniform_control[canvas_idx] = 0.0f;
    }
  }

  float wsum = 0.0f;
  float eff_n = 0.0f;
  int retained = sigma_clip(
      values, weights, k_effective,
      clip_sigma, clip_iterations, min_fraction, min_n_eff,
      &wsum, &eff_n);

  if (retained <= 0 || wsum <= 0.0f) {
    output[canvas_idx] = 0.0f;
    weight_sum[canvas_idx] = 0.0f;
    if (compute_uniform_control && reuse_control)
      uniform_control[canvas_idx] = 0.0f;
    return;
  }

  float accum = 0.0f;
  for (int i = 0; i < retained; ++i) accum += weights[i] * values[i];
  output[canvas_idx] = accum / wsum;
  weight_sum[canvas_idx] = wsum;
  if (compute_uniform_control && reuse_control)
    uniform_control[canvas_idx] = output[canvas_idx];
}
)ocl";

bool have_opencl() {
#if TILE_COMPILE_HAS_OPENCV_OPENCL
  try {
    return cv::ocl::haveOpenCL() && cv::ocl::useOpenCL();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

} // namespace

AqmhReconstructionResult reconstruct_aqmh_weighted_opencl(
    size_t frame_count,
    const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache,
    const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask,
    int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask,
    const AqmhProgressCallback &progress) {
  AqmhReconstructionResult result;
  result.output = Matrix2Df::Zero(height, width);
  result.weight_sum = Matrix2Df::Zero(height, width);
  if (cfg.compute_uniform_control)
    result.uniform_control_output = Matrix2Df::Zero(height, width);

  if (!have_opencl()) {
    std::cerr << "[OpenCL] OpenCL not available; falling back to CPU" << std::endl;
    result = reconstruct_aqmh_weighted(
        frame_count, load_frame, q_map_cache, global_weights, canvas_mask, width,
        height, cfg, load_frame_valid_mask, {}, {}, progress);
    result.acceleration_fallback = true;
    return result;
  }

  if (frame_count == 0 || !q_map_cache || width <= 0 || height <= 0) {
    result.acceleration_fallback = true;
    return result;
  }

  if (static_cast<int>(frame_count) > kMaxFramesCompile) {
    std::cerr << "[OpenCL] frame_count " << frame_count
              << " exceeds compile-time limit " << kMaxFramesCompile
              << "; falling back to CPU" << std::endl;
    result = reconstruct_aqmh_weighted(
        frame_count, load_frame, q_map_cache, global_weights, canvas_mask, width,
        height, cfg, load_frame_valid_mask, {}, {}, progress);
    result.acceleration_fallback = true;
    return result;
  }

  const bool cherry_enabled = cfg.cherry_pick &&
      static_cast<int>(frame_count) >= cfg.cherry_pick_k_min_required;
  if (cherry_enabled) {
    result.cherry_pick_k_map = Matrix2Df::Zero(height, width);
    result.cherry_pick_per_pixel_mode = true;
  }
  result.k_nominal_median = static_cast<float>(
      cherry_enabled ? cfg.cherry_pick_k_min_required : 0);

  // Use a moderate chunk size for OpenCL. Auto size based on a 2 GB budget.
  int chunk_rows;
  if (cfg.chunk_rows > 0) {
    chunk_rows = std::min(height, cfg.chunk_rows);
  } else {
    const size_t bytes_per_row =
        static_cast<size_t>(frame_count) * width * sizeof(float) * 2 +
        static_cast<size_t>(frame_count) * width * sizeof(uint8_t) +
        static_cast<size_t>(width) * sizeof(float) * 4;
    const size_t budget = 2u * 1024u * 1024u * 1024u;
    chunk_rows = std::max(1, std::min(height,
        static_cast<int>(budget / std::max<size_t>(1, bytes_per_row))));
  }
  result.chunk_rows = chunk_rows;
  result.chunk_count = (height + chunk_rows - 1) / chunk_rows;
  result.region_streaming_used = true;

  std::vector<float> h_global_weights(frame_count, 1.0f);
  for (Eigen::Index fi = 0; fi < global_weights.size() && fi < static_cast<Eigen::Index>(frame_count); ++fi) {
    const float w = global_weights[fi];
    h_global_weights[fi] = std::isfinite(w) && w > 0.0f ? w : 0.0f;
  }

  const size_t all_pixels = static_cast<size_t>(frame_count) * chunk_rows * width;
  const size_t chunk_pixels = static_cast<size_t>(chunk_rows) * width;

  cv::UMat u_frames, u_q_maps, u_frame_masks, u_canvas_mask, u_global_weights;
  cv::UMat u_output, u_weight_sum, u_uniform_control, u_cherry_k_map;

  cv::Mat(1, static_cast<int>(all_pixels), CV_32F).copyTo(u_frames);
  cv::Mat(1, static_cast<int>(all_pixels), CV_32F).copyTo(u_q_maps);
  cv::Mat(1, static_cast<int>(all_pixels), CV_8U).copyTo(u_frame_masks);
  cv::Mat(1, static_cast<int>(chunk_pixels), CV_8U).copyTo(u_canvas_mask);
  cv::Mat(1, static_cast<int>(frame_count), CV_32F).copyTo(u_global_weights);
  cv::Mat(1, static_cast<int>(chunk_pixels), CV_32F).copyTo(u_output);
  cv::Mat(1, static_cast<int>(chunk_pixels), CV_32F).copyTo(u_weight_sum);
  if (cfg.compute_uniform_control)
    cv::Mat(1, static_cast<int>(chunk_pixels), CV_32F).copyTo(u_uniform_control);
  if (cherry_enabled)
    cv::Mat(1, static_cast<int>(chunk_pixels), CV_32F).copyTo(u_cherry_k_map);

  cv::Mat(1, static_cast<int>(frame_count), CV_32F, h_global_weights.data())
      .copyTo(u_global_weights);

  cv::ocl::ProgramSource source(
      "tile_compile", "aqmh_reconstruction_kernel",
      kAqmhReconstructionKernelSrc, "");
  cv::ocl::Kernel kernel("aqmh_reconstruction_kernel", source);
  if (kernel.empty()) {
    std::cerr << "[OpenCL] Failed to create AQMH reconstruction kernel" << std::endl;
    result = reconstruct_aqmh_weighted(
        frame_count, load_frame, q_map_cache, global_weights, canvas_mask, width,
        height, cfg, load_frame_valid_mask, {}, {}, progress);
    result.acceleration_fallback = true;
    return result;
  }

  std::vector<float> h_frames(all_pixels, 0.0f);
  std::vector<float> h_q_maps(all_pixels, 0.0f);
  std::vector<uint8_t> h_masks(all_pixels, 0u);
  std::vector<uint8_t> h_canvas_mask(chunk_pixels, 0u);
  std::vector<float> h_output(chunk_pixels, 0.0f);
  std::vector<float> h_weight_sum(chunk_pixels, 0.0f);
  std::vector<float> h_uniform_control;
  if (cfg.compute_uniform_control)
    h_uniform_control.assign(chunk_pixels, 0.0f);
  std::vector<float> h_cherry_k_map;
  if (cherry_enabled)
    h_cherry_k_map.assign(chunk_pixels, 0.0f);

  for (int y0 = 0; y0 < height; y0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - y0);
    const size_t used_all_pixels = static_cast<size_t>(frame_count) * rows * width;

    std::fill(h_masks.begin(), h_masks.end(), 0u);

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
    cv::Mat(1, static_cast<int>(rows * width), CV_8U, h_canvas_mask.data())
        .copyTo(u_canvas_mask);

    for (size_t fi = 0; fi < frame_count; ++fi) {
      Matrix2Df full_frame;
      const bool frame_ok = load_frame(fi, full_frame);
      if (!frame_ok || full_frame.rows() != height || full_frame.cols() != width) {
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }
      Matrix2Df q = q_map_cache->read_cached(fi);
      if (q.rows() != height || q.cols() != width) {
        result.missing_map_samples += static_cast<uint64_t>(rows) * width;
        continue;
      }
      std::vector<uint8_t> fm;
      bool mask_ok = true;
      if (load_frame_valid_mask) {
        mask_ok = load_frame_valid_mask(fi, fm);
        if (!mask_ok || fm.size() != static_cast<size_t>(width * height)) {
          result.missing_map_samples += static_cast<uint64_t>(rows) * width;
          continue;
        }
      }
      for (int yy = 0; yy < rows; ++yy) {
        const int y = y0 + yy;
        for (int x = 0; x < width; ++x) {
          const size_t full_i = static_cast<size_t>(y) * width + x;
          const size_t local_i = static_cast<size_t>(yy) * width + x;
          const size_t idx = fi * chunk_rows * width + local_i;
          h_frames[idx] = full_frame(y, x);
          h_q_maps[idx] = q(y, x);
          if (fm.empty()) {
            h_masks[idx] = 1u;
          } else {
            h_masks[idx] = fm[full_i];
            if (fm[full_i] == 0u) continue;
          }
          if (!std::isfinite(q(y, x))) {
            ++result.missing_map_samples;
          } else {
            ++result.finite_map_samples;
          }
        }
      }
    }

    cv::Mat(1, static_cast<int>(used_all_pixels), CV_32F, h_frames.data())
        .copyTo(u_frames);
    cv::Mat(1, static_cast<int>(used_all_pixels), CV_32F, h_q_maps.data())
        .copyTo(u_q_maps);
    cv::Mat(1, static_cast<int>(used_all_pixels), CV_8U, h_masks.data())
        .copyTo(u_frame_masks);

    int arg_idx = 0;
    kernel.set(arg_idx++, cv::ocl::KernelArg::ReadOnly(u_frames));
    kernel.set(arg_idx++, cv::ocl::KernelArg::ReadOnly(u_q_maps));
    kernel.set(arg_idx++, cv::ocl::KernelArg::ReadOnly(u_canvas_mask));
    kernel.set(arg_idx++, cv::ocl::KernelArg::ReadOnly(u_frame_masks));
    kernel.set(arg_idx++, cv::ocl::KernelArg::ReadOnly(u_global_weights));
    kernel.set(arg_idx++, cv::ocl::KernelArg::WriteOnly(u_output));
    kernel.set(arg_idx++, cv::ocl::KernelArg::WriteOnly(u_weight_sum));
    if (cfg.compute_uniform_control)
      kernel.set(arg_idx++, cv::ocl::KernelArg::WriteOnly(u_uniform_control));
    if (cherry_enabled)
      kernel.set(arg_idx++, cv::ocl::KernelArg::WriteOnly(u_cherry_k_map));
    kernel.set(arg_idx++, width);
    kernel.set(arg_idx++, chunk_rows);
    kernel.set(arg_idx++, y0);
    kernel.set(arg_idx++, height);
    kernel.set(arg_idx++, static_cast<int>(frame_count));
    kernel.set(arg_idx++, cfg.clip_sigma);
    kernel.set(arg_idx++, cfg.clip_iterations);
    kernel.set(arg_idx++, cfg.min_fraction);
    kernel.set(arg_idx++, cfg.min_n_eff);
    kernel.set(arg_idx++, cherry_enabled ? 1 : 0);
    kernel.set(arg_idx++, cfg.cherry_pick_k_frac);
    kernel.set(arg_idx++, cfg.cherry_pick_k_min_required);
    kernel.set(arg_idx++, cfg.compute_uniform_control ? 1 : 0);
    kernel.set(arg_idx++, cfg.uniform_weights ? 1 : 0);

    size_t local_size[2] = {16, 16};
    size_t global_size[2] = {
        ((static_cast<size_t>(width) + local_size[0] - 1) / local_size[0]) * local_size[0],
        ((static_cast<size_t>(rows) + local_size[1] - 1) / local_size[1]) * local_size[1]};
    const size_t used_chunk_pixels = static_cast<size_t>(rows) * width;

    bool ok = kernel.run(2, global_size, local_size, true);
    if (!ok) {
      std::cerr << "[OpenCL] Kernel run failed; falling back to CPU" << std::endl;
      result = reconstruct_aqmh_weighted(
          frame_count, load_frame, q_map_cache, global_weights, canvas_mask, width,
          height, cfg, load_frame_valid_mask, {}, {}, progress);
      result.acceleration_fallback = true;
      return result;
    }

    cv::Mat out_host;
    u_output.copyTo(out_host);
    std::memcpy(h_output.data(), out_host.ptr<float>(),
                used_chunk_pixels * sizeof(float));
    cv::Mat wsum_host;
    u_weight_sum.copyTo(wsum_host);
    std::memcpy(h_weight_sum.data(), wsum_host.ptr<float>(),
                used_chunk_pixels * sizeof(float));
    if (cfg.compute_uniform_control) {
      cv::Mat uc_host;
      u_uniform_control.copyTo(uc_host);
      std::memcpy(h_uniform_control.data(), uc_host.ptr<float>(),
                  used_chunk_pixels * sizeof(float));
    }
    if (cherry_enabled) {
      cv::Mat k_host;
      u_cherry_k_map.copyTo(k_host);
      std::memcpy(h_cherry_k_map.data(), k_host.ptr<float>(),
                  used_chunk_pixels * sizeof(float));
    }

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

        if (h_canvas_mask[local_i] == 0u) continue;
        if (h_weight_sum[local_i] <= 0.0f) ++result.unsupported_pixels;
      }
    }

    if (progress) progress(y0 + rows, height);
  }

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
  return result;
}

} // namespace tile_compile::reconstruction
