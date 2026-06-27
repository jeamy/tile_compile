#pragma once

#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if __has_include(<opencv2/core/cuda.hpp>)
#include <opencv2/core/cuda.hpp>
#else
namespace cv::cuda { class Stream; }
#endif

namespace tile_compile::core {

enum class AccelerationBackend {
  cpu = 0,
  opencv_cuda,
  opencv_opencl,
  cuda,
};

enum class AccelerationPhase {
  prewarp = 0,
  aqmh_maps,
  aqmh_reconstruction,
  tile_reconstruction,
  stacking,
};

struct AccelerationCapabilities {
  bool tile_compile_with_cuda = false;
  bool opencv_cuda_runtime = false;
  bool opencv_opencl_headers = false;
  bool opencv_opencl_runtime = false;
  int device_id = 0;
  std::string device_name;
};

struct DeviceFrame {
  int rows = 0;
  int cols = 0;
  int channels = 1;
  size_t bytes = 0;
};

struct DeviceFrameBatch {
  size_t batch_size = 0;
  DeviceFrame frame;
  size_t total_bytes = 0;
};

struct DeviceTileBatch {
  size_t batch_size = 0;
  int channels = 1;
  size_t total_pixels = 0;
  size_t total_bytes = 0;
  int max_tile_width = 0;
  int max_tile_height = 0;
};

struct AccelerationSelection {
  AccelerationPhase phase = AccelerationPhase::prewarp;
  AccelerationBackend requested = AccelerationBackend::cpu;
  AccelerationBackend selected = AccelerationBackend::cpu;
  std::string requested_name = "cpu";
  bool auto_requested = false;
  bool request_honored = true;
  bool gpu_requested = false;
  bool using_gpu = false;
  bool tile_compile_with_cuda = false;
  bool opencv_cuda_headers = false;
  bool opencv_cuda_runtime = false;
  bool opencv_opencl_headers = false;
  bool opencv_opencl_runtime = false;
  std::string fallback_reason;
};

std::string acceleration_phase_name(AccelerationPhase phase);
std::string acceleration_backend_name(AccelerationBackend backend);
bool parse_acceleration_backend(const std::string &name,
                                AccelerationBackend &backend_out);

AccelerationSelection select_acceleration_backend(
    const std::string &requested_backend_name, AccelerationPhase phase);

json acceleration_selection_to_json(const AccelerationSelection &selection);
std::string acceleration_selection_summary(const AccelerationSelection &selection);

/// Run-scoped GPU/backend state. Runtime probing and device selection happen
/// once; every phase derives its supported backend from this immutable state.
class AccelerationContext {
public:
  explicit AccelerationContext(std::string requested_backend_name,
                               int device_id = 0);

  const AccelerationCapabilities &capabilities() const { return capabilities_; }
  AccelerationSelection selection_for(AccelerationPhase phase) const;
  json to_json() const;
  void synchronize() const;

private:
  std::string requested_backend_name_;
  AccelerationCapabilities capabilities_;
};

/// Owns one non-default CUDA stream per CPU worker. The implementation is
/// hidden so CPU/OpenCL callers do not depend on CUDA implementation details.
class WorkerCudaStreams {
public:
  WorkerCudaStreams(bool enabled, size_t worker_count);
  ~WorkerCudaStreams();
  WorkerCudaStreams(WorkerCudaStreams &&) noexcept;
  WorkerCudaStreams &operator=(WorkerCudaStreams &&) noexcept;
  WorkerCudaStreams(const WorkerCudaStreams &) = delete;
  WorkerCudaStreams &operator=(const WorkerCudaStreams &) = delete;

  cv::cuda::Stream *get(size_t worker_index) noexcept;
  size_t size() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

DeviceFrame make_device_frame(int rows, int cols, int channels = 1);
DeviceFrameBatch make_device_frame_batch(size_t batch_size, int rows, int cols,
                                         int channels = 1);
DeviceTileBatch make_device_tile_batch(const std::vector<Tile> &tiles,
                                       int channels = 1);

json device_frame_to_json(const DeviceFrame &frame);
json device_frame_batch_to_json(const DeviceFrameBatch &batch);
json device_tile_batch_to_json(const DeviceTileBatch &batch);

class AccelerationOps {
public:
  explicit AccelerationOps(AccelerationSelection selection);
  AccelerationOps(const AccelerationContext &context, AccelerationPhase phase);

  const AccelerationSelection &selection() const { return selection_; }

  bool warp_affine_frame(Matrix2Df img, const WarpMatrix &warp, ColorMode mode,
                         int canvas_height, int canvas_width, int offset_x,
                         int offset_y, Matrix2Df &warped_out,
                         std::vector<uint8_t> *valid_mask_out = nullptr,
                         bool *has_data_out = nullptr,
                         cv::cuda::Stream *stream = nullptr) const;

  reconstruction::WeightedTileResult sigma_clip_reduce(
      const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
      float sigma_low, float sigma_high, int max_iters, float min_fraction,
      float eps_weight,
      cv::cuda::Stream *stream = nullptr) const;

  Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df> &frames,
                             float sigma_low, float sigma_high, int max_iters,
                             float min_fraction,
                             cv::cuda::Stream *stream = nullptr) const;

  reconstruction::AqmhReconstructionResult reconstruct_aqmh(
      size_t frame_count,
      const reconstruction::AqmhFrameLoader &load_frame,
      metrics::QualityMapCache *q_map_cache,
      const VectorXf &global_weights,
      const std::vector<uint8_t> &canvas_mask, int width, int height,
      const reconstruction::AqmhReconstructionConfig &cfg,
      cv::cuda::Stream *stream = nullptr) const;

  void overlap_add(const Matrix2Df &tile, const Tile &tile_bounds,
                   const std::vector<float> &hann_x,
                   const std::vector<float> &hann_y,
                   const std::vector<uint8_t> &common_valid_mask,
                   int canvas_width, Matrix2Df &accum, Matrix2Df &weight_sum,
                   bool accumulate_weight = true) const;

  void overlap_add(const Matrix2Df &tile, const Tile &tile_bounds,
                   const Matrix2Df &coeff, Matrix2Df &accum,
                   Matrix2Df &weight_sum,
                   bool accumulate_weight = true) const;

  void overlap_add_preweighted(const Matrix2Df &weighted_tile,
                               const Tile &tile_bounds, Matrix2Df &accum,
                               Matrix2Df &weight_sum,
                               const Matrix2Df *weight_mask = nullptr,
                               bool accumulate_weight = true) const;

  bool normalize_overlap_accum(Matrix2Df &accum, Matrix2Df &weight_sum,
                               float eps_weight,
                               float invalid_value) const;
  void flush_overlap_state(Matrix2Df &accum, Matrix2Df &weight_sum) const;

  // --- GPU batch interface (B6) ---

  /// Input bundle for one tile in a batch sigma-clip dispatch.
  struct BatchSigmaClipInput {
      std::vector<Matrix2Df> tile_frames; // per-frame tile crops
      std::vector<float>     weights;     // per-frame quality weights
  };

  /// Process multiple tiles in a single GPU dispatch (reduces kernel-launch overhead).
  /// Falls back to sequential sigma_clip_reduce() calls on CPU or on OpenCL error.
  ///
  /// @param tile_inputs   One entry per tile.
  /// @param sigma_low     Lower sigma threshold for clipping.
  /// @param sigma_high    Upper sigma threshold for clipping.
  /// @param max_iters     Maximum sigma-clip iterations.
  /// @param min_fraction  Minimum surviving pixel fraction.
  /// @param eps_weight    Minimum weight to consider a frame.
  std::vector<reconstruction::WeightedTileResult> sigma_clip_reduce_batch(
      const std::vector<BatchSigmaClipInput>& tile_inputs,
      float sigma_low,
      float sigma_high,
      int   max_iters,
      float min_fraction,
      float eps_weight,
      cv::cuda::Stream *stream = nullptr) const;

private:
  struct OverlapAddState;
  AccelerationSelection selection_;
  mutable std::unordered_map<const Matrix2Df *,
                             std::shared_ptr<OverlapAddState>>
      overlap_add_states_;
  mutable std::unordered_map<const Matrix2Df *,
                             std::shared_ptr<OverlapAddState>>
      overlap_add_coeff_states_;
  mutable std::shared_mutex overlap_add_mutex_;
};

} // namespace tile_compile::core
