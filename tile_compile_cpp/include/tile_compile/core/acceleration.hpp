#pragma once

#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"

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
  // CFA-forward-drizzle single method (plan M6/M7). CUDA covers the droplet /
  // clipping / profile accumulation only; the a-trous fusion stays on the CPU
  // reference path (plan 19.2 stage 9).
  forward_drizzle,
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
  explicit AccelerationOps(AccelerationSelection selection,
                           std::string prewarp_interpolation = "cubic");
  AccelerationOps(const AccelerationContext &context, AccelerationPhase phase,
                  std::string prewarp_interpolation = "cubic");

  const AccelerationSelection &selection() const { return selection_; }

  bool warp_affine_frame(Matrix2Df img, const WarpMatrix &warp, ColorMode mode,
                         int canvas_height, int canvas_width, int offset_x,
                         int offset_y, Matrix2Df &warped_out,
                         std::vector<uint8_t> *valid_mask_out = nullptr,
                         bool *has_data_out = nullptr,
                         cv::cuda::Stream *stream = nullptr) const;

  bool warp_affine_rgb_frame(
      Matrix2Df img_r, Matrix2Df img_g, Matrix2Df img_b,
      const WarpMatrix &warp, int canvas_height, int canvas_width,
      int offset_x, int offset_y,
      Matrix2Df &warped_r_out, Matrix2Df &warped_g_out, Matrix2Df &warped_b_out,
      std::vector<uint8_t> *valid_mask_out = nullptr,
      bool *has_data_out = nullptr,
      cv::cuda::Stream *stream = nullptr) const;

private:
  AccelerationSelection selection_;
  std::string prewarp_interpolation_;
};

} // namespace tile_compile::core
