#pragma once

#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tile_compile::core {

enum class AccelerationBackend {
  cpu = 0,
  opencv_cuda,
  opencv_opencl,
  cuda,
};

enum class AccelerationPhase {
  prewarp = 0,
  tile_reconstruction,
  stacking,
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

  const AccelerationSelection &selection() const { return selection_; }

  bool warp_affine_frame(Matrix2Df img, const WarpMatrix &warp, ColorMode mode,
                         int canvas_height, int canvas_width, int offset_x,
                         int offset_y, Matrix2Df &warped_out,
                         std::vector<uint8_t> *valid_mask_out = nullptr,
                         bool *has_data_out = nullptr) const;

  reconstruction::WeightedTileResult sigma_clip_reduce(
      const std::vector<Matrix2Df> &tiles, const std::vector<float> &weights,
      float sigma_low, float sigma_high, int max_iters, float min_fraction,
      float eps_weight) const;

  Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df> &frames,
                             float sigma_low, float sigma_high, int max_iters,
                             float min_fraction) const;

  void overlap_add(const Matrix2Df &tile, const Tile &tile_bounds,
                   const std::vector<float> &hann_x,
                   const std::vector<float> &hann_y,
                   const std::vector<uint8_t> &common_valid_mask,
                   int canvas_width, Matrix2Df &accum, Matrix2Df &weight_sum,
                   bool accumulate_weight = true) const;

  bool normalize_overlap_accum(Matrix2Df &accum, Matrix2Df &weight_sum,
                               float eps_weight,
                               float invalid_value) const;
  void flush_overlap_state(Matrix2Df &accum, Matrix2Df &weight_sum) const;

private:
  struct OverlapAddState;
  AccelerationSelection selection_;
  mutable std::unordered_map<const Matrix2Df *,
                             std::shared_ptr<OverlapAddState>>
      overlap_add_states_;
};

} // namespace tile_compile::core
