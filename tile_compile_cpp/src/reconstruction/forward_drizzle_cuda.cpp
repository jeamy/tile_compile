#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

namespace tile_compile::reconstruction {

namespace {

// -1 = disabled. Initialised once from the environment, then overridable in
// process by set_forward_drizzle_cuda_fault_after_chunks() (tests).
std::atomic<int> g_fault_after_chunks{-2};  // -2 = "not yet read from env"

int read_env_fault() {
  const char *v = std::getenv("TILE_COMPILE_FORWARD_DRIZZLE_CUDA_FAULT_AFTER_CHUNKS");
  if (v == nullptr || *v == '\0') return -1;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n < 0 || n > 1'000'000) return -1;
  return static_cast<int>(n);
}

}  // namespace

// §30.81 step-5 baseline instrumentation (see the header). Process-global,
// gated by TC_FD_CUDA_PROFILE; a normal run never reads or writes it.
ForwardDrizzleCudaProfile &forward_drizzle_cuda_profile() {
  static ForwardDrizzleCudaProfile p;
  return p;
}
bool forward_drizzle_cuda_profile_enabled() {
  static const bool on = [] {
    const char *v = std::getenv("TC_FD_CUDA_PROFILE");
    return v != nullptr && *v != '\0';
  }();
  return on;
}

#if !TILE_COMPILE_WITH_CUDA
// CUDA-free build: no device at all. The CUDA build defines these in
// forward_drizzle_cuda_device.cu against the real runtime.
bool forward_drizzle_cuda_runtime_available() { return false; }
CudaDeviceMemory forward_drizzle_cuda_device_memory() { return {}; }
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *, const double *,
                                                  int, double *) {
  return false;
}
bool forward_drizzle_cuda_affine_leaf_corners_batch(const double *, int, double,
                                                    const double *, int,
                                                    double *) {
  return false;
}
bool forward_drizzle_cuda_affine_frame_contributions(
    const double *, int, double, int, int, int, int, int, int, int, int, int,
    const float *, int, int, int, bool, int, CudaDrizzleContribRecord *,
    long long, long long *out_written) {
  if (out_written) *out_written = 0;
  return false;
}
bool forward_drizzle_cuda_affine_target_gather(
    const double *, const double *, int, double, int, int, int, int, int, int,
    const float *, int, int, int, bool, double *, double *,
    unsigned long long *, unsigned long long *) {
  return false;
}
bool forward_drizzle_cuda_affine_coverage_gather(
    const double *, const double *, int, double, int, int, int, int, int, int,
    int, int, int, bool, double *) {
  return false;
}
bool forward_drizzle_cuda_affine_dense_scatter(
    const double *, int, double, int, int, int, int, int, int, const float *,
    int, int, int, bool, double *, double *, unsigned long long *) {
  return false;
}
bool forward_drizzle_cuda_local_dense_scatter(
    const double *, const ForwardDrizzleV2LocalWarp &, int, double, int, int,
    int, int, const float *, int, int, int, bool, int, int, double *,
    double *, double *, unsigned long long *, unsigned long long *) {
  return false;
}
ForwardDrizzleV2CudaWorkspace::ForwardDrizzleV2CudaWorkspace() = default;
ForwardDrizzleV2CudaWorkspace::~ForwardDrizzleV2CudaWorkspace() = default;
bool ForwardDrizzleV2CudaWorkspace::reserve(std::size_t, std::size_t, int) {
  return false;
}
bool ForwardDrizzleV2CudaWorkspace::run_dense_scatter(
    const double *, int, double, int, int, int, int, int, int, const float *,
    int, int, int, bool, double *, double *) {
  return false;
}
ForwardDrizzleV2CudaPrototypeKernel::ForwardDrizzleV2CudaPrototypeKernel() =
    default;
ForwardDrizzleV2CudaPrototypeKernel::~ForwardDrizzleV2CudaPrototypeKernel() =
    default;
bool ForwardDrizzleV2CudaPrototypeKernel::reserve(
    int, int, int, int, const ForwardDrizzleV2KernelConfig &) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::begin_band(
    int, const ForwardDrizzleV2KernelConfig &) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame(
    const double *, const float *, const float *, std::uint64_t,
    const ForwardDrizzleV2FrameQuality *,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_local(
    const double *, const ForwardDrizzleV2LocalWarp &, const float *,
    const float *, std::uint64_t, const ForwardDrizzleV2FrameQuality *,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_window(
    const double *, const ForwardDrizzleV2SourceWindow &, const float *,
    const float *, const ForwardDrizzleV2Sigma2FrameModel *, std::uint64_t,
    const ForwardDrizzleV2FrameQuality *,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_local_window(
    const double *, const ForwardDrizzleV2LocalWarp &,
    const ForwardDrizzleV2SourceWindow &, const float *, const float *,
    const ForwardDrizzleV2Sigma2FrameModel *, std::uint64_t,
    const ForwardDrizzleV2FrameQuality *,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_cached_leaves(
    const double *, const ForwardDrizzleV2SourceWindow &, const float *,
    const float *, const ForwardDrizzleV2Sigma2FrameModel *,
    const ForwardDrizzleV2CachedLeaf *, std::size_t, std::uint64_t,
    std::uint64_t, const ForwardDrizzleV2FrameQuality *,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_affine_samples(
    const double *, const ForwardDrizzleV2SourceSample *, std::size_t, bool,
    const ForwardDrizzleV2AlignedQuality *, std::uint64_t,
    const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::begin_affine_frame(
    std::uint64_t, const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_affine_piece(
    const double *, int, int, const ForwardDrizzleV2SourceWindow &,
    const float *, const float *, const ForwardDrizzleV2Sigma2FrameModel *,
    const ForwardDrizzleV2FrameQuality *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::finish_affine_frame(
    std::uint64_t) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::skip_frame(
    std::uint64_t, const ForwardDrizzleV2FrameMeta *) {
  return false;
}
bool ForwardDrizzleV2CudaPrototypeKernel::end_pilot() { return false; }
bool ForwardDrizzleV2CudaPrototypeKernel::finalize(
    ForwardDrizzleV2PixelResult *, ForwardDrizzleV2ProfileResult *,
    std::uint64_t *) {
  return false;
}
#endif

void set_forward_drizzle_cuda_fault_after_chunks(int n) {
  g_fault_after_chunks.store(n < 0 ? -1 : n, std::memory_order_relaxed);
}

int forward_drizzle_cuda_fault_after_chunks() {
  int cur = g_fault_after_chunks.load(std::memory_order_relaxed);
  if (cur == -2) {
    const int from_env = read_env_fault();
    // Only the first reader installs the env value; a concurrent setter wins.
    int expected = -2;
    if (g_fault_after_chunks.compare_exchange_strong(expected, from_env,
                                                     std::memory_order_relaxed))
      return from_env;
    cur = g_fault_after_chunks.load(std::memory_order_relaxed);
  }
  return cur;
}

}  // namespace tile_compile::reconstruction
