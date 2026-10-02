#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

namespace tile_compile::reconstruction {

#if !TILE_COMPILE_WITH_CUDA
// CUDA-free build: no device at all. The CUDA build defines these in
// forward_drizzle_cuda_device.cu against the real runtime.
bool forward_drizzle_cuda_runtime_available() { return false; }
CudaDeviceMemory forward_drizzle_cuda_device_memory() { return {}; }
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *, const double *,
                                                  int, double *) {
  return false;
}
bool forward_drizzle_cuda_affine_coverage_gather(
    const double *, const double *, int, double, int, int, int, int, int, int,
    int, int, int, bool, double *) {
  return false;
}
bool forward_drizzle_cuda_local_dense_scatter(
    const double *, const ForwardDrizzleV2LocalWarp &, int, double, int, int,
    int, int, const float *, int, int, int, bool, int, int, double *,
    double *, double *, unsigned long long *, unsigned long long *) {
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

}  // namespace tile_compile::reconstruction
