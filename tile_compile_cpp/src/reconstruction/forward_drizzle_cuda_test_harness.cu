// CFA forward drizzle v2 --- CUDA parity harness TU.
//
// Test-only device entry points that are not part of the production hot
// path live here so the production TU (forward_drizzle_cuda_device.cu)
// carries only the kernels the driver can reach. The harness exercises the
// SAME device code as production through forward_drizzle_cuda_clip.cuh (a
// kernel defined in another TU cannot be launched from here without -rdc,
// which is why the local-warp dense-scatter harness, whose kernel the
// production driver can also reach, stays in the production TU).
// Compiled with --fmad=false like the production TU (CMakeLists.txt).

#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#if TILE_COMPILE_WITH_CUDA

#include <cuda_runtime.h>

#include "forward_drizzle_cuda_clip.cuh"

#include <cstddef>

namespace tile_compile::reconstruction {

namespace {

__global__ void k_polygon_rect_area_batch(const double *quad_xy,
                                          const double *rect, int n,
                                          double *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double *q = quad_xy + static_cast<long long>(i) * 8;
  const double qx[4] = {q[0], q[2], q[4], q[6]};
  const double qy[4] = {q[1], q[3], q[5], q[7]};
  const double *r = rect + static_cast<long long>(i) * 4;
  out[i] = fd_cuda_clip::d_polygon_rect_area(qx, qy, r[0], r[1], r[2], r[3]);
}

}  // namespace

bool forward_drizzle_cuda_polygon_rect_area_batch(const double *quad_xy,
                                                  const double *rect, int n,
                                                  double *out_area) {
  if (n <= 0 || quad_xy == nullptr || rect == nullptr || out_area == nullptr)
    return false;
  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) {
    cudaGetLastError();
    return false;
  }
  struct ClearOnExit {
    ~ClearOnExit() { cudaGetLastError(); }
  } clear_on_exit;

  double *d_quad = nullptr, *d_rect = nullptr, *d_out = nullptr;
  const std::size_t nb_q = static_cast<std::size_t>(n) * 8 * sizeof(double);
  const std::size_t nb_r = static_cast<std::size_t>(n) * 4 * sizeof(double);
  const std::size_t nb_o = static_cast<std::size_t>(n) * sizeof(double);
  bool ok = cudaMalloc(&d_quad, nb_q) == cudaSuccess &&
            cudaMalloc(&d_rect, nb_r) == cudaSuccess &&
            cudaMalloc(&d_out, nb_o) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_quad, quad_xy, nb_q, cudaMemcpyHostToDevice) ==
             cudaSuccess &&
         cudaMemcpy(d_rect, rect, nb_r, cudaMemcpyHostToDevice) == cudaSuccess;
  if (ok) {
    const int block = 128;
    const int grid = (n + block - 1) / block;
    k_polygon_rect_area_batch<<<grid, block>>>(d_quad, d_rect, n, d_out);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  if (ok)
    ok = cudaMemcpy(out_area, d_out, nb_o, cudaMemcpyDeviceToHost) ==
         cudaSuccess;
  cudaFree(d_quad);
  cudaFree(d_rect);
  cudaFree(d_out);
  return ok;
}

}  // namespace tile_compile::reconstruction

#endif  // TILE_COMPILE_WITH_CUDA
