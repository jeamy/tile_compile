// CFA-forward-drizzle CUDA device probe --- milestone M7 slice 2, plan 19.4.
//
// This translation unit is compiled ONLY when the build has a CUDA toolchain
// (TILE_COMPILE_WITH_CUDA). It owns the functions that must talk to the CUDA
// runtime: the device-memory query that feeds plan_cuda_chunking(), and the
// (still conservative) runtime-availability flag.
//
// `forward_drizzle_cuda_runtime_available()` stays FALSE until the slice-2
// droplet / clipping / profile kernels and their parity matrix (plan 19.2/19.5)
// are in place: returning true here would make persist_forward_drizzle_multiband
// attempt the CUDA path, hit the "not implemented" ForwardDrizzleCudaError, and
// pay a full CPU restart on every run. The device-memory probe is safe to make
// real now (it only queries) and is what the auto-chunk planner will consume.

#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#if TILE_COMPILE_WITH_CUDA

#include <cuda_runtime.h>

#include <cstdlib>

namespace tile_compile::reconstruction {

namespace {

// --- 1:1 device port of the CPU polygon_rectangle_intersection_area math -----
// (src/reconstruction/forward_drizzle.cpp). Convex quad, clipped Sutherland-
// Hodgman against the 4 axis-aligned half-planes of [rx0,rx1] x [ry0,ry1],
// then shoelace. Static bound 8 = 4 quad vertices + <=1 new vertex per edge
// per plane, matching the CPU comment.

__device__ double d_shoelace_area(const double *x, const double *y, int n) {
  if (n < 3) return 0.0;
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1) % n;
    s += x[i] * y[j] - x[j] * y[i];
  }
  return fabs(s) * 0.5;
}

// plane: 0 => x>=c, 1 => x<=c, 2 => y>=c, 3 => y<=c
__device__ int d_clip_plane(const double *ix_, const double *iy_, int in_n,
                            int plane, double c, double *ox, double *oy) {
  if (in_n == 0) return 0;
  int out_n = 0;
  for (int i = 0; i < in_n; ++i) {
    const int j = (i + 1) % in_n;
    const double xi = ix_[i], yi = iy_[i], xj = ix_[j], yj = iy_[j];
    bool in_i, in_j;
    switch (plane) {
      case 0:  in_i = xi >= c; in_j = xj >= c; break;
      case 1:  in_i = xi <= c; in_j = xj <= c; break;
      case 2:  in_i = yi >= c; in_j = yj >= c; break;
      default: in_i = yi <= c; in_j = yj <= c; break;
    }
    if (in_i) { ox[out_n] = xi; oy[out_n] = yi; ++out_n; }
    if (in_i != in_j) {
      double px, py;
      if (plane <= 1) {  // vertical plane at x == c
        const double t = (c - xi) / (xj - xi);
        px = c;
        py = yi + t * (yj - yi);
      } else {           // horizontal plane at y == c
        const double t = (c - yi) / (yj - yi);
        px = xi + t * (xj - xi);
        py = c;
      }
      ox[out_n] = px; oy[out_n] = py; ++out_n;
    }
  }
  return out_n;
}

__device__ double d_polygon_rect_area(const double *qx, const double *qy,
                                      double rx0, double ry0, double rx1,
                                      double ry1) {
  double bx[8], by[8], cx[8], cy[8];
  int n = 4;
  for (int i = 0; i < 4; ++i) { bx[i] = qx[i]; by[i] = qy[i]; }
  n = d_clip_plane(bx, by, n, 0, rx0, cx, cy);
  for (int i = 0; i < n; ++i) { bx[i] = cx[i]; by[i] = cy[i]; }
  n = d_clip_plane(bx, by, n, 1, rx1, cx, cy);
  for (int i = 0; i < n; ++i) { bx[i] = cx[i]; by[i] = cy[i]; }
  n = d_clip_plane(bx, by, n, 2, ry0, cx, cy);
  for (int i = 0; i < n; ++i) { bx[i] = cx[i]; by[i] = cy[i]; }
  n = d_clip_plane(bx, by, n, 3, ry1, cx, cy);
  return d_shoelace_area(cx, cy, n);
}

__global__ void k_polygon_rect_area_batch(const double *quad_xy,
                                          const double *rect, int n,
                                          double *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double *q = quad_xy + static_cast<long long>(i) * 8;
  const double qx[4] = {q[0], q[2], q[4], q[6]};
  const double qy[4] = {q[1], q[3], q[5], q[7]};
  const double *r = rect + static_cast<long long>(i) * 4;
  out[i] = d_polygon_rect_area(qx, qy, r[0], r[1], r[2], r[3]);
}

// 1:1 with build_affine_leaf() + to_internal(): corner order
// (-h,-h) (+h,-h) (+h,+h) (-h,+h), affine then *internal_scale.
__global__ void k_affine_leaf_corners_batch(double a00, double a01, double a02,
                                            double a10, double a11, double a12,
                                            double sc, double half,
                                            const double *sample_xy, int n,
                                            double *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double sx = sample_xy[2 * i], sy = sample_xy[2 * i + 1];
  const double csx[4] = {sx - half, sx + half, sx + half, sx - half};
  const double csy[4] = {sy - half, sy - half, sy + half, sy + half};
  double *o = out + static_cast<long long>(i) * 8;
  for (int k = 0; k < 4; ++k) {
    const double qx = a00 * csx[k] + a01 * csy[k] + a02;
    const double qy = a10 * csx[k] + a11 * csy[k] + a12;
    o[2 * k] = qx * sc;
    o[2 * k + 1] = qy * sc;
  }
}

struct CudaScopedError {
  ~CudaScopedError() { cudaGetLastError(); }
};

// --- 1:1 device port of core/types.hpp cfa_channel_for_source_pixel ---------
// BayerPattern enum order: UNKNOWN=0, RGGB=1, BGGR=2, GRBG=3, GBRG=4.
// Returns 0=R, 1=G, 2=B to match forward_drizzle.cpp's channel mapping.
__device__ int d_cfa_channel(int sx, int sy, int bayer, int ox, int oy) {
  int rr, rc, br, bc;
  switch (bayer) {
    case 2:  rr = 1; rc = 1; br = 0; bc = 0; break;  // BGGR
    case 3:  rr = 0; rc = 1; br = 1; bc = 0; break;  // GRBG
    case 4:  rr = 1; rc = 0; br = 0; bc = 1; break;  // GBRG
    default: rr = 0; rc = 0; br = 1; bc = 1; break;  // RGGB / UNKNOWN
  }
  const int px = (sx + ox) & 1;
  const int py = (sy + oy) & 1;
  if (py == rr && px == rc) return 0;  // R
  if (py == br && px == bc) return 2;  // B
  return 1;                            // G
}

__device__ double d_clampd(double v, double lo, double hi) {
  return fmin(fmax(v, lo), hi);
}

// 1:1 with build_affine_leaf + the rasterize_drizzle_stripe bbox/area loop.
// One thread per source pixel of the band. Contributions are appended at a
// dense atomic offset --- the ORDER is arbitrary, which is fine: the host sorts
// by the (unique) canonical key afterwards, so the reduction is deterministic
// regardless of append order (plan 19.6). Areas are bit-identical to the CPU
// because this TU is compiled --fmad=false and the CPU path -ffp-contract=off.
__global__ void k_affine_frame_contribs(
    double a0, double a1, double a2, double a3, double a4, double a5, double sc,
    double half, int y_begin, int rows, int W, int band_sy0, int band_sy1,
    int source_w, const float *src_band, int bayer, int ox, int oy, int mono,
    int max_cells, CudaDrizzleContribRecord *recs, long long cap,
    unsigned long long *count, int *overflow) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long band_rows = band_sy1 - band_sy0;
  const long long total = band_rows * source_w;
  if (tid >= total) return;
  const long long row_in_band = tid / source_w;
  const int sy = band_sy0 + static_cast<int>(row_in_band);
  const int sx = static_cast<int>(tid % source_w);
  // `src_band` starts at source row band_sy0 (band-local buffer, not full image).
  const double v = static_cast<double>(src_band[row_in_band * source_w + sx]);
  if (!isfinite(v)) return;
  const int ch = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);

  const double x = sx + 0.5, y = sy + 0.5;
  const double csx[4] = {x - half, x + half, x + half, x - half};
  const double csy[4] = {y - half, y - half, y + half, y + half};
  double lx[4], ly[4];
  double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
  for (int i = 0; i < 4; ++i) {
    const double qx = a0 * csx[i] + a1 * csy[i] + a2;
    const double qy = a3 * csx[i] + a4 * csy[i] + a5;
    lx[i] = qx * sc;
    ly[i] = qy * sc;
    xmin = fmin(xmin, lx[i]); xmax = fmax(xmax, lx[i]);
    ymin = fmin(ymin, ly[i]); ymax = fmax(ymax, ly[i]);
  }
  const int x0 = static_cast<int>(d_clampd(floor(xmin), 0.0, (double)W));
  const int x1 = static_cast<int>(d_clampd(ceil(xmax), 0.0, (double)W));
  const int y0 = static_cast<int>(
      d_clampd(floor(ymin), (double)y_begin, (double)(y_begin + rows)));
  const int y1 = static_cast<int>(
      d_clampd(ceil(ymax), (double)y_begin, (double)(y_begin + rows)));

  int emitted = 0;
  for (int yy = y0; yy < y1; ++yy)
    for (int xx = x0; xx < x1; ++xx) {
      const double k =
          d_polygon_rect_area(lx, ly, (double)xx, (double)yy, xx + 1.0, yy + 1.0);
      if (k > 0.0) {
        if (emitted >= max_cells) { atomicExch(overflow, 1); return; }
        const unsigned long long idx = atomicAdd(count, 1ULL);
        if (idx >= static_cast<unsigned long long>(cap)) {
          atomicExch(overflow, 1);
          return;
        }
        CudaDrizzleContribRecord r;
        r.channel = static_cast<unsigned>(ch);
        r.target_y = static_cast<unsigned>(yy - y_begin);
        r.target_x = static_cast<unsigned>(xx);
        r.source_y = static_cast<unsigned>(sy);
        r.source_x = static_cast<unsigned>(sx);
        r.area = k;
        r.value = v;
        recs[idx] = r;
        ++emitted;
      }
    }
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
  CudaScopedError clear_on_exit;

  double *d_quad = nullptr, *d_rect = nullptr, *d_out = nullptr;
  const size_t nb_q = static_cast<size_t>(n) * 8 * sizeof(double);
  const size_t nb_r = static_cast<size_t>(n) * 4 * sizeof(double);
  const size_t nb_o = static_cast<size_t>(n) * sizeof(double);
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

bool forward_drizzle_cuda_affine_leaf_corners_batch(const double affine6[6],
                                                    int internal_scale,
                                                    double half,
                                                    const double *sample_xy,
                                                    int n, double *out_corners) {
  if (n <= 0 || affine6 == nullptr || sample_xy == nullptr ||
      out_corners == nullptr || internal_scale <= 0)
    return false;
  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;

  double *d_in = nullptr, *d_out = nullptr;
  const size_t nb_i = static_cast<size_t>(n) * 2 * sizeof(double);
  const size_t nb_o = static_cast<size_t>(n) * 8 * sizeof(double);
  bool ok = cudaMalloc(&d_in, nb_i) == cudaSuccess &&
            cudaMalloc(&d_out, nb_o) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_in, sample_xy, nb_i, cudaMemcpyHostToDevice) ==
         cudaSuccess;
  if (ok) {
    const int block = 128;
    const int grid = (n + block - 1) / block;
    k_affine_leaf_corners_batch<<<grid, block>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, d_in, n, d_out);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  if (ok)
    ok = cudaMemcpy(out_corners, d_out, nb_o, cudaMemcpyDeviceToHost) ==
         cudaSuccess;
  cudaFree(d_in);
  cudaFree(d_out);
  return ok;
}

bool forward_drizzle_cuda_affine_frame_contributions(
    const double affine6[6], int internal_scale, double half, int y_begin,
    int rows, int canvas_w_internal, int band_sy0, int band_sy1, int source_w,
    int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, int max_cells_per_pixel,
    CudaDrizzleContribRecord *records_out, long long records_capacity,
    long long *out_written) {
  if (!affine6 || !source_values || !records_out || !out_written ||
      internal_scale <= 0 || source_w <= 0 || source_h <= 0 ||
      max_cells_per_pixel <= 0 || records_capacity <= 0)
    return false;
  band_sy0 = band_sy0 < 0 ? 0 : band_sy0;
  band_sy1 = band_sy1 > source_h ? source_h : band_sy1;
  *out_written = 0;
  if (band_sy1 <= band_sy0 || rows <= 0) return true;  // empty band, no records

  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;

  const long long band_rows = band_sy1 - band_sy0;
  const long long total_threads = band_rows * source_w;
  const long long grid_ll = (total_threads + 127) / 128;
  if (grid_ll > 2000000000LL) return false;  // absurd band; caller uses CPU

  float *d_src = nullptr;
  CudaDrizzleContribRecord *d_recs = nullptr;
  unsigned long long *d_count = nullptr;
  int *d_overflow = nullptr;
  // `source_values` is already the band-local buffer (row 0 == source row
  // band_sy0), sized band_rows * source_w --- the caller never copies the whole
  // image.
  const size_t src_bytes =
      static_cast<size_t>(band_rows) * source_w * sizeof(float);
  const size_t rec_bytes =
      static_cast<size_t>(records_capacity) * sizeof(CudaDrizzleContribRecord);
  bool ok = cudaMalloc(&d_src, src_bytes) == cudaSuccess &&
            cudaMalloc(&d_recs, rec_bytes) == cudaSuccess &&
            cudaMalloc(&d_count, sizeof(unsigned long long)) == cudaSuccess &&
            cudaMalloc(&d_overflow, sizeof(int)) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_src, source_values, src_bytes, cudaMemcpyHostToDevice) ==
             cudaSuccess &&
         cudaMemset(d_count, 0, sizeof(unsigned long long)) == cudaSuccess &&
         cudaMemset(d_overflow, 0, sizeof(int)) == cudaSuccess;
  if (ok) {
    k_affine_frame_contribs<<<static_cast<unsigned>(grid_ll), 128>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, y_begin, rows,
        canvas_w_internal, band_sy0, band_sy1, source_w, d_src, bayer_pattern,
        cfa_origin_x, cfa_origin_y, mono ? 1 : 0, max_cells_per_pixel, d_recs,
        records_capacity, d_count, d_overflow);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  int overflow = 0;
  unsigned long long count = 0;
  if (ok)
    ok = cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(&count, d_count, sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  if (ok && overflow == 0 &&
      count <= static_cast<unsigned long long>(records_capacity)) {
    ok = cudaMemcpy(records_out, d_recs,
                    static_cast<size_t>(count) * sizeof(CudaDrizzleContribRecord),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) *out_written = static_cast<long long>(count);
  } else {
    ok = false;  // overflow or capacity exceeded -> caller falls back to CPU
  }
  cudaFree(d_src);
  cudaFree(d_recs);
  cudaFree(d_count);
  cudaFree(d_overflow);
  return ok;
}

CudaDeviceMemory forward_drizzle_cuda_device_memory() {
  CudaDeviceMemory m;
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
    cudaGetLastError();  // clear the sticky error
    return m;
  }
  std::size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
    cudaGetLastError();
    return m;
  }
  m.free_bytes = free_b;
  m.total_bytes = total_b;
  return m;
}

bool forward_drizzle_cuda_runtime_available() {
  // Plan 19.4/19.6: the affine device path is wired into
  // persist_forward_drizzle_multiband (§30.54) and store-level byte-identical
  // to the CPU reference. Enable it whenever a usable CUDA device is present;
  // persist_forward_drizzle_multiband still gates each attempt on
  // affine-only + not mode 2/1 and falls back to the CPU reference otherwise.
  return forward_drizzle_cuda_device_memory().free_bytes > 0;
}

}  // namespace tile_compile::reconstruction

#endif  // TILE_COMPILE_WITH_CUDA
