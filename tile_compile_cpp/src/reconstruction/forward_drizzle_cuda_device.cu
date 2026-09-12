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

#include <chrono>
#include <cfloat>
#include <cstdlib>
#include <limits>

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

__global__ void k_affine_target_gather(
    double a00, double a01, double a02, double a10, double a11, double a12,
    double i00, double i01, double i02, double i10, double i11, double i12,
    int internal_scale, double half, int tx0, int ty0, int cols, int rows,
    int source_w, int source_h, const float *source, int bayer, int ox, int oy,
    bool mono, double *out_a, double *out_b,
    unsigned long long *source_candidates,
    unsigned long long *positive_overlaps) {
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = cols * rows;
  if (local >= n) return;
  const int lx = local % cols;
  const int ly = local / cols;
  const int tx = tx0 + lx;
  const int ty = ty0 + ly;
  const double sc = static_cast<double>(internal_scale);

  double minx = DBL_MAX, maxx = -DBL_MAX;
  double miny = DBL_MAX, maxy = -DBL_MAX;
  for (int cy = 0; cy < 2; ++cy) {
    for (int cx = 0; cx < 2; ++cx) {
      const double x = static_cast<double>(tx + cx) / sc;
      const double y = static_cast<double>(ty + cy) / sc;
      const double sx = i00 * x + i01 * y + i02;
      const double sy = i10 * x + i11 * y + i12;
      minx = fmin(minx, sx); maxx = fmax(maxx, sx);
      miny = fmin(miny, sy); maxy = fmax(maxy, sy);
    }
  }
  const int sx0 = max(0, static_cast<int>(floor(minx - half - 0.5)) - 1);
  const int sx1 = min(source_w,
                      static_cast<int>(ceil(maxx + half - 0.5)) + 2);
  const int sy0 = max(0, static_cast<int>(floor(miny - half - 0.5)) - 1);
  const int sy1 = min(source_h,
                      static_cast<int>(ceil(maxy + half - 0.5)) + 2);

  double A[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};
  unsigned long long candidates = 0, overlaps = 0;
  for (int sy = sy0; sy < sy1; ++sy) {
    for (int sx = sx0; sx < sx1; ++sx) {
      ++candidates;
      const double value = static_cast<double>(source[sy * source_w + sx]);
      if (!isfinite(value)) continue;
      const double centre_x = static_cast<double>(sx) + 0.5;
      const double centre_y = static_cast<double>(sy) + 0.5;
      const double px[4] = {centre_x - half, centre_x + half,
                            centre_x + half, centre_x - half};
      const double py[4] = {centre_y - half, centre_y - half,
                            centre_y + half, centre_y + half};
      double qx[4], qy[4];
      for (int k = 0; k < 4; ++k) {
        qx[k] = (a00 * px[k] + a01 * py[k] + a02) * sc;
        qy[k] = (a10 * px[k] + a11 * py[k] + a12) * sc;
      }
      const double area = d_polygon_rect_area(qx, qy, tx, ty, tx + 1.0,
                                               ty + 1.0);
      if (!(area > 0.0)) continue;
      ++overlaps;
      const int c = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
      A[c] += area * value;
      B[c] += area;
    }
  }
  const int channels = mono ? 1 : 3;
  for (int c = 0; c < channels; ++c) {
    out_a[static_cast<long long>(c) * n + local] = A[c];
    out_b[static_cast<long long>(c) * n + local] = B[c];
  }
  if (candidates) atomicAdd(source_candidates, candidates);
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
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
    int band_sx0, int band_sx1, int source_w, const float *src_band,
    int bayer, int ox, int oy, int mono,
    int max_cells, CudaDrizzleContribRecord *recs, long long cap,
    unsigned long long *count, int *overflow) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long band_rows = band_sy1 - band_sy0;
  const long long band_cols = band_sx1 - band_sx0;
  const long long total = band_rows * band_cols;
  if (tid >= total) return;
  const long long row_in_band = tid / band_cols;
  const int sy = band_sy0 + static_cast<int>(row_in_band);
  const int sx = band_sx0 + static_cast<int>(tid % band_cols);
  // `src_band` is the band-local buffer: row 0 == source row band_sy0,
  // column 0 == source col band_sx0 (T5 X+Y windowing).
  const double v = static_cast<double>(src_band[row_in_band * band_cols +
                                                (sx - band_sx0)]);
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

__global__ void k_affine_dense_scatter(
    double a0, double a1, double a2, double a3, double a4, double a5,
    double sc, double half, int tx_begin, int ty_begin, int cols, int rows,
    int source_w, int source_h, const float *source, int bayer, int ox, int oy,
    int mono, double *out_a, double *out_b,
    unsigned long long *positive_overlaps) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long source_n = static_cast<long long>(source_w) * source_h;
  if (tid >= source_n) return;
  const int sy = static_cast<int>(tid / source_w);
  const int sx = static_cast<int>(tid % source_w);
  const double value = static_cast<double>(source[tid]);
  if (!isfinite(value)) return;
  const int channel = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
  const double x = sx + 0.5, y = sy + 0.5;
  const double px[4] = {x - half, x + half, x + half, x - half};
  const double py[4] = {y - half, y - half, y + half, y + half};
  double qx[4], qy[4];
  double minx = DBL_MAX, maxx = -DBL_MAX;
  double miny = DBL_MAX, maxy = -DBL_MAX;
  for (int k = 0; k < 4; ++k) {
    qx[k] = (a0 * px[k] + a1 * py[k] + a2) * sc;
    qy[k] = (a3 * px[k] + a4 * py[k] + a5) * sc;
    minx = fmin(minx, qx[k]); maxx = fmax(maxx, qx[k]);
    miny = fmin(miny, qy[k]); maxy = fmax(maxy, qy[k]);
  }
  const int x0 = max(tx_begin, static_cast<int>(floor(minx)));
  const int x1 = min(tx_begin + cols, static_cast<int>(ceil(maxx)));
  const int y0 = max(ty_begin, static_cast<int>(floor(miny)));
  const int y1 = min(ty_begin + rows, static_cast<int>(ceil(maxy)));
  unsigned long long overlaps = 0;
  const long long plane_n = static_cast<long long>(cols) * rows;
  for (int ty = y0; ty < y1; ++ty) {
    for (int tx = x0; tx < x1; ++tx) {
      const double area =
          d_polygon_rect_area(qx, qy, tx, ty, tx + 1.0, ty + 1.0);
      if (!(area > 0.0)) continue;
      const long long out = static_cast<long long>(channel) * plane_n +
                            static_cast<long long>(ty - ty_begin) * cols +
                            (tx - tx_begin);
      atomicAdd(out_a + out, area * value);
      atomicAdd(out_b + out, area);
      ++overlaps;
    }
  }
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
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
    int rows, int canvas_w_internal, int band_sy0, int band_sy1,
    int band_sx0, int band_sx1, int source_w,
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
  band_sx0 = band_sx0 < 0 ? 0 : band_sx0;
  band_sx1 = band_sx1 > source_w ? source_w : band_sx1;
  *out_written = 0;
  if (band_sy1 <= band_sy0 || band_sx1 <= band_sx0 || rows <= 0)
    return true;  // empty band, no records

  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;

  const long long band_rows = band_sy1 - band_sy0;
  const long long band_cols = band_sx1 - band_sx0;
  const long long total_threads = band_rows * band_cols;
  const long long grid_ll = (total_threads + 127) / 128;
  if (grid_ll > 2000000000LL) return false;  // absurd band; caller uses CPU

  float *d_src = nullptr;
  CudaDrizzleContribRecord *d_recs = nullptr;
  unsigned long long *d_count = nullptr;
  int *d_overflow = nullptr;
  // `source_values` is the band-local buffer (row 0 == source row band_sy0,
  // column 0 == source col band_sx0), sized band_rows * band_cols (T5 X+Y
  // windowing). The caller never copies the whole image.
  const size_t src_bytes =
      static_cast<size_t>(band_rows) * static_cast<size_t>(band_cols) *
      sizeof(float);
  const size_t rec_bytes =
      static_cast<size_t>(records_capacity) * sizeof(CudaDrizzleContribRecord);
  // §30.81 step-5 baseline: coarse device-phase timers, gated by
  // TC_FD_CUDA_PROFILE. cudaDeviceSynchronize() below makes the kernel window a
  // real wall measurement; malloc/upload/download are around blocking calls.
  const bool cprof = forward_drizzle_cuda_profile_enabled();
  using cclock = std::chrono::steady_clock;
  auto cnow = [] { return cclock::now(); };
  auto cadd = [&](std::atomic<double> &slot, cclock::time_point t0) {
    if (cprof)
      forward_drizzle_cuda_profile_add(
          slot, std::chrono::duration<double>(cnow() - t0).count());
  };
  auto t_phase = cnow();
  bool ok = cudaMalloc(&d_src, src_bytes) == cudaSuccess &&
            cudaMalloc(&d_recs, rec_bytes) == cudaSuccess &&
            cudaMalloc(&d_count, sizeof(unsigned long long)) == cudaSuccess &&
            cudaMalloc(&d_overflow, sizeof(int)) == cudaSuccess;
  cadd(forward_drizzle_cuda_profile().dev_malloc_s, t_phase);
  if (ok) {
    t_phase = cnow();
    ok = cudaMemcpy(d_src, source_values, src_bytes, cudaMemcpyHostToDevice) ==
             cudaSuccess &&
         cudaMemset(d_count, 0, sizeof(unsigned long long)) == cudaSuccess &&
         cudaMemset(d_overflow, 0, sizeof(int)) == cudaSuccess;
    cadd(forward_drizzle_cuda_profile().dev_upload_s, t_phase);
  }
  if (ok) {
    t_phase = cnow();
    k_affine_frame_contribs<<<static_cast<unsigned>(grid_ll), 128>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, y_begin, rows,
        canvas_w_internal, band_sy0, band_sy1, band_sx0, band_sx1, source_w,
        d_src, bayer_pattern, cfa_origin_x, cfa_origin_y, mono ? 1 : 0,
        max_cells_per_pixel, d_recs, records_capacity, d_count, d_overflow);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
    cadd(forward_drizzle_cuda_profile().dev_kernel_s, t_phase);
  }
  int overflow = 0;
  unsigned long long count = 0;
  t_phase = cnow();
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
  cadd(forward_drizzle_cuda_profile().dev_download_s, t_phase);
  cudaFree(d_src);
  cudaFree(d_recs);
  cudaFree(d_count);
  cudaFree(d_overflow);
  return ok;
}

bool forward_drizzle_cuda_affine_target_gather(
    const double affine6[6], const double inverse6[6], int internal_scale,
    double half, int target_x_begin, int target_y_begin, int target_cols,
    int target_rows, int source_w, int source_h, const float *source_values,
    int bayer_pattern, int cfa_origin_x, int cfa_origin_y, bool mono,
    double *out_a, double *out_b, unsigned long long *out_source_candidates,
    unsigned long long *out_positive_overlaps) {
  if (!affine6 || !inverse6 || !source_values || !out_a || !out_b ||
      internal_scale <= 0 || target_cols <= 0 || target_rows <= 0 ||
      source_w <= 0 || source_h <= 0)
    return false;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;
  const int channels = mono ? 1 : 3;
  const long long cells = static_cast<long long>(target_cols) * target_rows;
  if (cells <= 0 || cells > 2000000000LL) return false;
  const std::size_t src_bytes = static_cast<std::size_t>(source_w) * source_h *
                                sizeof(float);
  const std::size_t plane_bytes = static_cast<std::size_t>(channels) * cells *
                                  sizeof(double);
  float *d_source = nullptr;
  double *d_a = nullptr, *d_b = nullptr;
  unsigned long long *d_candidates = nullptr, *d_overlaps = nullptr;
  bool ok = cudaMalloc(&d_source, src_bytes) == cudaSuccess &&
            cudaMalloc(&d_a, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_b, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_candidates, sizeof(unsigned long long)) ==
                cudaSuccess &&
            cudaMalloc(&d_overlaps, sizeof(unsigned long long)) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_source, source_values, src_bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemset(d_candidates, 0, sizeof(unsigned long long)) ==
             cudaSuccess &&
         cudaMemset(d_overlaps, 0, sizeof(unsigned long long)) == cudaSuccess;
  if (ok) {
    const int block = 128;
    const int grid = (static_cast<int>(cells) + block - 1) / block;
    k_affine_target_gather<<<grid, block>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        inverse6[0], inverse6[1], inverse6[2], inverse6[3], inverse6[4],
        inverse6[5], internal_scale, half, target_x_begin, target_y_begin,
        target_cols, target_rows, source_w, source_h, d_source, bayer_pattern,
        cfa_origin_x, cfa_origin_y, mono, d_a, d_b, d_candidates, d_overlaps);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  unsigned long long candidates = 0, overlaps = 0;
  if (ok)
    ok = cudaMemcpy(out_a, d_a, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(out_b, d_b, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(&candidates, d_candidates, sizeof(candidates),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(&overlaps, d_overlaps, sizeof(overlaps),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_source);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_candidates);
  cudaFree(d_overlaps);
  if (ok) {
    if (out_source_candidates) *out_source_candidates = candidates;
    if (out_positive_overlaps) *out_positive_overlaps = overlaps;
  }
  return ok;
}

bool forward_drizzle_cuda_affine_dense_scatter(
    const double affine6[6], int internal_scale, double half,
    int target_x_begin, int target_y_begin, int target_cols, int target_rows,
    int source_w, int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, double *out_a,
    double *out_b, unsigned long long *out_positive_overlaps) {
  if (!affine6 || !source_values || !out_a || !out_b || internal_scale <= 0 ||
      target_cols <= 0 || target_rows <= 0 || source_w <= 0 || source_h <= 0)
    return false;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;
  const int channels = mono ? 1 : 3;
  const long long source_n = static_cast<long long>(source_w) * source_h;
  const long long target_n = static_cast<long long>(target_cols) * target_rows;
  if (source_n <= 0 || target_n <= 0 || source_n > 2000000000LL ||
      target_n > 2000000000LL)
    return false;
  const std::size_t src_bytes = static_cast<std::size_t>(source_n) * sizeof(float);
  const std::size_t plane_bytes =
      static_cast<std::size_t>(channels) * target_n * sizeof(double);
  float *d_source = nullptr;
  double *d_a = nullptr, *d_b = nullptr;
  unsigned long long *d_overlaps = nullptr;
  bool ok = cudaMalloc(&d_source, src_bytes) == cudaSuccess &&
            cudaMalloc(&d_a, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_b, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_overlaps, sizeof(unsigned long long)) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_source, source_values, src_bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemset(d_a, 0, plane_bytes) == cudaSuccess &&
         cudaMemset(d_b, 0, plane_bytes) == cudaSuccess &&
         cudaMemset(d_overlaps, 0, sizeof(unsigned long long)) == cudaSuccess;
  if (ok) {
    const int block = 128;
    const int grid = (static_cast<int>(source_n) + block - 1) / block;
    k_affine_dense_scatter<<<grid, block>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, target_x_begin,
        target_y_begin, target_cols, target_rows, source_w, source_h, d_source,
        bayer_pattern, cfa_origin_x, cfa_origin_y, mono ? 1 : 0, d_a, d_b,
        d_overlaps);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  unsigned long long overlaps = 0;
  if (ok)
    ok = cudaMemcpy(out_a, d_a, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(out_b, d_b, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(&overlaps, d_overlaps, sizeof(overlaps),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_source);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_overlaps);
  if (ok && out_positive_overlaps) *out_positive_overlaps = overlaps;
  return ok;
}

ForwardDrizzleV2CudaWorkspace::ForwardDrizzleV2CudaWorkspace() = default;

ForwardDrizzleV2CudaWorkspace::~ForwardDrizzleV2CudaWorkspace() {
  cudaFree(device_source_);
  cudaFree(device_a_);
  cudaFree(device_b_);
  cudaFree(device_overlaps_);
}

bool ForwardDrizzleV2CudaWorkspace::reserve(std::size_t source_elements,
                                            std::size_t target_plane_elements,
                                            int channels) {
  if (source_elements == 0 || target_plane_elements == 0 ||
      (channels != 1 && channels != 3))
    return false;
  if (source_elements <= source_capacity_ &&
      target_plane_elements <= plane_capacity_ &&
      channels <= channel_capacity_)
    return true;

  // Fail closed on any std::size_t overflow before a single byte is
  // allocated: an undersized buffer would silently corrupt the kernel's
  // dense-plane writes.
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (source_elements > kMax / sizeof(float) ||
      target_plane_elements >
          kMax / static_cast<std::size_t>(channels))
    return false;
  const std::size_t source_bytes = source_elements * sizeof(float);
  const std::size_t plane_values = target_plane_elements *
                                   static_cast<std::size_t>(channels);
  if (plane_values > kMax / sizeof(double)) return false;
  const std::size_t plane_bytes = plane_values * sizeof(double);
  if (plane_bytes > (kMax - sizeof(unsigned long long)) / 2 ||
      source_bytes > kMax - 2 * plane_bytes - sizeof(unsigned long long))
    return false;
  const std::size_t reserved_bytes = source_bytes + 2 * plane_bytes +
                                     sizeof(unsigned long long);

  void *new_source = nullptr, *new_a = nullptr, *new_b = nullptr,
       *new_overlaps = nullptr;
  bool ok = cudaMalloc(&new_source, source_bytes) == cudaSuccess &&
            cudaMalloc(&new_a, plane_bytes) == cudaSuccess &&
            cudaMalloc(&new_b, plane_bytes) == cudaSuccess &&
            cudaMalloc(&new_overlaps, sizeof(unsigned long long)) == cudaSuccess;
  if (!ok) {
    cudaFree(new_source);
    cudaFree(new_a);
    cudaFree(new_b);
    cudaFree(new_overlaps);
    cudaGetLastError();
    return false;
  }
  cudaFree(device_source_);
  cudaFree(device_a_);
  cudaFree(device_b_);
  cudaFree(device_overlaps_);
  device_source_ = new_source;
  device_a_ = new_a;
  device_b_ = new_b;
  device_overlaps_ = new_overlaps;
  source_capacity_ = source_elements;
  plane_capacity_ = target_plane_elements;
  channel_capacity_ = channels;
  ++stats_.allocations;
  stats_.reserved_device_bytes = reserved_bytes;
  return true;
}

bool ForwardDrizzleV2CudaWorkspace::run_dense_scatter(
    const double affine6[6], int internal_scale, double half,
    int target_x_begin, int target_y_begin, int target_cols, int target_rows,
    int source_w, int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, double *out_a,
    double *out_b) {
  if (!affine6 || !source_values || !out_a || !out_b || internal_scale <= 0 ||
      target_cols <= 0 || target_rows <= 0 || source_w <= 0 || source_h <= 0)
    return false;
  const int channels = mono ? 1 : 3;
  const std::size_t source_n = static_cast<std::size_t>(source_w) * source_h;
  const std::size_t target_n =
      static_cast<std::size_t>(target_cols) * target_rows;
  if (source_n > source_capacity_ || target_n > plane_capacity_ ||
      channels > channel_capacity_ || source_n > 2000000000ULL)
    return false;
  const std::size_t src_bytes = source_n * sizeof(float);
  const std::size_t plane_bytes =
      static_cast<std::size_t>(channels) * target_n * sizeof(double);
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  bool ok = cudaMemcpy(device_source_, source_values, src_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemset(device_a_, 0, plane_bytes) == cudaSuccess &&
            cudaMemset(device_b_, 0, plane_bytes) == cudaSuccess &&
            cudaMemset(device_overlaps_, 0, sizeof(unsigned long long)) ==
                cudaSuccess;
  auto t1 = clock::now();
  if (ok) {
    const int block = 128;
    const int grid = (static_cast<int>(source_n) + block - 1) / block;
    k_affine_dense_scatter<<<grid, block>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, target_x_begin,
        target_y_begin, target_cols, target_rows, source_w, source_h,
        static_cast<const float *>(device_source_), bayer_pattern, cfa_origin_x,
        cfa_origin_y, mono ? 1 : 0, static_cast<double *>(device_a_),
        static_cast<double *>(device_b_),
        static_cast<unsigned long long *>(device_overlaps_));
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  auto t2 = clock::now();
  unsigned long long overlaps = 0;
  if (ok)
    ok = cudaMemcpy(out_a, device_a_, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(out_b, device_b_, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(&overlaps, device_overlaps_, sizeof(overlaps),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  auto t3 = clock::now();
  if (!ok) return false;
  ++stats_.calls;
  stats_.source_bytes_uploaded += src_bytes;
  stats_.result_bytes_downloaded += 2 * plane_bytes;
  stats_.positive_overlaps += overlaps;
  stats_.upload_seconds += std::chrono::duration<double>(t1 - t0).count();
  stats_.kernel_seconds += std::chrono::duration<double>(t2 - t1).count();
  stats_.download_seconds += std::chrono::duration<double>(t3 - t2).count();
  return true;
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
