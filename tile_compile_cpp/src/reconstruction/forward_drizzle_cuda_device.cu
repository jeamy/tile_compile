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
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

#if TILE_COMPILE_WITH_CUDA

#include <cuda_runtime.h>

#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

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
  double ax[8], ay[8], bx[8], by[8];
  int n = 4;
  for (int i = 0; i < 4; ++i) {
    ax[i] = qx[i];
    ay[i] = qy[i];
  }
  n = d_clip_plane(ax, ay, n, 0, rx0, bx, by);
  n = d_clip_plane(bx, by, n, 1, rx1, ax, ay);
  n = d_clip_plane(ax, ay, n, 2, ry0, bx, by);
  n = d_clip_plane(bx, by, n, 3, ry1, ax, ay);
  return d_shoelace_area(ax, ay, n);
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

// --- Gate-6 prototype kernels ---------------------------------------------

// Bounded per-(pixel, channel) reservoir record. sigma2 must persist to fold
// time for the clip-accepted confidence contract (gate-6 role amendment).
struct V2ReservoirRecord {
  double x;
  double b;
  unsigned long long order;
  double sigma2;
};

// Hard device bound: slots = 2 * reservoir_size with reservoir_size <= 64.
constexpr int kV2MaxResSlots = 128;

__device__ unsigned long long d_splitmix64(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Dense scatter into the band-local internal planes. Unlike
// k_affine_dense_scatter the geometry weight is written for every positive
// overlap BEFORE the value check: a nonfinite source sample still owns its
// geometric support (footprint/B_geo), exactly like the CPU contract.
__global__ void k_scatter_v2(
    double a0, double a1, double a2, double a3, double a4, double a5,
    double sc, double half, int cols, int rows, int source_w, int source_h,
    const float *source, const float *sigma2, int bayer, int ox, int oy,
    int mono, double *fa, double *fbs, double *fbg, double *fs2,
    unsigned long long *positive_overlaps) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long source_n = static_cast<long long>(source_w) * source_h;
  if (tid >= source_n) return;
  const int sy = static_cast<int>(tid / source_w);
  const int sx = static_cast<int>(tid % source_w);
  const double value = static_cast<double>(source[tid]);
  const bool finite_value = isfinite(value);
  const double s2 = sigma2 != nullptr ? static_cast<double>(sigma2[tid]) : 0.0;
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
  const int x0 = max(0, static_cast<int>(floor(minx)));
  const int x1 = min(cols, static_cast<int>(ceil(maxx)));
  const int y0 = max(0, static_cast<int>(floor(miny)));
  const int y1 = min(rows, static_cast<int>(ceil(maxy)));
  unsigned long long overlaps = 0;
  const long long plane_n = static_cast<long long>(cols) * rows;
  for (int ty = y0; ty < y1; ++ty) {
    for (int tx = x0; tx < x1; ++tx) {
      const double area =
          d_polygon_rect_area(qx, qy, tx, ty, tx + 1.0, ty + 1.0);
      if (!(area > 0.0)) continue;
      const long long out = static_cast<long long>(channel) * plane_n +
                            static_cast<long long>(ty) * cols + tx;
      atomicAdd(fbg + out, area);
      if (finite_value) {
        atomicAdd(fa + out, area * value);
        atomicAdd(fbs + out, area);
        if (fs2 != nullptr) atomicAdd(fs2 + out, area * s2);
      }
      ++overlaps;
    }
  }
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
}

// One thread per NATIVE pixel of the band: folds the scale^2 frame-plane
// subpixels per channel into per-frame candidates and accumulates the
// full-stream statistics, coverage, support masks and the bounded hash
// reservoir. Same algebra as fold_native_pixel_v2 + the gate-3 stream loop.
// Support mask (u16, scale <= 2): bits 0..3 geometry, bits 4..7 source;
// estimator/profile layers equal the source layer on this minimal kernel and
// are derived at finalize.
__global__ void k_fold_accumulate_v2(
    const double *fa, const double *fbs, const double *fbg, const double *fs2,
    int icols, int irows, int scale, int ncols, int nrows, int channels,
    unsigned long long order, int keep_all, unsigned long long threshold,
    int res_slots, unsigned long long seed,
    double *accA, double *accB, double *accB2,
    double *covB, double *covB2, double *confS, double *confC,
    unsigned int *contrib, unsigned int *kept, unsigned int *footprint,
    unsigned short *supp, unsigned long long *degraded,
    V2ReservoirRecord *res) {
  const long long px =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (px >= nplane) return;
  const int nx = static_cast<int>(px % ncols);
  const int ny = static_cast<int>(px / ncols);
  const long long iplane = static_cast<long long>(icols) * irows;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  bool any_geo = false;
  for (int c = 0; c < channels; ++c) {
    const long long pc = static_cast<long long>(c) * nplane + px;
    double a = 0.0, b_src = 0.0, b_geo = 0.0, s2w = 0.0;
    unsigned int geo_bits = 0, src_bits = 0;
    for (int iy = 0; iy < scale; ++iy) {
      for (int ix = 0; ix < scale; ++ix) {
        const int j = iy * scale + ix;
        const long long ii =
            static_cast<long long>(ny * scale + iy) * icols + nx * scale + ix;
        const long long idx = static_cast<long long>(c) * iplane + ii;
        const double aj = fa[idx];
        const double bsj = fbs[idx];
        const double bgj = fbg[idx];
        a += inv_s2 * aj;
        b_src += inv_s2 * bsj;
        b_geo += inv_s2 * bgj;
        if (fs2 != nullptr) s2w += inv_s2 * fs2[idx];
        if (bgj > 0.0) geo_bits |= 1u << j;
        if (bsj > 0.0) src_bits |= 1u << j;
      }
    }
    supp[pc] |= static_cast<unsigned short>(geo_bits | (src_bits << 4));
    if (b_geo > 0.0) {
      covB[pc] += b_geo;
      covB2[pc] += b_geo * b_geo;
      any_geo = true;
    }
    if (!(b_src > 0.0)) continue;
    const double x = a / b_src;
    const double s2c = fs2 != nullptr ? s2w / b_src : 0.0;
    accA[pc] += a;
    accB[pc] += b_src;
    accB2[pc] += b_src * b_src;
    ++contrib[pc];
    // Gate-4 stream confidence: mirrors conf_sigma2_of. A present-but-invalid
    // sigma2 degrades; an absent plane contributes zero sigma silently.
    if (fs2 != nullptr && (!isfinite(s2c) || s2c < 0.0)) {
      ++degraded[pc];
    } else if (s2c > 0.0) {
      confS[pc] += b_src * sqrt(s2c);
      confC[pc] += b_src * b_src * s2c;
    }
    if (keep_all || d_splitmix64(order ^ seed) < threshold) {
      const unsigned int k = kept[pc]++;
      if (k < static_cast<unsigned int>(res_slots)) {
        V2ReservoirRecord r;
        r.x = x;
        r.b = b_src;
        r.order = order;
        r.sigma2 = s2c;
        res[pc * res_slots + k] = r;
      }
    }
  }
  if (any_geo) ++footprint[px];
}

// Band-end finalize: per (pixel, channel) runs the bit-exact gate-3 clip on
// the reservoir (identical evaluation order to robust_frame_oracle_v2:
// (x, order) sort, cumulative-weight median at >= total/2, (|x-med|, order)
// MAD order, asymmetric bounds, early stop on an unchanged mask) and maps the
// gate-4 confidence states. kept > res_slots is the deterministic overflow
// fallback: uniform stream value, support retained.
__global__ void k_finalize_v2(
    int ncols, int nrows, int channels, int subpixels, int res_slots,
    unsigned long long frames_processed, int min_candidates, int min_clip,
    int passes, double s_low, double s_high,
    const double *accA, const double *accB, const double *accB2,
    const double *confS, const double *confC,
    const unsigned int *contrib, const unsigned int *kept,
    const unsigned int *footprint, const unsigned short *supp,
    const unsigned long long *degraded, const V2ReservoirRecord *res,
    ForwardDrizzleV2PixelResult *out, unsigned long long *dense_overlap) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (tid >= nplane * channels) return;
  const long long pc = tid;
  const long long px = tid % nplane;
  const int c = static_cast<int>(tid / nplane);

  ForwardDrizzleV2PixelResult r;
  const double a_acc = accA[pc];
  const double b_acc = accB[pc];
  const double b2_acc = accB2[pc];
  r.contributors = contrib[pc];
  r.conf_degraded = degraded[pc];
  r.b = b_acc;
  r.n_eff = b2_acc > 0.0 ? b_acc * b_acc / b2_acc : 0.0;
  const unsigned int mask = supp[pc];
  r.geometry_fraction = static_cast<float>(
      __popc(mask & 0xFu) / static_cast<double>(subpixels));
  r.source_fraction = static_cast<float>(
      __popc((mask >> 4) & 0xFu) / static_cast<double>(subpixels));
  r.estimator_fraction = r.source_fraction;
  r.profile_fraction = r.source_fraction;

  if (c == 0 && footprint[px] == frames_processed && frames_processed > 0)
    atomicAdd(dense_overlap, 1ULL);

  const unsigned int n_kept = kept[pc];
  auto finish_conf = [&](double cb, double cs, double cc) {
    if (!(cb > 0.0)) {
      r.confidence = 0.0;
      r.confidence_state = static_cast<std::uint8_t>(
          ForwardDrizzleV2ConfidenceState::no_source_support);
      return;
    }
    if (cc > 0.0 && isfinite(cc)) {
      r.confidence = (cs * cs) / (cs * cs + cc);
      r.confidence_state =
          static_cast<std::uint8_t>(ForwardDrizzleV2ConfidenceState::modeled);
      return;
    }
    r.confidence = r.n_eff > 0.0 ? r.n_eff / (r.n_eff + 1.0) : 0.0;
    r.confidence_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2ConfidenceState::fallback_n_eff);
  };
  auto finish_uniform = [&](ForwardDrizzleV2RobustState state) {
    r.value = a_acc / b_acc;
    r.robust_state = static_cast<std::uint8_t>(state);
    finish_conf(b_acc, confS[pc], confC[pc]);
  };

  if (!(b_acc > 0.0)) {
    r.robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::no_source_support);
    finish_conf(0.0, 0.0, 0.0);
    out[pc] = r;
    return;
  }
  if (n_kept > static_cast<unsigned int>(res_slots)) {
    finish_uniform(ForwardDrizzleV2RobustState::reservoir_overflow_fallback);
    out[pc] = r;
    return;
  }
  if (r.contributors < static_cast<unsigned int>(min_candidates) ||
      n_kept < static_cast<unsigned int>(min_clip)) {
    finish_uniform(ForwardDrizzleV2RobustState::too_few_candidates_fallback);
    out[pc] = r;
    return;
  }

  // Load and insertion-sort the kept set by (x, order); identical ordering to
  // the CPU oracle.
  V2ReservoirRecord recs[kV2MaxResSlots];
  for (unsigned int i = 0; i < n_kept; ++i) recs[i] = res[pc * res_slots + i];
  for (unsigned int i = 1; i < n_kept; ++i) {
    const V2ReservoirRecord v = recs[i];
    unsigned int j = i;
    while (j > 0 &&
           (recs[j - 1].x > v.x ||
            (recs[j - 1].x == v.x && recs[j - 1].order > v.order))) {
      recs[j] = recs[j - 1];
      --j;
    }
    recs[j] = v;
  }
  unsigned long long acc_lo = ~0ULL, acc_hi = ~0ULL;  // accepted bitmask
  for (int pass = 0; pass < passes; ++pass) {
    double total_w = 0.0;
    unsigned int n_active = 0;
    for (unsigned int i = 0; i < n_kept; ++i) {
      const bool on = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
      if (on) {
        total_w += recs[i].b;
        ++n_active;
      }
    }
    if (n_active == 0) break;
    double median = 0.0;
    {
      double cum = 0.0;
      unsigned int last_on = 0;
      bool picked = false;
      for (unsigned int i = 0; i < n_kept; ++i) {
        const bool on =
            i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
        if (!on) continue;
        last_on = i;
        cum += recs[i].b;
        if (total_w > 0.0 && cum >= total_w / 2.0) {
          median = recs[i].x;
          picked = true;
          break;
        }
      }
      if (!picked) median = recs[last_on].x;
    }
    // Deviation order over the active set: (|x - median|, order).
    unsigned int ord[kV2MaxResSlots];
    unsigned int m = 0;
    for (unsigned int i = 0; i < n_kept; ++i) {
      const bool on = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
      if (on) ord[m++] = i;
    }
    for (unsigned int i = 1; i < m; ++i) {
      const unsigned int v = ord[i];
      const double dv = fabs(recs[v].x - median);
      unsigned int j = i;
      while (j > 0) {
        const unsigned int u = ord[j - 1];
        const double du = fabs(recs[u].x - median);
        if (du < dv || (du == dv && recs[u].order < recs[v].order)) break;
        ord[j] = u;
        --j;
      }
      ord[j] = v;
    }
    double mad = fabs(recs[ord[m - 1]].x - median);
    if (total_w > 0.0) {
      double cum = 0.0;
      for (unsigned int i = 0; i < m; ++i) {
        cum += recs[ord[i]].b;
        if (cum >= total_w / 2.0) {
          mad = fabs(recs[ord[i]].x - median);
          break;
        }
      }
    }
    const double lower = median - s_low * mad;
    const double upper = median + s_high * mad;
    bool changed = false;
    for (unsigned int i = 0; i < m; ++i) {
      const unsigned int idx = ord[i];
      const double x = recs[idx].x;
      if (!(x >= lower && x <= upper)) {
        if (idx < 64) acc_lo &= ~(1ULL << idx);
        else acc_hi &= ~(1ULL << (idx - 64));
        changed = true;
      }
    }
    if (!changed) break;
  }
  double ca = 0.0, cb = 0.0, cb2 = 0.0, cs = 0.0, cc = 0.0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    const bool on = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
    if (!on) continue;
    ca += recs[i].b * recs[i].x;
    cb += recs[i].b;
    cb2 += recs[i].b * recs[i].b;
    const double s2 = recs[i].sigma2;
    if (isfinite(s2) && s2 > 0.0) {
      cs += recs[i].b * sqrt(s2);
      cc += recs[i].b * recs[i].b * s2;
    }
  }
  r.value = cb > 0.0 ? ca / cb : 0.0;
  r.robust_state = static_cast<std::uint8_t>(
      ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
  finish_conf(cb, cs, cc);
  out[pc] = r;
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
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (!std::isfinite(half) || !(half > 0.0)) return false;
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
  // Reject non-finite transform/pixfrac parameters before the kernel:
  // floor(NaN) cast to int is undefined behaviour on device.
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (!std::isfinite(half) || !(half > 0.0)) return false;
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
  // No device-wide sync here: the stream only carries this workspace's own
  // operations, and destroying it implicitly waits for its queued work.
  cudaFree(device_source_);
  cudaFree(device_a_);
  cudaFree(device_b_);
  cudaFree(device_overlaps_);
  if (stream_ != nullptr)
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

bool ForwardDrizzleV2CudaWorkspace::reserve(std::size_t source_elements,
                                            std::size_t target_plane_elements,
                                            int channels) {
  if (source_elements == 0 || target_plane_elements == 0 ||
      (channels != 1 && channels != 3))
    return false;
  if (stream_ == nullptr) {
    // One non-blocking stream per workspace, created once before first use.
    // It is queue plumbing, not a hotpath buffer, so it is not counted in
    // stats_.allocations.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    stream_ = stream;
  }

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
  // Reject non-finite transform/pixfrac parameters before the kernel:
  // floor(NaN) cast to int is undefined behaviour on device.
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (!std::isfinite(half) || !(half > 0.0)) return false;
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
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  bool ok = cudaMemcpyAsync(device_source_, source_values, src_bytes,
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemsetAsync(device_a_, 0, plane_bytes, stream) ==
                cudaSuccess &&
            cudaMemsetAsync(device_b_, 0, plane_bytes, stream) ==
                cudaSuccess &&
            cudaMemsetAsync(device_overlaps_, 0, sizeof(unsigned long long),
                            stream) == cudaSuccess;
  ++stats_.stream_synchronizations;
  ok = cudaStreamSynchronize(stream) == cudaSuccess && ok;
  auto t1 = clock::now();
  if (ok) {
    const int block = 128;
    const int grid = (static_cast<int>(source_n) + block - 1) / block;
    k_affine_dense_scatter<<<grid, block, 0, stream>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(internal_scale), half, target_x_begin,
        target_y_begin, target_cols, target_rows, source_w, source_h,
        static_cast<const float *>(device_source_), bayer_pattern, cfa_origin_x,
        cfa_origin_y, mono ? 1 : 0, static_cast<double *>(device_a_),
        static_cast<double *>(device_b_),
        static_cast<unsigned long long *>(device_overlaps_));
    ok = cudaGetLastError() == cudaSuccess;
    ++stats_.stream_synchronizations;
    ok = cudaStreamSynchronize(stream) == cudaSuccess && ok;
  }
  auto t2 = clock::now();
  unsigned long long overlaps = 0;
  if (ok) {
    ok = cudaMemcpyAsync(out_a, device_a_, plane_bytes,
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
         cudaMemcpyAsync(out_b, device_b_, plane_bytes,
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
         cudaMemcpyAsync(&overlaps, device_overlaps_, sizeof(overlaps),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    ++stats_.stream_synchronizations;
    ok = cudaStreamSynchronize(stream) == cudaSuccess && ok;
  }
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

// Gate-2 test oracle: single-thread device port of the CPU
// fold_native_pixel_v2 algebra (src/reconstruction/forward_drizzle_v2.cpp).
// One thread owns the whole scalar fold so the per-frame and per-subpixel
// evaluation order is identical to the CPU reference. It proves the
// frame-before-square and four-support-layer arithmetic on the device; it is
// NOT the production batch kernel that Gate 6 selects later.
// Error codes mirror the CPU invalid_argument cases:
//   1 shape, 2 area, 3 frame order, 4 weight, 5 support order,
//   6 nonfinite numerator with positive denominator.
__global__ void k_fold_native_pixel_v2(
    const ForwardDrizzleV2FrameSubpixel *entries, unsigned long long frame_count,
    const double *area, unsigned long long subpixels,
    ForwardDrizzleV2FoldResult *out, int *error) {
  if (frame_count == 0 || subpixels == 0) { *error = 1; return; }
  double total_area = 0.0;
  for (unsigned long long j = 0; j < subpixels; ++j) {
    const double a = area[j];
    if (!isfinite(a) || a < 0.0) { *error = 2; return; }
    total_area += a;
  }
  if (!(total_area > 0.0)) { *error = 2; return; }

  ForwardDrizzleV2FoldResult r;
  for (unsigned long long f = 0; f < frame_count; ++f) {
    double af = 0.0;
    double bf = 0.0;
    for (unsigned long long j = 0; j < subpixels; ++j) {
      const ForwardDrizzleV2FrameSubpixel &v =
          entries[f * subpixels + j];
      if (v.frame_order != f) { *error = 3; return; }
      const bool bad_weight =
          !isfinite(v.geometry_b) || v.geometry_b < 0.0 ||
          !isfinite(v.source_b) || v.source_b < 0.0 ||
          !isfinite(v.estimator_b) || v.estimator_b < 0.0 ||
          !isfinite(v.b) || v.b < 0.0;
      if (bad_weight) { *error = 4; return; }
      if ((v.source_b > 0.0 && !(v.geometry_b > 0.0)) ||
          (v.estimator_b > 0.0 && !(v.source_b > 0.0)) ||
          (v.b > 0.0 && !(v.estimator_b > 0.0))) {
        *error = 5;
        return;
      }
      if (v.b > 0.0 && !isfinite(v.a)) { *error = 6; return; }
      if (!(v.b > 0.0)) continue;
      af += area[j] * v.a;
      bf += area[j] * v.b;
    }
    if (bf > 0.0) {
      r.a += af;
      r.b += bf;
      r.b2 += bf * bf;
    }
  }

  // Per-internal-subpixel support: count area[j] when at least one frame
  // carries the respective positive denominator there.
  double geometry_area = 0.0, source_area = 0.0, estimator_area = 0.0,
         profile_area = 0.0;
  for (unsigned long long j = 0; j < subpixels; ++j) {
    bool geo = false, src = false, est = false, prof = false;
    for (unsigned long long f = 0; f < frame_count; ++f) {
      const ForwardDrizzleV2FrameSubpixel &v =
          entries[f * subpixels + j];
      geo |= v.geometry_b > 0.0;
      src |= v.source_b > 0.0;
      est |= v.estimator_b > 0.0;
      prof |= v.b > 0.0;
    }
    if (geo) geometry_area += area[j];
    if (src) source_area += area[j];
    if (est) estimator_area += area[j];
    if (prof) profile_area += area[j];
  }
  r.geometry_area_fraction = fmin(fmax(geometry_area / total_area, 0.0), 1.0);
  r.source_area_fraction = fmin(fmax(source_area / total_area, 0.0), 1.0);
  r.estimator_area_fraction =
      fmin(fmax(estimator_area / total_area, 0.0), 1.0);
  r.profile_area_fraction = fmin(fmax(profile_area / total_area, 0.0), 1.0);
  r.geometry_support = r.geometry_area_fraction > 0.0;
  r.source_support = r.source_area_fraction > 0.0;
  r.estimator_support = r.estimator_area_fraction > 0.0;
  r.profile_support = r.b > 0.0 && isfinite(r.a) && isfinite(r.b) &&
                      isfinite(r.b2);
  if (r.profile_support) {
    r.value = r.a / r.b;
    r.n_eff = r.b2 > 0.0 ? r.b * r.b / r.b2 : 0.0;
  }
  *out = r;
}

bool fold_native_pixel_v2_cuda(
    std::span<const ForwardDrizzleV2FrameSubpixel> entries,
    std::size_t frame_count, std::span<const double> area,
    ForwardDrizzleV2FoldResult &out) {
  // Host-side shape guard: same shape contract as the CPU reference; any
  // violation (or overflow while sizing) leaves `out` untouched.
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (frame_count == 0 || area.empty() ||
      frame_count > kMax / area.size() ||
      entries.size() != frame_count * area.size() ||
      entries.size() > kMax / sizeof(ForwardDrizzleV2FrameSubpixel) ||
      area.size() > kMax / sizeof(double))
    return false;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;

  const std::size_t entries_bytes =
      entries.size() * sizeof(ForwardDrizzleV2FrameSubpixel);
  const std::size_t area_bytes = area.size() * sizeof(double);
  ForwardDrizzleV2FrameSubpixel *d_entries = nullptr;
  double *d_area = nullptr;
  ForwardDrizzleV2FoldResult *d_out = nullptr;
  int *d_error = nullptr;
  bool ok =
      cudaMalloc(&d_entries, entries_bytes) == cudaSuccess &&
      cudaMalloc(&d_area, area_bytes) == cudaSuccess &&
      cudaMalloc(&d_out, sizeof(ForwardDrizzleV2FoldResult)) == cudaSuccess &&
      cudaMalloc(&d_error, sizeof(int)) == cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_entries, entries.data(), entries_bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(d_area, area.data(), area_bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemset(d_error, 0, sizeof(int)) == cudaSuccess;
  if (ok) {
    k_fold_native_pixel_v2<<<1, 1>>>(
        d_entries, static_cast<unsigned long long>(frame_count), d_area,
        static_cast<unsigned long long>(area.size()), d_out, d_error);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  int error = 1;
  ForwardDrizzleV2FoldResult result;
  if (ok)
    ok = cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
  if (ok && error == 0)
    ok = cudaMemcpy(&result, d_out, sizeof(result),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_entries);
  cudaFree(d_area);
  cudaFree(d_out);
  cudaFree(d_error);
  if (ok && error == 0) {
    out = result;
    return true;
  }
  return false;
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

// --- Gate-6 prototype workspace --------------------------------------------

struct ForwardDrizzleV2CudaPrototypeKernel::Impl {
  cudaStream_t stream = nullptr;
  float *src = nullptr;
  float *s2 = nullptr;
  double *fa = nullptr;
  double *fbs = nullptr;
  double *fbg = nullptr;
  double *fs2 = nullptr;
  double *accA = nullptr;
  double *accB = nullptr;
  double *accB2 = nullptr;
  double *covB = nullptr;
  double *covB2 = nullptr;
  double *confS = nullptr;
  double *confC = nullptr;
  unsigned int *contrib = nullptr;
  unsigned int *kept = nullptr;
  unsigned int *footprint = nullptr;
  unsigned short *supp = nullptr;
  unsigned long long *degraded = nullptr;
  unsigned long long *scalars = nullptr;  // [0] overlaps, [1] dense overlap
  V2ReservoirRecord *res = nullptr;
  ForwardDrizzleV2PixelResult *out = nullptr;
  std::vector<cudaEvent_t> ev_up0, ev_up1, ev_k0, ev_k1;
  int ncols = 0, nrows = 0, icols = 0, irows = 0;
  int source_w = 0, source_h = 0, channels = 0, res_slots = 0;
  std::size_t nplane = 0, iplane = 0;
  ForwardDrizzleV2KernelConfig cfg;
  std::uint64_t frames = 0;
  bool finalized = false;

  ~Impl() {
    cudaFree(src);
    cudaFree(s2);
    cudaFree(fa);
    cudaFree(fbs);
    cudaFree(fbg);
    cudaFree(fs2);
    cudaFree(accA);
    cudaFree(accB);
    cudaFree(accB2);
    cudaFree(covB);
    cudaFree(covB2);
    cudaFree(confS);
    cudaFree(confC);
    cudaFree(contrib);
    cudaFree(kept);
    cudaFree(footprint);
    cudaFree(supp);
    cudaFree(degraded);
    cudaFree(scalars);
    cudaFree(res);
    cudaFree(out);
    for (auto *v : {&ev_up0, &ev_up1, &ev_k0, &ev_k1})
      for (cudaEvent_t e : *v) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
  }
};

namespace {

bool proto_checked_mul(std::size_t a, std::size_t b, std::size_t &out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

}  // namespace

ForwardDrizzleV2CudaPrototypeKernel::ForwardDrizzleV2CudaPrototypeKernel() =
    default;

ForwardDrizzleV2CudaPrototypeKernel::~ForwardDrizzleV2CudaPrototypeKernel() {
  delete impl_;
}

bool ForwardDrizzleV2CudaPrototypeKernel::reserve(
    int native_cols, int native_rows, int source_w, int source_h,
    const ForwardDrizzleV2KernelConfig &cfg) {
  if (impl_ != nullptr) return false;
  // Support-mask layout holds 2 layers x scale^2 bits in a u16 -> scale <= 2.
  // Local clip arrays bound reservoir_size to 64 (2*64 = kV2MaxResSlots).
  if (native_cols <= 0 || native_rows <= 0 || source_w <= 0 ||
      source_h <= 0 || cfg.internal_scale < 1 || cfg.internal_scale > 2 ||
      cfg.reservoir_size < 1 || cfg.reservoir_size > 64 ||
      cfg.stream_length == 0 || cfg.stream_length > 65536 ||
      cfg.min_clip_contributors < 1 || cfg.min_candidates < 1 ||
      cfg.robust_passes < 1 || !std::isfinite(cfg.sigma_low) ||
      !std::isfinite(cfg.sigma_high) || cfg.sigma_low <= 0.0 ||
      cfg.sigma_high <= 0.0 || !std::isfinite(cfg.half) ||
      !(cfg.half > 0.0) || cfg.bayer_pattern < 0 || cfg.bayer_pattern > 4)
    return false;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    cudaGetLastError();
    return false;
  }
  CudaScopedError clear_on_exit;

  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  const int scale = cfg.internal_scale;
  const long long icols = static_cast<long long>(native_cols) * scale;
  const long long irows = static_cast<long long>(native_rows) * scale;
  if (icols > 0x7fffffffLL || irows > 0x7fffffffLL) return false;
  const int channels = cfg.mono ? 1 : 3;
  const int res_slots = 2 * cfg.reservoir_size;
  std::size_t nplane = 0, iplane = 0, source_elems = 0, pc_elems = 0,
              frame_plane_elems = 0, total_bytes = 0, tmp = 0;
  if (!proto_checked_mul(static_cast<std::size_t>(native_cols),
                         static_cast<std::size_t>(native_rows), nplane) ||
      !proto_checked_mul(static_cast<std::size_t>(icols),
                         static_cast<std::size_t>(irows), iplane) ||
      !proto_checked_mul(static_cast<std::size_t>(source_w),
                         static_cast<std::size_t>(source_h), source_elems) ||
      !proto_checked_mul(nplane, static_cast<std::size_t>(channels),
                         pc_elems) ||
      !proto_checked_mul(iplane, static_cast<std::size_t>(channels),
                         frame_plane_elems) ||
      pc_elems > kMax / static_cast<std::size_t>(res_slots))
    return false;

  auto add_bytes = [&](std::size_t v) {
    if (v > kMax - total_bytes) { total_bytes = kMax; return false; }
    total_bytes += v;
    return true;
  };
  auto mul = [](std::size_t a, std::size_t b, std::size_t &o) {
    return proto_checked_mul(a, b, o);
  };
  const std::size_t frame_plane_count = cfg.sigma2_plane ? 4 : 3;
  if (!mul(source_elems, sizeof(float), tmp) || !add_bytes(tmp) ||
      (cfg.sigma2_plane &&
       (!mul(source_elems, sizeof(float), tmp) || !add_bytes(tmp))) ||
      !mul(frame_plane_elems, sizeof(double) * frame_plane_count, tmp) ||
      !add_bytes(tmp) ||
      !mul(pc_elems, sizeof(double) * 7, tmp) ||  // accA/B/B2 covB/2 confS/C
      !add_bytes(tmp) ||
      !mul(pc_elems, sizeof(unsigned int) * 2, tmp) ||  // contrib, kept
      !add_bytes(tmp) ||
      !mul(nplane, sizeof(unsigned int), tmp) || !add_bytes(tmp) ||
      !mul(pc_elems, sizeof(unsigned short), tmp) || !add_bytes(tmp) ||
      !mul(pc_elems, sizeof(unsigned long long), tmp) || !add_bytes(tmp) ||
      !mul(pc_elems * static_cast<std::size_t>(res_slots),
           sizeof(V2ReservoirRecord), tmp) ||
      !add_bytes(tmp) ||
      !mul(pc_elems, sizeof(ForwardDrizzleV2PixelResult), tmp) ||
      !add_bytes(tmp) ||
      !add_bytes(2 * sizeof(unsigned long long)))
    return false;

  Impl *im = new (std::nothrow) Impl();
  if (im == nullptr) return false;
  im->cfg = cfg;
  im->ncols = native_cols;
  im->nrows = native_rows;
  im->icols = static_cast<int>(icols);
  im->irows = static_cast<int>(irows);
  im->source_w = source_w;
  im->source_h = source_h;
  im->channels = channels;
  im->res_slots = res_slots;
  im->nplane = nplane;
  im->iplane = iplane;
  if (cudaStreamCreateWithFlags(&im->stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    cudaGetLastError();
    delete im;
    return false;
  }
  im->ev_up0.resize(cfg.stream_length);
  im->ev_up1.resize(cfg.stream_length);
  im->ev_k0.resize(cfg.stream_length);
  im->ev_k1.resize(cfg.stream_length);
  bool ok = true;
  for (auto *v : {&im->ev_up0, &im->ev_up1, &im->ev_k0, &im->ev_k1})
    for (cudaEvent_t &e : *v)
      ok = ok && cudaEventCreateWithFlags(&e, cudaEventDefault) == cudaSuccess;
  ok = ok &&
       cudaMalloc(&im->src, source_elems * sizeof(float)) == cudaSuccess &&
       (!cfg.sigma2_plane ||
        cudaMalloc(&im->s2, source_elems * sizeof(float)) == cudaSuccess) &&
       cudaMalloc(&im->fa, frame_plane_elems * sizeof(double)) ==
           cudaSuccess &&
       cudaMalloc(&im->fbs, frame_plane_elems * sizeof(double)) ==
           cudaSuccess &&
       cudaMalloc(&im->fbg, frame_plane_elems * sizeof(double)) ==
           cudaSuccess &&
       (!cfg.sigma2_plane ||
        cudaMalloc(&im->fs2, frame_plane_elems * sizeof(double)) ==
            cudaSuccess) &&
       cudaMalloc(&im->accA, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->accB, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->accB2, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->covB, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->covB2, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->confS, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->confC, pc_elems * sizeof(double)) == cudaSuccess &&
       cudaMalloc(&im->contrib, pc_elems * sizeof(unsigned int)) ==
           cudaSuccess &&
       cudaMalloc(&im->kept, pc_elems * sizeof(unsigned int)) == cudaSuccess &&
       cudaMalloc(&im->footprint, nplane * sizeof(unsigned int)) ==
           cudaSuccess &&
       cudaMalloc(&im->supp, pc_elems * sizeof(unsigned short)) ==
           cudaSuccess &&
       cudaMalloc(&im->degraded, pc_elems * sizeof(unsigned long long)) ==
           cudaSuccess &&
       cudaMalloc(&im->scalars, 2 * sizeof(unsigned long long)) ==
           cudaSuccess &&
       cudaMalloc(&im->res, pc_elems * static_cast<std::size_t>(res_slots) *
                              sizeof(V2ReservoirRecord)) == cudaSuccess &&
       cudaMalloc(&im->out, pc_elems * sizeof(ForwardDrizzleV2PixelResult)) ==
           cudaSuccess;
  if (!ok) {
    delete im;
    cudaGetLastError();
    return false;
  }
  // Queue zero-init on the workspace stream; ordered before any kernel.
  ok = cudaMemsetAsync(im->accA, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->accB, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->accB2, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->covB, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->covB2, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->confS, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->confC, 0, pc_elems * sizeof(double), im->stream) ==
           cudaSuccess &&
       cudaMemsetAsync(im->contrib, 0, pc_elems * sizeof(unsigned int),
                       im->stream) == cudaSuccess &&
       cudaMemsetAsync(im->kept, 0, pc_elems * sizeof(unsigned int),
                       im->stream) == cudaSuccess &&
       cudaMemsetAsync(im->footprint, 0, nplane * sizeof(unsigned int),
                       im->stream) == cudaSuccess &&
       cudaMemsetAsync(im->supp, 0, pc_elems * sizeof(unsigned short),
                       im->stream) == cudaSuccess &&
       cudaMemsetAsync(im->degraded, 0, pc_elems * sizeof(unsigned long long),
                       im->stream) == cudaSuccess &&
       cudaMemsetAsync(im->scalars, 0, 2 * sizeof(unsigned long long),
                       im->stream) == cudaSuccess;
  if (!ok) {
    delete im;
    cudaGetLastError();
    return false;
  }
  impl_ = im;
  stats_.allocations = 1;
  stats_.reserved_device_bytes = total_bytes;
  // Per-native-pixel footprint for the gate-5-plan comparison: everything
  // except the fixed source slot and scalars, divided by native pixels.
  bytes_per_native_pixel_ =
      (total_bytes - source_elems * sizeof(float) * (cfg.sigma2_plane ? 2 : 1) -
       2 * sizeof(unsigned long long)) /
      nplane;
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame(
    const double affine6[6], const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order) {
  if (impl_ == nullptr || affine6 == nullptr || source == nullptr ||
      impl_->finalized || impl_->frames >= impl_->cfg.stream_length)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  const std::size_t src_bytes =
      static_cast<std::size_t>(im.source_w) * im.source_h * sizeof(float);
  const std::size_t plane_bytes = im.iplane * sizeof(double) *
                                  static_cast<std::size_t>(im.channels);
  const std::uint64_t f = impl_->frames;
  const std::uint64_t n = im.cfg.stream_length;
  const bool keep_all =
      n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
  const std::uint64_t threshold =
      keep_all ? 0
               : static_cast<std::uint64_t>(
                     (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                      << 64) /
                     n);
  bool ok = cudaEventRecord(im.ev_up0[f], st) == cudaSuccess &&
            cudaMemcpyAsync(im.src, source, src_bytes, cudaMemcpyHostToDevice,
                            st) == cudaSuccess &&
            (!im.s2 || !sigma2_or_null ||
             cudaMemcpyAsync(im.s2, sigma2_or_null, src_bytes,
                             cudaMemcpyHostToDevice, st) == cudaSuccess) &&
            (!im.s2 || sigma2_or_null ||
             cudaMemsetAsync(im.s2, 0, src_bytes, st) == cudaSuccess) &&
            cudaEventRecord(im.ev_up1[f], st) == cudaSuccess &&
            cudaMemsetAsync(im.fa, 0, plane_bytes, st) == cudaSuccess &&
            cudaMemsetAsync(im.fbs, 0, plane_bytes, st) == cudaSuccess &&
            cudaMemsetAsync(im.fbg, 0, plane_bytes, st) == cudaSuccess &&
            (!im.fs2 ||
             cudaMemsetAsync(im.fs2, 0, plane_bytes, st) == cudaSuccess) &&
            cudaEventRecord(im.ev_k0[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  {
    const int block = 128;
    const long long source_n =
        static_cast<long long>(im.source_w) * im.source_h;
    const unsigned int grid =
        static_cast<unsigned int>((source_n + block - 1) / block);
    k_scatter_v2<<<grid, block, 0, st>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        static_cast<double>(im.cfg.internal_scale), im.cfg.half, im.icols,
        im.irows, im.source_w, im.source_h, im.src,
        im.fs2 ? im.s2 : nullptr, im.cfg.bayer_pattern, im.cfg.cfa_origin_x,
        im.cfg.cfa_origin_y, im.cfg.mono ? 1 : 0, im.fa, im.fbs, im.fbg,
        im.fs2, im.scalars);
  }
  {
    const int block = 128;
    const long long nplane = static_cast<long long>(im.nplane);
    const unsigned int grid =
        static_cast<unsigned int>((nplane + block - 1) / block);
    k_fold_accumulate_v2<<<grid, block, 0, st>>>(
        im.fa, im.fbs, im.fbg, im.fs2, im.icols, im.irows,
        im.cfg.internal_scale, im.ncols, im.nrows, im.channels, frame_order,
        keep_all ? 1 : 0, threshold, im.res_slots, im.cfg.reservoir_seed,
        im.accA, im.accB, im.accB2, im.covB, im.covB2, im.confS, im.confC,
        im.contrib, im.kept, im.footprint, im.supp, im.degraded, im.res);
  }
  ok = cudaGetLastError() == cudaSuccess &&
       cudaEventRecord(im.ev_k1[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  ++impl_->frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  stats_.source_bytes_uploaded += src_bytes * (sigma2_or_null ? 2 : 1);
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::finalize(
    ForwardDrizzleV2PixelResult *results, std::uint64_t *dense_overlap_count) {
  if (impl_ == nullptr || results == nullptr || impl_->finalized) return false;
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  const std::size_t n_pc = im.nplane * static_cast<std::size_t>(im.channels);
  const int block = 128;
  const long long total = static_cast<long long>(n_pc);
  const unsigned int grid =
      static_cast<unsigned int>((total + block - 1) / block);
  k_finalize_v2<<<grid, block, 0, st>>>(
      im.ncols, im.nrows, im.channels,
      im.cfg.internal_scale * im.cfg.internal_scale, im.res_slots, im.frames,
      im.cfg.min_candidates, im.cfg.min_clip_contributors,
      im.cfg.robust_passes, im.cfg.sigma_low, im.cfg.sigma_high, im.accA,
      im.accB, im.accB2, im.confS, im.confC, im.contrib, im.kept, im.footprint,
      im.supp, im.degraded, im.res, im.out, im.scalars + 1);
  if (cudaGetLastError() != cudaSuccess) return false;
  unsigned long long h_scalars[2] = {0, 0};
  std::vector<unsigned int> h_kept(n_pc), h_contrib(n_pc);
  const std::size_t out_bytes = n_pc * sizeof(ForwardDrizzleV2PixelResult);
  const std::size_t cnt_bytes = n_pc * sizeof(unsigned int);
  bool ok =
      cudaMemcpyAsync(results, im.out, out_bytes, cudaMemcpyDeviceToHost, st) ==
          cudaSuccess &&
      cudaMemcpyAsync(h_scalars, im.scalars, sizeof(h_scalars),
                      cudaMemcpyDeviceToHost, st) == cudaSuccess &&
      cudaMemcpyAsync(h_kept.data(), im.kept, cnt_bytes,
                      cudaMemcpyDeviceToHost, st) == cudaSuccess &&
      cudaMemcpyAsync(h_contrib.data(), im.contrib, cnt_bytes,
                      cudaMemcpyDeviceToHost, st) == cudaSuccess;
  ++stats_.stream_synchronizations;
  ok = cudaStreamSynchronize(st) == cudaSuccess && ok;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  im.finalized = true;
  for (std::uint64_t f = 0; f < im.frames; ++f) {
    float up_ms = 0.0f, k_ms = 0.0f;
    double frame_s = 0.0;
    if (cudaEventElapsedTime(&up_ms, im.ev_up0[f], im.ev_up1[f]) ==
        cudaSuccess) {
      stats_.upload_seconds += static_cast<double>(up_ms) * 1e-3;
      frame_s += static_cast<double>(up_ms) * 1e-3;
    }
    if (cudaEventElapsedTime(&k_ms, im.ev_k0[f], im.ev_k1[f]) == cudaSuccess) {
      stats_.kernel_seconds += static_cast<double>(k_ms) * 1e-3;
      frame_s += static_cast<double>(k_ms) * 1e-3;
    }
    stats_.max_frame_seconds = std::max(stats_.max_frame_seconds, frame_s);
  }
  stats_.positive_overlaps = h_scalars[0];
  if (dense_overlap_count) *dense_overlap_count = h_scalars[1];
  std::uint64_t kept_sum = 0, cand_sum = 0;
  for (std::size_t i = 0; i < n_pc; ++i) {
    kept_sum += h_kept[i];
    cand_sum += h_contrib[i];
  }
  stats_.reservoir_kept_total = kept_sum;
  stats_.candidates_streamed = cand_sum;
  stats_.result_bytes_downloaded += out_bytes + sizeof(h_scalars) +
                                    2 * cnt_bytes;
  return true;
}

}  // namespace tile_compile::reconstruction

#endif  // TILE_COMPILE_WITH_CUDA
