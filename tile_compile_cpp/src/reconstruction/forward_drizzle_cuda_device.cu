// CFA forward drizzle v2 --- CUDA device implementation.
//
// Compiled only with a CUDA toolchain (TILE_COMPILE_WITH_CUDA) and with
// --fmad=false so every fp64 expression rounds like the -ffp-contract=off CPU
// reference. Contents: the device probe, the SAMPLING_GEOMETRY coverage
// gather/scatter, the production v2 kernel workspace
// (ForwardDrizzleV2CudaPrototypeKernel: scatter, fold, finalize, shared frame
// rejection, pilot/full-frame estimator) and the local-warp scatter parity
// harness. The polygon clip itself is forward_drizzle_cuda_clip.cuh (shared
// with the clip harness TU forward_drizzle_cuda_test_harness.cu).

#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

#if TILE_COMPILE_WITH_CUDA

#include <cuda_runtime.h>

#include "forward_drizzle_cuda_clip.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

// Polygon clip + shoelace (one copy, shared with the parity harness TU).
using fd_cuda_clip::d_polygon_rect_area;
using fd_cuda_clip::d_shoelace_area;

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

// SAMPLING_GEOMETRY coverage gather: geometry only --- no source values, no
// A plane. One thread per target cell accumulates B[c] = sum of
// droplet overlap areas, scanning the conservative inverse-affine source
// neighbourhood in canonical (sy, sx) order --- the CPU rasterize's
// per-cell accumulation order, so the resulting B plane is bit-identical.
__global__ void k_affine_coverage_gather(
    double a00, double a01, double a02, double a10, double a11, double a12,
    double i00, double i01, double i02, double i10, double i11, double i12,
    int internal_scale, double half, int tx0, int ty0, int cols, int rows,
    int source_w, int source_h, int bayer, int ox, int oy, bool mono,
    double *out_b) {
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

  double B[3] = {0.0, 0.0, 0.0};
  for (int sy = sy0; sy < sy1; ++sy) {
    for (int sx = sx0; sx < sx1; ++sx) {
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
      const int c = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
      B[c] += area;
    }
  }
  const int channels = mono ? 1 : 3;
  for (int c = 0; c < channels; ++c)
    out_b[static_cast<long long>(c) * n + local] = B[c];
}

// Coverage scatter for the SAMPLING_GEOMETRY CFA pass, one parity class per
// launch. One thread per source sample with (sx&1)==par_x && (sy&1)==par_y
// inside the band; each ACCUMULATES its droplet overlap areas into the
// channel-major B plane. Within one parity class same-channel droplets are
// spaced 2 source pixels apart in both axes, so the caller's gate
// (2-2*half)*sigma_min(affine)*scale >= sqrt(2) guarantees every target cell
// receives at most ONE contribution per launch: the += is then race-free.
// Across the four launches the only multi-contributor cells are G cells hit
// by the two G parity classes (diagonal CFA neighbours); two addends sum
// commutatively, so the launch-order accumulation stays bit-identical to the
// CPU scan-order accumulate. MONO frames (up to 4 contributors per cell)
// must take the gather path --- pass mono=1 to skip the scatter there.
__global__ void k_affine_coverage_scatter(
    double a0, double a1, double a2, double a3, double a4, double a5,
    double sc, double half, int tx_begin, int ty_begin, int cols, int rows,
    int band_sy0, int band_sy1, int band_sx0, int band_sx1, int bayer, int ox,
    int oy, int par_x, int par_y, double *out_b) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int fx = band_sx0 + ((par_x - band_sx0) & 1);
  const int fy = band_sy0 + ((par_y - band_sy0) & 1);
  const long long pcols =
      fx < band_sx1 ? (band_sx1 - fx + 1) / 2 : 0;
  const long long prows =
      fy < band_sy1 ? (band_sy1 - fy + 1) / 2 : 0;
  const long long total = prows * pcols;
  if (tid >= total) return;
  const int sy = fy + 2 * static_cast<int>(tid / pcols);
  const int sx = fx + 2 * static_cast<int>(tid % pcols);
  const int ch = d_cfa_channel(sx, sy, bayer, ox, oy);

  const double x = sx + 0.5, y = sy + 0.5;
  const double px[4] = {x - half, x + half, x + half, x - half};
  const double py[4] = {y - half, y - half, y + half, y + half};
  double lx[4], ly[4];
  double xmin = DBL_MAX, xmax = -DBL_MAX, ymin = DBL_MAX, ymax = -DBL_MAX;
  for (int i = 0; i < 4; ++i) {
    lx[i] = (a0 * px[i] + a1 * py[i] + a2) * sc;
    ly[i] = (a3 * px[i] + a4 * py[i] + a5) * sc;
    xmin = fmin(xmin, lx[i]); xmax = fmax(xmax, lx[i]);
    ymin = fmin(ymin, ly[i]); ymax = fmax(ymax, ly[i]);
  }
  const int x0 = max(tx_begin, static_cast<int>(floor(xmin)));
  const int x1 = min(tx_begin + cols, static_cast<int>(ceil(xmax)));
  const int y0 = max(ty_begin, static_cast<int>(floor(ymin)));
  const int y1 = min(ty_begin + rows, static_cast<int>(ceil(ymax)));
  const long long plane_n = static_cast<long long>(cols) * rows;
  const long long base = static_cast<long long>(ch) * plane_n;
  for (int yy = y0; yy < y1; ++yy)
    for (int xx = x0; xx < x1; ++xx) {
      const double k = d_polygon_rect_area(lx, ly, (double)xx, (double)yy,
                                           xx + 1.0, yy + 1.0);
      if (k > 0.0)
        out_b[base + static_cast<long long>(yy - ty_begin) * cols +
              (xx - tx_begin)] += k;
    }
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

// Entries of the per-band device scalar block (see Impl::scalars).
constexpr std::size_t kV2ScalarCount = 5;

// Compact quality stream on device: raw uint16 cells + veto bytes over a
// storage-grid window (absolute storage coords). A stream supplies a float
// plane XOR a packed window; cells == nullptr means the stream is absent.
struct V2PackedQuality {
  const unsigned short *cells = nullptr;
  const unsigned char *veto = nullptr;
  int x_begin = 0;
  int y_begin = 0;
  int width = 0;
  int height = 0;
  int divisor = 1;
};

// Gate-9 quality I/O bundle for the scatter kernels: float planes XOR
// packed windows per stream (null = stream absent for this frame) and the
// five f64 frame accumulator planes (all null when profiles are disabled).
struct V2QualityIO {
  const float *qc = nullptr;
  const float *q0 = nullptr;
  const float *q1 = nullptr;
  const float *qa = nullptr;
  V2PackedQuality qc_p, q0_p, q1_p, qa_p;
  unsigned int qmask = 0;
  double *fqc = nullptr;
  double *fq0 = nullptr;
  double *fq1 = nullptr;
  double *fqa = nullptr;
  double *fqaf = nullptr;
};

// Decode one quality sample: float plane indexed by the active-local tid
// (compatibility path) or the packed storage-grid window decoded by the
// absolute source coordinate. Vetoed/zero cells yield NaN --- the same
// veto the float path reports.
__device__ inline float d_quality_sample(const float *plane,
                                         const V2PackedQuality &p,
                                         long long tid, int sx, int sy) {
  if (p.cells != nullptr) {
    const long long i =
        static_cast<long long>(sy / p.divisor - p.y_begin) * p.width +
        static_cast<long long>(sx / p.divisor - p.x_begin);
    if (p.veto != nullptr && p.veto[i]) return nanf("");
    const unsigned int c = p.cells[i];
    return c == 0u ? nanf("") : static_cast<float>(c) / 65535.0f;
  }
  return plane != nullptr ? plane[tid] : nanf("");
}

// Per-droplet quality accumulation, gated on finite source value by the
// caller (matching the CPU contract: the whole sample is skipped for
// nonfinite values). `qv` holds the decoded per-stream sample values
// (NaN = veto/absent): a NaN/<=0 sample contributes 0 to the area-weighted
// mean; fqaf collects the area weight of finite artifact samples only.
__device__ inline void d_scatter_quality(const V2QualityIO &q,
                                         const float qv[4],
                                         long long out, double area) {
  auto acc = [](float v, double *dst, long long o, double k) {
    if (dst == nullptr) return;
    const double d = static_cast<double>(v);
    atomicAdd(dst + o, k * (isfinite(d) && d > 0.0 ? d : 0.0));
  };
  if (q.qmask & 1u) acc(qv[0], q.fqc, out, area);
  if (q.qmask & 2u) acc(qv[1], q.fq0, out, area);
  if (q.qmask & 4u) acc(qv[2], q.fq1, out, area);
  if (q.qmask & 8u) acc(qv[3], q.fqa, out, area);
  if ((q.qmask & 8u) && q.fqaf != nullptr &&
      isfinite(static_cast<double>(qv[3])))
    atomicAdd(q.fqaf + out, area);
}

// Uploaded source buffer descriptor: packed row-major buffer at absolute
// origin (buf_x, buf_y), the ACTIVE rect (act_*) inside it that the launch
// iterates, and the reserved full source extent (src_*) that bounds the
// sigma-model halo reads.
struct V2Window {
  int buf_x = 0;
  int buf_y = 0;
  int buf_w = 0;
  int buf_h = 0;
  int act_x = 0;
  int act_y = 0;
  int act_w = 0;
  int act_h = 0;
  int src_w = 0;
  int src_h = 0;
};

// Inline sigma2 model parameters (mirror of
// ForwardDrizzleV2Sigma2FrameModel for kernel argument passing).
struct V2Sigma2Model {
  int enabled = 0;
  double noise = 0.0;
  double reg_px = 0.0;
  double half = 0.0;
};

// One absolute source sample from the window buffer; NaN outside the true
// source extent or outside the uploaded buffer (caller-provided halo makes
// the latter coincide with true borders only).
__device__ inline float d_window_at(const float *src, const V2Window &w,
                                    int ax, int ay) {
  if (ax < 0 || ay < 0 || ax >= w.src_w || ay >= w.src_h) return nanf("");
  const int bx = ax - w.buf_x, by = ay - w.buf_y;
  if (bx < 0 || by < 0 || bx >= w.buf_w || by >= w.buf_h) return nanf("");
  return src[static_cast<long long>(by) * w.buf_w + bx];
}

// Device mirror of forward_drizzle_v2_sigma2_model (same double ops).
__device__ inline double d_sigma2_model(double noise, double gx, double gy,
                                        double reg_px, double half) {
  if (!isfinite(noise) || noise < 0.0 || !isfinite(gx) || !isfinite(gy) ||
      !isfinite(reg_px) || reg_px < 0.0 || !isfinite(half) || !(half > 0.0))
    return nan("");
  const double g2 = gx * gx + gy * gy;
  return noise * noise + g2 * reg_px * reg_px + half * half / 3.0;
}

// Inline sigma2 at absolute (sx, sy) from the halo window buffer: central
// difference where both absolute neighbours are finite, one-sided fallback
// at true source borders or invalid neighbours, zero otherwise ---
// identical semantics to forward_drizzle_v2_sigma2_plane.
__device__ inline double d_sigma2_at(const float *src, const V2Window &w,
                                     const V2Sigma2Model &m, int sx,
                                     int sy) {
  auto diff = [&](bool x_axis) -> float {
    const float vm =
        d_window_at(src, w, sx - (x_axis ? 1 : 0), sy - (x_axis ? 0 : 1));
    const float vp =
        d_window_at(src, w, sx + (x_axis ? 1 : 0), sy + (x_axis ? 0 : 1));
    const bool fm = isfinite(vm), fp = isfinite(vp);
    if (fm && fp) return (vp - vm) * 0.5f;
    const float vc = d_window_at(src, w, sx, sy);
    if (fp && isfinite(vc)) return vp - vc;
    if (fm && isfinite(vc)) return vc - vm;
    return 0.0f;
  };
  // The explicit-plane path quantises to float; mirror that so both
  // sigma2 contracts agree bit for bit.
  return static_cast<double>(static_cast<float>(
      d_sigma2_model(m.noise, diff(true), diff(false), m.reg_px, m.half)));
}

// 16 B device row of ForwardDrizzleV2FrameMeta, indexed by frame order.
struct V2FrameMetaDev {
  float g_eff;
  float residual_factor;
  unsigned int is_direct;
  unsigned int pad;
};

// Full-frame estimator accumulator per (pixel, channel); mirrors CpuFullAcc.
// state: 0 no bounds, 1 active (frozen pilot bounds), 2 degenerate pilot.
// n_acc/n_rej count NON-pilot contributions only (diagnostics).
struct V2FullAcc {
  double wx[4], w[4], w2[4];
  double ca, cb, cb2, cs, cc, lo, hi;
  unsigned int n_acc, n_rej;
  unsigned char state;
};

__device__ __forceinline__ void d_full_add(V2FullAcc &f, double b, double x,
                                           double sigma2, double g, double qc,
                                           double q0, double q1,
                                           double fine_exp,
                                           double medium_exp) {
  const double wgt[4] = {b, b * g * qc, b * g * pow(q0, fine_exp),
                         b * g * pow(q1, medium_exp)};
  for (int k = 0; k < 4; ++k) {
    f.wx[k] += wgt[k] * x;
    f.w[k] += wgt[k];
    f.w2[k] += wgt[k] * wgt[k];
  }
  f.ca += b * x;
  f.cb += b;
  f.cb2 += b * b;
  if (isfinite(sigma2) && sigma2 > 0.0) {
    f.cs += b * sqrt(sigma2);
    f.cc += b * b * sigma2;
  }
}

__host__ __device__ unsigned long long d_splitmix64(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Clamped internal-cell bbox [x0,x1) x [y0,y1) of a band-local quad:
// floor(min)..ceil(max) of its vertices, clamped to [0,cols) x [0,rows).
// The fmin/fmax chain runs over k = 0..3 exactly as the kernels did inline.
__device__ __forceinline__ void d_quad_bbox(const double qx[4],
                                            const double qy[4], int cols,
                                            int rows, int &x0, int &x1,
                                            int &y0, int &y1) {
  double minx = DBL_MAX, maxx = -DBL_MAX;
  double miny = DBL_MAX, maxy = -DBL_MAX;
  for (int k = 0; k < 4; ++k) {
    minx = fmin(minx, qx[k]); maxx = fmax(maxx, qx[k]);
    miny = fmin(miny, qy[k]); maxy = fmax(maxy, qy[k]);
  }
  x0 = max(0, static_cast<int>(floor(minx)));
  x1 = min(cols, static_cast<int>(ceil(maxx)));
  y0 = max(0, static_cast<int>(floor(miny)));
  y1 = min(rows, static_cast<int>(ceil(maxy)));
}

// The clip + atomic inner loop every scatter kernel shares: accumulate the
// quad's overlap with each cell of [x0,x1) x [y0,y1) into the frame planes.
// The geometry weight is written for every positive overlap BEFORE the
// value check (a nonfinite source sample still owns its geometric support,
// footprint/B_geo, exactly like the CPU contract); value, source weight,
// sigma2 and quality only for finite values. `chan_base` is
// channel * plane_n. Expression order is the parity contract --- keep it.
__device__ __forceinline__ void d_scatter_quad(
    const double qx[4], const double qy[4], int x0, int x1, int y0, int y1,
    int cols, long long chan_base, double *fa, double *fbs, double *fbg,
    double *fs2, const V2QualityIO &qio, const float qv[4],
    bool finite_value, double value, double s2,
    unsigned long long &overlaps) {
  for (int ty = y0; ty < y1; ++ty) {
    for (int tx = x0; tx < x1; ++tx) {
      const double area =
          d_polygon_rect_area(qx, qy, tx, ty, tx + 1.0, ty + 1.0);
      if (!(area > 0.0)) continue;
      const long long out =
          chan_base + static_cast<long long>(ty) * cols + tx;
      atomicAdd(fbg + out, area);
      if (finite_value) {
        atomicAdd(fa + out, area * value);
        atomicAdd(fbs + out, area);
        if (fs2 != nullptr) atomicAdd(fs2 + out, area * s2);
        if (qio.qmask != 0u) d_scatter_quality(qio, qv, out, area);
      }
      ++overlaps;
    }
  }
}

// Dense scatter into the band-local internal planes. Unlike
// k_affine_dense_scatter the geometry weight is written for every positive
// overlap BEFORE the value check: a nonfinite source sample still owns its
// geometric support (footprint/B_geo), exactly like the CPU contract.
// affine6 is the canvas-coordinate source->canvas transform; the emitted
// internal coordinate is (q - band_origin) * sc.
__global__ void k_scatter_v2(
    double a0, double a1, double a2, double a3, double a4, double a5,
    double sc, double half, int cols, int rows, V2Window w,
    const float *source, const float *sigma2, V2Sigma2Model s2m, int bayer,
    int ox, int oy,
    int mono, double band_ox, double band_oy, int tile_x0, int tile_x1,
    double *fa, double *fbs,
    double *fbg, double *fs2,
    V2QualityIO qio, unsigned long long *positive_overlaps) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long source_n =
      static_cast<long long>(w.act_w) * w.act_h;
  if (tid >= source_n) return;
  // Packed-buffer indexing: tid iterates the ACTIVE rect inside the
  // uploaded buffer (optional halo); geometry keeps absolute source
  // coordinates and float sigma/Q planes are packed over the active rect.
  const int lx = static_cast<int>(tid % w.act_w);
  const int ly = static_cast<int>(tid / w.act_w);
  const int sx = w.buf_x + w.act_x + lx;
  const int sy = w.buf_y + w.act_y + ly;
  const long long bidx =
      static_cast<long long>(w.act_y + ly) * w.buf_w + (w.act_x + lx);
  const double value = static_cast<double>(source[bidx]);
  const bool finite_value = isfinite(value);
  const double s2 =
      sigma2 != nullptr
          ? static_cast<double>(sigma2[tid])
          : (s2m.enabled ? d_sigma2_at(source, w, s2m, sx, sy) : 0.0);
  float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (qio.qmask != 0u) {
    qv[0] = d_quality_sample(qio.qc, qio.qc_p, tid, sx, sy);
    qv[1] = d_quality_sample(qio.q0, qio.q0_p, tid, sx, sy);
    qv[2] = d_quality_sample(qio.q1, qio.q1_p, tid, sx, sy);
    qv[3] = d_quality_sample(qio.qa, qio.qa_p, tid, sx, sy);
  }
  const int channel = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
  const double x = sx + 0.5, y = sy + 0.5;
  const double px[4] = {x - half, x + half, x + half, x - half};
  const double py[4] = {y - half, y - half, y + half, y + half};
  double qx[4], qy[4];
  for (int k = 0; k < 4; ++k) {
    qx[k] = (a0 * px[k] + a1 * py[k] + a2 - band_ox) * sc;
    qy[k] = (a3 * px[k] + a4 * py[k] + a5 - band_oy) * sc;
  }
  int x0, x1, y0, y1;
  d_quad_bbox(qx, qy, cols, rows, x0, x1, y0, y1);
  // The affine piece path owns internal columns [tile_x0, tile_x1); the
  // bbox clamp drops emissions outside the tile so overlapping source scan
  // boxes never double-count.
  x0 = max(tile_x0, x0);
  x1 = min(tile_x1, x1);
  unsigned long long overlaps = 0;
  const long long plane_n = static_cast<long long>(cols) * rows;
  d_scatter_quad(qx, qy, x0, x1, y0, y1, cols,
                 static_cast<long long>(channel) * plane_n, fa, fbs, fbg,
                 fs2, qio, qv, finite_value, value, s2, overlaps);
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
}

// Geometry-cache scatter (tranche 6): one thread per committed leaf.
// Leaf corners are absolute internal canvas coordinates; subtracting the
// internal band origin (band_origin * sc) makes them band-local ---
// bit-identical to the local emit path's (native - band_origin) * sc since
// sc is a power of two. No inversion or subdivision runs here --- the
// cache build already finalised discards.
__global__ void k_scatter_v2_cached(
    const ForwardDrizzleV2CachedLeaf *leaves, long long leaf_count,
    double sc, int cols, int rows, V2Window w,
    const float *source, const float *sigma2, V2Sigma2Model s2m,
    double band_ox, double band_oy, double *fa, double *fbs, double *fbg,
    double *fs2, V2QualityIO qio, unsigned long long *positive_overlaps) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= leaf_count) return;
  const ForwardDrizzleV2CachedLeaf &L = leaves[tid];
  const int sx = static_cast<int>(L.source_x);
  const int sy = static_cast<int>(L.source_y);
  const long long bidx =
      static_cast<long long>(sy - w.buf_y) * w.buf_w + (sx - w.buf_x);
  // Float sigma/Q compatibility planes are packed over the ACTIVE rect;
  // the host validated that every leaf lies inside it.
  const long long tidx =
      static_cast<long long>(sy - (w.buf_y + w.act_y)) * w.act_w +
      (sx - (w.buf_x + w.act_x));
  const double value = static_cast<double>(source[bidx]);
  const bool finite_value = isfinite(value);
  const double s2 =
      sigma2 != nullptr
          ? static_cast<double>(sigma2[tidx])
          : (s2m.enabled ? d_sigma2_at(source, w, s2m, sx, sy) : 0.0);
  float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (qio.qmask != 0u) {
    qv[0] = d_quality_sample(qio.qc, qio.qc_p, tidx, sx, sy);
    qv[1] = d_quality_sample(qio.q0, qio.q0_p, tidx, sx, sy);
    qv[2] = d_quality_sample(qio.q1, qio.q1_p, tidx, sx, sy);
    qv[3] = d_quality_sample(qio.qa, qio.qa_p, tidx, sx, sy);
  }
  const int channel = static_cast<int>(L.channel);
  double qx[4], qy[4];
  for (int k = 0; k < 4; ++k) {
    qx[k] = L.x[k] - band_ox * sc;
    qy[k] = L.y[k] - band_oy * sc;
  }
  int x0, x1, y0, y1;
  d_quad_bbox(qx, qy, cols, rows, x0, x1, y0, y1);
  unsigned long long overlaps = 0;
  const long long plane_n = static_cast<long long>(cols) * rows;
  d_scatter_quad(qx, qy, x0, x1, y0, y1, cols,
                 static_cast<long long>(channel) * plane_n, fa, fbs, fbg,
                 fs2, qio, qv, finite_value, value, s2, overlaps);
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
}

// --- Tranche 8: canonical ragged affine sample list -----------------------

// Device mirror of ForwardDrizzleV2AlignedQuality: per-sample quantized
// codes/veto indexed by the sample's launch tid (never expanded floats).
struct V2AlignedQualityDev {
  const unsigned short *qc = nullptr, *q0 = nullptr, *q1 = nullptr,
                       *qa = nullptr;
  const unsigned char *vc = nullptr, *v0 = nullptr, *v1 = nullptr,
                      *va = nullptr;
  unsigned int qmask = 0;
};

__device__ inline float d_aligned_quality_sample(const unsigned short *codes,
                                                 const unsigned char *veto,
                                                 long long tid) {
  if ((veto != nullptr && veto[tid] != 0) || codes[tid] == 0)
    return nanf("");
  return static_cast<float>(codes[tid]) / 65535.0f;
}

// One thread per active source sample; absolute (x, y) come from the
// record, geometry is the same affine droplet as k_scatter_v2 with the
// full-band x clamp (no target tiling on this path).
__global__ void k_scatter_v2_samples(
    const ForwardDrizzleV2SourceSample *samples, long long sample_count,
    int sigma2_present, double a0, double a1, double a2, double a3,
    double a4, double a5, double sc, double half, int cols, int rows,
    int bayer, int ox, int oy, int mono, double band_ox, double band_oy,
    double *fa, double *fbs, double *fbg, double *fs2,
    V2AlignedQualityDev aq, V2QualityIO qio,
    unsigned long long *positive_overlaps) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= sample_count) return;
  const ForwardDrizzleV2SourceSample &s = samples[tid];
  const int sx = static_cast<int>(s.source_x);
  const int sy = static_cast<int>(s.source_y);
  const double value = static_cast<double>(s.value);
  const bool finite_value = isfinite(value);
  const double s2 = sigma2_present ? static_cast<double>(s.sigma2) : 0.0;
  float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (aq.qmask != 0u) {
    if ((aq.qmask & 1u) != 0u) qv[0] = d_aligned_quality_sample(aq.qc, aq.vc, tid);
    if ((aq.qmask & 2u) != 0u) qv[1] = d_aligned_quality_sample(aq.q0, aq.v0, tid);
    if ((aq.qmask & 4u) != 0u) qv[2] = d_aligned_quality_sample(aq.q1, aq.v1, tid);
    if ((aq.qmask & 8u) != 0u) qv[3] = d_aligned_quality_sample(aq.qa, aq.va, tid);
  }
  const int channel = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
  const double cx = static_cast<double>(sx) + 0.5;
  const double cy = static_cast<double>(sy) + 0.5;
  const double px[4] = {cx - half, cx + half, cx + half, cx - half};
  const double py[4] = {cy - half, cy - half, cy + half, cy + half};
  double qx[4], qy[4];
  for (int k = 0; k < 4; ++k) {
    qx[k] = (a0 * px[k] + a1 * py[k] + a2 - band_ox) * sc;
    qy[k] = (a3 * px[k] + a4 * py[k] + a5 - band_oy) * sc;
  }
  int x0, x1, y0, y1;
  d_quad_bbox(qx, qy, cols, rows, x0, x1, y0, y1);
  unsigned long long overlaps = 0;
  const long long plane_n = static_cast<long long>(cols) * rows;
  d_scatter_quad(qx, qy, x0, x1, y0, y1, cols,
                 static_cast<long long>(channel) * plane_n, fa, fbs, fbg,
                 fs2, qio, qv, finite_value, value, s2, overlaps);
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
}

// --- Gate-8 device port of the production local-warp contract -------------
//
// 1:1 ports of smooth_local_basis / evaluate_smooth_local_displacement /
// local_displacement_render_units / invert_local_source_to_canvas /
// subdivide_local from the CPU oracle (global_registration.cpp,
// registration_sampling_plan.cpp, forward_drizzle.cpp). Inversion internals
// run in fp32 like the oracle; geometry, errors and areas in fp64.

// d(q) in render units. Port of local_displacement_render_units +
// evaluate_smooth_local_displacement + smooth_local_basis: an invalid
// model or a non-positive/non-finite coordinate scale fails (the caller
// turns it into a per-sample discard, exactly like the CPU oracle); an
// out-of-image query is a true zero displacement.
__device__ bool d_local_displacement_render(
    const ForwardDrizzleV2LocalWarp &w, float qx, float qy, float &out_dx,
    float &out_dy) {
  out_dx = 0.0f;
  out_dy = 0.0f;
  if (!w.model_valid) return false;
  const float scale = w.model_coordinate_scale;
  if (!(scale > 0.0f) || !isfinite(scale)) return false;
  const float qmx = (qx - w.model_offset_x) * scale;
  const float qmy = (qy - w.model_offset_y) * scale;
  // smooth_local_basis: outside the model image the displacement is 0.
  if (w.image_rows <= 1 || w.image_cols <= 1 || qmx < 0.0f || qmy < 0.0f ||
      qmx > static_cast<float>(w.image_cols - 1) ||
      qmy > static_cast<float>(w.image_rows - 1)) {
    return true;
  }
  const float nx = qmx / static_cast<float>(w.image_cols - 1);
  const float ny = qmy / static_cast<float>(w.image_rows - 1);
  // std::min({nx, 1-nx, ny, 1-ny}) order of the oracle: the first element
  // stays unless a strictly smaller one is found (NaN in nx propagates).
  float edge = nx;
  if (1.0f - nx < edge) edge = 1.0f - nx;
  if (ny < edge) edge = ny;
  if (1.0f - ny < edge) edge = 1.0f - ny;
  const float v = edge / 0.08f;
  // std::clamp(v, 0, 1) semantics: NaN passes through unchanged.
  const float tt = (v < 0.0f) ? 0.0f : (1.0f < v ? 1.0f : v);
  const float taper = tt * tt * (3.0f - 2.0f * tt);
  if (taper <= 0.0f) return true;
  const float inv_two_sigma_sq = 1.0f / (2.0f * 0.28f * 0.28f);
  float basis[16];
  float sum = 0.0f;
  int index = 0;
  for (int gy = 0; gy < 4; ++gy) {
    const float cy = static_cast<float>(gy) / 3.0f;
    for (int gx = 0; gx < 4; ++gx, ++index) {
      const float cx = static_cast<float>(gx) / 3.0f;
      const float ddx = nx - cx, ddy = ny - cy;
      const float value = expf(-(ddx * ddx + ddy * ddy) * inv_two_sigma_sq);
      basis[index] = value;
      sum += value;
    }
  }
  // The oracle scales by taper/sum only for a usable sum; a degenerate sum
  // leaves the raw basis weights in place.
  const float factor = (sum > 1.0e-8f) ? taper / sum : 1.0f;
  float mx = 0.0f, my = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const float b = basis[i] * factor;
    mx += b * w.coeff_x[i];
    my += b * w.coeff_y[i];
  }
  const float inv_scale = 1.0f / scale;
  out_dx = mx * inv_scale;
  out_dy = my * inv_scale;
  return isfinite(out_dx) && isfinite(out_dy);
}

// Port of invert_local_source_to_canvas: u = affine(s) in fp32, then the
// bounded fixed-point iteration q_{n+1} = u - d(q_n).
__device__ bool d_invert_local(const ForwardDrizzleV2LocalWarp &w,
                               const double a[6], float sx, float sy,
                               int canvas_w_native, int canvas_h_native,
                               float &out_qx, float &out_qy) {
  const float a00 = static_cast<float>(a[0]), a01 = static_cast<float>(a[1]),
              a02 = static_cast<float>(a[2]), a10 = static_cast<float>(a[3]),
              a11 = static_cast<float>(a[4]), a12 = static_cast<float>(a[5]);
  const float ux = a00 * sx + a01 * sy + a02;
  const float uy = a10 * sx + a11 * sy + a12;
  if (!isfinite(ux) || !isfinite(uy)) return false;
  const float margin = w.safety_margin_px;
  const float cw = static_cast<float>(canvas_w_native);
  const float ch = static_cast<float>(canvas_h_native);
  if (ux < -margin || uy < -margin || ux > cw + margin || uy > ch + margin)
    return false;
  float qx = ux, qy = uy;
  bool converged = false;
  const int max_iter = w.max_iter > 0 ? w.max_iter : 1;
  for (int n = 0; n < max_iter; ++n) {
    float dx = 0.0f, dy = 0.0f;
    if (!d_local_displacement_render(w, qx, qy, dx, dy)) return false;
    const float nx = ux - dx, ny = uy - dy;
    if (!isfinite(nx) || !isfinite(ny)) return false;
    if (nx < -margin || ny < -margin || nx > cw + margin || ny > ch + margin)
      return false;
    const float step = fmaxf(fabsf(nx - qx), fabsf(ny - qy));
    qx = nx;
    qy = ny;
    if (step < w.tol_px) {
      converged = true;
      break;
    }
  }
  if (!converged) return false;
  out_qx = qx;
  out_qy = qy;
  return true;
}

// Grid coordinate i/2 of the interval [a0, a1], i in {0, 1, 2}. Exactly the
// CPU oracle's expression (subdivide_local: `x0 + (x1 - x0) * i / 2`); the
// node evaluator, the child split and the leaf-corner re-inversion must all
// use it so the scatter inverts bit-identical coordinates to the ones the
// evaluator accepted. Note a0 + (a1 - a0) * 2 / 2 need not equal a1.
__device__ __forceinline__ double d_node_grid(double a0, double a1, int i) {
  return a0 + (a1 - a0) * i / 2;
}

// Node bounds of the implicit subdivision tree. `id` enumerates 1+4+16
// nodes: 0 = root droplet box, 1..4 = depth-1 children (c = j*2+i of the
// parent's 2x2 split), 5..20 = depth-2 grandchildren. Child (i, j) spans
// [grid(i), grid(i + 1)] of its parent, as in subdivide_local.
__device__ void d_local_node_bounds(int id, double x0, double y0, double x1,
                                    double y1, double &nx0, double &ny0,
                                    double &nx1, double &ny1) {
  nx0 = x0;
  ny0 = y0;
  nx1 = x1;
  ny1 = y1;
  auto split = [](double &a0, double &a1, int i) {
    const double lo = d_node_grid(a0, a1, i);
    const double hi = d_node_grid(a0, a1, i + 1);
    a0 = lo;
    a1 = hi;
  };
  if (id >= 5) {
    const int p = (id - 5) / 4;  // depth-1 parent index
    split(nx0, nx1, p & 1);
    split(ny0, ny1, (p >> 1) & 1);
    const int c = (id - 5) % 4;
    split(nx0, nx1, c & 1);
    split(ny0, ny1, (c >> 1) & 1);
  } else if (id >= 1) {
    const int c = id - 1;
    split(nx0, nx1, c & 1);
    split(ny0, ny1, (c >> 1) & 1);
  }
}

// Port of subdivide_local's per-node evaluation: 3x3 inversions, parent
// quad, bilinear error and child-area tests. Returns 0 = accepted,
// 1 = rejected (subdivide), 2 = failure (discard the whole sample).
__device__ int d_eval_local_node(const ForwardDrizzleV2LocalWarp &w,
                                 const double a[6], double x0, double y0,
                                 double x1, double y1, int depth,
                                 int canvas_w_native, int canvas_h_native,
                                 double sc) {
  double gx[3][3], gy[3][3];
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      float qx = 0.0f, qy = 0.0f;
      if (!d_invert_local(w, a, static_cast<float>(d_node_grid(x0, x1, i)),
                          static_cast<float>(d_node_grid(y0, y1, j)),
                          canvas_w_native, canvas_h_native, qx, qy))
        return 2;
      gx[j][i] = static_cast<double>(qx) * sc;
      gy[j][i] = static_cast<double>(qy) * sc;
    }
  const double px[4] = {gx[0][0], gx[0][2], gx[2][2], gx[2][0]};
  const double py[4] = {gy[0][0], gy[0][2], gy[2][2], gy[2][0]};
  double error = 0.0, child_area = 0.0;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      const double u = i / 2.0, v = j / 2.0;
      const double bx = (1 - u) * (1 - v) * px[0] + u * (1 - v) * px[1] +
                        u * v * px[2] + (1 - u) * v * px[3];
      const double by = (1 - u) * (1 - v) * py[0] + u * (1 - v) * py[1] +
                        u * v * py[2] + (1 - u) * v * py[3];
      error = fmax(error, hypot(gx[j][i] - bx, gy[j][i] - by));
    }
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      const double cx[4] = {gx[j][i], gx[j][i + 1], gx[j + 1][i + 1],
                            gx[j + 1][i]};
      const double cy[4] = {gy[j][i], gy[j][i + 1], gy[j + 1][i + 1],
                            gy[j + 1][i]};
      child_area += d_shoelace_area(cx, cy, 4);
    }
  const double area = d_shoelace_area(px, py, 4);
  if (area > 0.0 && error <= w.position_epsilon_internal_px &&
      fabs(child_area - area) / area <= w.area_relative_epsilon)
    return 0;
  if (depth >= w.max_subdivision_depth) return 2;
  return 1;
}

// One thread per source sample, local-warp variant of k_scatter_v2: runs
// the implicit 21-node subdivision, then scatters every accepted leaf
// through the identical polygon-clip accumulation. All-or-nothing per
// sample exactly like the CPU oracle (a failure discards accepted leaves
// too). `discarded` counts failed samples.
__global__ void k_scatter_v2_local(
    double a0, double a1, double a2, double a3, double a4, double a5,
    ForwardDrizzleV2LocalWarp w, double sc, double half, int cols, int rows,
    V2Window win, const float *source,
    const float *sigma2, V2Sigma2Model s2m, int bayer, int ox, int oy,
    int mono,
    int canvas_w_native,
    int canvas_h_native, double band_ox, double band_oy, double *fa,
    double *fbs, double *fbg, double *fs2,
    V2QualityIO qio, unsigned long long *positive_overlaps,
    unsigned long long *discarded) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long source_n =
      static_cast<long long>(win.act_w) * win.act_h;
  if (tid >= source_n) return;
  const int lx = static_cast<int>(tid % win.act_w);
  const int ly = static_cast<int>(tid / win.act_w);
  const int sx = win.buf_x + win.act_x + lx;
  const int sy = win.buf_y + win.act_y + ly;
  const long long bidx =
      static_cast<long long>(win.act_y + ly) * win.buf_w + (win.act_x + lx);
  const double a[6] = {a0, a1, a2, a3, a4, a5};
  const double x = sx + 0.5, y = sy + 0.5;
  const double bx0 = x - half, by0 = y - half, bx1 = x + half,
               by1 = y + half;

  unsigned int visited = 0, accepted = 0;
  bool fail = false;
  for (int id = 0; id < 21 && !fail; ++id) {
    const int level = id == 0 ? 0 : (id < 5 ? 1 : 2);
    if (level > 0) {
      const int parent = level == 1 ? 0 : 1 + (id - 5) / 4;
      if (!((visited >> parent) & 1u) || ((accepted >> parent) & 1u))
        continue;  // parent accepted or never evaluated: node not visited
    }
    double nx0, ny0, nx1, ny1;
    d_local_node_bounds(id, bx0, by0, bx1, by1, nx0, ny0, nx1, ny1);
    const int status =
        d_eval_local_node(w, a, nx0, ny0, nx1, ny1, level, canvas_w_native,
                          canvas_h_native, sc);
    if (status == 2) {
      fail = true;
      break;
    }
    visited |= 1u << id;
    if (status == 0)
      accepted |= 1u << id;
    else if (level == 2)
      fail = true;  // rejected at max depth: whole sample discarded
  }
  if (fail || accepted == 0) {
    atomicAdd(discarded, 1ULL);
    return;
  }

  const double value = static_cast<double>(source[bidx]);
  const bool finite_value = isfinite(value);
  const double s2 =
      sigma2 != nullptr
          ? static_cast<double>(sigma2[tid])
          : (s2m.enabled ? d_sigma2_at(source, win, s2m, sx, sy) : 0.0);
  float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (qio.qmask != 0u) {
    qv[0] = d_quality_sample(qio.qc, qio.qc_p, tid, sx, sy);
    qv[1] = d_quality_sample(qio.q0, qio.q0_p, tid, sx, sy);
    qv[2] = d_quality_sample(qio.q1, qio.q1_p, tid, sx, sy);
    qv[3] = d_quality_sample(qio.qa, qio.qa_p, tid, sx, sy);
  }
  const int channel = mono ? 0 : d_cfa_channel(sx, sy, bayer, ox, oy);
  const long long plane_n = static_cast<long long>(cols) * rows;
  unsigned long long overlaps = 0;
  for (int id = 0; id < 21; ++id) {
    if (!((accepted >> id) & 1u)) continue;
    double nx0, ny0, nx1, ny1;
    d_local_node_bounds(id, bx0, by0, bx1, by1, nx0, ny0, nx1, ny1);
    // Re-evaluate only the four corner inversions of the accepted node.
    // The corners are the evaluator's grid points (0,0) (2,0) (2,2) (0,2)
    // computed by the same d_node_grid expression, so every inversion below
    // repeats one the evaluator already accepted: the failure branch cannot
    // fire after earlier leaves were scattered.
    const double gx0 = d_node_grid(nx0, nx1, 0), gx2 = d_node_grid(nx0, nx1, 2);
    const double gy0 = d_node_grid(ny0, ny1, 0), gy2 = d_node_grid(ny0, ny1, 2);
    const double csx[4] = {gx0, gx2, gx2, gx0};
    const double csy[4] = {gy0, gy0, gy2, gy2};
    double qx[4], qy[4];
    for (int k = 0; k < 4; ++k) {
      float fq = 0.0f, fq2 = 0.0f;
      if (!d_invert_local(w, a, static_cast<float>(csx[k]),
                          static_cast<float>(csy[k]), canvas_w_native,
                          canvas_h_native, fq, fq2)) {
        fail = true;
        break;
      }
      qx[k] = (static_cast<double>(fq) - band_ox) * sc;
      qy[k] = (static_cast<double>(fq2) - band_oy) * sc;
    }
    if (fail) break;
    int x0, x1, y0, y1;
    d_quad_bbox(qx, qy, cols, rows, x0, x1, y0, y1);
    d_scatter_quad(qx, qy, x0, x1, y0, y1, cols,
                   static_cast<long long>(channel) * plane_n, fa, fbs, fbg,
                   fs2, qio, qv, finite_value, value, s2, overlaps);
  }
  if (fail) {
    atomicAdd(discarded, 1ULL);
    return;
  }
  if (overlaps) atomicAdd(positive_overlaps, overlaps);
}

// Tranche-7 piece clear: zero the internal x columns [x0, x1) of every
// channel x row of the frame planes owned by one affine piece. Nullable
// planes are skipped (fs2 absent, Q planes when the frame is not
// quality-selected).
__global__ void k_clear_planes_x(
    double *fa, double *fbs, double *fbg, double *fs2,
    double *fqc, double *fq0, double *fq1, double *fqa, double *fqaf,
    int icols, int irows, int channels, int x0, int x1) {
  const long long t =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long w = x1 - x0;
  const long long total =
      static_cast<long long>(channels) * irows * w;
  if (t >= total) return;
  const int tx = x0 + static_cast<int>(t % w);
  const long long rc = t / w;
  const int ty = static_cast<int>(rc % irows);
  const int c = static_cast<int>(rc / irows);
  const long long idx =
      static_cast<long long>(c) * icols * irows +
      static_cast<long long>(ty) * icols + tx;
  fa[idx] = 0.0;
  fbs[idx] = 0.0;
  fbg[idx] = 0.0;
  if (fs2 != nullptr) fs2[idx] = 0.0;
  if (fqc != nullptr) fqc[idx] = 0.0;
  if (fq0 != nullptr) fq0[idx] = 0.0;
  if (fq1 != nullptr) fq1[idx] = 0.0;
  if (fqa != nullptr) fqa[idx] = 0.0;
  if (fqaf != nullptr) fqaf[idx] = 0.0;
}

// Band-end telemetry: sums of kept[] and contrib[] over all (pixel, channel)
// entries into out[0] / out[1]. Integer sums are exact and order-independent,
// so this replaces downloading both arrays for a host-side sum.
__global__ void k_sum_kept_contrib(const unsigned int *kept,
                                   const unsigned int *contrib, long long n,
                                   unsigned long long *out) {
  unsigned long long k = 0, c = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    k += kept[i];
    c += contrib[i];
  }
  for (int off = 16; off > 0; off >>= 1) {
    k += __shfl_down_sync(0xffffffffu, k, off);
    c += __shfl_down_sync(0xffffffffu, c, off);
  }
  if ((threadIdx.x & 31) == 0) {
    if (k) atomicAdd(out, k);
    if (c) atomicAdd(out + 1, c);
  }
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
    const double *fqc, const double *fq0, const double *fq1,
    const double *fqa, const double *fqaf, unsigned int qmask,
    int icols, int irows, int scale, int ncols, int nrows, int channels,
    int nx_begin, int nx_count, int ny_begin, int ny_count,
    unsigned long long order, int keep_all, unsigned long long threshold,
    int res_slots, unsigned long long seed,
    double *accA, double *accB, double *accB2,
    double *covB, double *covB2, double *confS, double *confC,
    unsigned int *contrib, unsigned int *kept, unsigned int *footprint,
    unsigned short *supp, unsigned long long *degraded,
    V2ReservoirRecord *res, float4 *resq,
    V2FullAcc *full, const V2FrameMetaDev *fmeta, double fine_exp,
    double medium_exp, int sfr, double consensus) {
  // Threads cover native columns [nx_begin, nx_begin + nx_count) and rows
  // [ny_begin, ny_begin + ny_count) only: the piece path passes its column
  // range with all rows, the sample/cached paths all columns with the
  // frame's row window (rows outside it were neither cleared nor written
  // for this frame and must not be read).
  const long long t =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  const long long piece_n = static_cast<long long>(nx_count) * ny_count;
  if (t >= piece_n) return;
  const int nx = nx_begin + static_cast<int>(t % nx_count);
  const int ny = ny_begin + static_cast<int>(t / nx_count);
  const long long px = static_cast<long long>(ny) * ncols + nx;
  const long long iplane = static_cast<long long>(icols) * irows;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  bool any_geo = false;
  // Full-frame mode: non-pilot candidates of this pixel, one per channel.
  const bool full_nonpilot =
      full != nullptr &&
      !(keep_all || d_splitmix64(order ^ seed) < threshold);
  bool f_has[3] = {false, false, false};
  double f_x[3] = {0.0, 0.0, 0.0}, f_b[3] = {0.0, 0.0, 0.0},
         f_s2[3] = {0.0, 0.0, 0.0}, f_qc[3] = {1.0, 1.0, 1.0},
         f_q0[3] = {1.0, 1.0, 1.0}, f_q1[3] = {1.0, 1.0, 1.0};
  for (int c = 0; c < channels; ++c) {
    const long long pc = static_cast<long long>(c) * nplane + px;
    double a = 0.0, b_src = 0.0, b_geo = 0.0, s2w = 0.0;
    double qc_s = 0.0, q0_s = 0.0, q1_s = 0.0, qa_s = 0.0, qaf_s = 0.0;
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
        if (resq != nullptr) {
          if (fqc != nullptr) qc_s += inv_s2 * fqc[idx];
          if (fq0 != nullptr) q0_s += inv_s2 * fq0[idx];
          if (fq1 != nullptr) q1_s += inv_s2 * fq1[idx];
          if (fqa != nullptr) qa_s += inv_s2 * fqa[idx];
          if (fqaf != nullptr) qaf_s += inv_s2 * fqaf[idx];
        }
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
    if (full_nonpilot) {
      f_has[c] = true;
      f_x[c] = x;
      f_b[c] = b_src;
      f_s2[c] = s2c;
      f_qc[c] = (qmask & 1u) ? qc_s / b_src : 1.0;
      f_q0[c] = (qmask & 2u) ? q0_s / b_src : 1.0;
      f_q1[c] = (qmask & 4u) ? q1_s / b_src : 1.0;
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
        if (resq != nullptr) {
          // Folded quality means: present stream => sum/b_src, absent => 1.0;
          // qa additionally encodes "no finite artifact datum" as -1.
          float4 qv;
          qv.x = (qmask & 1u) ? static_cast<float>(qc_s / b_src) : 1.0f;
          qv.y = (qmask & 2u) ? static_cast<float>(q0_s / b_src) : 1.0f;
          qv.z = (qmask & 4u) ? static_cast<float>(q1_s / b_src) : 1.0f;
          const bool qa_data =
              (qmask & 8u) != 0u && qaf_s > 0.0;
          qv.w = qa_data ? static_cast<float>(qa_s / b_src) : -1.0f;
          resq[pc * res_slots + k] = qv;
        }
      }
    }
  }
  if (full_nonpilot) {
    // Frozen pilot bounds + cross-channel consensus (see the CPU fold).
    bool rej[3] = {false, false, false};
    int voters = 0, rejected = 0;
    for (int c = 0; c < channels; ++c) {
      if (!f_has[c]) continue;
      const V2FullAcc &fa_ = full[static_cast<long long>(c) * nplane + px];
      if (fa_.state != 1) continue;
      ++voters;
      if (!(isfinite(f_x[c]) && f_x[c] >= fa_.lo && f_x[c] <= fa_.hi)) {
        rej[c] = true;
        ++rejected;
      }
    }
    if (sfr != 0 && voters > 1 &&
        static_cast<double>(rejected) / static_cast<double>(voters) >
            consensus) {
      for (int c = 0; c < channels; ++c) {
        if (!f_has[c] ||
            full[static_cast<long long>(c) * nplane + px].state != 1)
          continue;
        rej[c] = true;
      }
    }
    const double g = static_cast<double>(fmeta[order].g_eff);
    for (int c = 0; c < channels; ++c) {
      if (!f_has[c]) continue;
      V2FullAcc &fa_ = full[static_cast<long long>(c) * nplane + px];
      if (fa_.state != 1) continue;
      if (rej[c]) {
        ++fa_.n_rej;
      } else {
        d_full_add(fa_, f_b[c], f_x[c], f_s2[c], g, f_qc[c], f_q0[c],
                   f_q1[c], fine_exp, medium_exp);
        ++fa_.n_acc;
      }
    }
  }
  if (any_geo) ++footprint[px];
}

// Gate-9 device ports of the alpha-confidence helpers
// (alpha_confidence.cpp): Hazen-plotting-position weighted percentile over
// insertion-sorted (value, weight) pairs and the plan smoothstep. Explicit
// comparisons preserve the host's NaN propagation (std::clamp keeps NaN,
// fmin/fmax would drop it).
__device__ double d_smoothstep(double e0, double e1, double x) {
  if (!(e1 > e0)) return x >= e1 ? 1.0 : 0.0;
  double t = (x - e0) / (e1 - e0);
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  return t * t * (3.0 - 2.0 * t);
}

__device__ double d_hazen_percentile(double *vals, double *wts, int n,
                                     double p) {
  for (int i = 1; i < n; ++i) {
    const double v = vals[i], w = wts[i];
    int j = i;
    while (j > 0 && vals[j - 1] > v) {
      vals[j] = vals[j - 1];
      wts[j] = wts[j - 1];
      --j;
    }
    vals[j] = v;
    wts[j] = w;
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) total += wts[i];
  if (!(total > 0.0)) return vals[0];
  if (p < 0.0) p = 0.0;
  if (p > 1.0) p = 1.0;
  double cum = 0.0, prev_cdf = 0.0, prev_val = vals[0];
  for (int k = 0; k < n; ++k) {
    const double w = wts[k];
    cum += w;
    const double cdf = (cum - 0.5 * w) / total;
    const double val = vals[k];
    if (p <= cdf) {
      if (k == 0 || cdf <= prev_cdf) return val;
      double frac = (p - prev_cdf) / (cdf - prev_cdf);
      if (frac < 0.0) frac = 0.0;
      if (frac > 1.0) frac = 1.0;
      return prev_val + frac * (val - prev_val);
    }
    prev_cdf = cdf;
    prev_val = val;
  }
  return vals[n - 1];
}

// cfg.bimodal_veto (see forward_drizzle_cuda.hpp / cpu_bimodal_veto): on the
// currently accepted subset of `recs` (index range [0, n_kept), already
// x-sorted, `on` decoded from the acc_lo/acc_hi bitmask), find the largest
// gap between consecutive accepted values. If it exceeds gap_sigma*mad and
// splits off a minority side of >= 2 candidates holding < half the accepted
// weight, clear that side's bits. Mirrors cpu_bimodal_veto exactly (same
// order, same comparisons) for CPU/CUDA parity. Returns candidates rejected.
__device__ unsigned int d_bimodal_veto(const V2ReservoirRecord *recs,
                                       unsigned int n_kept,
                                       unsigned long long &acc_lo,
                                       unsigned long long &acc_hi, double mad,
                                       double gap_sigma) {
  if (!(mad > 0.0) || !(gap_sigma > 0.0)) return 0;
  unsigned int on[kV2MaxResSlots];
  unsigned int n_on = 0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    const bool bit = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
    if (bit) on[n_on++] = i;
  }
  if (n_on < 4) return 0;
  unsigned int split = 0;
  double best_gap = -1.0;
  for (unsigned int k = 1; k < n_on; ++k) {
    const double gap = recs[on[k]].x - recs[on[k - 1]].x;
    if (gap > best_gap) {
      best_gap = gap;
      split = k;
    }
  }
  if (!(best_gap > gap_sigma * mad)) return 0;
  double w_lo = 0.0, w_hi = 0.0;
  for (unsigned int k = 0; k < split; ++k) w_lo += recs[on[k]].b;
  for (unsigned int k = split; k < n_on; ++k) w_hi += recs[on[k]].b;
  const unsigned int n_lo = split, n_hi = n_on - split;
  const bool lo_is_minority = w_lo < w_hi;
  const unsigned int minority_n = lo_is_minority ? n_lo : n_hi;
  const double minority_w = lo_is_minority ? w_lo : w_hi;
  const double total_w = w_lo + w_hi;
  if (minority_n < 2 || !(total_w > 0.0) || !(minority_w < 0.5 * total_w))
    return 0;
  unsigned int rejected = 0;
  const unsigned int lo_bound = lo_is_minority ? split : n_on;
  const unsigned int begin = lo_is_minority ? 0 : split;
  for (unsigned int k = begin; k < lo_bound; ++k) {
    const unsigned int idx = on[k];
    if (idx < 64) acc_lo &= ~(1ULL << idx);
    else acc_hi &= ~(1ULL << (idx - 64));
    ++rejected;
  }
  return rejected;
}

// Recomputes the weighted median/MAD/asymmetric bounds from scratch on the
// CURRENT accepted set of `recs` (x-sorted). Used only to refresh the
// full-frame pilot's FROZEN bounds after d_bimodal_veto has removed a
// minority cluster from a pilot: the standard clip loop's bounds were
// captured before the veto ran, so without this they would still admit the
// rejected population's non-pilot frames later. Mirrors the per-pass
// median/MAD computation exactly (same weighted-median tie rule, same
// deviation-order MAD), just without the accept/reject sweep (the veto
// already produced the accepted set this recomputes bounds for).
__device__ void d_recompute_bounds(const V2ReservoirRecord *recs,
                                   unsigned int n_kept,
                                   unsigned long long acc_lo,
                                   unsigned long long acc_hi, double s_low,
                                   double s_high, double &out_lower,
                                   double &out_upper, double &out_mad) {
  double total_w = 0.0;
  unsigned int on[kV2MaxResSlots];
  unsigned int n_on = 0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    const bool bit = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
    if (!bit) continue;
    on[n_on++] = i;
    total_w += recs[i].b;
  }
  if (n_on == 0 || !(total_w > 0.0)) return;
  double median = recs[on[n_on - 1]].x;
  {
    double cum = 0.0;
    for (unsigned int k = 0; k < n_on; ++k) {
      cum += recs[on[k]].b;
      if (cum >= total_w / 2.0) {
        median = recs[on[k]].x;
        break;
      }
    }
  }
  unsigned int ord[kV2MaxResSlots];
  for (unsigned int k = 0; k < n_on; ++k) ord[k] = on[k];
  for (unsigned int i = 1; i < n_on; ++i) {
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
  double mad = fabs(recs[ord[n_on - 1]].x - median);
  {
    double cum = 0.0;
    for (unsigned int k = 0; k < n_on; ++k) {
      cum += recs[ord[k]].b;
      if (cum >= total_w / 2.0) {
        mad = fabs(recs[ord[k]].x - median);
        break;
      }
    }
  }
  out_lower = median - s_low * mad;
  out_upper = median + s_high * mad;
  out_mad = mad;
}

// Gate-9 profile reduction over the accepted subset of the x-sorted
// reservoir (bit `i` of acc_lo/acc_hi = sorted slot i): the four profiles
// (uniform b; raw b*g_eff*q; fine b*g_eff*q0^fe; medium b*g_eff*q1^me) and
// the v2 alpha factors. Shared by k_finalize_v2 and k_finalize_v2_sfr_reduce.
// The artifact and residual percentiles are computed one after the other in
// ONE (value, weight) scratch pair --- both are filled in ascending slot order
// exactly as before, so the results are unchanged while the per-thread local
// footprint of the finalize kernels drops by 2 KiB.
__device__ void d_reduce_profiles(
    const V2ReservoirRecord *recs, const float4 *qvs, unsigned int n_kept,
    unsigned long long acc_lo, unsigned long long acc_hi,
    const V2FrameMetaDev *meta, unsigned long long meta_capacity,
    double fine_exp, double medium_exp, const AlphaConfidenceParams &alpha,
    double confidence, std::uint32_t contributors, std::uint64_t degraded,
    ForwardDrizzleV2ProfileResult &pr) {
  auto on = [&](unsigned int i) {
    return i < 64 ? ((acc_lo >> i) & 1ULL) != 0
                  : ((acc_hi >> (i - 64)) & 1ULL) != 0;
  };
  auto meta_at = [&](unsigned long long order) {
    V2FrameMetaDev m{};
    if (meta != nullptr && order < meta_capacity) m = meta[order];
    return m;
  };
  double uwx = 0.0, uw = 0.0, uw2 = 0.0;
  double rwx = 0.0, rw = 0.0, rw2 = 0.0;
  double fwx = 0.0, fw = 0.0, fw2 = 0.0;
  double mwx = 0.0, mw = 0.0, mw2 = 0.0;
  double b_total = 0.0, b_direct = 0.0;
  double pv[kV2MaxResSlots], pw[kV2MaxResSlots];
  int n_art = 0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    if (!on(i)) continue;
    const float4 qv = qvs[i];
    const V2FrameMetaDev m = meta_at(recs[i].order);
    double g = static_cast<double>(m.g_eff);
    if (!(g >= 0.0)) g = 0.0;  // malformed meta: zero weight
    const double b = recs[i].b;
    const double x = recs[i].x;
    const double wu = b;
    const double wr = b * g * static_cast<double>(qv.x);
    const double wf = b * g * pow(static_cast<double>(qv.y), fine_exp);
    const double wm = b * g * pow(static_cast<double>(qv.z), medium_exp);
    uwx += wu * x; uw += wu; uw2 += wu * wu;
    rwx += wr * x; rw += wr; rw2 += wr * wr;
    fwx += wf * x; fw += wf; fw2 += wf * wf;
    mwx += wm * x; mw += wm; mw2 += wm * wm;
    b_total += b;
    if (m.is_direct != 0u) b_direct += b;
    if (qv.w >= 0.0f) {
      double av = static_cast<double>(qv.w);
      if (av < 0.0) av = 0.0;
      if (av > 1.0) av = 1.0;
      pv[n_art] = av;
      pw[n_art] = b;
      ++n_art;
    }
  }
  auto write = [](ForwardDrizzleV2ProfileOutput &o, double wx, double w,
                  double w2) {
    if (!(w > 0.0)) return;
    o.value = static_cast<float>(wx / w);
    o.weight_sum = static_cast<float>(w);
    o.n_eff = static_cast<float>(w2 > 0.0 ? (w * w) / w2 : 0.0);
    o.support = 1;
  };
  write(pr.uniform, uwx, uw, uw2);
  write(pr.raw, rwx, rw, rw2);
  write(pr.fine, fwx, fw, fw2);
  write(pr.medium, mwx, mw, mw2);
  // v2 contract: a_separation is the gate-4 confidence, not the
  // quality-percentile separation. When EVERY contributor carried an
  // invalid sigma2 there is no noise model underwriting separation:
  // collapse to 0 (the degraded calibration boundary).
  double sep = confidence;
  if (contributors > 0 && degraded == contributors) sep = 0.0;
  if (sep < 0.0) sep = 0.0;
  if (sep > 1.0) sep = 1.0;
  pr.a_separation = static_cast<float>(sep);
  if (n_art >= alpha.min_artifact_contributors) {
    const double a_p10 = d_hazen_percentile(pv, pw, n_art, 0.10);
    pr.a_artifact = static_cast<float>(
        d_smoothstep(alpha.artifact_lo, alpha.artifact_hi, a_p10));
    pr.artifact_applicable = true;
  }
  // Residual factors of the same accepted set, same slot order.
  int n_res = 0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    if (!on(i)) continue;
    pv[n_res] = static_cast<double>(meta_at(recs[i].order).residual_factor);
    pw[n_res] = recs[i].b;
    ++n_res;
  }
  // Empty accepted set (e.g. shared_frame_rejection voted every candidate of
  // this channel out): compute_alpha_confidence_channel returns all-zero
  // factors on the CPU. The previous device code ran the percentile on zero
  // entries and read an uninitialized slot; keep the zero default instead.
  if (n_res == 0) return;
  const double direct_fraction = b_total > 0.0 ? b_direct / b_total : 0.0;
  const double residual_p20 = d_hazen_percentile(pv, pw, n_res, 0.20);
  const double reg_dir = d_smoothstep(alpha.direct_fraction_lo,
                                      alpha.direct_fraction_hi,
                                      direct_fraction);
  const double reg_res = d_smoothstep(alpha.residual_p20_lo,
                                      alpha.residual_p20_hi, residual_p20);
  // std::min NaN semantics: (b < a) ? b : a.
  pr.a_registration = static_cast<float>(reg_res < reg_dir ? reg_res : reg_dir);
}

// Result records reach the host as raw bytes (the band artifacts are
// hashed byte-wise, and the driver may hand a device-filled result vector
// to the CPU restart): zero the whole object representation --- padding
// included --- before the member initializers run, so a device record is
// byte-identical to a value-initialized host record.
template <typename T>
__device__ __forceinline__ void d_init_record(T &t) {
  memset(&t, 0, sizeof(T));
  new (&t) T{};
}

// Confidence-state mapping shared by the finalize family (gate-4 contract,
// mirrors conf_sigma2_of's consumer): modeled when the sigma2 sums are
// usable, n_eff fallback otherwise, no_source_support without weight.
__device__ __forceinline__ void d_finish_conf(ForwardDrizzleV2PixelResult &r,
                                              double cb, double cs,
                                              double cc) {
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
}

// Load the kept reservoir slots of (pixel, channel) pc and insertion-sort
// them by (x, order) --- identical ordering to the CPU oracle. With kWithQ
// the quality side array moves WITH the records (it is indexed by reservoir
// slot, not by sorted position); it is loaded only when resq is non-null.
template <bool kWithQ>
__device__ __forceinline__ void d_load_sorted_reservoir(
    const V2ReservoirRecord *res, const float4 *resq, long long pc,
    int res_slots, unsigned int n_kept, V2ReservoirRecord *recs,
    float4 *qvs) {
  for (unsigned int i = 0; i < n_kept; ++i) {
    recs[i] = res[pc * res_slots + i];
    if constexpr (kWithQ) {
      if (resq != nullptr) qvs[i] = resq[pc * res_slots + i];
    }
  }
  for (unsigned int i = 1; i < n_kept; ++i) {
    const V2ReservoirRecord v = recs[i];
    float4 qv = {};
    if constexpr (kWithQ) qv = qvs[i];
    unsigned int j = i;
    while (j > 0 &&
           (recs[j - 1].x > v.x ||
            (recs[j - 1].x == v.x && recs[j - 1].order > v.order))) {
      recs[j] = recs[j - 1];
      if constexpr (kWithQ) qvs[j] = qvs[j - 1];
      --j;
    }
    recs[j] = v;
    if constexpr (kWithQ) qvs[j] = qv;
  }
}

// Band-end finalize core, shared by k_finalize_v2 (kEmitClipState = false:
// clip + reduce + gate-9 profiles in one pass) and k_finalize_v2_sfr_build
// (true: clip only, then publish has_clip / the accepted bitmask / the clip
// bounds for the vote, reduce and full-frame kernels). Per (pixel, channel)
// it runs the bit-exact gate-3 clip on the reservoir (identical evaluation
// order to robust_frame_oracle_v2: (x, order) sort, cumulative-weight median
// at >= total/2, (|x-med|, order) MAD order, asymmetric bounds, early stop
// on an unchanged mask) and maps the gate-4 confidence states. kept >
// res_slots is the deterministic overflow fallback: uniform stream value,
// support retained. When `pout` is non-null the same accepted mask
// additionally reduces the four gate-9 profiles (uniform b; raw b*g_eff*q;
// fine b*g_eff*q0^fe; medium b*g_eff*q1^me) and the v2 alpha factors
// (a_separation = clamp01(confidence); artifact / registration from the
// percentile contract) --- the mask is shared by construction. Early-exit
// tids (no support / overflow / too few candidates) finalize identically in
// both instantiations; the build variant additionally clears has_clip.
template <bool kEmitClipState>
__device__ __forceinline__ void d_finalize_core(
    long long tid, int ncols, int nrows, int channels, int subpixels,
    int res_slots, unsigned long long frames_processed,
    unsigned long long meta_capacity, int min_candidates, int min_clip,
    int passes, double s_low, double s_high, const double *accA,
    const double *accB, const double *accB2, const double *confS,
    const double *confC, const unsigned int *contrib,
    const unsigned int *kept, const unsigned int *footprint,
    const unsigned short *supp, const unsigned long long *degraded,
    const V2ReservoirRecord *res, const float4 *resq,
    const V2FrameMetaDev *meta, double fine_exp, double medium_exp,
    const AlphaConfidenceParams &alpha, ForwardDrizzleV2ProfileResult *pout,
    ForwardDrizzleV2PixelResult *out, unsigned long long *dense_overlap,
    unsigned char *has_clip, unsigned long long *sfr_accepted,
    double *bounds, int bimodal_veto, double bimodal_veto_gap_sigma,
    unsigned long long *bimodal_rejected_total,
    unsigned long long *bimodal_pixels_total) {
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (tid >= nplane * channels) return;
  const long long pc = tid;
  const long long px = tid % nplane;
  const int c = static_cast<int>(tid / nplane);

  ForwardDrizzleV2PixelResult r;
  d_init_record(r);
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
  auto finish_uniform = [&](ForwardDrizzleV2RobustState state) {
    r.value = a_acc / b_acc;
    r.robust_state = static_cast<std::uint8_t>(state);
    d_finish_conf(r, b_acc, confS[pc], confC[pc]);
  };
  // Gate-9: every fallback profile carries the uniform stream value and zero
  // alpha factors (spec profile_production_contract.fallback); without
  // source support the profiles stay unsupported.
  auto emit_fallback_profiles = [&]() {
    if (pout == nullptr) return;
    ForwardDrizzleV2ProfileResult pr;
    d_init_record(pr);
    if (b_acc > 0.0) {
      const float v = static_cast<float>(a_acc / b_acc);
      const float w = static_cast<float>(b_acc);
      const float ne = static_cast<float>(r.n_eff);
      for (auto *o : {&pr.uniform, &pr.raw, &pr.fine, &pr.medium}) {
        o->value = v;
        o->weight_sum = w;
        o->n_eff = ne;
        o->support = 1;
      }
    }
    pout[pc] = pr;
  };
  auto early_exit = [&]() {
    emit_fallback_profiles();
    out[pc] = r;
    if constexpr (kEmitClipState) has_clip[pc] = 0;
  };

  if (!(b_acc > 0.0)) {
    r.robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::no_source_support);
    d_finish_conf(r, 0.0, 0.0, 0.0);
    early_exit();
    return;
  }
  if (n_kept > static_cast<unsigned int>(res_slots)) {
    finish_uniform(ForwardDrizzleV2RobustState::reservoir_overflow_fallback);
    early_exit();
    return;
  }
  if (r.contributors < static_cast<unsigned int>(min_candidates) ||
      n_kept < static_cast<unsigned int>(min_clip)) {
    finish_uniform(ForwardDrizzleV2RobustState::too_few_candidates_fallback);
    early_exit();
    return;
  }

  // The build variant never reduces profiles: no quality side array.
  V2ReservoirRecord recs[kV2MaxResSlots];
  float4 qvs[kEmitClipState ? 1 : kV2MaxResSlots];
  d_load_sorted_reservoir<!kEmitClipState>(res, resq, pc, res_slots, n_kept,
                                           recs, qvs);
  unsigned long long acc_lo = ~0ULL, acc_hi = ~0ULL;  // accepted bitmask
  double last_lower = 0.0, last_upper = 0.0, last_mad = -1.0;
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
    last_lower = lower;
    last_upper = upper;
    last_mad = mad;
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
  // last_mad > 0 implies a clip pass ran; d_bimodal_veto itself rejects
  // nothing for a non-positive MAD, so this is the same gate as before.
  if (bimodal_veto != 0 && last_mad > 0.0) {
    const unsigned int rej = d_bimodal_veto(recs, n_kept, acc_lo, acc_hi,
                                            last_mad, bimodal_veto_gap_sigma);
    if (rej > 0) {
      atomicAdd(bimodal_rejected_total,
               static_cast<unsigned long long>(rej));
      atomicAdd(bimodal_pixels_total, 1ULL);
      if constexpr (kEmitClipState) {
        // The frozen bounds (below) were captured from the standard clip's
        // last pass, before this veto ran: without refreshing them here, a
        // full-frame pilot's frozen bounds would still admit the just-removed
        // population's non-pilot frames later (see d_recompute_bounds).
        if (bounds != nullptr)
          d_recompute_bounds(recs, n_kept, acc_lo, acc_hi, s_low, s_high,
                             last_lower, last_upper, last_mad);
      }
    }
  }
  if constexpr (kEmitClipState) {
    // .value/.robust_state/.confidence/.confidence_state are not yet valid
    // (the accepted set can still change in kernel B) -- everything else
    // computed above (contributors, conf_degraded, b, n_eff, the *_fraction
    // fields) is already final and must reach kernel C, which starts from
    // out[pc] rather than recomputing it.
    if (bounds != nullptr) {
      bounds[3 * pc] = last_lower;
      bounds[3 * pc + 1] = last_upper;
      bounds[3 * pc + 2] = last_mad;
    }
    out[pc] = r;
    has_clip[pc] = 1;
    sfr_accepted[2 * pc] = acc_lo;
    sfr_accepted[2 * pc + 1] = acc_hi;
    return;
  } else {
    double ca = 0.0, cb = 0.0, cs = 0.0, cc = 0.0;
    for (unsigned int i = 0; i < n_kept; ++i) {
      const bool on = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
      if (!on) continue;
      ca += recs[i].b * recs[i].x;
      cb += recs[i].b;
      const double s2 = recs[i].sigma2;
      if (isfinite(s2) && s2 > 0.0) {
        cs += recs[i].b * sqrt(s2);
        cc += recs[i].b * recs[i].b * s2;
      }
    }
    r.value = cb > 0.0 ? ca / cb : 0.0;
    r.robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
    d_finish_conf(r, cb, cs, cc);

    if (pout != nullptr) {
      ForwardDrizzleV2ProfileResult pr;
    d_init_record(pr);
      d_reduce_profiles(recs, qvs, n_kept, acc_lo, acc_hi, meta,
                        meta_capacity, fine_exp, medium_exp, alpha,
                        r.confidence, r.contributors, degraded[pc], pr);
      pout[pc] = pr;
    }
    out[pc] = r;
  }
}

// Band-end finalize (no shared frame rejection / full-frame estimator):
// one thread per (pixel, channel), clip + reduce in one pass. See
// d_finalize_core.
__global__ void k_finalize_v2(
    int ncols, int nrows, int channels, int subpixels, int res_slots,
    unsigned long long frames_processed, unsigned long long meta_capacity,
    int min_candidates, int min_clip,
    int passes, double s_low, double s_high,
    const double *accA, const double *accB, const double *accB2,
    const double *confS, const double *confC,
    const unsigned int *contrib, const unsigned int *kept,
    const unsigned int *footprint, const unsigned short *supp,
    const unsigned long long *degraded, const V2ReservoirRecord *res,
    const float4 *resq, const V2FrameMetaDev *meta,
    double fine_exp, double medium_exp, AlphaConfidenceParams alpha,
    ForwardDrizzleV2ProfileResult *pout,
    ForwardDrizzleV2PixelResult *out, unsigned long long *dense_overlap,
    int bimodal_veto, double bimodal_veto_gap_sigma,
    unsigned long long *bimodal_rejected_total,
    unsigned long long *bimodal_pixels_total) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  d_finalize_core<false>(
      tid, ncols, nrows, channels, subpixels, res_slots, frames_processed,
      meta_capacity, min_candidates, min_clip, passes, s_low, s_high, accA,
      accB, accB2, confS, confC, contrib, kept, footprint, supp, degraded,
      res, resq, meta, fine_exp, medium_exp, alpha, pout, out, dense_overlap,
      nullptr, nullptr, nullptr, bimodal_veto, bimodal_veto_gap_sigma,
      bimodal_rejected_total, bimodal_pixels_total);
}

// shared_frame_rejection (config::ReconstructionClippingConfig::
// shared_frame_rejection, mirrored on ForwardDrizzleV2KernelConfig) device
// path: reconciles each channel's independent sigma-clip decision against
// the other channels that saw the same frame as a candidate at the same
// pixel, since R/G/B are sampled from disjoint CFA sensor positions and can
// otherwise reject a frame in one channel while keeping it in another --
// producing anti-correlated chroma noise. Split into three kernels (build /
// vote / reduce) instead of one thread per pixel doing all channels: a
// single thread holding the full per-channel clip state (V2ReservoirRecord
// recs[128] + float4 qvs[128], per channel) for up to 3 channels would
// multiply k_finalize_v2's already-heavy per-thread local-memory footprint
// threefold. Three launches give a free global barrier between phases
// instead, and only the per-pixel vote kernel (B) needs cross-channel data,
// kept to the minimum (x, order) it actually needs for that.
struct V2SortKey {
  double x;
  unsigned long long order;
};

__device__ void v2_sort_keys(V2SortKey *keys, unsigned int n) {
  for (unsigned int i = 1; i < n; ++i) {
    const V2SortKey v = keys[i];
    unsigned int j = i;
    while (j > 0 && (keys[j - 1].x > v.x ||
                     (keys[j - 1].x == v.x && keys[j - 1].order > v.order))) {
      keys[j] = keys[j - 1];
      --j;
    }
    keys[j] = v;
  }
}

// Kernel A (build): identical to k_finalize_v2 through the clip passes for
// every (pixel, channel) tid (same d_finalize_core instantiation path).
// Early-exit tids (no support / overflow / too few candidates) finalize
// exactly as k_finalize_v2 does and are done. Clip-eligible tids stop right
// after the clip loop instead of reducing: they record has_clip=1 and the
// accepted bitmask (2 x u64, res_slots can reach 128 = 2*reservoir_size)
// for kernel B to read and possibly revise, and kernel C to reduce from.
__global__ void k_finalize_v2_sfr_build(
    int ncols, int nrows, int channels, int subpixels, int res_slots,
    unsigned long long frames_processed,
    int min_candidates, int min_clip,
    int passes, double s_low, double s_high,
    const double *accA, const double *accB, const double *accB2,
    const double *confS, const double *confC,
    const unsigned int *contrib, const unsigned int *kept,
    const unsigned int *footprint, const unsigned short *supp,
    const unsigned long long *degraded, const V2ReservoirRecord *res,
    ForwardDrizzleV2ProfileResult *pout,
    ForwardDrizzleV2PixelResult *out, unsigned long long *dense_overlap,
    unsigned char *has_clip, unsigned long long *sfr_accepted,
    double *bounds, int bimodal_veto, double bimodal_veto_gap_sigma,
    unsigned long long *bimodal_rejected_total,
    unsigned long long *bimodal_pixels_total) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const AlphaConfidenceParams alpha{};
  d_finalize_core<true>(
      tid, ncols, nrows, channels, subpixels, res_slots, frames_processed,
      0ULL, min_candidates, min_clip, passes, s_low, s_high, accA, accB,
      accB2, confS, confC, contrib, kept, footprint, supp, degraded, res,
      nullptr, nullptr, 0.0, 0.0, alpha, pout, out, dense_overlap, has_clip,
      sfr_accepted, bounds, bimodal_veto, bimodal_veto_gap_sigma,
      bimodal_rejected_total, bimodal_pixels_total);
}

// Kernel B (vote): one thread per NATIVE PIXEL (not pixel*channel), looping
// its up to 3 channels itself -- consensus is inherently a cross-channel
// operation, so the channels of one pixel must be visible to the same
// thread. Only (x, order) per candidate is needed to reproduce kernel A's
// sort (and so map each accepted-bitmask bit back to its frame_order), not
// the full V2ReservoirRecord/quality state kernel A and C need -- keeping
// this thread's local footprint well under three full copies of theirs.
__global__ void k_finalize_v2_sfr_vote(
    int ncols, int nrows, int channels, int res_slots,
    double consensus_threshold, const unsigned int *kept,
    const V2ReservoirRecord *res, const unsigned char *has_clip,
    unsigned long long *sfr_accepted) {
  const long long px =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (px >= nplane || channels <= 1) return;

  V2SortKey keys[3][kV2MaxResSlots];
  unsigned long long acc_lo[3] = {0, 0, 0}, acc_hi[3] = {0, 0, 0};
  unsigned int nk[3] = {0, 0, 0};
  bool present[3] = {false, false, false};
  for (int ch = 0; ch < channels; ++ch) {
    const long long pc = static_cast<long long>(ch) * nplane + px;
    if (!has_clip[pc]) continue;
    present[ch] = true;
    nk[ch] = kept[pc];
    for (unsigned int i = 0; i < nk[ch]; ++i) {
      const V2ReservoirRecord &rec = res[pc * res_slots + i];
      keys[ch][i] = V2SortKey{rec.x, rec.order};
    }
    v2_sort_keys(keys[ch], nk[ch]);
    acc_lo[ch] = sfr_accepted[2 * pc];
    acc_hi[ch] = sfr_accepted[2 * pc + 1];
  }

  auto bit = [](unsigned long long lo, unsigned long long hi,
               unsigned int i) {
    return i < 64 ? (lo >> i) & 1ULL : (hi >> (i - 64)) & 1ULL;
  };

  // Per (channel, sorted-slot) vote tally: how many channels saw this exact
  // candidate's frame_order at all, and how many of those rejected it.
  // Bounded by channels(<=3) * res_slots(<=128) candidates per pixel -- a
  // small, fixed nested search, no hash map needed.
  for (int ch = 0; ch < channels; ++ch) {
    if (!present[ch]) continue;
    for (unsigned int i = 0; i < nk[ch]; ++i) {
      if (!bit(acc_lo[ch], acc_hi[ch], i))
        continue;  // already rejected by its own channel's clip
      const unsigned long long order = keys[ch][i].order;
      int seen = 0, rejected = 0;
      for (int och = 0; och < channels; ++och) {
        if (!present[och]) continue;
        for (unsigned int j = 0; j < nk[och]; ++j) {
          if (keys[och][j].order != order) continue;
          ++seen;
          if (!bit(acc_lo[och], acc_hi[och], j)) ++rejected;
          break;  // a frame contributes at most one candidate per channel
        }
      }
      if (seen <= 1) continue;  // only this channel saw it: nothing to vote
      const double frac =
          static_cast<double>(rejected) / static_cast<double>(seen);
      if (frac > consensus_threshold) {
        if (i < 64) acc_lo[ch] &= ~(1ULL << i);
        else acc_hi[ch] &= ~(1ULL << (i - 64));
      }
    }
  }

  for (int ch = 0; ch < channels; ++ch) {
    if (!present[ch]) continue;
    const long long pc = static_cast<long long>(ch) * nplane + px;
    sfr_accepted[2 * pc] = acc_lo[ch];
    sfr_accepted[2 * pc + 1] = acc_hi[ch];
  }
}

// Kernel C (reduce): identical grid/threading to kernel A and to
// k_finalize_v2 (one thread per (pixel, channel)); for has_clip tids only
// (early-exit tids were already finalized by kernel A), reloads and
// re-sorts the reservoir exactly as kernel A did -- deterministic, so this
// reproduces the same bit-index <-> record mapping without kernel A having
// to persist the sorted order -- then reduces using the (possibly
// consensus-revised) accepted bitmask instead of recomputing the clip.
__global__ void k_finalize_v2_sfr_reduce(
    int ncols, int nrows, int channels, int res_slots,
    unsigned long long meta_capacity,
    const unsigned int *kept, const V2ReservoirRecord *res,
    const float4 *resq, const V2FrameMetaDev *meta,
    double fine_exp, double medium_exp, AlphaConfidenceParams alpha,
    const unsigned char *has_clip, const unsigned long long *sfr_accepted,
    ForwardDrizzleV2ProfileResult *pout, ForwardDrizzleV2PixelResult *out) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (tid >= nplane * channels) return;
  const long long pc = tid;
  if (!has_clip[pc]) return;  // kernel A already wrote out[pc]/pout[pc]

  const unsigned int n_kept = kept[pc];
  V2ReservoirRecord recs[kV2MaxResSlots];
  float4 qvs[kV2MaxResSlots];
  d_load_sorted_reservoir<true>(res, resq, pc, res_slots, n_kept, recs, qvs);
  const unsigned long long acc_lo = sfr_accepted[2 * pc];
  const unsigned long long acc_hi = sfr_accepted[2 * pc + 1];
  auto bit = [&](unsigned int i) {
    return i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
  };

  ForwardDrizzleV2PixelResult r = out[pc];
  double ca = 0.0, cb = 0.0, cs = 0.0, cc = 0.0;
  for (unsigned int i = 0; i < n_kept; ++i) {
    if (!bit(i)) continue;
    ca += recs[i].b * recs[i].x;
    cb += recs[i].b;
    const double s2 = recs[i].sigma2;
    if (isfinite(s2) && s2 > 0.0) {
      cs += recs[i].b * sqrt(s2);
      cc += recs[i].b * recs[i].b * s2;
    }
  }
  r.value = cb > 0.0 ? ca / cb : 0.0;
  r.robust_state = static_cast<std::uint8_t>(
      ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
  d_finish_conf(r, cb, cs, cc);

  if (pout != nullptr) {
    ForwardDrizzleV2ProfileResult pr;
    d_init_record(pr);
    d_reduce_profiles(recs, qvs, n_kept, acc_lo, acc_hi, meta, meta_capacity,
                      fine_exp, medium_exp, alpha, r.confidence,
                      r.contributors, r.conf_degraded, pr);
    pout[pc] = pr;
  }
  out[pc] = r;
}

// Full-frame estimator pilot barrier, kernel D: after sfr_build (bounds
// requested) and sfr_vote, every clip-eligible (pixel, channel) tid holds the
// pilot's accepted bitmask (sorted-position indexed) and its final clip
// bounds. This freezes the bounds and folds the accepted pilot candidates
// into the full-frame sums in (x, order) order, exactly like the CPU
// end_pilot(). Counters: [2] degenerate pilots, [3] pixel-channels without
// bounds that had candidates.
__global__ void k_full_seed(
    int ncols, int nrows, int channels, int res_slots, const unsigned int *kept,
    const unsigned int *contrib, const V2ReservoirRecord *res,
    const float4 *resq, const V2FrameMetaDev *meta,
    unsigned long long meta_capacity, double fine_exp, double medium_exp,
    const unsigned char *has_clip, const unsigned long long *sfr_accepted,
    const double *bounds, V2FullAcc *full, unsigned long long *counters) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (tid >= nplane * channels) return;
  const long long pc = tid;
  if (!has_clip[pc]) {
    if (contrib[pc] > 0) atomicAdd(counters + 3, 1ULL);
    return;
  }
  const unsigned int n_kept = kept[pc];
  V2ReservoirRecord recs[kV2MaxResSlots];
  float4 qvs[kV2MaxResSlots];
  d_load_sorted_reservoir<true>(res, resq, pc, res_slots, n_kept, recs, qvs);
  const unsigned long long acc_lo = sfr_accepted[2 * pc];
  const unsigned long long acc_hi = sfr_accepted[2 * pc + 1];
  V2FullAcc f = full[pc];
  const double mad = bounds[3 * pc + 2];
  if (mad > 0.0) {
    f.state = 1;
    f.lo = bounds[3 * pc];
    f.hi = bounds[3 * pc + 1];
  } else {
    f.state = 2;
    atomicAdd(counters + 2, 1ULL);
  }
  for (unsigned int i = 0; i < n_kept; ++i) {
    const bool on = i < 64 ? (acc_lo >> i) & 1ULL : (acc_hi >> (i - 64)) & 1ULL;
    if (!on) continue;
    V2FrameMetaDev m{};
    if (meta != nullptr && recs[i].order < meta_capacity) m = meta[recs[i].order];
    d_full_add(f, recs[i].b, recs[i].x, recs[i].sigma2,
               static_cast<double>(m.g_eff), static_cast<double>(qvs[i].x),
               static_cast<double>(qvs[i].y), static_cast<double>(qvs[i].z),
               fine_exp, medium_exp);
  }
  full[pc] = f;
}

// Full-frame estimator finalize override, kernel E (after sfr_reduce): the
// device twin of the CPU finalize override. Counters: [0] non-pilot accepted,
// [1] non-pilot rejected.
__global__ void k_full_apply(
    int ncols, int nrows, int channels, const unsigned int *contrib,
    const unsigned long long *degraded, const V2FullAcc *full,
    ForwardDrizzleV2PixelResult *out, ForwardDrizzleV2ProfileResult *pout,
    unsigned long long *counters) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long nplane = static_cast<long long>(ncols) * nrows;
  if (tid >= nplane * channels) return;
  const long long pc = tid;
  const V2FullAcc f = full[pc];
  if (f.n_acc != 0) atomicAdd(counters + 0, static_cast<unsigned long long>(f.n_acc));
  if (f.n_rej != 0) atomicAdd(counters + 1, static_cast<unsigned long long>(f.n_rej));
  ForwardDrizzleV2PixelResult r = out[pc];
  if (f.state == 0 ||
      r.robust_state !=
          static_cast<std::uint8_t>(
              ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip))
    return;
  if (f.state == 2) {
    r.robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::degenerate_scale);
    out[pc] = r;
    return;
  }
  if (!(f.cb > 0.0)) return;
  r.value = f.ca / f.cb;
  r.robust_state = static_cast<std::uint8_t>(
      ForwardDrizzleV2RobustState::primary_reservoir_pilot_full_frame);
  r.n_eff = f.cb2 > 0.0 ? f.cb * f.cb / f.cb2 : 0.0;
  if (f.cc > 0.0 && isfinite(f.cc)) {
    r.confidence = (f.cs * f.cs) / (f.cs * f.cs + f.cc);
    r.confidence_state =
        static_cast<std::uint8_t>(ForwardDrizzleV2ConfidenceState::modeled);
  } else {
    r.confidence = r.n_eff > 0.0 ? r.n_eff / (r.n_eff + 1.0) : 0.0;
    r.confidence_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2ConfidenceState::fallback_n_eff);
  }
  out[pc] = r;
  if (pout == nullptr) return;
  ForwardDrizzleV2ProfileResult pr = pout[pc];
  ForwardDrizzleV2ProfileOutput *outs[4] = {&pr.uniform, &pr.raw, &pr.fine,
                                            &pr.medium};
  for (int k = 0; k < 4; ++k) {
    *outs[k] = ForwardDrizzleV2ProfileOutput{};
    if (!(f.w[k] > 0.0)) continue;
    outs[k]->value = static_cast<float>(f.wx[k] / f.w[k]);
    outs[k]->weight_sum = static_cast<float>(f.w[k]);
    outs[k]->n_eff =
        static_cast<float>(f.w2[k] > 0.0 ? (f.w[k] * f.w[k]) / f.w2[k] : 0.0);
    outs[k]->support = 1;
  }
  double sep = (r.contributors > 0 && degraded[pc] == r.contributors)
                   ? 0.0
                   : r.confidence;
  if (sep < 0.0) sep = 0.0;
  if (sep > 1.0) sep = 1.0;
  pr.a_separation = static_cast<float>(sep);
  pout[pc] = pr;
}

// Host-side executability contract of a local-warp descriptor. Everything
// the CPU oracle can express (invalid model, non-finite values, negative
// margin or depth) degrades on device to the oracle's own per-sample
// failure path; only a subdivision depth beyond the implicit 21-node tree
// is not executable and rejected here.
bool local_warp_valid(const ForwardDrizzleV2LocalWarp &w) {
  return w.max_subdivision_depth <= 2;
}

}  // namespace

// forward_drizzle_cuda_polygon_rect_area_batch (clip parity harness) lives
// in forward_drizzle_cuda_test_harness.cu; it shares the clip code through
// forward_drizzle_cuda_clip.cuh.

bool forward_drizzle_cuda_affine_coverage_gather(
    const double affine6[6], const double inverse6[6], int internal_scale,
    double half, int target_x_begin, int target_y_begin, int target_cols,
    int target_rows, int source_w, int source_h, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, double *out_b) {
  if (!affine6 || !inverse6 || !out_b || internal_scale <= 0 ||
      target_cols <= 0 || target_rows <= 0 || source_w <= 0 ||
      source_h <= 0)
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
  const std::size_t plane_bytes =
      static_cast<std::size_t>(channels) * cells * sizeof(double);
  // Grow-only static output plane + dedicated non-blocking stream + pinned
  // host staging: the SAMPLING_GEOMETRY caller serializes every call
  // through its own mutex (this function's statics are NOT thread-safe), and
  // per-call cudaMalloc/cudaFree of a ~30MB buffer dominated the old records
  // pipeline's call overhead. Nothing here touches the legacy default stream
  // or calls cudaDeviceSynchronize, so a concurrently open forward-drizzle
  // workspace stream is never stalled. Capacity is tracked in BYTES: a MONO
  // call (1 channel) followed by a CFA call (3 channels) over the same cell
  // count needs three times the plane.
  static cudaStream_t cov_stream = nullptr;
  static double *d_b = nullptr;
  static std::size_t d_b_capacity_bytes = 0;
  static double *h_b = nullptr;  // pinned; null => pageable download
  static std::size_t h_b_capacity_bytes = 0;
  if (cov_stream == nullptr &&
      cudaStreamCreateWithFlags(&cov_stream, cudaStreamNonBlocking) !=
          cudaSuccess) {
    cudaGetLastError();
    cov_stream = nullptr;
    return false;
  }
  if (plane_bytes > d_b_capacity_bytes) {
    cudaFree(d_b);
    d_b = nullptr;
    d_b_capacity_bytes = 0;
    if (cudaMalloc(&d_b, plane_bytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    d_b_capacity_bytes = plane_bytes;
  }
  if (plane_bytes > h_b_capacity_bytes) {
    // Best effort: without pinned staging the download stays pageable.
    if (h_b != nullptr) cudaFreeHost(h_b);
    h_b = nullptr;
    h_b_capacity_bytes = 0;
    void *p = nullptr;
    if (cudaMallocHost(&p, plane_bytes) == cudaSuccess) {
      h_b = static_cast<double *>(p);
      h_b_capacity_bytes = plane_bytes;
    } else {
      cudaGetLastError();
    }
  }
  bool ok = true;
  const bool force_gather =
      std::getenv("TC_COVERAGE_FORCE_GATHER") != nullptr;
  // Scatter is only race-free when every (channel, cell) pair has at most one
  // contributing droplet per parity-class launch. A target cell can straddle
  // the gap between two same-parity droplets only if the gap (measured in
  // target space along its worst-case direction) is at least the cell
  // diagonal sqrt(2). Same-parity source spacing is 2 px in both axes, so
  // the source-space gap is (2 - 2*half); the forward affine can shrink it
  // --- use its smallest singular value. MONO has no parity sublattices
  // (every pixel is channel 0), so it always takes the gather path.
  const double t2 = affine6[0] * affine6[0] + affine6[1] * affine6[1] +
                    affine6[3] * affine6[3] + affine6[4] * affine6[4];
  const double det =
      std::abs(affine6[0] * affine6[4] - affine6[1] * affine6[3]);
  const double disc =
      std::sqrt(std::max(0.0, t2 * t2 - 4.0 * det * det));
  const double sigma_min = std::sqrt(std::max(0.0, 0.5 * (t2 - disc)));
  const double gap_cells =
      (2.0 - 2.0 * half) * sigma_min * internal_scale;
  const bool scatter_ok =
      !mono && half <= 0.5 && gap_cells >= std::sqrt(2.0);
  if (scatter_ok && !force_gather) {
    // Scatter direction: one thread per source sample of a parity class,
    // accumulating droplet areas directly into the dense plane (see the
    // kernel comment for why this is race-free and bit-identical). Needs the
    // plane zeroed and the conservative inverse-mapped source band.
    ok = cudaMemsetAsync(d_b, 0, plane_bytes, cov_stream) == cudaSuccess;
    if (!ok) return false;
    double sx_lo = DBL_MAX, sx_hi = -DBL_MAX, sy_lo = DBL_MAX,
           sy_hi = -DBL_MAX;
    for (double dx : {static_cast<double>(target_x_begin) / internal_scale,
                      static_cast<double>(target_x_begin + target_cols) /
                          internal_scale})
      for (double dy : {static_cast<double>(target_y_begin) / internal_scale,
                        static_cast<double>(target_y_begin + target_rows) /
                            internal_scale}) {
        const double sx =
            inverse6[0] * dx + inverse6[1] * dy + inverse6[2];
        const double sy =
            inverse6[3] * dx + inverse6[4] * dy + inverse6[5];
        sx_lo = fmin(sx_lo, sx); sx_hi = fmax(sx_hi, sx);
        sy_lo = fmin(sy_lo, sy); sy_hi = fmax(sy_hi, sy);
      }
    const int band_sy0 = max(0, static_cast<int>(floor(sy_lo - 1)));
    const int band_sy1 = min(source_h, static_cast<int>(ceil(sy_hi + 1)));
    const int band_sx0 = max(0, static_cast<int>(floor(sx_lo - 1)));
    const int band_sx1 = min(source_w, static_cast<int>(ceil(sx_hi + 1)));
    if (band_sy1 <= band_sy0 || band_sx1 <= band_sx0) {
      std::memset(out_b, 0, plane_bytes);  // frame misses the band entirely
      return true;
    }
    // Four sequential parity-class launches (same stream -> serialized). The
    // plane was memset once; each launch accumulates its sublattice. One
    // parity class holds at most ceil(rows/2) * ceil(cols/2) samples, so the
    // grid covers that instead of the whole band (the kernel bounds-checks).
    const long long total =
        static_cast<long long>((band_sy1 - band_sy0 + 1) / 2) *
        ((band_sx1 - band_sx0 + 1) / 2);
    const int block = 128;
    const int grid = static_cast<int>((total + block - 1) / block);
    for (int py = 0; py < 2 && ok; ++py)
      for (int px = 0; px < 2; ++px) {
        k_affine_coverage_scatter<<<grid, block, 0, cov_stream>>>(
            affine6[0], affine6[1], affine6[2], affine6[3], affine6[4],
            affine6[5], static_cast<double>(internal_scale), half,
            target_x_begin, target_y_begin, target_cols, target_rows,
            band_sy0, band_sy1, band_sx0, band_sx1, bayer_pattern,
            cfa_origin_x, cfa_origin_y, px, py, d_b);
      }
    ok = cudaGetLastError() == cudaSuccess;
  } else {
    const int block = 128;
    const int grid = (static_cast<int>(cells) + block - 1) / block;
    k_affine_coverage_gather<<<grid, block, 0, cov_stream>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        inverse6[0], inverse6[1], inverse6[2], inverse6[3], inverse6[4],
        inverse6[5], internal_scale, half, target_x_begin, target_y_begin,
        target_cols, target_rows, source_w, source_h, bayer_pattern,
        cfa_origin_x, cfa_origin_y, mono, d_b);
    ok = cudaGetLastError() == cudaSuccess;
  }
  // Download on the same stream (ordered after the launches) and wait for
  // this call's work only; the pinned bounce buffer keeps the copy a true
  // DMA transfer.
  double *dst = h_b != nullptr ? h_b : out_b;
  ok = ok &&
       cudaMemcpyAsync(dst, d_b, plane_bytes, cudaMemcpyDeviceToHost,
                       cov_stream) == cudaSuccess;
  ok = cudaStreamSynchronize(cov_stream) == cudaSuccess && ok;
  if (ok && dst != out_b) std::memcpy(out_b, h_b, plane_bytes);
  return ok;
}


bool forward_drizzle_cuda_local_dense_scatter(
    const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
    int internal_scale, double half, int target_cols, int target_rows,
    int source_w, int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, int canvas_w_native,
    int canvas_h_native, double *out_a, double *out_b_src, double *out_b_geo,
    unsigned long long *out_positive_overlaps,
    unsigned long long *out_discarded) {
  if (!affine6 || !source_values || !out_a || !out_b_src || !out_b_geo ||
      internal_scale <= 0 || target_cols <= 0 || target_rows <= 0 ||
      source_w <= 0 || source_h <= 0 || canvas_w_native <= 0 ||
      canvas_h_native <= 0)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (!std::isfinite(half) || !(half > 0.0)) return false;
  if (!local_warp_valid(warp)) return false;
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
  double *d_a = nullptr, *d_bs = nullptr, *d_bg = nullptr;
  unsigned long long *d_scalars = nullptr;
  bool ok = cudaMalloc(&d_source, src_bytes) == cudaSuccess &&
            cudaMalloc(&d_a, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_bs, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_bg, plane_bytes) == cudaSuccess &&
            cudaMalloc(&d_scalars, 2 * sizeof(unsigned long long)) ==
                cudaSuccess;
  if (ok)
    ok = cudaMemcpy(d_source, source_values, src_bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemset(d_a, 0, plane_bytes) == cudaSuccess &&
         cudaMemset(d_bs, 0, plane_bytes) == cudaSuccess &&
         cudaMemset(d_bg, 0, plane_bytes) == cudaSuccess &&
         cudaMemset(d_scalars, 0, 2 * sizeof(unsigned long long)) ==
             cudaSuccess;
  if (ok) {
    const int block = 128;
    const int grid = (static_cast<int>(source_n) + block - 1) / block;
    V2Window win{};
    win.buf_w = source_w;
    win.buf_h = source_h;
    win.act_w = source_w;
    win.act_h = source_h;
    win.src_w = source_w;
    win.src_h = source_h;
    k_scatter_v2_local<<<grid, block>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4], affine6[5],
        warp, static_cast<double>(internal_scale), half, target_cols,
        target_rows, win, d_source, nullptr, V2Sigma2Model{}, bayer_pattern,
        cfa_origin_x, cfa_origin_y, mono ? 1 : 0, canvas_w_native,
        canvas_h_native, 0.0, 0.0, d_a, d_bs, d_bg, nullptr, V2QualityIO{},
        d_scalars, d_scalars + 1);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
  }
  unsigned long long scalars[2] = {0, 0};
  if (ok)
    ok = cudaMemcpy(out_a, d_a, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(out_b_src, d_bs, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(out_b_geo, d_bg, plane_bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         cudaMemcpy(scalars, d_scalars, sizeof(scalars),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_source);
  cudaFree(d_a);
  cudaFree(d_bs);
  cudaFree(d_bg);
  cudaFree(d_scalars);
  if (ok) {
    if (out_positive_overlaps) *out_positive_overlaps = scalars[0];
    if (out_discarded) *out_discarded = scalars[1];
  }
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

// --- Gate-6 prototype workspace --------------------------------------------

namespace {

// Owning device buffer: cudaMalloc in alloc(), cudaFree in the destructor.
// Converts implicitly to T* so kernel and runtime call sites read like raw
// pointers; pointer arithmetic goes through operator+ / get().
template <typename T>
class DevBuf {
 public:
  DevBuf() = default;
  ~DevBuf() { reset(); }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
  // Allocates `count` elements and returns the cudaMalloc status; count == 0
  // leaves the buffer null and succeeds.
  cudaError_t alloc(std::size_t count) {
    reset();
    if (count == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t e = cudaMalloc(&p, count * sizeof(T));
    if (e == cudaSuccess) p_ = static_cast<T *>(p);
    return e;
  }
  void reset() {
    if (p_ != nullptr) cudaFree(p_);
    p_ = nullptr;
  }
  T *get() const { return p_; }
  operator T *() const { return p_; }
  T *operator+(std::ptrdiff_t n) const { return p_ + n; }

 private:
  T *p_ = nullptr;
};

// Owning page-locked host buffer (cudaMallocHost / cudaFreeHost), used for
// the upload staging slots.
template <typename T>
class PinnedBuf {
 public:
  PinnedBuf() = default;
  ~PinnedBuf() { reset(); }
  PinnedBuf(const PinnedBuf &) = delete;
  PinnedBuf &operator=(const PinnedBuf &) = delete;
  cudaError_t alloc(std::size_t count) {
    reset();
    if (count == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t e = cudaMallocHost(&p, count * sizeof(T));
    if (e == cudaSuccess) p_ = static_cast<T *>(p);
    return e;
  }
  void reset() {
    if (p_ != nullptr) cudaFreeHost(p_);
    p_ = nullptr;
  }
  T *get() const { return p_; }
  operator T *() const { return p_; }

 private:
  T *p_ = nullptr;
};

// Zero the internal rows [off/icols, (off+len)/icols) of every (plane,
// channel) pair of the contiguous frame-plane block: pair p*channels+c
// starts at pair * iplane doubles; `off` and `len` are in doubles. One
// launch replaces up to 9 x channels row-window memsets.
__global__ void k_clear_plane_rows(double *block, int pairs, long long iplane,
                                   long long off, long long len) {
  const long long total = static_cast<long long>(pairs) * len;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       t < total; t += stride)
    block[(t / len) * iplane + off + (t % len)] = 0.0;
}

// Internal-row window [r0, r1) of the band that bounds every cell a frame's
// scatter can touch, from the frame's extreme band-local internal y
// coordinates. Padded by one row on each side and aligned to native rows so
// (a) it stays a superset regardless of host/device rounding differences
// (the kernels' own floor/ceil clamp is the authority, this window only has
// to contain it) and (b) the fold reads only rows this frame cleared. A
// non-finite or inverted range yields the whole band.
struct V2RowWindow {
  int r0 = 0;  // internal rows
  int r1 = 0;
  int ny0 = 0;  // native rows
  int ny1 = 0;
};

V2RowWindow v2_row_window(double qy_lo, double qy_hi, int irows, int scale) {
  V2RowWindow w;
  w.r0 = 0;
  w.r1 = irows;
  if (std::isfinite(qy_lo) && std::isfinite(qy_hi) && qy_lo <= qy_hi) {
    const double lo = std::floor(qy_lo) - 1.0;
    const double hi = std::ceil(qy_hi) + 1.0;
    const double irows_d = static_cast<double>(irows);
    w.r0 = lo <= 0.0 ? 0 : (lo >= irows_d ? irows : static_cast<int>(lo));
    w.r1 = hi >= irows_d ? irows : (hi <= 0.0 ? 0 : static_cast<int>(hi));
    if (w.r1 < w.r0) w.r1 = w.r0;
  }
  w.ny0 = w.r0 / scale;
  w.ny1 = (w.r1 + scale - 1) / scale;
  w.r0 = w.ny0 * scale;
  w.r1 = std::min(irows, w.ny1 * scale);
  return w;
}

}  // namespace

struct ForwardDrizzleV2CudaPrototypeKernel::Impl {
  cudaStream_t stream = nullptr;
  DevBuf<float> src;
  DevBuf<float> s2;
  DevBuf<float> qc, q0, q1, qa;
  // Compact Q windows: raw uint16 cells + veto bytes, full storage-grid
  // upper bound (divisor >= 1 -> at most source_elems cells).
  DevBuf<unsigned short> pqc, pq0, pq1, pqa;
  DevBuf<unsigned char> pvc, pv0, pv1, pva;
  // Geometry-cache leaf buffer (reserved to cfg.cached_leaf_capacity).
  DevBuf<ForwardDrizzleV2CachedLeaf> dleaves;
  // Tranche-8 ragged affine sample list (reserved to source_w*source_h).
  DevBuf<ForwardDrizzleV2SourceSample> dsamples;
  // Per-frame planes as ONE contiguous block, plane-major: fa, fbs, fbg,
  // [fs2], [fqc, fq0, fq1, fqa, fqaf]; each plane holds channels x iplane
  // doubles of the ACTIVE band (bind_planes rebinds the offsets per band),
  // so the per-frame clear is a single memset / one row-window launch.
  DevBuf<double> fplanes;
  int plane_count = 0;       // planes bound in fplanes
  int base_plane_count = 0;  // fa/fbs/fbg[/fs2]: always cleared
  double *fa = nullptr;
  double *fbs = nullptr;
  double *fbg = nullptr;
  double *fs2 = nullptr;
  double *fqc = nullptr;
  double *fq0 = nullptr;
  double *fq1 = nullptr;
  double *fqa = nullptr;
  double *fqaf = nullptr;
  DevBuf<double> accA, accB, accB2, covB, covB2, confS, confC;
  DevBuf<unsigned int> contrib, kept, footprint;
  DevBuf<unsigned short> supp;
  DevBuf<unsigned long long> degraded;
  // [0] overlaps, [1] dense overlap, [2] local discards, [3] sum of kept,
  // [4] sum of contrib (kV2ScalarCount entries).
  DevBuf<unsigned long long> scalars;
  DevBuf<V2ReservoirRecord> res;
  DevBuf<float4> resq;
  DevBuf<V2FrameMetaDev> meta;
  DevBuf<ForwardDrizzleV2PixelResult> out;
  DevBuf<ForwardDrizzleV2ProfileResult> pout;
  // shared_frame_rejection only: has_clip[pc] (1 byte) and the accepted
  // bitmask (2 x u64 per pc, res_slots can reach 128) that kernel A writes,
  // kernel B may revise, and kernel C reduces from. Null when the feature
  // is off for this workspace.
  DevBuf<unsigned char> sfr_has_clip;
  DevBuf<unsigned long long> sfr_accepted;
  // Full-frame estimator (cfg.full_frame_estimator): per-pc accumulators,
  // pilot clip bounds (lower, upper, mad per pc) and counters
  // [0] accepted, [1] rejected, [2] degenerate pilot, [3] no bounds,
  // [4] dense-overlap scratch for the pilot-barrier build launch.
  DevBuf<V2FullAcc> dfull;
  DevBuf<double> dbounds;
  DevBuf<unsigned long long> fcounters;
  // cfg.bimodal_veto counters, always allocated (like `scalars`): [0]
  // candidates rejected, [1] (pixel, channel) instances that rejected at
  // least once.
  DevBuf<unsigned long long> bimodal_counters;
  bool pilot_done = false;
  // Pinned host staging for the per-frame uploads: two slots so the host
  // never waits for the device to drain before the async copy (a
  // cudaMemcpyAsync from pageable memory blocks the host behind queued
  // kernels). Slot s is reusable once the uploads of the frame that last
  // used it (pin_last[s], event ev_up1) completed. The affine-sample path
  // stages the sample records + aligned Q codes/veto, the cached-leaf and
  // local-warp paths the source window, the leaf records and the packed Q
  // windows (same hpq/hpv slots: one frame per slot at a time). pinned_ok is
  // false when a page-locked allocation failed: the pageable path is used
  // unchanged. The compatibility float sigma2/Q input planes and the dense
  // affine-piece path (several uploads per frame) stay pageable.
  bool pinned_ok = false;
  PinnedBuf<ForwardDrizzleV2SourceSample> hsamples[2];
  PinnedBuf<unsigned short> hpq[2][4];
  PinnedBuf<unsigned char> hpv[2][4];
  PinnedBuf<float> hsrc[2];
  PinnedBuf<ForwardDrizzleV2CachedLeaf> hleaves[2];
  long long pin_last[2] = {-1, -1};
  std::vector<cudaEvent_t> ev_up0, ev_up1, ev_k0, ev_k1;
  // Host flag per frame order: the four events were recorded for a real
  // accumulate in the CURRENT band (skip frames leave it false so stale
  // prior-band timings never leak into finalize).
  std::vector<char> event_recorded;
  // max_nrows is the reserved capacity; nrows/irows/nplane/iplane are the
  // ACTIVE band dimensions (<= reserved).
  int max_nrows = 0;
  int ncols = 0, nrows = 0, icols = 0, irows = 0;
  int source_w = 0, source_h = 0, channels = 0, res_slots = 0;
  std::size_t nplane = 0, iplane = 0;
  ForwardDrizzleV2KernelConfig cfg;
  std::uint64_t frames = 0;
  bool finalized = false;

  // Bind the per-frame plane pointers to the ACTIVE band size (contiguous
  // in fplanes). Called by reserve() and begin_band().
  void bind_planes() {
    const std::size_t stride = iplane * static_cast<std::size_t>(channels);
    double *p = fplanes.get();
    int k = 0;
    auto next = [&]() { return p + static_cast<std::size_t>(k++) * stride; };
    fa = next();
    fbs = next();
    fbg = next();
    fs2 = cfg.sigma2_plane ? next() : nullptr;
    base_plane_count = k;
    if (cfg.emit_profiles) {
      fqc = next();
      fq0 = next();
      fq1 = next();
      fqa = next();
      fqaf = next();
    } else {
      fqc = fq0 = fq1 = fqa = fqaf = nullptr;
    }
    plane_count = k;
  }
  // Stage a provider buffer into pinned slot memory when pinning is
  // available; returns the pointer the async copy must read from.
  template <typename T>
  const T *stage(PinnedBuf<T> &slot, const T *src, std::size_t count) const {
    if (!pinned_ok || src == nullptr || count == 0) return src;
    std::memcpy(slot.get(), src, count * sizeof(T));
    return slot.get();
  }
  // Wait until the frame that last used pin slot `slot` finished its
  // uploads (its ev_up1), so the slot's pinned memory may be overwritten.
  bool pin_slot_ready(int slot) {
    if (!pinned_ok || pin_last[slot] < 0) return true;
    return cudaEventSynchronize(ev_up1[static_cast<std::size_t>(
               pin_last[slot])]) == cudaSuccess;
  }
  // Queue the per-frame plane clear on the workspace stream: the whole
  // block (one memset) when the row window covers the band, else one
  // row-window launch over the planes this frame writes.
  bool clear_planes(const V2RowWindow &rw, bool q_frame,
                    cudaStream_t st) const {
    const int planes = q_frame ? plane_count : base_plane_count;
    const long long stride =
        static_cast<long long>(iplane) * channels;  // doubles per plane
    if (rw.r1 <= rw.r0) return true;
    if (rw.r0 == 0 && rw.r1 == irows) {
      return cudaMemsetAsync(fa, 0,
                             static_cast<std::size_t>(planes) * stride *
                                 sizeof(double),
                             st) == cudaSuccess;
    }
    const long long off = static_cast<long long>(rw.r0) * icols;
    const long long len = static_cast<long long>(rw.r1 - rw.r0) * icols;
    const int pairs = planes * channels;
    const long long total = static_cast<long long>(pairs) * len;
    const int block = 128;
    const unsigned int grid = static_cast<unsigned int>(
        std::min<long long>((total + block - 1) / block, 4096));
    k_clear_plane_rows<<<grid, block, 0, st>>>(
        fa, pairs, static_cast<long long>(iplane), off, len);
    return cudaGetLastError() == cudaSuccess;
  }

  ~Impl() {
    for (auto *v : {&ev_up0, &ev_up1, &ev_k0, &ev_k1})
      for (cudaEvent_t e : *v)
        if (e != nullptr) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
  }
};

namespace {

bool proto_checked_mul(std::size_t a, std::size_t b, std::size_t &out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

// Fields begin_band may change: native_rows and band_origin_y_native.
// Every other config field is part of the reserved workspace's fixed
// contract (same rule as the CPU port).
bool v2_fixed_cfg_equal(const ForwardDrizzleV2KernelConfig &a,
                        const ForwardDrizzleV2KernelConfig &b) {
  return a.internal_scale == b.internal_scale &&
         a.reservoir_size == b.reservoir_size &&
         a.reservoir_seed == b.reservoir_seed &&
         a.stream_length == b.stream_length &&
         a.min_clip_contributors == b.min_clip_contributors &&
         a.min_candidates == b.min_candidates &&
         a.robust_passes == b.robust_passes && a.sigma_low == b.sigma_low &&
         a.sigma_high == b.sigma_high && a.half == b.half &&
         a.bayer_pattern == b.bayer_pattern &&
         a.cfa_origin_x == b.cfa_origin_x && a.cfa_origin_y == b.cfa_origin_y &&
         a.mono == b.mono && a.sigma2_plane == b.sigma2_plane &&
         a.emit_profiles == b.emit_profiles &&
         a.fine_quality_exponent == b.fine_quality_exponent &&
         a.medium_quality_exponent == b.medium_quality_exponent &&
         a.canvas_width_native == b.canvas_width_native &&
         a.canvas_height_native == b.canvas_height_native &&
         a.band_origin_x_native == b.band_origin_x_native &&
         a.cached_leaf_capacity == b.cached_leaf_capacity &&
         a.full_frame_estimator == b.full_frame_estimator &&
         a.bimodal_veto == b.bimodal_veto &&
         a.bimodal_veto_gap_sigma == b.bimodal_veto_gap_sigma &&
         a.shared_frame_rejection == b.shared_frame_rejection &&
         a.shared_frame_rejection_consensus ==
             b.shared_frame_rejection_consensus &&
         a.float_plane_inputs == b.float_plane_inputs;
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
      (cfg.full_frame_estimator && !cfg.emit_profiles) ||
      (cfg.bimodal_veto &&
       (!std::isfinite(cfg.bimodal_veto_gap_sigma) ||
        cfg.bimodal_veto_gap_sigma <= 0.0)) ||
      cfg.stream_length == 0 || cfg.stream_length > 65536 ||
      cfg.min_clip_contributors < 1 || cfg.min_candidates < 1 ||
      cfg.robust_passes < 1 || !std::isfinite(cfg.sigma_low) ||
      !std::isfinite(cfg.sigma_high) || cfg.sigma_low <= 0.0 ||
      cfg.sigma_high <= 0.0 || !std::isfinite(cfg.half) ||
      !(cfg.half > 0.0) || cfg.bayer_pattern < 0 || cfg.bayer_pattern > 4 ||
      cfg.band_origin_x_native < 0 || cfg.band_origin_y_native < 0 ||
      // When the full canvas dims are supplied the band window must be
      // contained in the canvas; a band origin without canvas dims is only
      // meaningful as 0 (band == canvas).
      (cfg.canvas_width_native > 0 &&
       cfg.band_origin_x_native + native_cols > cfg.canvas_width_native) ||
      (cfg.canvas_height_native > 0 &&
       cfg.band_origin_y_native + native_rows > cfg.canvas_height_native) ||
      (cfg.canvas_width_native == 0 && cfg.band_origin_x_native != 0) ||
      (cfg.canvas_height_native == 0 && cfg.band_origin_y_native != 0) ||
      (cfg.emit_profiles &&
       (!std::isfinite(cfg.fine_quality_exponent) ||
        !std::isfinite(cfg.medium_quality_exponent) ||
        cfg.fine_quality_exponent < 0.0f ||
        cfg.medium_quality_exponent < 0.0f)))
    return false;
  last_device_error_.clear();
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      last_device_error_ =
          std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e);
    return false;
  }
  CudaScopedError clear_on_exit;

  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  const int scale = cfg.internal_scale;
  const long long icols = static_cast<long long>(native_cols) * scale;
  const long long irows = static_cast<long long>(native_rows) * scale;
  if (icols > 0x7fffffffLL || irows > 0x7fffffffLL) return false;
  const int channels = cfg.mono ? 1 : 3;
  // Exact global keep set bounds the retained candidates per pixel; the
  // historical 2*R cap is preserved so rare keep sets larger than 2R still
  // drive the kept > res_slots overflow fallback at finalize (2R <= 128 =
  // kV2MaxResSlots always holds since reservoir_size <= 64).
  const auto selected = forward_drizzle_v2_selected_frame_orders(
      cfg.stream_length, cfg.reservoir_size, cfg.reservoir_seed);
  const std::size_t historical_cap =
      2 * static_cast<std::size_t>(cfg.reservoir_size);
  const int res_slots = static_cast<int>(
      std::max<std::size_t>(1, std::min(selected.size(), historical_cap)));
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
  const std::size_t frame_plane_count =
      (cfg.sigma2_plane ? 4 : 3) + (cfg.emit_profiles ? 5 : 0);
  // Float compatibility input planes (see float_plane_inputs).
  const bool float_s2 = cfg.sigma2_plane && cfg.float_plane_inputs;
  const bool float_q = cfg.emit_profiles && cfg.float_plane_inputs;
  std::size_t leaf_bytes = 0;
  if (!mul(static_cast<std::size_t>(cfg.cached_leaf_capacity),
           sizeof(ForwardDrizzleV2CachedLeaf), leaf_bytes))
    return false;
  if (!mul(source_elems, sizeof(float), tmp) || !add_bytes(tmp) ||
      // Tranche-8 canonical ragged affine sample list (full-source cap).
      !mul(source_elems, sizeof(ForwardDrizzleV2SourceSample), tmp) ||
      !add_bytes(tmp) ||
      (float_s2 &&
       (!mul(source_elems, sizeof(float), tmp) || !add_bytes(tmp))) ||
      (float_q &&
       (!mul(source_elems, sizeof(float) * 4, tmp) || !add_bytes(tmp))) ||
      (cfg.emit_profiles &&
       // Compact Q windows: uint16 cells + veto bytes per stream.
       (!mul(source_elems, 12u, tmp) || !add_bytes(tmp))) ||
      !mul(frame_plane_elems, sizeof(double) * frame_plane_count, tmp) ||
      !add_bytes(tmp) ||
      (cfg.emit_profiles &&
       (!mul(static_cast<std::size_t>(cfg.stream_length),
             sizeof(V2FrameMetaDev), tmp) ||
        !add_bytes(tmp))) ||
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
      (cfg.emit_profiles &&
       (!mul(pc_elems * static_cast<std::size_t>(res_slots),
             sizeof(float4), tmp) ||
        !add_bytes(tmp))) ||
      !mul(pc_elems, sizeof(ForwardDrizzleV2PixelResult), tmp) ||
      !add_bytes(tmp) ||
      (cfg.emit_profiles &&
       (!mul(pc_elems, sizeof(ForwardDrizzleV2ProfileResult), tmp) ||
        !add_bytes(tmp))) ||
      ((cfg.shared_frame_rejection || cfg.full_frame_estimator) &&
       (!mul(pc_elems, sizeof(unsigned char), tmp) || !add_bytes(tmp) ||
        !mul(pc_elems, 2 * sizeof(unsigned long long), tmp) ||
        !add_bytes(tmp))) ||
      (cfg.full_frame_estimator &&
       (!mul(pc_elems, sizeof(V2FullAcc) + 3 * sizeof(double), tmp) ||
        !add_bytes(tmp) || !add_bytes(5 * sizeof(unsigned long long)))) ||
      !add_bytes(leaf_bytes) ||
      !add_bytes(kV2ScalarCount * sizeof(unsigned long long)) ||
      !add_bytes(2 * sizeof(unsigned long long)))
    return false;

  Impl *im = new (std::nothrow) Impl();
  if (im == nullptr) return false;
  im->cfg = cfg;
  im->max_nrows = native_rows;
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
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      last_device_error_ =
          std::string("cudaStreamCreate: ") + cudaGetErrorString(e);
    delete im;
    return false;
  }
  im->ev_up0.resize(cfg.stream_length);
  im->ev_up1.resize(cfg.stream_length);
  im->ev_k0.resize(cfg.stream_length);
  im->ev_k1.resize(cfg.stream_length);
  im->event_recorded.assign(cfg.stream_length, 0);
  cudaError_t err = cudaSuccess;
  for (auto *v : {&im->ev_up0, &im->ev_up1, &im->ev_k0, &im->ev_k1})
    for (cudaEvent_t &e : *v)
      if (err == cudaSuccess)
        err = cudaEventCreateWithFlags(&e, cudaEventDefault);
  if (err != cudaSuccess) {
    last_device_error_ =
        std::string("reserve/cudaEventCreate: ") + cudaGetErrorString(err);
    cudaGetLastError();
    delete im;
    return false;
  }
  // The single allocation batch: every role, in reserve order. The first
  // failing cudaMalloc is recorded by its own status, so the destructor's
  // cudaFree calls cannot mask it.
  auto A = [&](auto &buf, std::size_t count) {
    if (err == cudaSuccess) err = buf.alloc(count);
  };
  const std::size_t res_elems = pc_elems * static_cast<std::size_t>(res_slots);
  const bool sfr_or_full = cfg.shared_frame_rejection || cfg.full_frame_estimator;
  A(im->src, source_elems);
  if (float_s2) A(im->s2, source_elems);
  if (float_q) {
    A(im->qc, source_elems);
    A(im->q0, source_elems);
    A(im->q1, source_elems);
    A(im->qa, source_elems);
  }
  if (cfg.emit_profiles) {
    A(im->pqc, source_elems);
    A(im->pq0, source_elems);
    A(im->pq1, source_elems);
    A(im->pqa, source_elems);
    A(im->pvc, source_elems);
    A(im->pv0, source_elems);
    A(im->pv1, source_elems);
    A(im->pva, source_elems);
  }
  A(im->dleaves, static_cast<std::size_t>(cfg.cached_leaf_capacity));
  A(im->dsamples, source_elems);
  A(im->fplanes, frame_plane_elems * frame_plane_count);
  A(im->accA, pc_elems);
  A(im->accB, pc_elems);
  A(im->accB2, pc_elems);
  A(im->covB, pc_elems);
  A(im->covB2, pc_elems);
  A(im->confS, pc_elems);
  A(im->confC, pc_elems);
  A(im->contrib, pc_elems);
  A(im->kept, pc_elems);
  A(im->footprint, nplane);
  A(im->supp, pc_elems);
  A(im->degraded, pc_elems);
  A(im->scalars, kV2ScalarCount);
  A(im->bimodal_counters, 2);
  A(im->res, res_elems);
  if (cfg.emit_profiles) {
    A(im->resq, res_elems);
    A(im->meta, static_cast<std::size_t>(cfg.stream_length));
    A(im->pout, pc_elems);
  }
  A(im->out, pc_elems);
  if (sfr_or_full) {
    A(im->sfr_has_clip, pc_elems);
    A(im->sfr_accepted, pc_elems * 2);
  }
  if (cfg.full_frame_estimator) {
    A(im->dfull, pc_elems);
    A(im->dbounds, pc_elems * 3);
    A(im->fcounters, 5);
  }
  if (err != cudaSuccess) {
    last_device_error_ =
        std::string("reserve/cudaMalloc: ") + cudaGetErrorString(err);
    cudaGetLastError();
    delete im;
    return false;
  }
  im->bind_planes();
  // Best-effort pinned staging slots (see Impl): all or nothing.
  {
    cudaError_t pin_err = cudaSuccess;
    auto P = [&](auto &buf, std::size_t count) {
      if (pin_err == cudaSuccess) pin_err = buf.alloc(count);
    };
    for (int sl = 0; sl < 2; ++sl) {
      P(im->hsamples[sl], source_elems);
      P(im->hsrc[sl], source_elems);
      P(im->hleaves[sl], static_cast<std::size_t>(cfg.cached_leaf_capacity));
      if (!cfg.emit_profiles) continue;
      for (int k = 0; k < 4; ++k) {
        P(im->hpq[sl][k], source_elems);
        P(im->hpv[sl][k], source_elems);
      }
    }
    if (pin_err != cudaSuccess) {
      cudaGetLastError();
      for (int sl = 0; sl < 2; ++sl) {
        im->hsamples[sl].reset();
        im->hsrc[sl].reset();
        im->hleaves[sl].reset();
        for (int k = 0; k < 4; ++k) {
          im->hpq[sl][k].reset();
          im->hpv[sl][k].reset();
        }
      }
    }
    im->pinned_ok = pin_err == cudaSuccess;
  }
  // Queue zero-init on the workspace stream; ordered before any kernel.
  bool ok =
      cudaMemsetAsync(im->accA, 0, pc_elems * sizeof(double), im->stream) ==
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
      cudaMemsetAsync(im->scalars, 0,
                      kV2ScalarCount * sizeof(unsigned long long),
                      im->stream) == cudaSuccess &&
      cudaMemsetAsync(im->bimodal_counters, 0,
                      2 * sizeof(unsigned long long),
                      im->stream) == cudaSuccess &&
      (!cfg.full_frame_estimator ||
       (cudaMemsetAsync(im->dfull, 0, pc_elems * sizeof(V2FullAcc),
                        im->stream) == cudaSuccess &&
        cudaMemsetAsync(im->fcounters, 0, 5 * sizeof(unsigned long long),
                        im->stream) == cudaSuccess));
  if (!ok) {
    const cudaError_t memset_err = cudaGetLastError();
    if (memset_err != cudaSuccess)
      last_device_error_ = std::string("reserve/cudaMemsetAsync: ") +
                           cudaGetErrorString(memset_err);
    delete im;
    return false;
  }
  impl_ = im;
  stats_.allocations = 1;
  stats_.workspace_reservations = 1;
  stats_.reserved_device_bytes = total_bytes;
  capacity_bytes_ = total_bytes;
  pending_reservation_ = true;
  // Per-native-pixel footprint for the gate-5-plan comparison: everything
  // except the fixed pipeline slots (source + sigma2 + gate-9 quality
  // planes), the frame-meta table and scalars, divided by native pixels.
  std::size_t fixed_slot_bytes =
      source_elems * sizeof(float) * (float_s2 ? 2 : 1);
  if (float_q) fixed_slot_bytes += source_elems * sizeof(float) * 4;
  if (cfg.emit_profiles) {
    fixed_slot_bytes += source_elems * 12u +
                        static_cast<std::size_t>(cfg.stream_length) *
                            sizeof(V2FrameMetaDev);
  }
  bytes_per_native_pixel_ =
      (total_bytes - fixed_slot_bytes -
       kV2ScalarCount * sizeof(unsigned long long)) /
      nplane;
  return true;
}

// Rebind the workspace to the next band: recompute the ACTIVE dimensions
// from native_rows, rezero every per-band role on the workspace stream over
// the active byte ranges only, and clear the host event flags. No cudaMalloc
// / cudaFree / event creation; all buffers and the stream persist.
bool ForwardDrizzleV2CudaPrototypeKernel::begin_band(
    int native_rows, const ForwardDrizzleV2KernelConfig &cfg) {
  last_device_error_.clear();
  if (impl_ == nullptr) return false;
  Impl &im = *impl_;
  // Legal only right after reserve() (frames==0, not yet finalized) or
  // after a successful finalize(); a partial/failed stream cannot rebound.
  if (frame_open_ || (!im.finalized && im.frames != 0)) return false;
  if (native_rows < 1 || native_rows > im.max_nrows ||
      !v2_fixed_cfg_equal(im.cfg, cfg) || cfg.band_origin_y_native < 0 ||
      (cfg.canvas_height_native > 0 &&
       cfg.band_origin_y_native + native_rows > cfg.canvas_height_native) ||
      (cfg.canvas_height_native == 0 && cfg.band_origin_y_native != 0))
    return false;
  CudaScopedError clear_on_exit;
  im.cfg = cfg;
  im.nrows = native_rows;
  im.irows = native_rows * cfg.internal_scale;
  im.nplane = static_cast<std::size_t>(im.ncols) * im.nrows;
  im.iplane = static_cast<std::size_t>(im.icols) * im.irows;
  im.bind_planes();
  const std::size_t pc_elems =
      im.nplane * static_cast<std::size_t>(im.channels);
  const bool ok =
      cudaMemsetAsync(im.accA, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.accB, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.accB2, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.covB, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.covB2, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.confS, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.confC, 0, pc_elems * sizeof(double), im.stream) ==
          cudaSuccess &&
      cudaMemsetAsync(im.contrib, 0, pc_elems * sizeof(unsigned int),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.kept, 0, pc_elems * sizeof(unsigned int),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.footprint, 0, im.nplane * sizeof(unsigned int),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.supp, 0, pc_elems * sizeof(unsigned short),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.degraded, 0, pc_elems * sizeof(unsigned long long),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.scalars, 0,
                      kV2ScalarCount * sizeof(unsigned long long),
                      im.stream) == cudaSuccess &&
      cudaMemsetAsync(im.bimodal_counters, 0, 2 * sizeof(unsigned long long),
                      im.stream) == cudaSuccess &&
      (!im.cfg.full_frame_estimator ||
       (cudaMemsetAsync(im.dfull, 0, pc_elems * sizeof(V2FullAcc),
                        im.stream) == cudaSuccess &&
        cudaMemsetAsync(im.fcounters, 0, 5 * sizeof(unsigned long long),
                        im.stream) == cudaSuccess));
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  im.pilot_done = false;
  im.pin_last[0] = im.pin_last[1] = -1;
  std::fill(im.event_recorded.begin(), im.event_recorded.end(), char{0});
  im.frames = 0;
  im.finalized = false;
  // stats() is a current-band delta: the single workspace reservation is
  // reported exactly once, by the first begin_band.
  stats_ = ForwardDrizzleV2PrototypeStats{};
  stats_.band_resets = 1;
  if (pending_reservation_) {
    stats_.workspace_reservations = 1;
    stats_.allocations = 1;
    pending_reservation_ = false;
  }
  stats_.reserved_device_bytes = capacity_bytes_;
  return true;
}

// Shared frame pipeline for the cached-leaf and the gate-8 local-warp
// scatter: upload (pinned-staged), row-window plane clear, geometry
// scatter, fold+accumulate over the same row window. `leaves != nullptr`
// selects the cached kernel, otherwise `warp` must be set. All work is
// queued on the workspace stream.
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_impl(
    const double affine6[6], const ForwardDrizzleV2LocalWarp *warp,
    const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
    std::uint64_t unique_source_samples, std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  // The one-shot paths must not interleave with an open affine piece frame.
  if (frame_open_) return false;
  if (leaves == nullptr && warp == nullptr) return false;
  // Packed buffer (optional halo) must lie inside the reserved full source
  // extent; only the ACTIVE rect is launched.
  if (window.x_begin < 0 || window.y_begin < 0 || window.width <= 0 ||
      window.height <= 0 ||
      window.x_begin + window.width > im.source_w ||
      window.y_begin + window.height > im.source_h)
    return false;
  V2Window w;
  w.buf_x = window.x_begin;
  w.buf_y = window.y_begin;
  w.buf_w = window.width;
  w.buf_h = window.height;
  w.act_x = window.active_x;
  w.act_y = window.active_y;
  w.act_w = window.active_width;
  w.act_h = window.active_height;
  if (w.act_w == 0 && w.act_h == 0 && w.act_x == 0 && w.act_y == 0) {
    w.act_w = w.buf_w;
    w.act_h = w.buf_h;
  }
  if (w.act_x < 0 || w.act_y < 0 || w.act_w <= 0 || w.act_h <= 0 ||
      w.act_x + w.act_w > w.buf_w || w.act_y + w.act_h > w.buf_h)
    return false;
  const int abs_x0 = w.buf_x + w.act_x;
  const int abs_y0 = w.buf_y + w.act_y;
  w.src_w = im.source_w;
  w.src_h = im.source_h;
  const double sc = static_cast<double>(im.cfg.internal_scale);
  const double band_oy = static_cast<double>(im.cfg.band_origin_y_native);
  // Row window of this frame (see v2_row_window): the leaf extremes bound
  // the cached scatter; the local-warp inversion runs on device, so that
  // path keeps the whole band.
  double qy_lo = -std::numeric_limits<double>::infinity();
  double qy_hi = std::numeric_limits<double>::infinity();
  // Cached-geometry path: leaves carry the committed geometry; a local warp
  // must not be combined with it, the count is bound by the reserved
  // capacity, and every leaf's source sample must lie inside the ACTIVE
  // rect (float sigma/Q planes index by active-local position).
  if (leaves != nullptr) {
    if (warp != nullptr || leaf_count == 0 ||
        leaf_count > im.cfg.cached_leaf_capacity)
      return false;
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();
    for (std::size_t li = 0; li < leaf_count; ++li) {
      const ForwardDrizzleV2CachedLeaf &L = leaves[li];
      if (L.source_x >= static_cast<std::uint32_t>(im.source_w) ||
          L.source_y >= static_cast<std::uint32_t>(im.source_h) ||
          L.channel >= static_cast<std::uint16_t>(im.channels))
        return false;
      const int sx = static_cast<int>(L.source_x);
      const int sy = static_cast<int>(L.source_y);
      if (sx < abs_x0 || sx >= abs_x0 + w.act_w || sy < abs_y0 ||
          sy >= abs_y0 + w.act_h)
        return false;
      for (int k = 0; k < 4; ++k) {
        if (!std::isfinite(L.x[k]) || !std::isfinite(L.y[k])) return false;
        y_min = std::min(y_min, L.y[k]);
        y_max = std::max(y_max, L.y[k]);
      }
    }
    // Same expression as k_scatter_v2_cached's qy[k] = L.y[k] - band_oy*sc.
    qy_lo = y_min - band_oy * sc;
    qy_hi = y_max - band_oy * sc;
  }
  const V2RowWindow rw =
      v2_row_window(qy_lo, qy_hi, im.irows, im.cfg.internal_scale);
  // An explicit sigma2 plane and an enabled inline model are mutually
  // exclusive (the model reads the halo buffer on device).
  V2Sigma2Model s2m{};
  if (sigma2_model_or_null != nullptr && sigma2_model_or_null->enabled) {
    s2m.enabled = 1;
    s2m.noise = sigma2_model_or_null->sigma_noise;
    s2m.reg_px = sigma2_model_or_null->sigma_reg_px;
    s2m.half = sigma2_model_or_null->droplet_half;
  }
  if (sigma2_or_null != nullptr && s2m.enabled) return false;
  // Without float_plane_inputs no float sigma2/quality input planes exist.
  if (!im.cfg.float_plane_inputs &&
      (sigma2_or_null != nullptr ||
       (quality_or_null != nullptr &&
        (quality_or_null->q_composite != nullptr ||
         quality_or_null->q_scale0 != nullptr ||
         quality_or_null->q_scale1 != nullptr ||
         quality_or_null->q_artifact != nullptr))))
    return false;
  const std::size_t src_elems =
      static_cast<std::size_t>(w.buf_w) * w.buf_h;
  const std::size_t src_bytes = src_elems * sizeof(float);
  const std::size_t act_bytes =
      static_cast<std::size_t>(w.act_w) * w.act_h * sizeof(float);
  const std::size_t src_cap =
      static_cast<std::size_t>(im.source_w) * im.source_h;
  const std::uint64_t f = impl_->frames;
  const std::uint64_t n = im.cfg.stream_length;
  // The finalize kernel indexes the meta table by candidate order.
  if (frame_order >= n) return false;
  if (im.cfg.emit_profiles) {
    // The profile contract requires a valid meta row per frame; an absent
    // or malformed row fails the call (the CPU reduce throws).
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
  }
  const bool keep_all =
      n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
  const std::uint64_t threshold =
      keep_all ? 0
               : static_cast<std::uint64_t>(
                     (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                      << 64) /
                     n);
  const bool qp = im.cfg.emit_profiles;
  // Quality work is relevant only for frames the reservoir keep set can
  // retain; non-selected orders carry qmask = 0 and skip all Q uploads,
  // clears and scatter work.
  const bool keep_frame =
      keep_all ||
      d_splitmix64(frame_order ^ im.cfg.reservoir_seed) < threshold;
  // Full-frame mode: pilot frames strictly before end_pilot(), all other
  // frames strictly after; every frame carries quality.
  if (im.cfg.full_frame_estimator && keep_frame == im.pilot_done) return false;
  const bool q_frame = qp && (keep_frame || im.cfg.full_frame_estimator);
  // Each stream supplies a float plane XOR a packed storage-grid window;
  // both set or a malformed descriptor fails the call. Packed descriptors
  // must cover every storage cell the active rect maps to and fit the
  // reserved (source_elems) device/staging window.
  ForwardDrizzleV2FrameQuality h_q{};
  if (q_frame && quality_or_null != nullptr) h_q = *quality_or_null;
  const float *h_qf[4] = {h_q.q_composite, h_q.q_scale0, h_q.q_scale1,
                          h_q.q_artifact};
  const ForwardDrizzleV2PackedQualityPlane *h_qp[4] = {
      &h_q.qc_packed, &h_q.q0_packed, &h_q.q1_packed, &h_q.qa_packed};
  for (int k = 0; k < 4; ++k) {
    const ForwardDrizzleV2PackedQualityPlane &p = *h_qp[k];
    if (p.cells == nullptr) continue;  // absent or float stream
    if (h_qf[k] != nullptr) return false;  // never both
    if (p.storage_width <= 0 || p.storage_height <= 0 ||
        p.storage_divisor <= 0)
      return false;
    const int d = p.storage_divisor;
    if (!(p.storage_x_begin <= abs_x0 / d &&
          p.storage_x_begin + p.storage_width > (abs_x0 + w.act_w - 1) / d &&
          p.storage_y_begin <= abs_y0 / d &&
          p.storage_y_begin + p.storage_height > (abs_y0 + w.act_h - 1) / d))
      return false;
    if (static_cast<std::size_t>(p.storage_width) * p.storage_height >
        src_cap)
      return false;
  }
  unsigned int qmask = 0;
  for (int k = 0; k < 4; ++k)
    if (h_qf[k] != nullptr || h_qp[k]->cells != nullptr) qmask |= 1u << k;
  V2FrameMetaDev h_meta{};
  if (meta_or_null != nullptr) {
    h_meta.g_eff = meta_or_null->g_eff;
    h_meta.residual_factor = meta_or_null->residual_factor;
    h_meta.is_direct = meta_or_null->is_direct;
  }
  // Pinned staging (see Impl): wait only for the uploads of the frame that
  // last used this slot, copy the provider buffers into page-locked memory
  // and issue the async copies from there so the host returns immediately.
  const int pin_slot = static_cast<int>(f & 1u);
  if (!im.pin_slot_ready(pin_slot)) {
    cudaGetLastError();
    return false;
  }
  const float *up_source = im.stage(im.hsrc[pin_slot], source, src_elems);
  const ForwardDrizzleV2CachedLeaf *up_leaves =
      leaves != nullptr ? im.stage(im.hleaves[pin_slot], leaves, leaf_count)
                        : nullptr;
  const unsigned short *up_cells[4] = {nullptr, nullptr, nullptr, nullptr};
  const unsigned char *up_veto[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int k = 0; k < 4; ++k) {
    const ForwardDrizzleV2PackedQualityPlane &p = *h_qp[k];
    if (p.cells == nullptr) continue;
    const std::size_t cn =
        static_cast<std::size_t>(p.storage_width) * p.storage_height;
    up_cells[k] = im.stage(im.hpq[pin_slot][k], p.cells, cn);
    up_veto[k] = im.stage(im.hpv[pin_slot][k], p.veto, cn);
  }
  float *d_qf[4] = {im.qc.get(), im.q0.get(), im.q1.get(), im.qa.get()};
  unsigned short *d_cells[4] = {im.pqc.get(), im.pq0.get(), im.pq1.get(),
                                im.pqa.get()};
  unsigned char *d_veto[4] = {im.pvc.get(), im.pv0.get(), im.pv1.get(),
                              im.pva.get()};
  bool ok = cudaEventRecord(im.ev_up0[f], st) == cudaSuccess &&
            cudaMemcpyAsync(im.src, up_source, src_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess &&
            (!im.s2 || !sigma2_or_null ||
             cudaMemcpyAsync(im.s2, sigma2_or_null, act_bytes,
                             cudaMemcpyHostToDevice, st) == cudaSuccess);
  for (int k = 0; k < 4 && ok; ++k) {
    // Float compatibility plane (packed over the active rect) XOR the
    // compact storage cells + veto bytes.
    if (h_qf[k] != nullptr)
      ok = cudaMemcpyAsync(d_qf[k], h_qf[k], act_bytes,
                           cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (ok && up_cells[k] != nullptr) {
      const std::size_t cn = static_cast<std::size_t>(h_qp[k]->storage_width) *
                             h_qp[k]->storage_height;
      ok = cudaMemcpyAsync(d_cells[k], up_cells[k],
                           cn * sizeof(unsigned short),
                           cudaMemcpyHostToDevice, st) == cudaSuccess &&
           (up_veto[k] == nullptr ||
            cudaMemcpyAsync(d_veto[k], up_veto[k], cn,
                            cudaMemcpyHostToDevice, st) == cudaSuccess);
    }
  }
  ok = ok &&
       (leaves == nullptr ||
        cudaMemcpyAsync(im.dleaves, up_leaves,
                        leaf_count * sizeof(ForwardDrizzleV2CachedLeaf),
                        cudaMemcpyHostToDevice, st) == cudaSuccess) &&
       (!qp ||
        cudaMemcpyAsync(im.meta + frame_order, &h_meta,
                        sizeof(V2FrameMetaDev), cudaMemcpyHostToDevice,
                        st) == cudaSuccess) &&
       cudaEventRecord(im.ev_up1[f], st) == cudaSuccess &&
       im.clear_planes(rw, q_frame, st) &&
       cudaEventRecord(im.ev_k0[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  // ev_up1[f] marks the end of this frame's uploads: the pinned slot is
  // free again once it completes.
  if (im.pinned_ok) im.pin_last[pin_slot] = static_cast<long long>(f);
  V2QualityIO qio;
  if (q_frame) {
    qio.qc = h_qf[0] != nullptr ? d_qf[0] : nullptr;
    qio.q0 = h_qf[1] != nullptr ? d_qf[1] : nullptr;
    qio.q1 = h_qf[2] != nullptr ? d_qf[2] : nullptr;
    qio.qa = h_qf[3] != nullptr ? d_qf[3] : nullptr;
    V2PackedQuality *dq[4] = {&qio.qc_p, &qio.q0_p, &qio.q1_p, &qio.qa_p};
    for (int k = 0; k < 4; ++k) {
      const ForwardDrizzleV2PackedQualityPlane &p = *h_qp[k];
      if (p.cells == nullptr) continue;
      dq[k]->cells = d_cells[k];
      dq[k]->veto = p.veto != nullptr ? d_veto[k] : nullptr;
      dq[k]->x_begin = p.storage_x_begin;
      dq[k]->y_begin = p.storage_y_begin;
      dq[k]->width = p.storage_width;
      dq[k]->height = p.storage_height;
      dq[k]->divisor = p.storage_divisor;
    }
    qio.qmask = qmask;
    qio.fqc = im.fqc;
    qio.fq0 = im.fq0;
    qio.fq1 = im.fq1;
    qio.fqa = im.fqa;
    qio.fqaf = im.fqaf;
  }
  // An empty row window (the frame misses the band entirely) has nothing
  // to scatter or fold; the bookkeeping below still advances the stream.
  if (rw.r1 > rw.r0) {
    const int block = 128;
    const long long source_n =
        static_cast<long long>(w.act_w) * w.act_h;
    const unsigned int grid =
        static_cast<unsigned int>((source_n + block - 1) / block);
    const float *d_s2 =
        (im.fs2 && !s2m.enabled && sigma2_or_null != nullptr) ? im.s2.get()
                                                               : nullptr;
    if (leaves != nullptr) {
      const unsigned int lgrid = static_cast<unsigned int>(
          (static_cast<long long>(leaf_count) + block - 1) / block);
      k_scatter_v2_cached<<<lgrid, block, 0, st>>>(
          im.dleaves, static_cast<long long>(leaf_count), sc, im.icols,
          im.irows, w, im.src, d_s2, s2m,
          static_cast<double>(im.cfg.band_origin_x_native), band_oy, im.fa,
          im.fbs, im.fbg, im.fs2, qio, im.scalars);
    } else {
      // The inversion bounds check uses the FULL native canvas, not the
      // band height the workspace planes were reserved for.
      const int canvas_w = im.cfg.canvas_width_native > 0
                               ? im.cfg.canvas_width_native
                               : im.ncols;
      const int canvas_h = im.cfg.canvas_height_native > 0
                               ? im.cfg.canvas_height_native
                               : im.nrows;
      k_scatter_v2_local<<<grid, block, 0, st>>>(
          affine6[0], affine6[1], affine6[2], affine6[3], affine6[4],
          affine6[5], *warp, sc, im.cfg.half, im.icols, im.irows, w, im.src,
          d_s2, s2m, im.cfg.bayer_pattern, im.cfg.cfa_origin_x,
          im.cfg.cfa_origin_y, im.cfg.mono ? 1 : 0, canvas_w, canvas_h,
          static_cast<double>(im.cfg.band_origin_x_native), band_oy, im.fa,
          im.fbs, im.fbg, im.fs2, qio, im.scalars, im.scalars + 2);
    }
    const long long fold_n =
        static_cast<long long>(im.ncols) * (rw.ny1 - rw.ny0);
    const unsigned int fgrid =
        static_cast<unsigned int>((fold_n + block - 1) / block);
    k_fold_accumulate_v2<<<fgrid, block, 0, st>>>(
        im.fa, im.fbs, im.fbg, im.fs2,
        q_frame ? im.fqc : nullptr, q_frame ? im.fq0 : nullptr,
        q_frame ? im.fq1 : nullptr, q_frame ? im.fqa : nullptr,
        q_frame ? im.fqaf : nullptr, qmask, im.icols, im.irows,
        im.cfg.internal_scale, im.ncols, im.nrows, im.channels, 0, im.ncols,
        rw.ny0, rw.ny1 - rw.ny0, frame_order,
        keep_all ? 1 : 0, threshold, im.res_slots, im.cfg.reservoir_seed,
        im.accA, im.accB, im.accB2, im.covB, im.covB2, im.confS, im.confC,
        im.contrib, im.kept, im.footprint, im.supp, im.degraded, im.res,
        im.resq, im.dfull, im.meta,
        static_cast<double>(im.cfg.fine_quality_exponent),
        static_cast<double>(im.cfg.medium_quality_exponent),
        im.cfg.shared_frame_rejection ? 1 : 0,
        static_cast<double>(im.cfg.shared_frame_rejection_consensus));
  }
  ok = cudaGetLastError() == cudaSuccess &&
       cudaEventRecord(im.ev_k1[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  // All four events of this frame were recorded in the current band;
  // finalize only reads timings for flagged frames.
  im.event_recorded[f] = 1;
  ++impl_->frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  stats_.source_bytes_uploaded +=
      src_bytes + (sigma2_or_null != nullptr ? act_bytes : 0);
  // Cached-geometry frames launch the cache-counted covered samples, not
  // the dense active rect (the build-side discards are already final).
  stats_.source_samples_launched +=
      leaves != nullptr
          ? unique_source_samples
          : static_cast<std::uint64_t>(w.act_w) * w.act_h;
  if (leaves != nullptr) {
    stats_.cached_leaf_records_launched += leaf_count;
    stats_.cached_leaf_bytes_uploaded +=
        static_cast<std::uint64_t>(leaf_count) *
        sizeof(ForwardDrizzleV2CachedLeaf);
  }
  if (q_frame) {
    ++stats_.quality_frames_processed;
    for (int k = 0; k < 4; ++k) {
      if (h_qp[k]->cells != nullptr)
        stats_.quality_bytes_uploaded +=
            static_cast<std::uint64_t>(h_qp[k]->storage_width) *
            h_qp[k]->storage_height * 3u;
      else if (h_qf[k] != nullptr)
        stats_.quality_bytes_uploaded += act_bytes;
    }
  }
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame(
    const double affine6[6], const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr) return false;
  const ForwardDrizzleV2SourceWindow full{0, 0, impl_->source_w,
                                        impl_->source_h};
  return accumulate_frame_window(affine6, full, source, sigma2_or_null,
                                 nullptr, frame_order, quality_or_null,
                                 meta_or_null);
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_window(
    const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  // Affine one-shot: begin + one full-width piece + finish.
  if (impl_ == nullptr) return false;
  if (!begin_affine_frame(frame_order, meta_or_null)) return false;
  if (!accumulate_affine_piece(affine6, 0, impl_->ncols, window, source,
                               sigma2_or_null, sigma2_model_or_null,
                               quality_or_null))
    return false;
  return finish_affine_frame(frame_order);
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_local(
    const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
    const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr) return false;
  const ForwardDrizzleV2SourceWindow full{0, 0, impl_->source_w,
                                        impl_->source_h};
  return accumulate_frame_local_window(affine6, warp, full, source,
                                       sigma2_or_null, nullptr, frame_order,
                                       quality_or_null, meta_or_null);
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_local_window(
    const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
    const ForwardDrizzleV2SourceWindow &window, const float *source,
    const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || affine6 == nullptr || source == nullptr ||
      impl_->finalized || impl_->frames >= impl_->cfg.stream_length)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (!local_warp_valid(warp)) return false;
  return accumulate_frame_impl(affine6, &warp, window, source, sigma2_or_null,
                               sigma2_model_or_null, nullptr, 0, 0,
                               frame_order, quality_or_null, meta_or_null);
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_cached_leaves(
    const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
    std::uint64_t unique_source_samples, std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || affine6 == nullptr || source == nullptr ||
      leaves == nullptr || impl_->finalized ||
      impl_->frames >= impl_->cfg.stream_length)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  return accumulate_frame_impl(affine6, nullptr, window, source,
                               sigma2_or_null, sigma2_model_or_null, leaves,
                               leaf_count, unique_source_samples, frame_order,
                               quality_or_null, meta_or_null);
}

// Tranche 8 canonical ragged affine path: one-shot full-target frame fed by
// the per-sample list. Uploads only the sample records and (on selected
// frames) the aligned packed quality arrays; the scatter kernel runs one
// thread per sample with the full-band x clamp. The plane clear and the
// fold cover only the band rows the frame can touch (row window from the
// sample bbox corners mapped through the affine, see v2_row_window).
bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_frame_affine_samples(
    const double affine6[6], const ForwardDrizzleV2SourceSample *samples,
    std::size_t sample_count, bool sigma2_present,
    const ForwardDrizzleV2AlignedQuality *quality_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || frame_open_ || impl_->finalized ||
      impl_->frames >= impl_->cfg.stream_length ||
      frame_order >= impl_->cfg.stream_length || affine6 == nullptr ||
      sample_count == 0 || samples == nullptr)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  const std::size_t src_cap =
      static_cast<std::size_t>(im.source_w) * im.source_h;
  if (sample_count > src_cap) return false;
  // Host-side canonical-order validation + span-row count (samples must be
  // strictly increasing in (y, x) and inside the reserved source extent);
  // the same scan collects the source bbox for the row window.
  std::uint64_t span_rows = 0;
  std::uint32_t prev_y = 0, prev_x = 0;
  std::uint32_t sx_min = 0, sx_max = 0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const auto &s = samples[i];
    if (s.source_x >= static_cast<std::uint32_t>(im.source_w) ||
        s.source_y >= static_cast<std::uint32_t>(im.source_h) ||
        (i > 0 && (s.source_y < prev_y ||
                   (s.source_y == prev_y && s.source_x <= prev_x))))
      return false;
    if (i == 0 || s.source_y != prev_y) ++span_rows;
    if (i == 0 || s.source_x < sx_min) sx_min = s.source_x;
    if (i == 0 || s.source_x > sx_max) sx_max = s.source_x;
    prev_y = s.source_y;
    prev_x = s.source_x;
  }
  const std::uint64_t f = impl_->frames;
  const std::uint64_t n = im.cfg.stream_length;
  if (im.cfg.emit_profiles) {
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
  }
  const bool keep_all =
      n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
  const std::uint64_t threshold =
      keep_all ? 0
               : static_cast<std::uint64_t>(
                     (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                      << 64) /
                     n);
  const bool qp = im.cfg.emit_profiles;
  const bool pilot_frame =
      keep_all ||
      d_splitmix64(frame_order ^ im.cfg.reservoir_seed) < threshold;
  if (im.cfg.full_frame_estimator && pilot_frame == im.pilot_done)
    return false;
  const bool q_frame = qp && (pilot_frame || im.cfg.full_frame_estimator);
  const ForwardDrizzleV2AlignedQuality h_q =
      (q_frame && quality_or_null != nullptr) ? *quality_or_null
                                            : ForwardDrizzleV2AlignedQuality{};
  const unsigned int qmask = q_frame ? h_q.presence_mask : 0u;
  const std::uint16_t *aqc[4] = {h_q.qc, h_q.q0, h_q.q1, h_q.qa};
  const std::uint8_t *avc[4] = {h_q.vc, h_q.v0, h_q.v1, h_q.va};
  unsigned short *dpq[4] = {im.pqc.get(), im.pq0.get(), im.pq1.get(),
                            im.pqa.get()};
  unsigned char *dpv[4] = {im.pvc.get(), im.pv0.get(), im.pv1.get(),
                           im.pva.get()};
  for (int k = 0; k < 4; ++k)
    if ((qmask & (1u << k)) != 0 && (aqc[k] == nullptr || dpq[k] == nullptr))
      return false;
  V2FrameMetaDev h_meta{};
  if (meta_or_null != nullptr) {
    h_meta.g_eff = meta_or_null->g_eff;
    h_meta.residual_factor = meta_or_null->residual_factor;
    h_meta.is_direct = meta_or_null->is_direct;
  }
  const std::size_t rec_bytes =
      sample_count * sizeof(ForwardDrizzleV2SourceSample);
  // Row window: the extreme band-local internal y over the sample set is
  // attained at a corner of the source bbox (the mapped droplet corner
  // expression is monotone in px and py), evaluated with the kernel's own
  // formula; v2_row_window pads it by a row on each side.
  const double sc = static_cast<double>(im.cfg.internal_scale);
  const double half = im.cfg.half;
  const double band_oy = static_cast<double>(im.cfg.band_origin_y_native);
  const double px_lo = (static_cast<double>(sx_min) + 0.5) - half;
  const double px_hi = (static_cast<double>(sx_max) + 0.5) + half;
  const double py_lo = (static_cast<double>(samples[0].source_y) + 0.5) - half;
  const double py_hi =
      (static_cast<double>(samples[sample_count - 1].source_y) + 0.5) + half;
  double qy_lo = std::numeric_limits<double>::infinity();
  double qy_hi = -std::numeric_limits<double>::infinity();
  for (double px : {px_lo, px_hi})
    for (double py : {py_lo, py_hi}) {
      const double qy =
          (affine6[3] * px + affine6[4] * py + affine6[5] - band_oy) * sc;
      qy_lo = std::min(qy_lo, qy);
      qy_hi = std::max(qy_hi, qy);
    }
  const V2RowWindow rw =
      v2_row_window(qy_lo, qy_hi, im.irows, im.cfg.internal_scale);
  // Pinned staging (see Impl): wait only for the uploads of the frame that
  // last used this slot, copy the provider buffers into page-locked memory
  // and issue the async copies from there so the host returns immediately.
  const int pin_slot = static_cast<int>(f & 1u);
  if (!im.pin_slot_ready(pin_slot)) {
    cudaGetLastError();
    return false;
  }
  const ForwardDrizzleV2SourceSample *up_samples =
      im.stage(im.hsamples[pin_slot], samples, sample_count);
  const std::uint16_t *up_q[4] = {aqc[0], aqc[1], aqc[2], aqc[3]};
  const std::uint8_t *up_v[4] = {avc[0], avc[1], avc[2], avc[3]};
  if (q_frame) {
    for (int k = 0; k < 4; ++k) {
      if ((qmask & (1u << k)) == 0) continue;
      up_q[k] = im.stage(im.hpq[pin_slot][k], aqc[k], sample_count);
      up_v[k] = im.stage(im.hpv[pin_slot][k], avc[k], sample_count);
    }
  }
  bool ok = cudaEventRecord(im.ev_up0[f], st) == cudaSuccess &&
            cudaMemcpyAsync(im.dsamples, up_samples, rec_bytes,
                            cudaMemcpyHostToDevice, st) == cudaSuccess;
  if (ok && q_frame) {
    for (int k = 0; k < 4 && ok; ++k) {
      if ((qmask & (1u << k)) == 0) continue;
      ok = cudaMemcpyAsync(dpq[k], up_q[k],
                           sample_count * sizeof(unsigned short),
                           cudaMemcpyHostToDevice, st) == cudaSuccess &&
           (up_v[k] == nullptr ||
            cudaMemcpyAsync(dpv[k], up_v[k], sample_count,
                            cudaMemcpyHostToDevice, st) == cudaSuccess);
    }
  }
  ok = ok &&
       (!qp || cudaMemcpyAsync(im.meta + frame_order, &h_meta,
                               sizeof(V2FrameMetaDev), cudaMemcpyHostToDevice,
                               st) == cudaSuccess) &&
       cudaEventRecord(im.ev_up1[f], st) == cudaSuccess &&
       im.clear_planes(rw, q_frame, st) &&
       cudaEventRecord(im.ev_k0[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  // ev_up1[f] marks the end of this frame's uploads: the pinned slot is
  // free again once it completes.
  if (im.pinned_ok) im.pin_last[pin_slot] = static_cast<long long>(f);
  V2AlignedQualityDev aq{};
  V2QualityIO qio;
  if (q_frame && qmask != 0u) {
    aq.qc = (qmask & 1u) ? dpq[0] : nullptr;
    aq.q0 = (qmask & 2u) ? dpq[1] : nullptr;
    aq.q1 = (qmask & 4u) ? dpq[2] : nullptr;
    aq.qa = (qmask & 8u) ? dpq[3] : nullptr;
    aq.vc = ((qmask & 1u) && avc[0] != nullptr) ? dpv[0] : nullptr;
    aq.v0 = ((qmask & 2u) && avc[1] != nullptr) ? dpv[1] : nullptr;
    aq.v1 = ((qmask & 4u) && avc[2] != nullptr) ? dpv[2] : nullptr;
    aq.va = ((qmask & 8u) && avc[3] != nullptr) ? dpv[3] : nullptr;
    aq.qmask = qmask;
    qio.qmask = qmask;
    qio.fqc = im.fqc;
    qio.fq0 = im.fq0;
    qio.fq1 = im.fq1;
    qio.fqa = im.fqa;
    qio.fqaf = im.fqaf;
  }
  // An empty row window (the frame misses the band entirely) has nothing
  // to scatter or fold; the bookkeeping below still advances the stream.
  if (rw.r1 > rw.r0) {
    const int block = 128;
    const unsigned int grid = static_cast<unsigned int>(
        (static_cast<long long>(sample_count) + block - 1) / block);
    k_scatter_v2_samples<<<grid, block, 0, st>>>(
        im.dsamples, static_cast<long long>(sample_count),
        sigma2_present ? 1 : 0, affine6[0], affine6[1], affine6[2],
        affine6[3], affine6[4], affine6[5], sc, half, im.icols, im.irows,
        im.cfg.bayer_pattern, im.cfg.cfa_origin_x, im.cfg.cfa_origin_y,
        im.cfg.mono ? 1 : 0,
        static_cast<double>(im.cfg.band_origin_x_native), band_oy, im.fa,
        im.fbs, im.fbg, im.fs2, aq, qio, im.scalars);
    const long long fold_n =
        static_cast<long long>(im.ncols) * (rw.ny1 - rw.ny0);
    const unsigned int fgrid =
        static_cast<unsigned int>((fold_n + block - 1) / block);
    k_fold_accumulate_v2<<<fgrid, block, 0, st>>>(
        im.fa, im.fbs, im.fbg, im.fs2,
        q_frame ? im.fqc : nullptr, q_frame ? im.fq0 : nullptr,
        q_frame ? im.fq1 : nullptr, q_frame ? im.fqa : nullptr,
        q_frame ? im.fqaf : nullptr, qmask, im.icols, im.irows,
        im.cfg.internal_scale, im.ncols, im.nrows, im.channels, 0, im.ncols,
        rw.ny0, rw.ny1 - rw.ny0, frame_order, keep_all ? 1 : 0, threshold,
        im.res_slots, im.cfg.reservoir_seed, im.accA, im.accB, im.accB2,
        im.covB, im.covB2, im.confS, im.confC, im.contrib, im.kept,
        im.footprint, im.supp, im.degraded, im.res, im.resq, im.dfull,
        im.meta, static_cast<double>(im.cfg.fine_quality_exponent),
        static_cast<double>(im.cfg.medium_quality_exponent),
        im.cfg.shared_frame_rejection ? 1 : 0,
        static_cast<double>(im.cfg.shared_frame_rejection_consensus));
  }
  ok = cudaGetLastError() == cudaSuccess &&
       cudaEventRecord(im.ev_k1[f], st) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  im.event_recorded[f] = 1;
  ++impl_->frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  stats_.source_samples_launched += sample_count;
  stats_.affine_samples_processed += sample_count;
  stats_.affine_span_rows += span_rows;
  stats_.source_bytes_uploaded += static_cast<std::uint64_t>(rec_bytes);
  if (q_frame) {
    ++stats_.quality_frames_processed;
    const std::uint64_t per_stream =
        static_cast<std::uint64_t>(sample_count) * 3u;
    for (int k = 0; k < 4; ++k)
      if ((qmask & (1u << k)) != 0) stats_.quality_bytes_uploaded += per_stream;
  }
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::begin_affine_frame(
    std::uint64_t frame_order, const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || frame_open_ || impl_->finalized ||
      impl_->frames >= impl_->cfg.stream_length ||
      frame_order >= impl_->cfg.stream_length)
    return false;
  Impl &im = *impl_;
  const std::uint64_t f = im.frames;
  if (im.cfg.emit_profiles) {
    // Same meta contract as the one-shot path; the row is uploaded once at
    // frame begin and timed by ev_up0 -> ev_up1.
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
    V2FrameMetaDev h_meta{};
    h_meta.g_eff = meta_or_null->g_eff;
    h_meta.residual_factor = meta_or_null->residual_factor;
    h_meta.is_direct = meta_or_null->is_direct;
    if (cudaEventRecord(im.ev_up0[f], im.stream) != cudaSuccess ||
        cudaMemcpyAsync(im.meta + frame_order, &h_meta,
                        sizeof(V2FrameMetaDev), cudaMemcpyHostToDevice,
                        im.stream) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  } else if (cudaEventRecord(im.ev_up0[f], im.stream) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const std::uint64_t n = im.cfg.stream_length;
  const bool keep_all =
      n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
  const std::uint64_t threshold =
      keep_all ? 0
               : static_cast<std::uint64_t>(
                     (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                      << 64) /
                     n);
  frame_open_ = true;
  open_order_ = frame_order;
  open_pieces_ = 0;
  open_tx_end_ = 0;
  open_qmask_ = 0;
  const bool pilot_frame =
      keep_all ||
      d_splitmix64(frame_order ^ im.cfg.reservoir_seed) < threshold;
  if (im.cfg.full_frame_estimator && pilot_frame == im.pilot_done) {
    frame_open_ = false;
    return false;
  }
  open_qframe_ = im.cfg.emit_profiles &&
                 (pilot_frame || im.cfg.full_frame_estimator);
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::accumulate_affine_piece(
    const double affine6[6], int target_x_begin_native,
    int target_cols_native, const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    const ForwardDrizzleV2FrameQuality *quality_or_null) {
  if (impl_ == nullptr || !frame_open_ || affine6 == nullptr ||
      source == nullptr)
    return false;
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  // Ordered, non-overlapping, in-bounds native target range.
  if (target_x_begin_native < 0 || target_cols_native <= 0 ||
      target_x_begin_native < open_tx_end_ ||
      target_x_begin_native + target_cols_native > im.ncols)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  // Same window contract as accumulate_frame_impl.
  if (window.x_begin < 0 || window.y_begin < 0 || window.width <= 0 ||
      window.height <= 0 ||
      window.x_begin + window.width > im.source_w ||
      window.y_begin + window.height > im.source_h)
    return false;
  V2Window w;
  w.buf_x = window.x_begin;
  w.buf_y = window.y_begin;
  w.buf_w = window.width;
  w.buf_h = window.height;
  w.act_x = window.active_x;
  w.act_y = window.active_y;
  w.act_w = window.active_width;
  w.act_h = window.active_height;
  if (w.act_w == 0 && w.act_h == 0 && w.act_x == 0 && w.act_y == 0) {
    w.act_w = w.buf_w;
    w.act_h = w.buf_h;
  }
  if (w.act_x < 0 || w.act_y < 0 || w.act_w <= 0 || w.act_h <= 0 ||
      w.act_x + w.act_w > w.buf_w || w.act_y + w.act_h > w.buf_h)
    return false;
  const int abs_x0 = w.buf_x + w.act_x;
  const int abs_y0 = w.buf_y + w.act_y;
  w.src_w = im.source_w;
  w.src_h = im.source_h;
  V2Sigma2Model s2m{};
  if (sigma2_model_or_null != nullptr && sigma2_model_or_null->enabled) {
    s2m.enabled = 1;
    s2m.noise = sigma2_model_or_null->sigma_noise;
    s2m.reg_px = sigma2_model_or_null->sigma_reg_px;
    s2m.half = sigma2_model_or_null->droplet_half;
  }
  if (sigma2_or_null != nullptr && s2m.enabled) return false;
  // Without float_plane_inputs no float sigma2/quality input planes exist.
  if (!im.cfg.float_plane_inputs &&
      (sigma2_or_null != nullptr ||
       (quality_or_null != nullptr &&
        (quality_or_null->q_composite != nullptr ||
         quality_or_null->q_scale0 != nullptr ||
         quality_or_null->q_scale1 != nullptr ||
         quality_or_null->q_artifact != nullptr))))
    return false;
  const std::size_t src_bytes =
      static_cast<std::size_t>(w.buf_w) * w.buf_h * sizeof(float);
  const std::size_t act_bytes =
      static_cast<std::size_t>(w.act_w) * w.act_h * sizeof(float);
  const std::uint64_t f = impl_->frames;
  const bool q_frame = open_qframe_;
  ForwardDrizzleV2FrameQuality h_q{};
  if (q_frame && quality_or_null != nullptr) h_q = *quality_or_null;
  auto pq_ok = [&](const ForwardDrizzleV2PackedQualityPlane &p,
                   const float *fp) {
    if (p.cells == nullptr) return true;
    if (fp != nullptr) return false;
    if (p.storage_width <= 0 || p.storage_height <= 0 ||
        p.storage_divisor <= 0)
      return false;
    const int d = p.storage_divisor;
    return p.storage_x_begin <= abs_x0 / d &&
           p.storage_x_begin + p.storage_width >
               (abs_x0 + w.act_w - 1) / d &&
           p.storage_y_begin <= abs_y0 / d &&
           p.storage_y_begin + p.storage_height >
               (abs_y0 + w.act_h - 1) / d;
  };
  if (!pq_ok(h_q.qc_packed, h_q.q_composite) ||
      !pq_ok(h_q.q0_packed, h_q.q_scale0) ||
      !pq_ok(h_q.q1_packed, h_q.q_scale1) ||
      !pq_ok(h_q.qa_packed, h_q.q_artifact))
    return false;
  const unsigned int qmask =
      ((h_q.q_composite != nullptr || h_q.qc_packed.cells != nullptr) ? 1u
                                                                    : 0u) |
      ((h_q.q_scale0 != nullptr || h_q.q0_packed.cells != nullptr) ? 2u
                                                                 : 0u) |
      ((h_q.q_scale1 != nullptr || h_q.q1_packed.cells != nullptr) ? 4u
                                                                 : 0u) |
      ((h_q.q_artifact != nullptr || h_q.qa_packed.cells != nullptr) ? 8u
                                                                   : 0u);
  // The stream presence is a per-frame contract: every piece must carry the
  // same qmask.
  if (open_pieces_ == 0) {
    open_qmask_ = qmask;
  } else if (qmask != open_qmask_) {
    return false;
  }
  auto upload = [&](float *dst, const float *src) {
    return src == nullptr ||
           cudaMemcpyAsync(dst, src, act_bytes, cudaMemcpyHostToDevice, st) ==
               cudaSuccess;
  };
  const std::size_t src_cap =
      static_cast<std::size_t>(im.source_w) * im.source_h;
  auto upload_packed = [&](unsigned short *d_cells, unsigned char *d_veto,
                           const ForwardDrizzleV2PackedQualityPlane &p) {
    if (p.cells == nullptr) return true;
    const std::size_t cn = static_cast<std::size_t>(p.storage_width) *
                           p.storage_height;
    if (cn > src_cap) return false;  // reserved device window bound
    return cudaMemcpyAsync(d_cells, p.cells, cn * sizeof(unsigned short),
                           cudaMemcpyHostToDevice, st) == cudaSuccess &&
           (p.veto == nullptr ||
            cudaMemcpyAsync(d_veto, p.veto, cn, cudaMemcpyHostToDevice, st) ==
                cudaSuccess);
  };
  bool ok =
      cudaMemcpyAsync(im.src, source, src_bytes, cudaMemcpyHostToDevice,
                      st) == cudaSuccess &&
      (!im.s2 || !sigma2_or_null ||
       cudaMemcpyAsync(im.s2, sigma2_or_null, act_bytes,
                       cudaMemcpyHostToDevice, st) == cudaSuccess) &&
      upload(im.qc, h_q.q_composite) && upload(im.q0, h_q.q_scale0) &&
      upload(im.q1, h_q.q_scale1) && upload(im.qa, h_q.q_artifact) &&
      upload_packed(im.pqc, im.pvc, h_q.qc_packed) &&
      upload_packed(im.pq0, im.pv0, h_q.q0_packed) &&
      upload_packed(im.pq1, im.pv1, h_q.q1_packed) &&
      upload_packed(im.pqa, im.pva, h_q.qa_packed);
  const int block = 128;
  const int scale = im.cfg.internal_scale;
  // Internal x columns owned by this piece.
  const int ix0 = target_x_begin_native * scale;
  const int ix1 = (target_x_begin_native + target_cols_native) * scale;
  if (ok) {
    // Clear ONLY this tile's internal columns of the frame planes.
    const long long clear_n =
        static_cast<long long>(im.channels) * im.irows * (ix1 - ix0);
    const unsigned int clear_grid =
        static_cast<unsigned int>((clear_n + block - 1) / block);
    k_clear_planes_x<<<clear_grid, block, 0, st>>>(
        im.fa, im.fbs, im.fbg, im.fs2,
        q_frame ? im.fqc : nullptr, q_frame ? im.fq0 : nullptr,
        q_frame ? im.fq1 : nullptr, q_frame ? im.fqa : nullptr,
        q_frame ? im.fqaf : nullptr, im.icols, im.irows, im.channels, ix0,
        ix1);
    // Aggregate event quartet: upload-end/kernel-start only at the first
    // piece; later pieces' uploads land inside kernel_seconds.
    ok = cudaGetLastError() == cudaSuccess &&
         (open_pieces_ > 0 ||
          cudaEventRecord(im.ev_up1[f], st) == cudaSuccess) &&
         (open_pieces_ > 0 ||
          cudaEventRecord(im.ev_k0[f], st) == cudaSuccess);
  }
  if (ok) {
    const long long source_n =
        static_cast<long long>(w.act_w) * w.act_h;
    const unsigned int grid =
        static_cast<unsigned int>((source_n + block - 1) / block);
    V2QualityIO qio;
    if (q_frame) {
      qio.qc = h_q.q_composite != nullptr ? im.qc.get() : nullptr;
      qio.q0 = h_q.q_scale0 != nullptr ? im.q0.get() : nullptr;
      qio.q1 = h_q.q_scale1 != nullptr ? im.q1.get() : nullptr;
      qio.qa = h_q.q_artifact != nullptr ? im.qa.get() : nullptr;
      auto set_packed = [](V2PackedQuality &d, unsigned short *cells,
                           unsigned char *veto,
                           const ForwardDrizzleV2PackedQualityPlane &p) {
        if (p.cells == nullptr) return;
        d.cells = cells;
        d.veto = p.veto != nullptr ? veto : nullptr;
        d.x_begin = p.storage_x_begin;
        d.y_begin = p.storage_y_begin;
        d.width = p.storage_width;
        d.height = p.storage_height;
        d.divisor = p.storage_divisor;
      };
      set_packed(qio.qc_p, im.pqc, im.pvc, h_q.qc_packed);
      set_packed(qio.q0_p, im.pq0, im.pv0, h_q.q0_packed);
      set_packed(qio.q1_p, im.pq1, im.pv1, h_q.q1_packed);
      set_packed(qio.qa_p, im.pqa, im.pva, h_q.qa_packed);
      qio.qmask = qmask;
      qio.fqc = im.fqc;
      qio.fq0 = im.fq0;
      qio.fq1 = im.fq1;
      qio.fqa = im.fqa;
      qio.fqaf = im.fqaf;
    }
    const float *d_s2 =
        (im.fs2 && !s2m.enabled && sigma2_or_null != nullptr) ? im.s2.get()
                                                               : nullptr;
    k_scatter_v2<<<grid, block, 0, st>>>(
        affine6[0], affine6[1], affine6[2], affine6[3], affine6[4],
        affine6[5], static_cast<double>(scale), im.cfg.half, im.icols,
        im.irows, w, im.src, d_s2, s2m, im.cfg.bayer_pattern,
        im.cfg.cfa_origin_x, im.cfg.cfa_origin_y, im.cfg.mono ? 1 : 0,
        static_cast<double>(im.cfg.band_origin_x_native),
        static_cast<double>(im.cfg.band_origin_y_native), ix0, ix1, im.fa,
        im.fbs, im.fbg, im.fs2, qio, im.scalars);
    // Fold only this tile's native columns; each output pixel is folded
    // exactly once for the frame across its pieces.
    const long long fold_n =
        static_cast<long long>(target_cols_native) * im.nrows;
    const unsigned int fold_grid =
        static_cast<unsigned int>((fold_n + block - 1) / block);
    const std::uint64_t n = im.cfg.stream_length;
    const bool keep_all =
        n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
    const std::uint64_t threshold =
        keep_all ? 0
                 : static_cast<std::uint64_t>(
                       (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                        << 64) /
                       n);
    k_fold_accumulate_v2<<<fold_grid, block, 0, st>>>(
        im.fa, im.fbs, im.fbg, im.fs2, q_frame ? im.fqc : nullptr,
        q_frame ? im.fq0 : nullptr, q_frame ? im.fq1 : nullptr,
        q_frame ? im.fqa : nullptr, q_frame ? im.fqaf : nullptr, qmask,
        im.icols, im.irows, scale, im.ncols, im.nrows, im.channels,
        target_x_begin_native, target_cols_native, 0, im.nrows, open_order_,
        keep_all ? 1 : 0, threshold, im.res_slots, im.cfg.reservoir_seed,
        im.accA, im.accB, im.accB2, im.covB, im.covB2, im.confS, im.confC,
        im.contrib, im.kept, im.footprint, im.supp, im.degraded, im.res,
        im.resq, im.dfull, im.meta,
        static_cast<double>(im.cfg.fine_quality_exponent),
        static_cast<double>(im.cfg.medium_quality_exponent),
        im.cfg.shared_frame_rejection ? 1 : 0,
        static_cast<double>(im.cfg.shared_frame_rejection_consensus));
    ok = cudaGetLastError() == cudaSuccess &&
         cudaEventRecord(im.ev_k1[f], st) == cudaSuccess;
  }
  if (!ok) {
    cudaGetLastError();
    return false;
  }
  im.event_recorded[f] = 1;
  ++open_pieces_;
  open_tx_end_ = target_x_begin_native + target_cols_native;
  ++stats_.affine_pieces_processed;
  const std::size_t source_n_sz =
      static_cast<std::size_t>(w.act_w) * w.act_h;
  stats_.source_bytes_uploaded += src_bytes +
      (sigma2_or_null != nullptr ? act_bytes : 0);
  stats_.source_samples_launched += source_n_sz;
  if (q_frame) {
    const std::uint64_t qfb =
        static_cast<std::uint64_t>(w.act_w) * w.act_h * sizeof(float);
    auto qb = [&](const float *fp,
                  const ForwardDrizzleV2PackedQualityPlane &p)
        -> std::uint64_t {
      if (p.cells != nullptr)
        return static_cast<std::uint64_t>(p.storage_width) *
               p.storage_height * 3u;
      return fp != nullptr ? qfb : 0u;
    };
    stats_.quality_bytes_uploaded +=
        qb(h_q.q_composite, h_q.qc_packed) +
        qb(h_q.q_scale0, h_q.q0_packed) +
        qb(h_q.q_scale1, h_q.q1_packed) +
        qb(h_q.q_artifact, h_q.qa_packed);
  }
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::finish_affine_frame(
    std::uint64_t frame_order) {
  if (impl_ == nullptr || !frame_open_ || frame_order != open_order_ ||
      open_pieces_ == 0)
    return false;
  frame_open_ = false;
  ++impl_->frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  if (open_qframe_) ++stats_.quality_frames_processed;
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::skip_frame(
    std::uint64_t frame_order, const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || frame_open_ || impl_->finalized ||
      impl_->frames >= impl_->cfg.stream_length ||
      frame_order >= impl_->cfg.stream_length)
    return false;
  if (impl_->cfg.emit_profiles) {
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
    V2FrameMetaDev h_meta{};
    h_meta.g_eff = meta_or_null->g_eff;
    h_meta.residual_factor = meta_or_null->residual_factor;
    h_meta.is_direct = meta_or_null->is_direct;
    if (cudaMemcpyAsync(impl_->meta + frame_order, &h_meta,
                        sizeof(V2FrameMetaDev), cudaMemcpyHostToDevice,
                        impl_->stream) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  }
  ++impl_->frames;
  ++stats_.slot_transitions;
  ++stats_.frames_skipped_empty_window;
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::end_pilot() {
  last_device_error_.clear();
  if (impl_ == nullptr || frame_open_ || impl_->finalized ||
      !impl_->cfg.full_frame_estimator || impl_->pilot_done)
    return false;
  Impl &im = *impl_;
  cudaStream_t st = im.stream;
  const std::size_t n_pc = im.nplane * static_cast<std::size_t>(im.channels);
  const int block = 128;
  const unsigned int grid = static_cast<unsigned int>(
      (static_cast<long long>(n_pc) + block - 1) / block);
  // Build with min_candidates = 0: the all-frame contributor guard is
  // evaluated at finalize, exactly like the CPU end_pilot().
  k_finalize_v2_sfr_build<<<grid, block, 0, st>>>(
      im.ncols, im.nrows, im.channels,
      im.cfg.internal_scale * im.cfg.internal_scale, im.res_slots, im.frames,
      0, im.cfg.min_clip_contributors, im.cfg.robust_passes,
      im.cfg.sigma_low, im.cfg.sigma_high, im.accA, im.accB, im.accB2,
      im.confS, im.confC, im.contrib, im.kept, im.footprint, im.supp,
      im.degraded, im.res, im.pout, im.out, im.fcounters + 4,
      im.sfr_has_clip, im.sfr_accepted, im.dbounds,
      im.cfg.bimodal_veto ? 1 : 0, im.cfg.bimodal_veto_gap_sigma,
      im.bimodal_counters, im.bimodal_counters + 1);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    last_device_error_ = std::string("end_pilot build launch: ") +
                         cudaGetErrorString(e);
    return false;
  }
  if (im.channels > 1 && im.cfg.shared_frame_rejection) {
    const unsigned int grid_px = static_cast<unsigned int>(
        (static_cast<long long>(im.nplane) + block - 1) / block);
    k_finalize_v2_sfr_vote<<<grid_px, block, 0, st>>>(
        im.ncols, im.nrows, im.channels, im.res_slots,
        static_cast<double>(im.cfg.shared_frame_rejection_consensus), im.kept,
        im.res, im.sfr_has_clip, im.sfr_accepted);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      last_device_error_ = std::string("end_pilot vote launch: ") +
                           cudaGetErrorString(e);
      return false;
    }
  }
  k_full_seed<<<grid, block, 0, st>>>(
      im.ncols, im.nrows, im.channels, im.res_slots, im.kept, im.contrib,
      im.res, im.resq, im.meta, im.cfg.stream_length,
      static_cast<double>(im.cfg.fine_quality_exponent),
      static_cast<double>(im.cfg.medium_quality_exponent), im.sfr_has_clip,
      im.sfr_accepted, im.dbounds, im.dfull, im.fcounters);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    last_device_error_ = std::string("end_pilot seed launch: ") +
                         cudaGetErrorString(e);
    return false;
  }
  im.pilot_done = true;
  return true;
}

bool ForwardDrizzleV2CudaPrototypeKernel::finalize(
    ForwardDrizzleV2PixelResult *results,
    ForwardDrizzleV2ProfileResult *profiles_or_null,
    std::uint64_t *dense_overlap_count) {
  if (impl_ == nullptr || results == nullptr || impl_->finalized ||
      frame_open_)
    return false;
  Impl &im = *impl_;
  if (im.cfg.emit_profiles && profiles_or_null == nullptr) return false;
  if (im.cfg.full_frame_estimator && !im.pilot_done) return false;
  cudaStream_t st = im.stream;
  const std::size_t n_pc = im.nplane * static_cast<std::size_t>(im.channels);
  const int block = 128;
  const long long total = static_cast<long long>(n_pc);
  const unsigned int grid =
      static_cast<unsigned int>((total + block - 1) / block);
  // The full-frame estimator always finalizes through the build/vote/reduce
  // kernels (the pilot bounds come from the same clip); the consensus vote
  // only runs when shared_frame_rejection is on.
  const bool use_sfr =
      im.cfg.shared_frame_rejection || im.cfg.full_frame_estimator;
  if (!use_sfr) {
    k_finalize_v2<<<grid, block, 0, st>>>(
        im.ncols, im.nrows, im.channels,
        im.cfg.internal_scale * im.cfg.internal_scale, im.res_slots,
        im.frames, im.cfg.stream_length,
        im.cfg.min_candidates, im.cfg.min_clip_contributors,
        im.cfg.robust_passes, im.cfg.sigma_low, im.cfg.sigma_high, im.accA,
        im.accB, im.accB2, im.confS, im.confC, im.contrib, im.kept,
        im.footprint, im.supp, im.degraded, im.res, im.resq, im.meta,
        static_cast<double>(im.cfg.fine_quality_exponent),
        static_cast<double>(im.cfg.medium_quality_exponent),
        AlphaConfidenceParams{}, im.pout, im.out, im.scalars + 1,
        im.cfg.bimodal_veto ? 1 : 0, im.cfg.bimodal_veto_gap_sigma,
        im.bimodal_counters, im.bimodal_counters + 1);
    if (const cudaError_t launch_err = cudaGetLastError();
        launch_err != cudaSuccess) {
      last_device_error_ = std::string("k_finalize_v2 launch: ") +
                           cudaGetErrorString(launch_err);
      return false;
    }
  } else {
    k_finalize_v2_sfr_build<<<grid, block, 0, st>>>(
        im.ncols, im.nrows, im.channels,
        im.cfg.internal_scale * im.cfg.internal_scale, im.res_slots,
        im.frames,
        im.cfg.min_candidates, im.cfg.min_clip_contributors,
        im.cfg.robust_passes, im.cfg.sigma_low, im.cfg.sigma_high, im.accA,
        im.accB, im.accB2, im.confS, im.confC, im.contrib, im.kept,
        im.footprint, im.supp, im.degraded, im.res, im.pout, im.out,
        im.scalars + 1, im.sfr_has_clip, im.sfr_accepted, nullptr,
        im.cfg.bimodal_veto ? 1 : 0, im.cfg.bimodal_veto_gap_sigma,
        im.bimodal_counters, im.bimodal_counters + 1);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      last_device_error_ =
          std::string("k_finalize_v2_sfr_build launch: ") +
          cudaGetErrorString(e);
      return false;
    }
    if (im.channels > 1 && im.cfg.shared_frame_rejection) {
      const unsigned int grid_px = static_cast<unsigned int>(
          (static_cast<long long>(im.nplane) + block - 1) / block);
      k_finalize_v2_sfr_vote<<<grid_px, block, 0, st>>>(
          im.ncols, im.nrows, im.channels, im.res_slots,
          static_cast<double>(im.cfg.shared_frame_rejection_consensus),
          im.kept, im.res, im.sfr_has_clip, im.sfr_accepted);
      if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        last_device_error_ = std::string("k_finalize_v2_sfr_vote launch: ") +
                             cudaGetErrorString(e);
        return false;
      }
    }
    k_finalize_v2_sfr_reduce<<<grid, block, 0, st>>>(
        im.ncols, im.nrows, im.channels, im.res_slots, im.cfg.stream_length,
        im.kept, im.res, im.resq, im.meta,
        static_cast<double>(im.cfg.fine_quality_exponent),
        static_cast<double>(im.cfg.medium_quality_exponent),
        AlphaConfidenceParams{}, im.sfr_has_clip, im.sfr_accepted, im.pout,
        im.out);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      last_device_error_ = std::string("k_finalize_v2_sfr_reduce launch: ") +
                           cudaGetErrorString(e);
      return false;
    }
    if (im.cfg.full_frame_estimator) {
      k_full_apply<<<grid, block, 0, st>>>(
          im.ncols, im.nrows, im.channels, im.contrib, im.degraded,
          im.dfull, im.out, im.pout, im.fcounters);
      if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        last_device_error_ = std::string("k_full_apply launch: ") +
                             cudaGetErrorString(e);
        return false;
      }
    }
  }
  {
    // Block size must stay a multiple of 32 (full-warp shuffles).
    const unsigned int sum_grid = static_cast<unsigned int>(std::min<long long>(
        (total + block - 1) / block, 1024));
    k_sum_kept_contrib<<<sum_grid, block, 0, st>>>(im.kept, im.contrib, total,
                                                  im.scalars + 3);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      last_device_error_ = std::string("k_sum_kept_contrib launch: ") +
                           cudaGetErrorString(e);
      return false;
    }
  }
  unsigned long long h_scalars[kV2ScalarCount] = {};
  unsigned long long h_fcounters[5] = {0, 0, 0, 0, 0};
  unsigned long long h_bimodal[2] = {0, 0};
  const std::size_t out_bytes = n_pc * sizeof(ForwardDrizzleV2PixelResult);
  const std::size_t prof_bytes =
      im.cfg.emit_profiles ? n_pc * sizeof(ForwardDrizzleV2ProfileResult) : 0;
  cudaError_t first_err = cudaSuccess;
  auto step = [&](const char *api, cudaError_t e) {
    if (e != cudaSuccess && first_err == cudaSuccess) {
      first_err = e;
      last_device_error_ =
          std::string(api) + ": " + cudaGetErrorString(e);
    }
    return e == cudaSuccess;
  };
  bool ok =
      step("cudaMemcpyAsync(results)",
           cudaMemcpyAsync(results, im.out, out_bytes, cudaMemcpyDeviceToHost,
                           st)) &&
      (!im.cfg.emit_profiles ||
       step("cudaMemcpyAsync(profiles)",
            cudaMemcpyAsync(profiles_or_null, im.pout, prof_bytes,
                            cudaMemcpyDeviceToHost, st))) &&
      step("cudaMemcpyAsync(scalars)",
           cudaMemcpyAsync(h_scalars, im.scalars, sizeof(h_scalars),
                           cudaMemcpyDeviceToHost, st)) &&
      (!im.cfg.full_frame_estimator ||
       step("cudaMemcpyAsync(full counters)",
            cudaMemcpyAsync(h_fcounters, im.fcounters, sizeof(h_fcounters),
                            cudaMemcpyDeviceToHost, st))) &&
      step("cudaMemcpyAsync(bimodal counters)",
           cudaMemcpyAsync(h_bimodal, im.bimodal_counters, sizeof(h_bimodal),
                           cudaMemcpyDeviceToHost, st));
  ++stats_.stream_synchronizations;
  ok = step("cudaStreamSynchronize", cudaStreamSynchronize(st)) && ok;
  if (!ok) {
    // Sticky errors from earlier async work in this band surface here; when
    // no API call above reported one, keep whatever the runtime has pending.
    if (first_err == cudaSuccess) {
      const cudaError_t sticky = cudaGetLastError();
      if (sticky != cudaSuccess)
        last_device_error_ =
            std::string("finalize pending error: ") +
            cudaGetErrorString(sticky);
    } else {
      cudaGetLastError();
    }
    return false;
  }
  im.finalized = true;
  for (std::uint64_t f = 0; f < im.frames; ++f) {
    // Skip frames and orders never recorded this band have no valid event
    // pair; never read stale prior-band timings.
    if (!im.event_recorded[f]) continue;
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
  if (im.cfg.full_frame_estimator) {
    stats_.full_frame_accepted = h_fcounters[0];
    stats_.full_frame_rejected = h_fcounters[1];
    stats_.full_frame_degenerate_pilot = h_fcounters[2];
    stats_.full_frame_no_bounds = h_fcounters[3];
  }
  stats_.bimodal_veto_rejected_candidates = h_bimodal[0];
  stats_.bimodal_veto_pixels = h_bimodal[1];
  stats_.positive_overlaps = h_scalars[0];
  if (dense_overlap_count) *dense_overlap_count = h_scalars[1];
  stats_.local_samples_discarded = h_scalars[2];
  stats_.reservoir_kept_total = h_scalars[3];
  stats_.candidates_streamed = h_scalars[4];
  stats_.result_bytes_downloaded += out_bytes + prof_bytes + sizeof(h_scalars);
  return true;
}

}  // namespace tile_compile::reconstruction

#endif  // TILE_COMPILE_WITH_CUDA
