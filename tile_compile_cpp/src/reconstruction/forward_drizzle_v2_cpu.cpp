// Gate-10 host CPU port of the forward-drizzle v2 band kernel.
//
// The routines below are 1:1 host ports of k_scatter_v2 /
// k_scatter_v2_local / k_fold_accumulate_v2 / k_finalize_v2 from
// forward_drizzle_cuda_device.cu: same accumulation order, same keep
// predicate, same clip and the same shared profile reduction. The local-warp
// scatter calls the production inversion oracle invert_local_source_to_canvas
// directly (the device d_invert_local is its bit-faithful port); subdivision
// mirrors subdivide_local's accept/subdivide/fail contract.

#include "tile_compile/reconstruction/forward_drizzle_v2_cpu.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "tile_compile/core/types.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

namespace tile_compile::reconstruction {

namespace {

constexpr int kV2MaxResSlots = 128;

struct CpuReservoirRecord {
  double x = 0.0;
  double b = 0.0;
  std::uint64_t order = 0;
  double sigma2 = 0.0;
};

// Per-kept-candidate quality means (slot-parallel to the reservoir record):
// qc/q0/q1 are folded means or 1.0 when the stream is absent; qa carries -1
// for "no finite artifact sample" exactly like the device float4 side array.
struct CpuReservoirQuality {
  float qc = 1.0f;
  float q0 = 1.0f;
  float q1 = 1.0f;
  float qa = -1.0f;
};

std::uint64_t cpu_splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

bool cpu_checked_mul(std::size_t a, std::size_t b, std::size_t &out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

// Decode one quality sample of a stream: float plane indexed by the
// active-local tid (compatibility path) or the compact storage-grid
// descriptor decoded by the absolute source coordinate. Absent streams and
// vetoed/zero cells yield NaN (the same veto the float path reports).
float cpu_quality_sample(const float *plane,
                         const ForwardDrizzleV2PackedQualityPlane &packed,
                         std::size_t tid, int sx, int sy) {
  if (packed.cells != nullptr) {
    const std::size_t i =
        static_cast<std::size_t>(sy / packed.storage_divisor -
                                 packed.storage_y_begin) *
            static_cast<std::size_t>(packed.storage_width) +
        static_cast<std::size_t>(sx / packed.storage_divisor -
                                 packed.storage_x_begin);
    if (packed.veto != nullptr && packed.veto[i])
      return std::numeric_limits<float>::quiet_NaN();
    const std::uint16_t c = packed.cells[i];
    return c == 0u ? std::numeric_limits<float>::quiet_NaN()
                   : static_cast<float>(c) / 65535.0f;
  }
  return plane != nullptr ? plane[tid]
                          : std::numeric_limits<float>::quiet_NaN();
}

// A packed quality descriptor is well-formed iff its storage window covers
// every storage cell the active rect maps to (absolute coords, divisor).
bool cpu_packed_q_covers(const ForwardDrizzleV2PackedQualityPlane &p,
                         int abs_x0, int abs_y0, int aw, int ah) {
  if (p.cells == nullptr) return true;  // absent stream
  if (p.storage_width <= 0 || p.storage_height <= 0 || p.storage_divisor <= 0)
    return false;
  const int d = p.storage_divisor;
  return p.storage_x_begin <= abs_x0 / d &&
         p.storage_x_begin + p.storage_width > (abs_x0 + aw - 1) / d &&
         p.storage_y_begin <= abs_y0 / d &&
         p.storage_y_begin + p.storage_height > (abs_y0 + ah - 1) / d;
}

// Quality accumulation of one droplet/cell pair, gated on finite source
// value by the caller (device d_scatter_quality port). `qv` holds the
// already-decoded per-stream sample values (NaN = veto/absent).
void cpu_scatter_quality(const float qv[4], unsigned int qmask,
                         std::size_t out, double area,
                         std::vector<double> &vqc, std::vector<double> &vq0,
                         std::vector<double> &vq1, std::vector<double> &vqa,
                         std::vector<double> &vqaf) {
  auto acc = [](float v, std::vector<double> &dst, std::size_t o, double k) {
    if (dst.empty()) return;
    const double d = static_cast<double>(v);
    dst[o] += k * (std::isfinite(d) && d > 0.0 ? d : 0.0);
  };
  if (qmask & 1u) acc(qv[0], vqc, out, area);
  if (qmask & 2u) acc(qv[1], vq0, out, area);
  if (qmask & 4u) acc(qv[2], vq1, out, area);
  if (qmask & 8u) acc(qv[3], vqa, out, area);
  if ((qmask & 8u) && std::isfinite(static_cast<double>(qv[3])))
    vqaf[out] += area;
}

// Rasterize one accepted leaf quad (internal coordinates) into the frame
// planes, clipped to the internal x column range [tile_x0, tile_x1)
// (tile_x1 <= cols). Returns the number of positive-overlap cells.
std::uint64_t cpu_scatter_leaf(const double *qx, const double *qy, int cols,
                               int rows, int channel, double value,
                               bool finite_value, double s2,
                               const float *qv, unsigned int qmask,
                               int iplane, int tile_x0, int tile_x1,
                               std::vector<double> &fa,
                               std::vector<double> &fbs,
                               std::vector<double> &fbg,
                               std::vector<double> &fs2,
                               std::vector<double> &fqc,
                               std::vector<double> &fq0,
                               std::vector<double> &fq1,
                               std::vector<double> &fqa,
                               std::vector<double> &fqaf) {
  double minx = std::numeric_limits<double>::max();
  double maxx = -std::numeric_limits<double>::max();
  double miny = std::numeric_limits<double>::max();
  double maxy = -std::numeric_limits<double>::max();
  for (int k = 0; k < 4; ++k) {
    minx = std::min(minx, qx[k]);
    maxx = std::max(maxx, qx[k]);
    miny = std::min(miny, qy[k]);
    maxy = std::max(maxy, qy[k]);
  }
  const int x0 =
      std::max(tile_x0, std::max(0, static_cast<int>(std::floor(minx))));
  const int x1 =
      std::min(tile_x1, std::min(cols, static_cast<int>(std::ceil(maxx))));
  const int y0 = std::max(0, static_cast<int>(std::floor(miny)));
  const int y1 = std::min(rows, static_cast<int>(std::ceil(maxy)));
  std::uint64_t overlaps = 0;
  const std::size_t base = static_cast<std::size_t>(channel) * iplane;
  for (int ty = y0; ty < y1; ++ty) {
    for (int tx = x0; tx < x1; ++tx) {
      const double area = polygon_rectangle_intersection_area(
          qx, qy, tx, ty, tx + 1.0, ty + 1.0);
      if (!(area > 0.0)) continue;
      const std::size_t out =
          base + static_cast<std::size_t>(ty) * cols + tx;
      fbg[out] += area;
      if (finite_value) {
        fa[out] += area * value;
        fbs[out] += area;
        if (!fs2.empty()) fs2[out] += area * s2;
        if (qmask != 0u && qv != nullptr)
          cpu_scatter_quality(qv, qmask, out, area, fqc, fq0, fq1, fqa, fqaf);
      }
      ++overlaps;
    }
  }
  return overlaps;
}

// subdivide_local port: evaluates a node via a 3x3 inversion grid, accepts
// when the bilinear error and child-area tests pass, rejects to the four
// children, fails at maximum depth. Returns 0=accepted (leaf corners in
// out_x/out_y, RAW native canvas coordinates --- the emit path applies the
// band-origin shift and internal scaling exactly like the device kernel),
// 1=needs subdivision, 2=sample failure.
int cpu_eval_local_node(const registration::FrameSamplingTransform &f,
                        const registration::LocalInversionParams &inv,
                        double x0, double y0, double x1, double y1, int depth,
                        int canvas_w_native, int canvas_h_native, double sc,
                        const ForwardDrizzleV2LocalWarp &w, double *out_x,
                        double *out_y) {
  double gx[3][3], gy[3][3];
  float fx[3][3], fy[3][3];
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      float qx = 0.0f, qy = 0.0f;
      if (!registration::invert_local_source_to_canvas(
              f, static_cast<float>(x0 + (x1 - x0) * i / 2),
              static_cast<float>(y0 + (y1 - y0) * j / 2), canvas_w_native,
              canvas_h_native, inv, qx, qy))
        return 2;
      fx[j][i] = qx;
      fy[j][i] = qy;
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
      error = std::max(error, std::hypot(gx[j][i] - bx, gy[j][i] - by));
    }
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      const double cx[4] = {gx[j][i], gx[j][i + 1], gx[j + 1][i + 1],
                            gx[j + 1][i]};
      const double cy[4] = {gy[j][i], gy[j][i + 1], gy[j + 1][i + 1],
                            gy[j + 1][i]};
      child_area += shoelace_area(cx, cy, 4);
    }
  const double area = shoelace_area(px, py, 4);
  if (area > 0.0 && error <= w.position_epsilon_internal_px &&
      std::fabs(child_area - area) / area <= w.area_relative_epsilon) {
    const int ci[4] = {0, 2, 2, 0};
    const int cj[4] = {0, 0, 2, 2};
    for (int k = 0; k < 4; ++k) {
      out_x[k] = static_cast<double>(fx[cj[k]][ci[k]]);
      out_y[k] = static_cast<double>(fy[cj[k]][ci[k]]);
    }
    return 0;
  }
  if (depth >= w.max_subdivision_depth) return 2;
  return 1;
}

// All-or-nothing leaf collection mirroring subdivide_local: a node failure
// anywhere discards the whole sample. Returns false on failure.
bool cpu_collect_local_leaves(
    const registration::FrameSamplingTransform &f,
    const registration::LocalInversionParams &inv,
    const ForwardDrizzleV2LocalWarp &w, double x0, double y0, double x1,
    double y1, int depth, int canvas_w_native, int canvas_h_native, double sc,
    std::vector<Leaf> &leaves) {
  Leaf leaf;
  const int status =
      cpu_eval_local_node(f, inv, x0, y0, x1, y1, depth, canvas_w_native,
                          canvas_h_native, sc, w, leaf.x, leaf.y);
  if (status == 2) return false;
  if (status == 0) {
    leaves.push_back(leaf);
    return true;
  }
  const double mx = x0 + (x1 - x0) * 0.5, my = y0 + (y1 - y0) * 0.5;
  return cpu_collect_local_leaves(f, inv, w, x0, y0, mx, my, depth + 1,
                                  canvas_w_native, canvas_h_native, sc,
                                  leaves) &&
         cpu_collect_local_leaves(f, inv, w, mx, y0, x1, my, depth + 1,
                                  canvas_w_native, canvas_h_native, sc,
                                  leaves) &&
         cpu_collect_local_leaves(f, inv, w, x0, my, mx, y1, depth + 1,
                                  canvas_w_native, canvas_h_native, sc,
                                  leaves) &&
         cpu_collect_local_leaves(f, inv, w, mx, my, x1, y1, depth + 1,
                                  canvas_w_native, canvas_h_native, sc,
                                  leaves);
}

}  // namespace

struct ForwardDrizzleV2CpuKernel::Impl {
  ForwardDrizzleV2KernelConfig cfg;
  // max_nrows is the reserved capacity; nrows/irows/nplane/iplane/pc_elems
  // are the ACTIVE band dimensions (<= reserved).
  int max_nrows = 0;
  int ncols = 0, nrows = 0, icols = 0, irows = 0;
  int source_w = 0, source_h = 0, channels = 0, res_slots = 0;
  std::size_t nplane = 0, iplane = 0, pc_elems = 0;
  std::uint64_t frames = 0;
  bool finalized = false;

  // Per-frame internal planes (channel-major).
  std::vector<double> fa, fbs, fbg, fs2;
  std::vector<double> fqc, fq0, fq1, fqa, fqaf;
  // Band accumulators, per (pixel, channel).
  std::vector<double> accA, accB, accB2, covB, covB2, confS, confC;
  std::vector<unsigned int> contrib, kept, footprint;
  std::vector<unsigned short> supp;
  std::vector<std::uint64_t> degraded;
  std::vector<CpuReservoirRecord> res;
  std::vector<CpuReservoirQuality> resq;
  std::vector<ForwardDrizzleV2FrameMeta> meta;

  ~Impl() = default;
};

ForwardDrizzleV2CpuKernel::ForwardDrizzleV2CpuKernel() = default;
ForwardDrizzleV2CpuKernel::~ForwardDrizzleV2CpuKernel() { delete impl_; }

bool ForwardDrizzleV2CpuKernel::reserve(int native_cols, int native_rows,
                                        int source_w, int source_h,
                                        const ForwardDrizzleV2KernelConfig &cfg) {
  if (impl_ != nullptr) return false;
  if (native_cols <= 0 || native_rows <= 0 || source_w <= 0 || source_h <= 0 ||
      cfg.internal_scale < 1 || cfg.internal_scale > 2 ||
      cfg.reservoir_size < 1 || cfg.reservoir_size > 64 ||
      cfg.stream_length == 0 || cfg.stream_length > 65536 ||
      cfg.min_clip_contributors < 1 || cfg.min_candidates < 1 ||
      cfg.robust_passes < 1 || !std::isfinite(cfg.sigma_low) ||
      !std::isfinite(cfg.sigma_high) || cfg.sigma_low <= 0.0 ||
      cfg.sigma_high <= 0.0 || !std::isfinite(cfg.half) ||
      !(cfg.half > 0.0) || cfg.bayer_pattern < 0 || cfg.bayer_pattern > 4 ||
      cfg.band_origin_x_native < 0 || cfg.band_origin_y_native < 0 ||
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

  const int scale = cfg.internal_scale;
  const long long icols = static_cast<long long>(native_cols) * scale;
  const long long irows = static_cast<long long>(native_rows) * scale;
  if (icols > 0x7fffffffLL || irows > 0x7fffffffLL) return false;
  const int channels = cfg.mono ? 1 : 3;
  // Exact global keep set bounds the retained candidates per pixel; the
  // historical 2*R cap is preserved so rare keep sets larger than 2R still
  // drive the kept > res_slots overflow fallback at finalize.
  const auto selected = forward_drizzle_v2_selected_frame_orders(
      cfg.stream_length, cfg.reservoir_size, cfg.reservoir_seed);
  const std::size_t historical_cap =
      2 * static_cast<std::size_t>(cfg.reservoir_size);
  const int res_slots = static_cast<int>(
      std::max<std::size_t>(1, std::min(selected.size(), historical_cap)));
  std::size_t nplane = 0, iplane = 0, pc_elems = 0, frame_plane_elems = 0;
  if (!cpu_checked_mul(static_cast<std::size_t>(native_cols),
                       static_cast<std::size_t>(native_rows), nplane) ||
      !cpu_checked_mul(static_cast<std::size_t>(icols),
                       static_cast<std::size_t>(irows), iplane) ||
      !cpu_checked_mul(nplane, static_cast<std::size_t>(channels),
                       pc_elems) ||
      !cpu_checked_mul(iplane, static_cast<std::size_t>(channels),
                       frame_plane_elems) ||
      pc_elems > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(res_slots))
    return false;

  auto im = std::make_unique<Impl>();
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
  im->pc_elems = pc_elems;

  const std::size_t res_elems =
      pc_elems * static_cast<std::size_t>(res_slots);
  std::size_t total_bytes = 0;
  auto account = [&](std::size_t elems, std::size_t sz) {
    std::size_t tmp = 0;
    if (!cpu_checked_mul(elems, sz, tmp) ||
        tmp > std::numeric_limits<std::size_t>::max() - total_bytes)
      return false;
    total_bytes += tmp;
    return true;
  };
  const std::size_t frame_planes =
      (cfg.sigma2_plane ? 4 : 3) + (cfg.emit_profiles ? 5 : 0);
  if (!account(frame_plane_elems, sizeof(double) * frame_planes) ||
      !account(pc_elems, sizeof(double) * 7) ||
      !account(pc_elems, sizeof(unsigned int) * 2) ||
      !account(nplane, sizeof(unsigned int)) ||
      !account(pc_elems, sizeof(unsigned short)) ||
      !account(pc_elems, sizeof(std::uint64_t)) ||
      !account(res_elems, sizeof(CpuReservoirRecord)) ||
      (cfg.emit_profiles &&
       (!account(res_elems, sizeof(CpuReservoirQuality)) ||
        !account(static_cast<std::size_t>(cfg.stream_length),
                 sizeof(ForwardDrizzleV2FrameMeta)) ||
        !account(pc_elems, sizeof(ForwardDrizzleV2ProfileResult)))) ||
      !account(pc_elems, sizeof(ForwardDrizzleV2PixelResult)))
    return false;

  try {
    im->fa.assign(frame_plane_elems, 0.0);
    im->fbs.assign(frame_plane_elems, 0.0);
    im->fbg.assign(frame_plane_elems, 0.0);
    if (cfg.sigma2_plane) im->fs2.assign(frame_plane_elems, 0.0);
    if (cfg.emit_profiles) {
      im->fqc.assign(frame_plane_elems, 0.0);
      im->fq0.assign(frame_plane_elems, 0.0);
      im->fq1.assign(frame_plane_elems, 0.0);
      im->fqa.assign(frame_plane_elems, 0.0);
      im->fqaf.assign(frame_plane_elems, 0.0);
    }
    im->accA.assign(pc_elems, 0.0);
    im->accB.assign(pc_elems, 0.0);
    im->accB2.assign(pc_elems, 0.0);
    im->covB.assign(pc_elems, 0.0);
    im->covB2.assign(pc_elems, 0.0);
    im->confS.assign(pc_elems, 0.0);
    im->confC.assign(pc_elems, 0.0);
    im->contrib.assign(pc_elems, 0u);
    im->kept.assign(pc_elems, 0u);
    im->footprint.assign(nplane, 0u);
    im->supp.assign(pc_elems, 0u);
    im->degraded.assign(pc_elems, 0ull);
    im->res.assign(res_elems, CpuReservoirRecord{});
    if (cfg.emit_profiles) {
      im->resq.assign(res_elems, CpuReservoirQuality{});
      im->meta.assign(static_cast<std::size_t>(cfg.stream_length),
                      ForwardDrizzleV2FrameMeta{});
    }
  } catch (const std::bad_alloc &) {
    return false;
  }

  stats_ = ForwardDrizzleV2PrototypeStats{};
  stats_.allocations = 1;
  stats_.workspace_reservations = 1;
  stats_.reserved_device_bytes = total_bytes;
  bytes_per_native_pixel_ =
      nplane > 0 ? total_bytes / nplane : total_bytes;
  capacity_bytes_ = total_bytes;
  pending_reservation_ = true;
  impl_ = im.release();
  return true;
}

// Fields begin_band may change: native_rows and band_origin_y_native. Every
// other config field is part of the reserved workspace's fixed contract.
static bool v2_fixed_cfg_equal(const ForwardDrizzleV2KernelConfig &a,
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
         a.cached_leaf_capacity == b.cached_leaf_capacity;
}

bool ForwardDrizzleV2CpuKernel::begin_band(
    int native_rows, const ForwardDrizzleV2KernelConfig &cfg) {
  if (impl_ == nullptr) return false;
  Impl &im = *impl_;
  // Legal only right after reserve() (frames==0, not yet finalized) or
  // after a successful finalize(). A partial/failed frame stream cannot be
  // rebound --- nor can a band end mid piece frame.
  if (frame_open_ || (!im.finalized && im.frames != 0)) return false;
  if (native_rows < 1 || native_rows > im.max_nrows ||
      !v2_fixed_cfg_equal(im.cfg, cfg) || cfg.band_origin_y_native < 0 ||
      (cfg.canvas_height_native > 0 &&
       cfg.band_origin_y_native + native_rows > cfg.canvas_height_native) ||
      (cfg.canvas_height_native == 0 && cfg.band_origin_y_native != 0))
    return false;
  const int scale = im.cfg.internal_scale;
  im.cfg = cfg;
  im.nrows = native_rows;
  im.irows = native_rows * scale;
  im.nplane = static_cast<std::size_t>(im.ncols) * im.nrows;
  im.iplane = static_cast<std::size_t>(im.icols) * im.irows;
  im.pc_elems = im.nplane * static_cast<std::size_t>(im.channels);
  // Clear the active prefix of every per-band role; no vector is resized.
  std::fill_n(im.accA.begin(), im.pc_elems, 0.0);
  std::fill_n(im.accB.begin(), im.pc_elems, 0.0);
  std::fill_n(im.accB2.begin(), im.pc_elems, 0.0);
  std::fill_n(im.covB.begin(), im.pc_elems, 0.0);
  std::fill_n(im.covB2.begin(), im.pc_elems, 0.0);
  std::fill_n(im.confS.begin(), im.pc_elems, 0.0);
  std::fill_n(im.confC.begin(), im.pc_elems, 0.0);
  std::fill_n(im.contrib.begin(), im.pc_elems, 0u);
  std::fill_n(im.kept.begin(), im.pc_elems, 0u);
  std::fill_n(im.footprint.begin(), im.nplane, 0u);
  std::fill_n(im.supp.begin(), im.pc_elems, 0u);
  std::fill_n(im.degraded.begin(), im.pc_elems, 0ull);
  if (cfg.emit_profiles)
    std::fill(im.meta.begin(), im.meta.end(), ForwardDrizzleV2FrameMeta{});
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

bool ForwardDrizzleV2CpuKernel::accumulate_frame_impl(
    const double affine6[6], const ForwardDrizzleV2LocalWarp *warp,
    const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
    std::uint64_t unique_source_samples, std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  Impl &im = *impl_;
  const std::uint64_t n = im.cfg.stream_length;
  // The one-shot paths must not interleave with an open affine piece frame.
  if (frame_open_ || frame_order >= n) return false;
  // The window is a packed row-major buffer (optional halo) inside the
  // reserved full source extent; only the ACTIVE rect is iterated and
  // geometry keeps absolute source coordinates.
  if (window.x_begin < 0 || window.y_begin < 0 || window.width <= 0 ||
      window.height <= 0 ||
      window.x_begin + window.width > im.source_w ||
      window.y_begin + window.height > im.source_h)
    return false;
  int act_x = window.active_x, act_y = window.active_y;
  int act_w = window.active_width, act_h = window.active_height;
  if (act_w == 0 && act_h == 0 && act_x == 0 && act_y == 0) {
    act_w = window.width;
    act_h = window.height;
  }
  if (act_x < 0 || act_y < 0 || act_w <= 0 || act_h <= 0 ||
      act_x + act_w > window.width || act_y + act_h > window.height)
    return false;
  const int abs_x0 = window.x_begin + act_x;
  const int abs_y0 = window.y_begin + act_y;
  // Cached-geometry path: the leaves carry the committed geometry; a local
  // warp must not be combined with it, the count is bound by the reserved
  // capacity, and every leaf's source sample must lie inside the ACTIVE
  // rect (packed float sigma/Q planes index by active-local position).
  if (leaves != nullptr) {
    if (warp != nullptr || leaf_count == 0 ||
        leaf_count > im.cfg.cached_leaf_capacity)
      return false;
    for (std::size_t li = 0; li < leaf_count; ++li) {
      const ForwardDrizzleV2CachedLeaf &L = leaves[li];
      if (L.source_x >= static_cast<std::uint32_t>(im.source_w) ||
          L.source_y >= static_cast<std::uint32_t>(im.source_h) ||
          L.channel >= static_cast<std::uint16_t>(im.channels))
        return false;
      const int sx = static_cast<int>(L.source_x);
      const int sy = static_cast<int>(L.source_y);
      if (sx < abs_x0 || sx >= abs_x0 + act_w || sy < abs_y0 ||
          sy >= abs_y0 + act_h)
        return false;
      for (int k = 0; k < 4; ++k)
        if (!std::isfinite(L.x[k]) || !std::isfinite(L.y[k])) return false;
    }
  }
  // An explicit sigma2 plane and an enabled inline model are mutually
  // exclusive (the compact model path reads the halo source buffer).
  const ForwardDrizzleV2Sigma2FrameModel *s2m =
      (sigma2_model_or_null != nullptr && sigma2_model_or_null->enabled)
          ? sigma2_model_or_null
          : nullptr;
  if (sigma2_or_null != nullptr && s2m != nullptr) return false;
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
  // Quality work is relevant only for frames the reservoir keep set can
  // retain; non-selected orders carry qmask = 0 and skip all Q planes.
  const bool keep_frame =
      keep_all ||
      cpu_splitmix64(frame_order ^ im.cfg.reservoir_seed) < threshold;
  const bool q_frame = qp && keep_frame;
  // Each stream supplies a float plane XOR a packed storage-grid window;
  // both set or a malformed descriptor fails the call.
  ForwardDrizzleV2FrameQuality q_in{};
  if (q_frame && quality_or_null != nullptr) q_in = *quality_or_null;
  const bool qc_ok = !(q_in.q_composite != nullptr &&
                       q_in.qc_packed.cells != nullptr);
  const bool q0_ok =
      !(q_in.q_scale0 != nullptr && q_in.q0_packed.cells != nullptr);
  const bool q1_ok =
      !(q_in.q_scale1 != nullptr && q_in.q1_packed.cells != nullptr);
  const bool qa_ok =
      !(q_in.q_artifact != nullptr && q_in.qa_packed.cells != nullptr);
  if (!qc_ok || !q0_ok || !q1_ok || !qa_ok ||
      !cpu_packed_q_covers(q_in.qc_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.q0_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.q1_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.qa_packed, abs_x0, abs_y0, act_w, act_h))
    return false;
  const unsigned int qmask =
      ((q_in.q_composite != nullptr || q_in.qc_packed.cells != nullptr)
           ? 1u
           : 0u) |
      ((q_in.q_scale0 != nullptr || q_in.q0_packed.cells != nullptr) ? 2u
                                                                   : 0u) |
      ((q_in.q_scale1 != nullptr || q_in.q1_packed.cells != nullptr) ? 4u
                                                                   : 0u) |
      ((q_in.q_artifact != nullptr || q_in.qa_packed.cells != nullptr) ? 8u
                                                                     : 0u);

  if (qp && meta_or_null != nullptr) im.meta[frame_order] = *meta_or_null;

  // --- scatter: clear the frame planes, then droplet-rasterize the frame ---
  std::fill(im.fa.begin(), im.fa.end(), 0.0);
  std::fill(im.fbs.begin(), im.fbs.end(), 0.0);
  std::fill(im.fbg.begin(), im.fbg.end(), 0.0);
  if (!im.fs2.empty()) std::fill(im.fs2.begin(), im.fs2.end(), 0.0);
  if (q_frame) {
    std::fill(im.fqc.begin(), im.fqc.end(), 0.0);
    std::fill(im.fq0.begin(), im.fq0.end(), 0.0);
    std::fill(im.fq1.begin(), im.fq1.end(), 0.0);
    std::fill(im.fqa.begin(), im.fqa.end(), 0.0);
    std::fill(im.fqaf.begin(), im.fqaf.end(), 0.0);
  }
  const double sc = static_cast<double>(im.cfg.internal_scale);
  const double half = im.cfg.half;
  const double band_ox =
      static_cast<double>(im.cfg.band_origin_x_native);
  const double band_oy =
      static_cast<double>(im.cfg.band_origin_y_native);
  const int cols = im.icols, rows = im.irows;
  const std::size_t plane_n = im.iplane;
  const BayerPattern bayer =
      static_cast<BayerPattern>(im.cfg.bayer_pattern);
  const bool mono = im.cfg.mono;
  const int canvas_w = im.cfg.canvas_width_native > 0
                           ? im.cfg.canvas_width_native
                           : im.ncols;
  const int canvas_h = im.cfg.canvas_height_native > 0
                           ? im.cfg.canvas_height_native
                           : im.nrows;

  // Synthesized frame transform for the production local inversion oracle.
  registration::FrameSamplingTransform ftrans;
  registration::LocalInversionParams inv;
  if (warp != nullptr) {
    ftrans.valid = true;
    ftrans.has_smooth_local_model = true;
    auto &s2c = ftrans.source_to_canvas;
    s2c(0, 0) = static_cast<float>(affine6[0]);
    s2c(0, 1) = static_cast<float>(affine6[1]);
    s2c(0, 2) = static_cast<float>(affine6[2]);
    s2c(1, 0) = static_cast<float>(affine6[3]);
    s2c(1, 1) = static_cast<float>(affine6[4]);
    s2c(1, 2) = static_cast<float>(affine6[5]);
    ftrans.source_to_canvas_affine_valid = true;
    auto &m = ftrans.smooth_local_model;
    for (int i = 0; i < 16; ++i) {
      m.coeff_x[i] = warp->coeff_x[i];
      m.coeff_y[i] = warp->coeff_y[i];
    }
    m.image_rows = warp->image_rows;
    m.image_cols = warp->image_cols;
    m.valid = warp->model_valid != 0;
    ftrans.model_coordinate_scale = warp->model_coordinate_scale;
    ftrans.model_offset_x = warp->model_offset_x;
    ftrans.model_offset_y = warp->model_offset_y;
    inv.max_iter = warp->max_iter;
    inv.tol_px = warp->tol_px;
    inv.safety_margin_px = warp->safety_margin_px;
  }

  std::vector<Leaf> local_leaves;
  std::uint64_t positive = 0, discarded = 0;
  const std::size_t source_n =
      static_cast<std::size_t>(act_w) * static_cast<std::size_t>(act_h);
  // Inline sigma2 model: absolute-neighbour access against the uploaded
  // (halo-extended) buffer; outside the true source extent counts as a
  // missing neighbour, exactly like the whole-plane oracle.
  auto sigma2_at = [&](int sx, int sy) -> double {
    auto at = [&](int ax, int ay) -> float {
      if (ax < 0 || ay < 0 || ax >= im.source_w || ay >= im.source_h)
        return std::numeric_limits<float>::quiet_NaN();
      const int bx = ax - window.x_begin, by = ay - window.y_begin;
      if (bx < 0 || by < 0 || bx >= window.width || by >= window.height)
        return std::numeric_limits<float>::quiet_NaN();
      return source[static_cast<std::size_t>(by) * window.width + bx];
    };
    auto diff = [&](bool x_axis) -> float {
      const float vm = at(sx - (x_axis ? 1 : 0), sy - (x_axis ? 0 : 1));
      const float vp = at(sx + (x_axis ? 1 : 0), sy + (x_axis ? 0 : 1));
      const bool fm = std::isfinite(vm), fp = std::isfinite(vp);
      if (fm && fp) return (vp - vm) * 0.5f;
      const float vc = at(sx, sy);
      if (fp && std::isfinite(vc)) return vp - vc;
      if (fm && std::isfinite(vc)) return vc - vm;
      return 0.0f;
    };
    // The explicit-plane path quantises to float; mirror that so both
    // sigma2 contracts are bit-identical.
    return static_cast<double>(static_cast<float>(
        forward_drizzle_v2_sigma2_model(s2m->sigma_noise, diff(true),
                                        diff(false), s2m->sigma_reg_px,
                                        s2m->droplet_half)));
  };
  if (leaves != nullptr) {
    // Geometry-cache path: one iteration per committed leaf. Corners are
    // absolute internal-canvas coordinates; subtracting the internal band
    // origin (band_origin * sc) makes them band-local --- bit-identical to
    // the local emit path's (native - band_origin) * sc because sc is a
    // power of two. All validation ran in the preamble, so the
    // buffer/sample indices below are provably in bounds.
    for (std::size_t li = 0; li < leaf_count; ++li) {
      const ForwardDrizzleV2CachedLeaf &L = leaves[li];
      const int sx = static_cast<int>(L.source_x);
      const int sy = static_cast<int>(L.source_y);
      const std::size_t tid =
          static_cast<std::size_t>(sy - abs_y0) *
              static_cast<std::size_t>(act_w) +
          static_cast<std::size_t>(sx - abs_x0);
      const std::size_t bidx =
          static_cast<std::size_t>(sy - window.y_begin) *
              static_cast<std::size_t>(window.width) +
          static_cast<std::size_t>(sx - window.x_begin);
      const double value = static_cast<double>(source[bidx]);
      const bool finite_value = std::isfinite(value);
      const double s2 =
          im.fs2.empty()
              ? 0.0
              : (sigma2_or_null != nullptr
                     ? static_cast<double>(sigma2_or_null[tid])
                     : (s2m != nullptr ? sigma2_at(sx, sy) : 0.0));
      float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (qmask != 0u) {
        qv[0] = cpu_quality_sample(q_in.q_composite, q_in.qc_packed, tid, sx,
                                   sy);
        qv[1] = cpu_quality_sample(q_in.q_scale0, q_in.q0_packed, tid, sx,
                                   sy);
        qv[2] = cpu_quality_sample(q_in.q_scale1, q_in.q1_packed, tid, sx,
                                   sy);
        qv[3] = cpu_quality_sample(q_in.q_artifact, q_in.qa_packed, tid, sx,
                                   sy);
      }
      double ex[4], ey[4];
      for (int k = 0; k < 4; ++k) {
        ex[k] = L.x[k] - band_ox * sc;
        ey[k] = L.y[k] - band_oy * sc;
      }
      positive += cpu_scatter_leaf(
          ex, ey, cols, rows, static_cast<int>(L.channel), value,
          finite_value, s2, qv, qmask, static_cast<int>(plane_n), 0, cols,
          im.fa, im.fbs, im.fbg, im.fs2, im.fqc, im.fq0, im.fq1, im.fqa,
          im.fqaf);
    }
    stats_.cached_leaf_records_launched += leaf_count;
  } else
  for (std::size_t tid = 0; tid < source_n; ++tid) {
    const int lx = static_cast<int>(tid % static_cast<std::size_t>(act_w));
    const int ly = static_cast<int>(tid / static_cast<std::size_t>(act_w));
    const int sy = abs_y0 + ly;
    const int sx = abs_x0 + lx;
    const std::size_t bidx =
        static_cast<std::size_t>(act_y + ly) *
            static_cast<std::size_t>(window.width) +
        static_cast<std::size_t>(act_x + lx);
    const double value = static_cast<double>(source[bidx]);
    const bool finite_value = std::isfinite(value);
    const double s2 =
        im.fs2.empty()
            ? 0.0
            : (sigma2_or_null != nullptr
                   ? static_cast<double>(sigma2_or_null[tid])
                   : (s2m != nullptr ? sigma2_at(sx, sy) : 0.0));
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (qmask != 0u) {
      qv[0] = cpu_quality_sample(q_in.q_composite, q_in.qc_packed, tid, sx,
                                 sy);
      qv[1] = cpu_quality_sample(q_in.q_scale0, q_in.q0_packed, tid, sx, sy);
      qv[2] = cpu_quality_sample(q_in.q_scale1, q_in.q1_packed, tid, sx, sy);
      qv[3] = cpu_quality_sample(q_in.q_artifact, q_in.qa_packed, tid, sx,
                                 sy);
    }
    const int channel =
        mono ? 0
             : static_cast<int>(cfa_channel_for_source_pixel(
                   sx, sy, bayer, im.cfg.cfa_origin_x, im.cfg.cfa_origin_y));
    const double x = sx + 0.5, y = sy + 0.5;
    if (warp == nullptr) {
      double qx[4], qy[4];
      const double px[4] = {x - half, x + half, x + half, x - half};
      const double py[4] = {y - half, y - half, y + half, y + half};
      for (int k = 0; k < 4; ++k) {
        qx[k] = (affine6[0] * px[k] + affine6[1] * py[k] + affine6[2] -
                 band_ox) *
                sc;
        qy[k] = (affine6[3] * px[k] + affine6[4] * py[k] + affine6[5] -
                 band_oy) *
                sc;
      }
      positive += cpu_scatter_leaf(
          qx, qy, cols, rows, channel, value, finite_value, s2, qv, qmask,
          static_cast<int>(plane_n), 0, cols, im.fa, im.fbs, im.fbg,
          im.fs2, im.fqc, im.fq0, im.fq1, im.fqa, im.fqaf);
    } else {
      // All-or-nothing subdivision, then scatter each accepted leaf. The
      // leaf corners are deterministic re-evaluations of the node corners
      // (same inversion call sites as the evaluator).
      local_leaves.clear();
      if (!cpu_collect_local_leaves(ftrans, inv, *warp, x - half, y - half,
                                    x + half, y + half, 0, canvas_w,
                                    canvas_h, sc, local_leaves)) {
        ++discarded;
        continue;
      }
      if (local_leaves.empty()) {
        ++discarded;
        continue;
      }
      std::uint64_t ov = 0;
      for (const auto &leaf : local_leaves) {
        // Leaf corners are raw canvas coordinates; the device emit path
        // applies (q - band_origin) * sc, mirrored here bit-exactly.
        double ex[4], ey[4];
        for (int k = 0; k < 4; ++k) {
          ex[k] = (leaf.x[k] - band_ox) * sc;
          ey[k] = (leaf.y[k] - band_oy) * sc;
        }
        ov += cpu_scatter_leaf(
            ex, ey, cols, rows, channel, value, finite_value, s2, qv,
            qmask, static_cast<int>(plane_n), 0, cols, im.fa, im.fbs,
            im.fbg, im.fs2, im.fqc, im.fq0, im.fq1, im.fqa, im.fqaf);
      }
      positive += ov;
    }
  }
  stats_.positive_overlaps += positive;
  stats_.local_samples_discarded += discarded;

  // --- fold: k-fold the frame planes into accumulators + reservoir --------
  fold_native_range(0, im.ncols, q_frame, qmask, frame_order, keep_all,
                    threshold);
  ++im.frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  // Uploaded source bytes cover the whole buffer incl. halo; the explicit
  // sigma2 compatibility plane is packed over the active rect only.
  stats_.source_bytes_uploaded +=
      static_cast<std::uint64_t>(window.width) * window.height *
          sizeof(float) +
      (sigma2_or_null != nullptr ? source_n * sizeof(float) : 0);
  // Cached-geometry frames launch the cache-counted covered samples, not
  // the dense active rect (the build-side discards are already final).
  stats_.source_samples_launched +=
      leaves != nullptr ? unique_source_samples : source_n;
  if (q_frame) {
    ++stats_.quality_frames_processed;
    // Host-side equivalent of the device Q uploads: compact windows count
    // cells+veto (3 bytes/storage cell), float planes the active rect.
    const std::uint64_t qfb = source_n * sizeof(float);
    auto qb = [&](const float *f,
                  const ForwardDrizzleV2PackedQualityPlane &p)
        -> std::uint64_t {
      if (p.cells != nullptr)
        return static_cast<std::uint64_t>(p.storage_width) *
               p.storage_height * 3u;
      return f != nullptr ? qfb : 0u;
    };
    stats_.quality_bytes_uploaded +=
        qb(q_in.q_composite, q_in.qc_packed) +
        qb(q_in.q_scale0, q_in.q0_packed) +
        qb(q_in.q_scale1, q_in.q1_packed) +
        qb(q_in.q_artifact, q_in.qa_packed);
  }
  return true;
}

// k-fold the frame planes into accumulators + reservoir for native target
// columns [nx0, nx1) of the active band (every row, every channel).
void ForwardDrizzleV2CpuKernel::fold_native_range(
    int nx0, int nx1, bool q_frame, unsigned int qmask,
    std::uint64_t frame_order, bool keep_all, std::uint64_t threshold) {
  Impl &im = *impl_;
  const int scale = im.cfg.internal_scale;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  const int ncols = im.ncols, nrows = im.nrows;
  const std::size_t np = im.nplane;
  const unsigned int slots = static_cast<unsigned int>(im.res_slots);
  for (int ny = 0; ny < nrows; ++ny)
   for (int nx = nx0; nx < nx1; ++nx) {
    const std::size_t px =
        static_cast<std::size_t>(ny) * ncols + static_cast<std::size_t>(nx);
    bool any_geo = false;
    for (int c = 0; c < im.channels; ++c) {
      const std::size_t pc = static_cast<std::size_t>(c) * np + px;
      double a = 0.0, b_src = 0.0, b_geo = 0.0, s2w = 0.0;
      double qc_s = 0.0, q0_s = 0.0, q1_s = 0.0, qa_s = 0.0, qaf_s = 0.0;
      unsigned int geo_bits = 0, src_bits = 0;
      for (int iy = 0; iy < scale; ++iy) {
        for (int ix = 0; ix < scale; ++ix) {
          const int j = iy * scale + ix;
          const std::size_t ii =
              static_cast<std::size_t>(ny * scale + iy) * im.icols +
              nx * scale + ix;
          const std::size_t idx = static_cast<std::size_t>(c) * im.iplane + ii;
          const double aj = im.fa[idx];
          const double bsj = im.fbs[idx];
          const double bgj = im.fbg[idx];
          a += inv_s2 * aj;
          b_src += inv_s2 * bsj;
          b_geo += inv_s2 * bgj;
          if (!im.fs2.empty()) s2w += inv_s2 * im.fs2[idx];
          if (q_frame) {
            qc_s += inv_s2 * im.fqc[idx];
            q0_s += inv_s2 * im.fq0[idx];
            q1_s += inv_s2 * im.fq1[idx];
            qa_s += inv_s2 * im.fqa[idx];
            qaf_s += inv_s2 * im.fqaf[idx];
          }
          if (bgj > 0.0) geo_bits |= 1u << j;
          if (bsj > 0.0) src_bits |= 1u << j;
        }
      }
      im.supp[pc] |=
          static_cast<unsigned short>(geo_bits | (src_bits << 4));
      if (b_geo > 0.0) {
        im.covB[pc] += b_geo;
        im.covB2[pc] += b_geo * b_geo;
        any_geo = true;
      }
      if (!(b_src > 0.0)) continue;
      const double xv = a / b_src;
      const double s2c = im.fs2.empty() ? 0.0 : s2w / b_src;
      im.accA[pc] += a;
      im.accB[pc] += b_src;
      im.accB2[pc] += b_src * b_src;
      ++im.contrib[pc];
      ++stats_.candidates_streamed;
      if (!im.fs2.empty() && (!std::isfinite(s2c) || s2c < 0.0)) {
        ++im.degraded[pc];
      } else if (s2c > 0.0) {
        im.confS[pc] += b_src * std::sqrt(s2c);
        im.confC[pc] += b_src * b_src * s2c;
      }
      if (keep_all || cpu_splitmix64(frame_order ^ im.cfg.reservoir_seed) <
                          threshold) {
        const unsigned int k = im.kept[pc]++;
        ++stats_.reservoir_kept_total;
        if (k < slots) {
          CpuReservoirRecord &r = im.res[pc * slots + k];
          r.x = xv;
          r.b = b_src;
          r.order = frame_order;
          r.sigma2 = s2c;
          if (q_frame) {
            CpuReservoirQuality &qv = im.resq[pc * slots + k];
            qv.qc = (qmask & 1u) ? static_cast<float>(qc_s / b_src) : 1.0f;
            qv.q0 = (qmask & 2u) ? static_cast<float>(q0_s / b_src) : 1.0f;
            qv.q1 = (qmask & 4u) ? static_cast<float>(q1_s / b_src) : 1.0f;
            const bool qa_data = (qmask & 8u) != 0u && qaf_s > 0.0;
            qv.qa =
                qa_data ? static_cast<float>(qa_s / b_src) : -1.0f;
          }
        }
      }
    }
    if (any_geo) ++im.footprint[px];
  }
}

bool ForwardDrizzleV2CpuKernel::accumulate_frame(
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

bool ForwardDrizzleV2CpuKernel::accumulate_frame_window(
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

bool ForwardDrizzleV2CpuKernel::accumulate_frame_local(
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

bool ForwardDrizzleV2CpuKernel::accumulate_frame_local_window(
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
  if (warp.max_subdivision_depth > 2) return false;
  return accumulate_frame_impl(affine6, &warp, window, source, sigma2_or_null,
                               sigma2_model_or_null, nullptr, 0, 0,
                               frame_order, quality_or_null, meta_or_null);
}

bool ForwardDrizzleV2CpuKernel::accumulate_frame_cached_leaves(
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

bool ForwardDrizzleV2CpuKernel::accumulate_frame_affine_samples(
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
  const int sw = im.source_w, sh = im.source_h;
  const int cols = im.icols, rows = im.irows;
  if (sample_count > static_cast<std::size_t>(sw) * sh) return false;
  // Canonical order (strictly increasing (y, x)) + in-extent coordinates;
  // the same scan counts the distinct span rows.
  std::uint64_t span_rows = 0;
  std::uint32_t prev_y = 0, prev_x = 0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const ForwardDrizzleV2SourceSample &s = samples[i];
    if (s.source_x >= static_cast<std::uint32_t>(sw) ||
        s.source_y >= static_cast<std::uint32_t>(sh) ||
        (i > 0 && (s.source_y < prev_y ||
                   (s.source_y == prev_y && s.source_x <= prev_x))))
      return false;
    if (i == 0 || s.source_y != prev_y) ++span_rows;
    prev_y = s.source_y;
    prev_x = s.source_x;
  }
  const std::uint64_t n = im.cfg.stream_length;
  if (im.cfg.emit_profiles) {
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
    im.meta[frame_order] = *meta_or_null;
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
  const bool keep_frame =
      keep_all ||
      cpu_splitmix64(frame_order ^ im.cfg.reservoir_seed) < threshold;
  const bool q_frame = qp && keep_frame;
  const ForwardDrizzleV2AlignedQuality aq =
      (q_frame && quality_or_null != nullptr)
          ? *quality_or_null
          : ForwardDrizzleV2AlignedQuality{};
  const unsigned int qmask = q_frame ? aq.presence_mask : 0u;
  // A set presence bit requires the code array; veto may stay null.
  const std::uint16_t *aqc[4] = {aq.qc, aq.q0, aq.q1, aq.qa};
  const std::uint8_t *aqv[4] = {aq.vc, aq.v0, aq.v1, aq.va};
  for (int k = 0; k < 4; ++k)
    if ((qmask & (1u << k)) != 0 && aqc[k] == nullptr) return false;
  const double sc = static_cast<double>(im.cfg.internal_scale);
  const double half = im.cfg.half;
  const double band_ox =
      static_cast<double>(im.cfg.band_origin_x_native);
  const double band_oy =
      static_cast<double>(im.cfg.band_origin_y_native);
  const BayerPattern bayer =
      static_cast<BayerPattern>(im.cfg.bayer_pattern);
  const bool mono = im.cfg.mono;
  const std::size_t plane_n = im.iplane;
  // One-shot full-target frame: clear all frame planes, then droplet-
  // rasterize every canonical sample over the whole band.
  std::fill(im.fa.begin(), im.fa.end(), 0.0);
  std::fill(im.fbs.begin(), im.fbs.end(), 0.0);
  std::fill(im.fbg.begin(), im.fbg.end(), 0.0);
  if (!im.fs2.empty()) std::fill(im.fs2.begin(), im.fs2.end(), 0.0);
  if (q_frame) {
    std::fill(im.fqc.begin(), im.fqc.end(), 0.0);
    std::fill(im.fq0.begin(), im.fq0.end(), 0.0);
    std::fill(im.fq1.begin(), im.fq1.end(), 0.0);
    std::fill(im.fqa.begin(), im.fqa.end(), 0.0);
    std::fill(im.fqaf.begin(), im.fqaf.end(), 0.0);
  }
  std::uint64_t positive = 0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const ForwardDrizzleV2SourceSample &s = samples[i];
    const int sx = static_cast<int>(s.source_x);
    const int sy = static_cast<int>(s.source_y);
    const double value = static_cast<double>(s.value);
    const bool finite_value = std::isfinite(value);
    const double s2 = sigma2_present ? static_cast<double>(s.sigma2) : 0.0;
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 4; ++k) {
      if ((qmask & (1u << k)) == 0) continue;
      qv[k] = ((aqv[k] != nullptr && aqv[k][i] != 0) || aqc[k][i] == 0)
                  ? std::numeric_limits<float>::quiet_NaN()
                  : static_cast<float>(aqc[k][i]) / 65535.0f;
    }
    const int channel =
        mono ? 0
             : static_cast<int>(cfa_channel_for_source_pixel(
                   sx, sy, bayer, im.cfg.cfa_origin_x, im.cfg.cfa_origin_y));
    const double x = sx + 0.5, y = sy + 0.5;
    double qx[4], qy[4];
    const double px[4] = {x - half, x + half, x + half, x - half};
    const double py[4] = {y - half, y - half, y + half, y + half};
    for (int k = 0; k < 4; ++k) {
      qx[k] = (affine6[0] * px[k] + affine6[1] * py[k] + affine6[2] -
               band_ox) *
              sc;
      qy[k] = (affine6[3] * px[k] + affine6[4] * py[k] + affine6[5] -
               band_oy) *
              sc;
    }
    positive += cpu_scatter_leaf(
        qx, qy, cols, rows, channel, value, finite_value, s2, qv, qmask,
        static_cast<int>(plane_n), 0, cols, im.fa, im.fbs, im.fbg, im.fs2,
        im.fqc, im.fq0, im.fq1, im.fqa, im.fqaf);
  }
  stats_.positive_overlaps += positive;
  fold_native_range(0, im.ncols, q_frame, qmask, frame_order, keep_all,
                    threshold);
  ++im.frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  stats_.source_samples_launched += sample_count;
  stats_.affine_samples_processed += sample_count;
  stats_.affine_span_rows += span_rows;
  stats_.source_bytes_uploaded +=
      static_cast<std::uint64_t>(sample_count) *
      sizeof(ForwardDrizzleV2SourceSample);
  if (q_frame) {
    ++stats_.quality_frames_processed;
    const std::uint64_t per_stream =
        static_cast<std::uint64_t>(sample_count) * 3u;
    for (int k = 0; k < 4; ++k)
      if ((qmask & (1u << k)) != 0)
        stats_.quality_bytes_uploaded += per_stream;
  }
  return true;
}

bool ForwardDrizzleV2CpuKernel::begin_affine_frame(
    std::uint64_t frame_order, const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || frame_open_ || impl_->finalized ||
      impl_->frames >= impl_->cfg.stream_length ||
      frame_order >= impl_->cfg.stream_length)
    return false;
  Impl &im = *impl_;
  if (im.cfg.emit_profiles) {
    if (meta_or_null == nullptr ||
        !std::isfinite(meta_or_null->g_eff) || meta_or_null->g_eff < 0.0f ||
        !std::isfinite(meta_or_null->residual_factor))
      return false;
    im.meta[frame_order] = *meta_or_null;
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
  open_qframe_ =
      im.cfg.emit_profiles &&
      (keep_all || cpu_splitmix64(frame_order ^ im.cfg.reservoir_seed) <
                       threshold);
  return true;
}

bool ForwardDrizzleV2CpuKernel::accumulate_affine_piece(
    const double affine6[6], int target_x_begin_native,
    int target_cols_native, const ForwardDrizzleV2SourceWindow &window,
    const float *source, const float *sigma2_or_null,
    const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
    const ForwardDrizzleV2FrameQuality *quality_or_null) {
  if (impl_ == nullptr || !frame_open_ || affine6 == nullptr ||
      source == nullptr)
    return false;
  Impl &im = *impl_;
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
  int act_x = window.active_x, act_y = window.active_y;
  int act_w = window.active_width, act_h = window.active_height;
  if (act_w == 0 && act_h == 0 && act_x == 0 && act_y == 0) {
    act_w = window.width;
    act_h = window.height;
  }
  if (act_x < 0 || act_y < 0 || act_w <= 0 || act_h <= 0 ||
      act_x + act_w > window.width || act_y + act_h > window.height)
    return false;
  const int abs_x0 = window.x_begin + act_x;
  const int abs_y0 = window.y_begin + act_y;
  const std::size_t source_n =
      static_cast<std::size_t>(act_w) * static_cast<std::size_t>(act_h);
  const ForwardDrizzleV2Sigma2FrameModel *s2m =
      (sigma2_model_or_null != nullptr && sigma2_model_or_null->enabled)
          ? sigma2_model_or_null
          : nullptr;
  if (sigma2_or_null != nullptr && s2m != nullptr) return false;
  const std::uint64_t n = im.cfg.stream_length;
  const bool keep_all =
      n <= static_cast<std::uint64_t>(im.cfg.reservoir_size);
  const std::uint64_t threshold =
      keep_all ? 0
               : static_cast<std::uint64_t>(
                     (static_cast<unsigned __int128>(im.cfg.reservoir_size)
                      << 64) /
                     n);
  const bool q_frame = open_qframe_;
  ForwardDrizzleV2FrameQuality q_in{};
  if (q_frame && quality_or_null != nullptr) q_in = *quality_or_null;
  const bool qc_ok = !(q_in.q_composite != nullptr &&
                       q_in.qc_packed.cells != nullptr);
  const bool q0_ok =
      !(q_in.q_scale0 != nullptr && q_in.q0_packed.cells != nullptr);
  const bool q1_ok =
      !(q_in.q_scale1 != nullptr && q_in.q1_packed.cells != nullptr);
  const bool qa_ok =
      !(q_in.q_artifact != nullptr && q_in.qa_packed.cells != nullptr);
  if (!qc_ok || !q0_ok || !q1_ok || !qa_ok ||
      !cpu_packed_q_covers(q_in.qc_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.q0_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.q1_packed, abs_x0, abs_y0, act_w, act_h) ||
      !cpu_packed_q_covers(q_in.qa_packed, abs_x0, abs_y0, act_w, act_h))
    return false;
  const unsigned int qmask =
      ((q_in.q_composite != nullptr || q_in.qc_packed.cells != nullptr)
           ? 1u
           : 0u) |
      ((q_in.q_scale0 != nullptr || q_in.q0_packed.cells != nullptr) ? 2u
                                                                   : 0u) |
      ((q_in.q_scale1 != nullptr || q_in.q1_packed.cells != nullptr) ? 4u
                                                                   : 0u) |
      ((q_in.q_artifact != nullptr || q_in.qa_packed.cells != nullptr) ? 8u
                                                                     : 0u);
  // The stream presence is a per-frame contract: every piece must carry the
  // same qmask.
  if (open_pieces_ == 0) {
    open_qmask_ = qmask;
  } else if (qmask != open_qmask_) {
    return false;
  }

  const int scale = im.cfg.internal_scale;
  const double sc = static_cast<double>(scale);
  const double half = im.cfg.half;
  const double band_ox =
      static_cast<double>(im.cfg.band_origin_x_native);
  const double band_oy =
      static_cast<double>(im.cfg.band_origin_y_native);
  const int cols = im.icols, rows = im.irows;
  const std::size_t plane_n = im.iplane;
  const BayerPattern bayer =
      static_cast<BayerPattern>(im.cfg.bayer_pattern);
  const bool mono = im.cfg.mono;
  // Internal x columns owned by this piece.
  const int ix0 = target_x_begin_native * scale;
  const int ix1 = (target_x_begin_native + target_cols_native) * scale;

  // Clear ONLY this tile's internal columns of the frame planes.
  for (int c = 0; c < im.channels; ++c) {
    const std::size_t base = static_cast<std::size_t>(c) * plane_n;
    for (int ty = 0; ty < rows; ++ty) {
      const std::size_t off =
          base + static_cast<std::size_t>(ty) * cols + ix0;
      const int nw = ix1 - ix0;
      std::fill_n(im.fa.begin() + static_cast<std::ptrdiff_t>(off), nw, 0.0);
      std::fill_n(im.fbs.begin() + static_cast<std::ptrdiff_t>(off), nw, 0.0);
      std::fill_n(im.fbg.begin() + static_cast<std::ptrdiff_t>(off), nw, 0.0);
      if (!im.fs2.empty())
        std::fill_n(im.fs2.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
      if (q_frame) {
        std::fill_n(im.fqc.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
        std::fill_n(im.fq0.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
        std::fill_n(im.fq1.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
        std::fill_n(im.fqa.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
        std::fill_n(im.fqaf.begin() + static_cast<std::ptrdiff_t>(off), nw,
                    0.0);
      }
    }
  }

  // Inline sigma2 model: absolute-neighbour access against the uploaded
  // (halo-extended) buffer; identical semantics to the one-shot path.
  auto sigma2_at = [&](int sx, int sy) -> double {
    auto at = [&](int ax, int ay) -> float {
      if (ax < 0 || ay < 0 || ax >= im.source_w || ay >= im.source_h)
        return std::numeric_limits<float>::quiet_NaN();
      const int bx = ax - window.x_begin, by = ay - window.y_begin;
      if (bx < 0 || by < 0 || bx >= window.width || by >= window.height)
        return std::numeric_limits<float>::quiet_NaN();
      return source[static_cast<std::size_t>(by) * window.width + bx];
    };
    auto diff = [&](bool x_axis) -> float {
      const float vm = at(sx - (x_axis ? 1 : 0), sy - (x_axis ? 0 : 1));
      const float vp = at(sx + (x_axis ? 1 : 0), sy + (x_axis ? 0 : 1));
      const bool fm = std::isfinite(vm), fp = std::isfinite(vp);
      if (fm && fp) return (vp - vm) * 0.5f;
      const float vc = at(sx, sy);
      if (fp && std::isfinite(vc)) return vp - vc;
      if (fm && std::isfinite(vc)) return vc - vm;
      return 0.0f;
    };
    // The explicit-plane path quantises to float; mirror that so both
    // sigma2 contracts are bit-identical.
    return static_cast<double>(static_cast<float>(
        forward_drizzle_v2_sigma2_model(s2m->sigma_noise, diff(true),
                                        diff(false), s2m->sigma_reg_px,
                                        s2m->droplet_half)));
  };

  // Scatter this piece's active source rect, clipped to the tile's
  // internal x columns.
  std::uint64_t positive = 0;
  for (std::size_t tid = 0; tid < source_n; ++tid) {
    const int lx = static_cast<int>(tid % static_cast<std::size_t>(act_w));
    const int ly = static_cast<int>(tid / static_cast<std::size_t>(act_w));
    const int sy = abs_y0 + ly;
    const int sx = abs_x0 + lx;
    const std::size_t bidx =
        static_cast<std::size_t>(act_y + ly) *
            static_cast<std::size_t>(window.width) +
        static_cast<std::size_t>(act_x + lx);
    const double value = static_cast<double>(source[bidx]);
    const bool finite_value = std::isfinite(value);
    const double s2 =
        im.fs2.empty()
            ? 0.0
            : (sigma2_or_null != nullptr
                   ? static_cast<double>(sigma2_or_null[tid])
                   : (s2m != nullptr ? sigma2_at(sx, sy) : 0.0));
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (qmask != 0u) {
      qv[0] = cpu_quality_sample(q_in.q_composite, q_in.qc_packed, tid, sx,
                                 sy);
      qv[1] = cpu_quality_sample(q_in.q_scale0, q_in.q0_packed, tid, sx, sy);
      qv[2] = cpu_quality_sample(q_in.q_scale1, q_in.q1_packed, tid, sx, sy);
      qv[3] = cpu_quality_sample(q_in.q_artifact, q_in.qa_packed, tid, sx,
                                 sy);
    }
    const int channel =
        mono ? 0
             : static_cast<int>(cfa_channel_for_source_pixel(
                   sx, sy, bayer, im.cfg.cfa_origin_x, im.cfg.cfa_origin_y));
    const double x = sx + 0.5, y = sy + 0.5;
    double qx[4], qy[4];
    const double px[4] = {x - half, x + half, x + half, x - half};
    const double py[4] = {y - half, y - half, y + half, y + half};
    for (int k = 0; k < 4; ++k) {
      qx[k] = (affine6[0] * px[k] + affine6[1] * py[k] + affine6[2] -
               band_ox) *
              sc;
      qy[k] = (affine6[3] * px[k] + affine6[4] * py[k] + affine6[5] -
               band_oy) *
              sc;
    }
    positive += cpu_scatter_leaf(
        qx, qy, cols, rows, channel, value, finite_value, s2, qv, qmask,
        static_cast<int>(plane_n), ix0, ix1, im.fa, im.fbs, im.fbg, im.fs2,
        im.fqc, im.fq0, im.fq1, im.fqa, im.fqaf);
  }
  stats_.positive_overlaps += positive;

  // Fold only this tile's native columns; each output pixel is folded
  // exactly once for the frame across its pieces.
  fold_native_range(target_x_begin_native,
                    target_x_begin_native + target_cols_native, q_frame,
                    qmask, open_order_, keep_all, threshold);
  ++open_pieces_;
  open_tx_end_ = target_x_begin_native + target_cols_native;
  ++stats_.affine_pieces_processed;
  stats_.source_bytes_uploaded +=
      static_cast<std::uint64_t>(window.width) * window.height *
          sizeof(float) +
      (sigma2_or_null != nullptr ? source_n * sizeof(float) : 0);
  stats_.source_samples_launched += source_n;
  if (q_frame) {
    const std::uint64_t qfb = source_n * sizeof(float);
    auto qb = [&](const float *f,
                  const ForwardDrizzleV2PackedQualityPlane &p)
        -> std::uint64_t {
      if (p.cells != nullptr)
        return static_cast<std::uint64_t>(p.storage_width) *
               p.storage_height * 3u;
      return f != nullptr ? qfb : 0u;
    };
    stats_.quality_bytes_uploaded +=
        qb(q_in.q_composite, q_in.qc_packed) +
        qb(q_in.q_scale0, q_in.q0_packed) +
        qb(q_in.q_scale1, q_in.q1_packed) +
        qb(q_in.q_artifact, q_in.qa_packed);
  }
  return true;
}

bool ForwardDrizzleV2CpuKernel::finish_affine_frame(
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

bool ForwardDrizzleV2CpuKernel::skip_frame(
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
    impl_->meta[frame_order] = *meta_or_null;
  }
  ++impl_->frames;
  ++stats_.slot_transitions;
  ++stats_.frames_skipped_empty_window;
  return true;
}

bool ForwardDrizzleV2CpuKernel::finalize(
    ForwardDrizzleV2PixelResult *results,
    ForwardDrizzleV2ProfileResult *profiles_or_null,
    std::uint64_t *dense_overlap_count) {
  if (impl_ == nullptr || results == nullptr || impl_->finalized ||
      frame_open_)
    return false;
  Impl &im = *impl_;
  if (im.cfg.emit_profiles && profiles_or_null == nullptr) return false;
  const std::size_t np = im.nplane;
  const unsigned int slots = static_cast<unsigned int>(im.res_slots);
  const int subpixels = im.cfg.internal_scale * im.cfg.internal_scale;
  const double fine_exp =
      static_cast<double>(im.cfg.fine_quality_exponent);
  const double medium_exp =
      static_cast<double>(im.cfg.medium_quality_exponent);
  ForwardDrizzleV2ProfileConfig pcfg;
  pcfg.fine_quality_exponent = im.cfg.fine_quality_exponent;
  pcfg.medium_quality_exponent = im.cfg.medium_quality_exponent;
  std::uint64_t dense = 0;

  std::vector<ForwardDrizzleV2RobustCandidate> cands;
  std::vector<double> sigma2s;
  std::vector<std::uint8_t> accepted;
  std::vector<ForwardDrizzleV2ProfileCandidate> pcands;
  std::vector<std::uint64_t> kept_order;

  for (std::size_t tid = 0; tid < im.pc_elems; ++tid) {
    const std::size_t px = tid % np;
    ForwardDrizzleV2PixelResult r;
    const double a_acc = im.accA[tid];
    const double b_acc = im.accB[tid];
    const double b2_acc = im.accB2[tid];
    r.contributors = im.contrib[tid];
    r.conf_degraded = im.degraded[tid];
    r.b = b_acc;
    r.n_eff = b2_acc > 0.0 ? b_acc * b_acc / b2_acc : 0.0;
    const unsigned int mask = im.supp[tid];
    r.geometry_fraction = static_cast<float>(
        static_cast<double>(std::popcount(mask & 0xFu)) / subpixels);
    r.source_fraction = static_cast<float>(
        static_cast<double>(std::popcount((mask >> 4) & 0xFu)) / subpixels);
    r.estimator_fraction = r.source_fraction;
    r.profile_fraction = r.source_fraction;
    if ((tid / np) == 0 && im.footprint[px] == im.frames && im.frames > 0)
      ++dense;

    auto finish_conf = [&](double cb, double cs, double cc) {
      if (!(cb > 0.0)) {
        r.confidence = 0.0;
        r.confidence_state = static_cast<std::uint8_t>(
            ForwardDrizzleV2ConfidenceState::no_source_support);
        return;
      }
      if (cc > 0.0 && std::isfinite(cc)) {
        r.confidence = (cs * cs) / (cs * cs + cc);
        r.confidence_state = static_cast<std::uint8_t>(
            ForwardDrizzleV2ConfidenceState::modeled);
        return;
      }
      r.confidence = r.n_eff > 0.0 ? r.n_eff / (r.n_eff + 1.0) : 0.0;
      r.confidence_state = static_cast<std::uint8_t>(
          ForwardDrizzleV2ConfidenceState::fallback_n_eff);
    };
    auto emit_fallback_profiles = [&]() {
      if (profiles_or_null == nullptr) return;
      ForwardDrizzleV2ProfileResult pr;
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
      profiles_or_null[tid] = pr;
    };

    const unsigned int n_kept = im.kept[tid];
    const bool no_support = !(b_acc > 0.0);
    const bool overflow = n_kept > slots;
    const bool too_few =
        r.contributors <
            static_cast<unsigned int>(im.cfg.min_candidates) ||
        n_kept < static_cast<unsigned int>(im.cfg.min_clip_contributors);
    if (no_support || overflow || too_few) {
      if (no_support) {
        r.robust_state = static_cast<std::uint8_t>(
            ForwardDrizzleV2RobustState::no_source_support);
        finish_conf(0.0, 0.0, 0.0);
      } else {
        r.value = a_acc / b_acc;
        r.robust_state = static_cast<std::uint8_t>(
            overflow
                ? ForwardDrizzleV2RobustState::reservoir_overflow_fallback
                : ForwardDrizzleV2RobustState::too_few_candidates_fallback);
        finish_conf(b_acc, im.confS[tid], im.confC[tid]);
      }
      emit_fallback_profiles();
      results[tid] = r;
      continue;
    }

    // Load + sort the kept set by (x, order); the quality side array moves
    // with the records (it is indexed by reservoir slot).
    cands.clear();
    sigma2s.clear();
    pcands.clear();
    kept_order.clear();
    cands.reserve(n_kept);
    sigma2s.reserve(n_kept);
    kept_order.reserve(n_kept);
    for (unsigned int i = 0; i < n_kept; ++i) kept_order.push_back(i);
    std::sort(kept_order.begin(), kept_order.end(), [&](std::uint64_t i,
                                                      std::uint64_t j) {
      const auto &ri = im.res[tid * slots + i];
      const auto &rj = im.res[tid * slots + j];
      if (ri.x != rj.x) return ri.x < rj.x;
      return ri.order < rj.order;
    });
    for (unsigned int pos = 0; pos < n_kept; ++pos) {
      const auto &rc = im.res[tid * slots + kept_order[pos]];
      ForwardDrizzleV2RobustCandidate cand;
      cand.frame_order = static_cast<std::size_t>(rc.order);
      cand.x = rc.x;
      cand.b = rc.b;
      cands.push_back(cand);
      sigma2s.push_back(rc.sigma2);
      if (im.cfg.emit_profiles) {
        const auto &qv = im.resq[tid * slots + kept_order[pos]];
        ForwardDrizzleV2ProfileCandidate pc;
        pc.frame_order = cand.frame_order;
        pc.x = rc.x;
        pc.b = rc.b;
        pc.q = qv.qc;
        pc.q0 = qv.q0;
        pc.q1 = qv.q1;
        pc.qa = qv.qa >= 0.0f ? static_cast<double>(qv.qa) : 1.0;
        pc.qa_has_data = qv.qa >= 0.0f;
        pcands.push_back(pc);
      }
    }

    // Iterative clip identical to k_finalize_v2 / robust_frame_oracle_v2:
    // (x, order) order, weighted median at >= total/2, deviation-ordered
    // weighted MAD, asymmetric bounds, early stop on an unchanged mask.
    accepted.assign(n_kept, std::uint8_t{1});
    for (int pass = 0; pass < im.cfg.robust_passes; ++pass) {
      double total_w = 0.0;
      unsigned int n_active = 0;
      for (unsigned int i = 0; i < n_kept; ++i)
        if (accepted[i]) {
          total_w += cands[i].b;
          ++n_active;
        }
      if (n_active == 0) break;
      double median = 0.0;
      {
        double cum = 0.0;
        unsigned int last_on = 0;
        bool picked = false;
        for (unsigned int i = 0; i < n_kept; ++i) {
          if (!accepted[i]) continue;
          last_on = i;
          cum += cands[i].b;
          if (total_w > 0.0 && cum >= total_w / 2.0) {
            median = cands[i].x;
            picked = true;
            break;
          }
        }
        if (!picked) median = cands[last_on].x;
      }
      std::vector<unsigned int> ord;
      ord.reserve(n_kept);
      for (unsigned int i = 0; i < n_kept; ++i)
        if (accepted[i]) ord.push_back(i);
      std::sort(ord.begin(), ord.end(), [&](unsigned int i, unsigned int j) {
        const double di = std::fabs(cands[i].x - median);
        const double dj = std::fabs(cands[j].x - median);
        if (di != dj) return di < dj;
        return cands[i].frame_order < cands[j].frame_order;
      });
      double mad = std::fabs(cands[ord.back()].x - median);
      if (total_w > 0.0) {
        double cum = 0.0;
        for (unsigned int idx : ord) {
          cum += cands[idx].b;
          if (cum >= total_w / 2.0) {
            mad = std::fabs(cands[idx].x - median);
            break;
          }
        }
      }
      const double lower = median - im.cfg.sigma_low * mad;
      const double upper = median + im.cfg.sigma_high * mad;
      bool changed = false;
      for (unsigned int idx : ord) {
        const double xv = cands[idx].x;
        if (!(xv >= lower && xv <= upper)) {
          accepted[idx] = 0;
          changed = true;
        }
      }
      if (!changed) break;
    }

    double ca = 0.0, cb = 0.0, cs = 0.0, cc = 0.0;
    for (unsigned int i = 0; i < n_kept; ++i) {
      if (!accepted[i]) continue;
      ca += cands[i].b * cands[i].x;
      cb += cands[i].b;
      const double s2v = sigma2s[i];
      if (std::isfinite(s2v) && s2v > 0.0) {
        cs += cands[i].b * std::sqrt(s2v);
        cc += cands[i].b * cands[i].b * s2v;
      }
    }
    r.value = cb > 0.0 ? ca / cb : 0.0;
    r.robust_state = static_cast<std::uint8_t>(
        ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip);
    finish_conf(cb, cs, cc);

    if (profiles_or_null != nullptr) {
      const bool fully_degraded =
          r.contributors > 0 && im.degraded[tid] == r.contributors;
      try {
        profiles_or_null[tid] = forward_drizzle_v2_profile_reduce(
            pcands, accepted, im.meta, pcfg, r.confidence, fully_degraded);
      } catch (const std::invalid_argument &) {
        return false;
      }
    }
    results[tid] = r;
  }
  if (dense_overlap_count != nullptr) *dense_overlap_count = dense;
  im.finalized = true;
  stats_.stream_synchronizations = 1;
  stats_.result_bytes_downloaded =
      im.pc_elems * sizeof(ForwardDrizzleV2PixelResult);
  return true;
}

}  // namespace tile_compile::reconstruction
