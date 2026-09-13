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

// Quality accumulation of one droplet/cell pair, gated on finite source
// value by the caller (device d_scatter_quality port).
void cpu_scatter_quality(const ForwardDrizzleV2FrameQuality &q,
                         std::size_t out, double area, std::size_t tid,
                         std::vector<double> &vqc, std::vector<double> &vq0,
                         std::vector<double> &vq1, std::vector<double> &vqa,
                         std::vector<double> &vqaf) {
  auto acc = [](const float *src, std::vector<double> &dst, std::size_t o,
                double k, std::size_t t) {
    if (src == nullptr || dst.empty()) return;
    const double v = static_cast<double>(src[t]);
    dst[o] += k * (std::isfinite(v) && v > 0.0 ? v : 0.0);
  };
  acc(q.q_composite, vqc, out, area, tid);
  acc(q.q_scale0, vq0, out, area, tid);
  acc(q.q_scale1, vq1, out, area, tid);
  acc(q.q_artifact, vqa, out, area, tid);
  if (q.q_artifact != nullptr &&
      std::isfinite(static_cast<double>(q.q_artifact[tid])))
    vqaf[out] += area;
}

// Rasterize one accepted leaf quad (internal coordinates) into the frame
// planes. Returns the number of positive-overlap cells.
std::uint64_t cpu_scatter_leaf(const double *qx, const double *qy, int cols,
                               int rows, int channel, double value,
                               bool finite_value, double s2,
                               const ForwardDrizzleV2FrameQuality *q,
                               bool qp, std::size_t tid, int iplane,
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
  const int x0 = std::max(0, static_cast<int>(std::floor(minx)));
  const int x1 = std::min(cols, static_cast<int>(std::ceil(maxx)));
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
        if (qp && q != nullptr)
          cpu_scatter_quality(*q, out, area, tid, fqc, fq0, fq1, fqa, fqaf);
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
  const int res_slots = 2 * cfg.reservoir_size;
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
  stats_.reserved_device_bytes = total_bytes;
  bytes_per_native_pixel_ =
      nplane > 0 ? total_bytes / nplane : total_bytes;
  impl_ = im.release();
  return true;
}

bool ForwardDrizzleV2CpuKernel::accumulate_frame_impl(
    const double affine6[6], const ForwardDrizzleV2LocalWarp *warp,
    const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  Impl &im = *impl_;
  const std::uint64_t n = im.cfg.stream_length;
  if (frame_order >= n) return false;
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
  const float *h_qc =
      qp && quality_or_null ? quality_or_null->q_composite : nullptr;
  const float *h_q0 =
      qp && quality_or_null ? quality_or_null->q_scale0 : nullptr;
  const float *h_q1 =
      qp && quality_or_null ? quality_or_null->q_scale1 : nullptr;
  const float *h_qa =
      qp && quality_or_null ? quality_or_null->q_artifact : nullptr;
  const unsigned int qmask = (h_qc != nullptr ? 1u : 0u) |
                             (h_q0 != nullptr ? 2u : 0u) |
                             (h_q1 != nullptr ? 4u : 0u) |
                             (h_qa != nullptr ? 8u : 0u);

  if (qp && meta_or_null != nullptr) im.meta[frame_order] = *meta_or_null;

  // --- scatter: clear the frame planes, then droplet-rasterize the frame ---
  std::fill(im.fa.begin(), im.fa.end(), 0.0);
  std::fill(im.fbs.begin(), im.fbs.end(), 0.0);
  std::fill(im.fbg.begin(), im.fbg.end(), 0.0);
  if (!im.fs2.empty()) std::fill(im.fs2.begin(), im.fs2.end(), 0.0);
  if (qp) {
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

  std::vector<Leaf> leaves;
  std::uint64_t positive = 0, discarded = 0;
  ForwardDrizzleV2FrameQuality q_for_scatter{};
  if (qp && quality_or_null != nullptr) q_for_scatter = *quality_or_null;
  const ForwardDrizzleV2FrameQuality *qs =
      qp && quality_or_null != nullptr ? &q_for_scatter : nullptr;
  const std::size_t source_n =
      static_cast<std::size_t>(im.source_w) * im.source_h;
  for (std::size_t tid = 0; tid < source_n; ++tid) {
    const int sy = static_cast<int>(tid / im.source_w);
    const int sx = static_cast<int>(tid % im.source_w);
    const double value = static_cast<double>(source[tid]);
    const bool finite_value = std::isfinite(value);
    const double s2 = im.fs2.empty()
                          ? 0.0
                          : static_cast<double>(
                                sigma2_or_null != nullptr ? sigma2_or_null[tid]
                                                          : 0.0f);
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
          qx, qy, cols, rows, channel, value, finite_value, s2, qs, qp, tid,
          static_cast<int>(plane_n), im.fa, im.fbs, im.fbg, im.fs2, im.fqc,
          im.fq0, im.fq1, im.fqa, im.fqaf);
    } else {
      // All-or-nothing subdivision, then scatter each accepted leaf. The
      // leaf corners are deterministic re-evaluations of the node corners
      // (same inversion call sites as the evaluator).
      leaves.clear();
      if (!cpu_collect_local_leaves(ftrans, inv, *warp, x - half, y - half,
                                    x + half, y + half, 0, canvas_w,
                                    canvas_h, sc, leaves)) {
        ++discarded;
        continue;
      }
      if (leaves.empty()) {
        ++discarded;
        continue;
      }
      std::uint64_t ov = 0;
      for (const auto &leaf : leaves) {
        // Leaf corners are raw canvas coordinates; the device emit path
        // applies (q - band_origin) * sc, mirrored here bit-exactly.
        double ex[4], ey[4];
        for (int k = 0; k < 4; ++k) {
          ex[k] = (leaf.x[k] - band_ox) * sc;
          ey[k] = (leaf.y[k] - band_oy) * sc;
        }
        ov += cpu_scatter_leaf(
            ex, ey, cols, rows, channel, value, finite_value, s2, qs,
            qp, tid, static_cast<int>(plane_n), im.fa, im.fbs, im.fbg,
            im.fs2, im.fqc, im.fq0, im.fq1, im.fqa, im.fqaf);
      }
      positive += ov;
    }
  }
  stats_.positive_overlaps += positive;
  stats_.local_samples_discarded += discarded;

  // --- fold: k-fold the frame planes into accumulators + reservoir --------
  const int scale = im.cfg.internal_scale;
  const double inv_s2 = 1.0 / (static_cast<double>(scale) * scale);
  const int ncols = im.ncols, nrows = im.nrows;
  const std::size_t np = im.nplane;
  const unsigned int slots = static_cast<unsigned int>(im.res_slots);
  for (std::size_t px = 0; px < np; ++px) {
    const int nx = static_cast<int>(px % ncols);
    const int ny = static_cast<int>(px / ncols);
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
          if (qp) {
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
          if (qp) {
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
  ++im.frames;
  ++stats_.frames_processed;
  ++stats_.slot_transitions;
  stats_.source_bytes_uploaded +=
      source_n * sizeof(float) * (sigma2_or_null != nullptr ? 2 : 1);
  return true;
}

bool ForwardDrizzleV2CpuKernel::accumulate_frame(
    const double affine6[6], const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || affine6 == nullptr || source == nullptr ||
      impl_->finalized || impl_->frames >= impl_->cfg.stream_length)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  return accumulate_frame_impl(affine6, nullptr, source, sigma2_or_null,
                               frame_order, quality_or_null, meta_or_null);
}

bool ForwardDrizzleV2CpuKernel::accumulate_frame_local(
    const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
    const float *source, const float *sigma2_or_null,
    std::uint64_t frame_order,
    const ForwardDrizzleV2FrameQuality *quality_or_null,
    const ForwardDrizzleV2FrameMeta *meta_or_null) {
  if (impl_ == nullptr || affine6 == nullptr || source == nullptr ||
      impl_->finalized || impl_->frames >= impl_->cfg.stream_length)
    return false;
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(affine6[i])) return false;
  if (warp.max_subdivision_depth > 2) return false;
  return accumulate_frame_impl(affine6, &warp, source, sigma2_or_null,
                               frame_order, quality_or_null, meta_or_null);
}

bool ForwardDrizzleV2CpuKernel::finalize(
    ForwardDrizzleV2PixelResult *results,
    ForwardDrizzleV2ProfileResult *profiles_or_null,
    std::uint64_t *dense_overlap_count) {
  if (impl_ == nullptr || results == nullptr || impl_->finalized) return false;
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
