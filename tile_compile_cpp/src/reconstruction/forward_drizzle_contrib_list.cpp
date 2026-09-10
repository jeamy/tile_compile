#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"

#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tile_compile::reconstruction {

namespace {
using registration::RegistrationSamplingPlan;

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
           std::uint32_t, std::uint32_t, std::uint32_t>
as_tuple(const DrizzleContribKey &k) {
  return {k.frame_order, k.channel,  k.target_y, k.target_x,
          k.source_y,    k.source_x, k.leaf_order};
}

struct StripeGeom {
  int scale = 1;
  int W = 0;
  int H = 0;
  int channels = 1;
};

StripeGeom stripe_geom(const RegistrationSamplingPlan &plan,
                       const config::ReconstructionDrizzleConfig &cfg,
                       int y_begin, int rows) {
  StripeGeom g;
  g.scale = cfg.internal_scale;
  g.W = plan.canvas_width_native * g.scale;
  g.H = plan.canvas_height_native * g.scale;
  g.channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  if (g.scale < 1 || g.W <= 0 || rows <= 0 || y_begin < 0 ||
      y_begin + rows > g.H)
    throw std::invalid_argument("DRIZZLE_CONTRIB_LIST_BAD_STRIPE");
  return g;
}

// Materialise one frame's contributions for this stripe: pre-count with the
// finite-source filter (overflow-safe), reserve exactly, then fill. NOT sorted
// here --- the caller sorts (once globally, or per frame).
std::vector<DrizzleContrib> build_frame_records(
    const RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &f, std::uint32_t frame_order,
    const Matrix2Df &src, const config::ReconstructionDrizzleConfig &cfg,
    const StripeGeom &g, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &sub, std::size_t mem_budget_bytes,
    int x_begin = 0, int cols = -1) {
  if (src.rows() != plan.source_height || src.cols() != plan.source_width)
    throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");

  // §30.81 step 3a: the rasterizer sink indexes the window row-major (`win_w`
  // wide, origin `xb`); decode target_x/y back to ABSOLUTE canvas coords.
  const int xb = std::clamp(x_begin, 0, g.W);
  const int win_w = cols < 0 ? g.W : std::clamp(x_begin + cols, xb, g.W) - xb;
  if (win_w <= 0) return {};

  unsigned long long count = 0;
  const unsigned long long kCountCeiling =
      static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max() /
                                      sizeof(DrizzleContrib));
  {
    geomstats::ScopedVariant _v(geomstats::Variant::kContribCount, cfg.pixfrac);
    geomstats::ScopedGeometryTimer _t;
    rasterize_drizzle_stripe(
        plan, f, g.scale, cfg.pixfrac, y_begin, rows,
        [&](int sx, int sy, int, int, std::size_t, double) {
          if (std::isfinite(static_cast<double>(src(sy, sx)))) ++count;
        },
        sub, xb, win_w);
  }
  if (count >= kCountCeiling ||
      static_cast<std::size_t>(count) * sizeof(DrizzleContrib) > mem_budget_bytes)
    throw std::runtime_error("DRIZZLE_CONTRIB_LIST_BUDGET");

  std::vector<DrizzleContrib> out;
  out.reserve(static_cast<std::size_t>(count));
  const auto Wwin = static_cast<std::size_t>(win_w);
  const auto xb_u = static_cast<std::uint32_t>(xb);
  geomstats::ScopedVariant _v(geomstats::Variant::kContribFill, cfg.pixfrac);
  geomstats::ScopedGeometryTimer _t;
  rasterize_drizzle_stripe(
      plan, f, g.scale, cfg.pixfrac, y_begin, rows,
      [&](int sx, int sy, int c, int leaf, std::size_t i, double k) {
        const double v = static_cast<double>(src(sy, sx));
        if (!std::isfinite(v)) return;
        DrizzleContrib rec;
        rec.key.frame_order = frame_order;
        rec.key.channel = static_cast<std::uint32_t>(c);
        rec.key.target_y = static_cast<std::uint32_t>(i / Wwin);
        rec.key.target_x = xb_u + static_cast<std::uint32_t>(i % Wwin);
        rec.key.source_y = static_cast<std::uint32_t>(sy);
        rec.key.source_x = static_cast<std::uint32_t>(sx);
        rec.key.leaf_order = static_cast<std::uint32_t>(leaf);
        rec.area = k;
        rec.value = v;
        out.push_back(rec);
      },
      sub, xb, win_w);
  if (out.size() != static_cast<std::size_t>(count))
    throw std::runtime_error("DRIZZLE_CONTRIB_LIST_COUNT_DRIFT");
  return out;
}

// plan 19.6.2 hybrid CPU-geometry -> GPU-rasterization for a LOCAL-WARP frame.
// The CPU reference (`enumerate_drizzle_stripe_leaf_cells` -> `sample_leaves` ->
// `subdivide_local` / `invert_local_source_to_canvas`) authoritatively fixes the
// displacement, the fixed-point inversion, the bounds test, the adaptive
// subdivision and the exact leaf corners AND the clamped bbox cell set. The GPU
// only computes the exact polygon/cell overlap AREA for each (leaf, cell), via
// the same bit-exact kernel the affine device path uses
// (forward_drizzle_cuda_polygon_rect_area_batch). leaf_order, source index and
// channel ride along on the host. Downstream (sort, Q fold, clip, profile) is
// unchanged shared host code, so the result is byte-identical to the pure CPU
// reference for MONO and OSC, every chunk height, and any `max_batch_items`.
std::vector<DrizzleContrib> build_frame_records_hybrid_local(
    const RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &f, std::uint32_t frame_order,
    const Matrix2Df &src, const config::ReconstructionDrizzleConfig &cfg,
    const StripeGeom &g, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &sub, std::size_t mem_budget_bytes,
    std::size_t max_batch_items, HybridPathStats *stats) {
  if (src.rows() != plan.source_height || src.cols() != plan.source_width)
    throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");
  if (max_batch_items == 0) max_batch_items = 1;

  using hclock = std::chrono::steady_clock;
  const auto t_total0 = hclock::now();
  double gpu_seconds = 0.0;
  long long gpu_calls = 0, cells = 0;

  struct WorkItem {
    std::uint32_t sx, sy, ty, tx, channel, leaf_order;
  };
  std::vector<double> quad;  // 8 doubles / item: CCW corners x0,y0..x3,y3
  std::vector<double> rect;  // 4 doubles / item: x, y, x+1, y+1
  std::vector<WorkItem> meta;
  quad.reserve(max_batch_items * 8);
  rect.reserve(max_batch_items * 4);
  meta.reserve(max_batch_items);

  std::vector<DrizzleContrib> out;
  std::size_t batch_cap = max_batch_items;
  constexpr std::size_t kBatchFloor = 4096;

  auto flush = [&]() {
    const std::size_t n = meta.size();
    if (n == 0) return;
    std::vector<double> area(n);
    const auto t_gpu0 = hclock::now();
    for (;;) {
      bool ok = true;
      for (std::size_t off = 0; off < n && ok; off += batch_cap) {
        const int m = static_cast<int>(std::min(batch_cap, n - off));
        ok = forward_drizzle_cuda_polygon_rect_area_batch(
            quad.data() + off * 8, rect.data() + off * 4, m, area.data() + off);
        ++gpu_calls;
      }
      if (ok) break;
      if (batch_cap <= kBatchFloor)
        throw ForwardDrizzleCudaError(
            "forward_drizzle CUDA: polygon-area batch failed at the batch floor "
            "(hybrid local-warp path)");
      batch_cap /= 2;
    }
    gpu_seconds +=
        std::chrono::duration<double>(hclock::now() - t_gpu0).count();
    cells += static_cast<long long>(n);
    for (std::size_t t = 0; t < n; ++t) {
      if (!(area[t] > 0.0)) continue;
      const WorkItem &w = meta[t];
      const double v = static_cast<double>(
          src(static_cast<int>(w.sy), static_cast<int>(w.sx)));
      if (!std::isfinite(v)) continue;
      DrizzleContrib rec;
      rec.key.frame_order = frame_order;
      rec.key.channel = w.channel;
      rec.key.target_y = w.ty;
      rec.key.target_x = w.tx;
      rec.key.source_y = w.sy;
      rec.key.source_x = w.sx;
      rec.key.leaf_order = w.leaf_order;
      rec.area = area[t];
      rec.value = v;
      out.push_back(rec);
    }
    if (out.size() * sizeof(DrizzleContrib) > mem_budget_bytes)
      throw std::runtime_error("DRIZZLE_CONTRIB_LIST_BUDGET");
    quad.clear();
    rect.clear();
    meta.clear();
  };

  // No ScopedGeometryTimer here: HybridPathStats already separates CPU geometry
  // from GPU raster time on this path, and flush() (GPU work) runs inside the
  // enumerate sink.
  geomstats::ScopedVariant _v(geomstats::Variant::kHybridCpuGeometry,
                              cfg.pixfrac);
  enumerate_drizzle_stripe_leaf_cells(
      plan, f, g.scale, cfg.pixfrac, y_begin, rows,
      [&](int sx, int sy, int c, int leaf_order, int cx, int cy,
          const double *lx, const double *ly) {
        quad.insert(quad.end(), {lx[0], ly[0], lx[1], ly[1], lx[2], ly[2],
                                 lx[3], ly[3]});
        const double dcx = static_cast<double>(cx);
        const double dcy = static_cast<double>(cy);
        rect.insert(rect.end(), {dcx, dcy, dcx + 1.0, dcy + 1.0});
        meta.push_back(WorkItem{static_cast<std::uint32_t>(sx),
                                static_cast<std::uint32_t>(sy),
                                static_cast<std::uint32_t>(cy - y_begin),
                                static_cast<std::uint32_t>(cx),
                                static_cast<std::uint32_t>(c),
                                static_cast<std::uint32_t>(leaf_order)});
        if (meta.size() >= max_batch_items) flush();
      },
      sub);
  flush();
  if (stats) {
    const double total =
        std::chrono::duration<double>(hclock::now() - t_total0).count();
    stats->gpu_raster_seconds += gpu_seconds;
    stats->cpu_seconds += std::max(0.0, total - gpu_seconds);
    stats->gpu_batch_calls += gpu_calls;
    stats->leaf_cells += cells;
    stats->records += static_cast<long long>(out.size());
  }
  return out;
}

// Segment-reduce an already-canonically-sorted record range [begin, end) into
// wx/w/w2: for each maximal run of equal (frame, channel, target_y, target_x),
// one double accumulator pair in record order, then wx += A / w += B / w2 += B*B.
void reduce_sorted_range(const DrizzleContrib *begin, const DrizzleContrib *end,
                         int W, DrizzleUniformAccum &acc) {
  const DrizzleContrib *s = begin;
  while (s < end) {
    const DrizzleContrib *e = s + 1;
    const auto &k0 = s->key;
    while (e < end) {
      const auto &k = e->key;
      if (k.frame_order != k0.frame_order || k.channel != k0.channel ||
          k.target_y != k0.target_y || k.target_x != k0.target_x)
        break;
      ++e;
    }
    double A = 0.0, B = 0.0;
    for (const DrizzleContrib *t = s; t < e; ++t) {
      A += t->area * t->value;
      B += t->area;
    }
    const std::size_t i = static_cast<std::size_t>(k0.target_y) *
                              static_cast<std::size_t>(W) +
                          static_cast<std::size_t>(k0.target_x);
    const int c = static_cast<int>(k0.channel);
    acc.wx[c][i] += A;
    acc.w[c][i] += B;
    acc.w2[c][i] += B * B;
    s = e;
  }
}

DrizzleUniformAccum make_accum(const StripeGeom &g, int rows) {
  DrizzleUniformAccum acc;
  acc.width = g.W;
  acc.rows = rows;
  acc.channels = g.channels;
  const std::size_t n =
      static_cast<std::size_t>(g.W) * static_cast<std::size_t>(rows);
  for (int c = 0; c < g.channels; ++c) {
    acc.wx[c].assign(n, 0.0);
    acc.w[c].assign(n, 0.0);
    acc.w2[c].assign(n, 0.0);
  }
  return acc;
}

}  // namespace

bool contrib_key_less(const DrizzleContribKey &a, const DrizzleContribKey &b) {
  return as_tuple(a) < as_tuple(b);
}

DrizzleContribList build_uniform_contrib_list(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision,
    std::size_t mem_budget_bytes) {
  const StripeGeom g = stripe_geom(plan, cfg, y_begin, rows);
  const auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);

  DrizzleContribList list;
  list.width = g.W;
  list.rows = rows;
  list.channels = g.channels;
  list.y_begin = y_begin;

  for (std::size_t fo = 0; fo < prepared.frames.size(); ++fo) {
    const auto &f = *prepared.frames[fo];
    auto recs = build_frame_records(
        plan, f, static_cast<std::uint32_t>(fo), source_of(f.source_index), cfg,
        g, y_begin, rows, subdivision, mem_budget_bytes);
    // The running total must also respect the budget (whole-stripe form).
    if ((list.records.size() + recs.size()) * sizeof(DrizzleContrib) >
        mem_budget_bytes)
      throw std::runtime_error("DRIZZLE_CONTRIB_LIST_BUDGET");
    list.records.insert(list.records.end(), recs.begin(), recs.end());
  }
  list.predicted_count = list.records.size();

  std::sort(list.records.begin(), list.records.end(),
            [](const DrizzleContrib &a, const DrizzleContrib &b) {
              return contrib_key_less(a.key, b.key);
            });
  return list;
}

DrizzleUniformAccum reduce_uniform_contrib_list(const DrizzleContribList &list) {
  DrizzleUniformAccum acc;
  acc.width = list.width;
  acc.rows = list.rows;
  acc.channels = list.channels;
  const std::size_t n =
      static_cast<std::size_t>(list.width) * static_cast<std::size_t>(list.rows);
  for (int c = 0; c < list.channels; ++c) {
    acc.wx[c].assign(n, 0.0);
    acc.w[c].assign(n, 0.0);
    acc.w2[c].assign(n, 0.0);
  }
  if (!list.records.empty())
    reduce_sorted_range(list.records.data(),
                        list.records.data() + list.records.size(), list.width,
                        acc);
  return acc;
}

DrizzleUniformAccum accumulate_uniform_by_frame(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision,
    std::size_t per_frame_mem_budget_bytes) {
  const StripeGeom g = stripe_geom(plan, cfg, y_begin, rows);
  const auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  DrizzleUniformAccum acc = make_accum(g, rows);

  // frame_order leads the canonical key, so a frame's records form a
  // self-contained block: build -> sort -> reduce one frame at a time, in
  // prepared-frame order. Bit-identical to the whole-stripe build+sort+reduce.
  for (std::size_t fo = 0; fo < prepared.frames.size(); ++fo) {
    const auto &f = *prepared.frames[fo];
    auto recs = build_frame_records(
        plan, f, static_cast<std::uint32_t>(fo), source_of(f.source_index), cfg,
        g, y_begin, rows, subdivision, per_frame_mem_budget_bytes);
    std::sort(recs.begin(), recs.end(),
              [](const DrizzleContrib &a, const DrizzleContrib &b) {
                return contrib_key_less(a.key, b.key);
              });
    if (!recs.empty())
      reduce_sorted_range(recs.data(), recs.data() + recs.size(), g.W, acc);
  }
  return acc;
}

namespace {

// (isfinite(qv) && qv > 0) ? qv : 0.0  --- the plan-11.9 fold applied per
// contribution, before summing, exactly as the streaming sink does.
double q_fold(double qv) {
  return (std::isfinite(qv) && qv > 0.0) ? qv : 0.0;
}
double map_at(const Matrix2Df *m, std::uint32_t sy, std::uint32_t sx) {
  return m ? static_cast<double>((*m)(static_cast<int>(sy),
                                      static_cast<int>(sx)))
           : 1.0;
}

// Produces one frame's UNSORTED contribution records for the stripe. The CPU
// producer runs the reference rasterizer; the CUDA producer runs the device
// rasterizer and converts. Everything downstream (sort, Q fold, clip, profile)
// is shared, so the CPU and CUDA pair results are bit-identical exactly when
// the records match.
//
// §30.81 step 3a: `x_begin`/`cols` scope the producer to a target-column window
// (`cols < 0` => full width). The CPU producer enumerates only that window; the
// CUDA producer narrows the uploaded source-row band and drops out-of-window
// records (the device kernel X-window is step 3). target_x keys stay ABSOLUTE.
using PairFrameRecordProducer = std::function<std::vector<DrizzleContrib>(
    std::size_t fo, const registration::FrameSamplingTransform &f,
    const StripeGeom &g, int x_begin, int cols)>;

ForwardDrizzleUniformAndRawResult accumulate_pair_impl(
    const RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb,
    std::size_t mem_budget_bytes, const PairFrameRecordProducer &produce,
    int x_begin, int cols, const PairTileSink *tile_sink, int tile_cols) {
  const StripeGeom g = stripe_geom(plan, cfg, y_begin, rows);
  const auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  const int channels = g.channels;
  const std::size_t frame_count = plan.frames.size();
  const bool tiled = tile_sink != nullptr && *tile_sink && tile_cols > 0;

  const bool need_q0 = mb.emit_fine;
  const bool need_q1 = mb.emit_medium;
  const bool need_qa = mb.emit_alpha_confidence;
  if ((mb.emit_fine || mb.emit_medium || need_qa) && !quality_of)
    throw std::invalid_argument("DRIZZLE_MULTIBAND_REQUIRES_QUALITY_PROVIDER");

  std::size_t source_count = 0;
  for (const auto &f : plan.frames)
    source_count = std::max(source_count, f.source_index + 1);

  bool need_qc = false;
  if (quality_of)
    for (const auto &f : plan.frames)
      if (quality_of(f.source_index).composite) { need_qc = true; break; }

  std::vector<std::pair<std::uint8_t, float>> reg_by_source;
  if (need_qa) {
    reg_by_source.assign(source_count, {0u, 1.0f});
    for (const auto &f : plan.frames) {
      const bool direct =
          !f.model_predicted && f.model_prediction_factor == 1.0f;
      reg_by_source[f.source_index] = {direct ? std::uint8_t{1}
                                             : std::uint8_t{0},
                                       f.registration_residual_factor};
    }
  }
  auto g_eff_for = [&](std::size_t si) -> double {
    if (g_eff_by_source_index.empty()) return 1.0;
    return static_cast<double>(g_eff_by_source_index[si]);
  };

  const DrizzleProfileReduceConfig reduce_cfg{
      cfg.min_clip_contributors,   cfg.robust_passes,
      clip_cfg.clip_sigma_low,     clip_cfg.clip_sigma_high,
      clip_cfg.min_fraction,       clip_cfg.min_n_eff,
      mb.emit_fine,                mb.emit_medium,
      need_qa,                     mb.fine_quality_exponent,
      mb.medium_quality_exponent,  mb.alpha_confidence};

  // §30.81 step-5 baseline timers (see forward_drizzle_cuda.hpp). Zero cost
  // unless TC_FD_CUDA_PROFILE.
  const bool cprof = forward_drizzle_cuda_profile_enabled();
  auto &cp = forward_drizzle_cuda_profile();
  using cclock = std::chrono::steady_clock;
  auto cnow = [] { return cclock::now(); };
  auto cadd = [&](std::atomic<double> &slot, cclock::time_point t0) {
    if (cprof)
      forward_drizzle_cuda_profile_add(
          slot, std::chrono::duration<double>(cnow() - t0).count());
  };
  if (cprof) cp.calls.fetch_add(1, std::memory_order_relaxed);

  auto planes_of = [&](ForwardDrizzleUniformResult &p) {
    return channels == 1
               ? std::array<ProfilePlane *, 3>{&p.L, nullptr, nullptr}
               : std::array<ProfilePlane *, 3>{&p.R, &p.G, &p.B};
  };

  // Build one frame's records for a target-column window (`ww < 0` => full
  // width): produce (device or CPU) then the canonical sort. target_x keys are
  // absolute, so the sort order and every downstream step are identical to the
  // full-width records restricted to the window.
  auto produce_sorted = [&](std::size_t fo, int wx0, int ww) {
    auto tp = cnow();
    auto recs = produce(fo, *prepared.frames[fo], g, wx0, ww);
    cadd(cp.produce_s, tp);
    if (cprof)
      cp.records_sorted.fetch_add(recs.size(), std::memory_order_relaxed);
    tp = cnow();
    std::sort(recs.begin(), recs.end(),
              [](const DrizzleContrib &a, const DrizzleContrib &b) {
                return contrib_key_less(a.key, b.key);
              });
    cadd(cp.sort_s, tp);
    return recs;
  };

  // Reduce one target-column window [wx0, wx0+ww) into a fresh result of that
  // width. `get_recs(fo)` yields frame fo's already-sorted records as a live
  // reference (the memo entry, or a reused scratch on the single-window path).
  // Bit-identical to the pre-step-5 body over the same window --- the only
  // change is that the produce + sort moved into `get_recs`.
  auto reduce_window =
      [&](int wx0, int ww,
          const auto &get_recs) -> ForwardDrizzleUniformAndRawResult {
    const std::size_t n =
        static_cast<std::size_t>(ww) * static_cast<std::size_t>(rows);
    {  // fail closed before the flat ClipCandidate buffer allocation
      const std::size_t per_ch =
          (frame_count &&
           n > std::numeric_limits<std::size_t>::max() / frame_count)
              ? std::numeric_limits<std::size_t>::max()
              : n * frame_count;
      const std::size_t elems =
          (channels && per_ch > std::numeric_limits<std::size_t>::max() /
                                    static_cast<std::size_t>(channels))
              ? std::numeric_limits<std::size_t>::max()
              : per_ch * static_cast<std::size_t>(channels);
      const std::size_t bytes =
          (elems >
           std::numeric_limits<std::size_t>::max() / sizeof(ClipCandidate))
              ? std::numeric_limits<std::size_t>::max()
              : elems * sizeof(ClipCandidate);
      if (bytes > mem_budget_bytes)
        throw std::runtime_error("DRIZZLE_CONTRIB_LIST_BUDGET");
    }
    const std::uint32_t wxb_u = static_cast<std::uint32_t>(wx0);
    const std::uint32_t wxe_u = static_cast<std::uint32_t>(wx0 + ww);
    const auto Wsz = static_cast<std::size_t>(ww);

    ForwardDrizzleUniformAndRawResult r;
    auto init_profile = [&](ForwardDrizzleUniformResult &p, bool on) {
      p.color_mode = plan.color_mode;
      p.internal_width = ww;
      p.internal_height = on ? rows : 0;
      if (!on) return;
      if (channels == 1) {
        p.L.allocate(ww, rows);
      } else {
        p.R.allocate(ww, rows);
        p.G.allocate(ww, rows);
        p.B.allocate(ww, rows);
      }
    };
    init_profile(r.uniform, true);
    init_profile(r.raw, true);
    init_profile(r.fine, mb.emit_fine);
    init_profile(r.medium, mb.emit_medium);
    std::vector<double> ac_sep, ac_art, ac_reg;
    if (need_qa) {
      r.a_separation.assign(n, std::numeric_limits<float>::quiet_NaN());
      r.a_artifact.assign(n, std::numeric_limits<float>::quiet_NaN());
      r.a_registration.assign(n, std::numeric_limits<float>::quiet_NaN());
      r.alpha_confidence_support.assign(n, 0u);
      ac_sep.assign(n, std::numeric_limits<double>::infinity());
      ac_art.assign(n, std::numeric_limits<double>::infinity());
      ac_reg.assign(n, std::numeric_limits<double>::infinity());
    }
    const auto up = planes_of(r.uniform);
    const auto rp = planes_of(r.raw);
    const auto fp = planes_of(r.fine);
    const auto mp = planes_of(r.medium);

    std::vector<std::vector<ClipCandidate>> cand(channels);
    std::vector<std::vector<std::size_t>> counts(channels);
    for (int c = 0; c < channels; ++c) {
      cand[c].resize(n * frame_count);
      counts[c].assign(n, 0);
    }

    for (std::size_t fo = 0; fo < prepared.frames.size(); ++fo) {
      const auto &f = *prepared.frames[fo];
      FrameQualityMaps qm;
      if (quality_of) qm = quality_of(f.source_index);
      const Matrix2Df *qc = need_qc ? qm.composite : nullptr;
      const Matrix2Df *q0 = need_q0 ? qm.scale0 : nullptr;
      const Matrix2Df *q1 = need_q1 ? qm.scale1 : nullptr;
      const Matrix2Df *qa = need_qa ? qm.artifact : nullptr;
      const std::vector<DrizzleContrib> &recs = get_recs(fo);

      std::size_t s = 0;
      while (s < recs.size()) {
        std::size_t e = s + 1;
        const auto &k0 = recs[s].key;
        while (e < recs.size()) {
          const auto &k = recs[e].key;
          if (k.channel != k0.channel || k.target_y != k0.target_y ||
              k.target_x != k0.target_x)
            break;
          ++e;
        }
        // A segment shares one (channel, target_y, target_x) key, so a record
        // outside [wx0, wx0+ww) means the whole segment is; skip it before the
        // Q fold. Full-width window => never taken.
        if (k0.target_x < wxb_u || k0.target_x >= wxe_u) {
          s = e;
          continue;
        }
        double A = 0.0, B = 0.0, QA = 0.0, QA0 = 0.0, QA1 = 0.0, QAA = 0.0,
               QAF = 0.0;
        for (std::size_t t = s; t < e; ++t) {
          const auto &rr = recs[t];
          A += rr.area * rr.value;
          B += rr.area;
          const std::uint32_t sy = rr.key.source_y, sx = rr.key.source_x;
          if (qc) QA += rr.area * q_fold(map_at(qc, sy, sx));
          if (q0) QA0 += rr.area * q_fold(map_at(q0, sy, sx));
          if (q1) QA1 += rr.area * q_fold(map_at(q1, sy, sx));
          if (qa) {
            const double av = map_at(qa, sy, sx);
            QAA += rr.area * q_fold(av);
            if (std::isfinite(av)) QAF += rr.area;
          }
        }
        if (B > 0.0) {
          const int c = static_cast<int>(k0.channel);
          const std::size_t i = static_cast<std::size_t>(k0.target_y) * Wsz +
                                (k0.target_x - wxb_u);
          cand[c][i * frame_count + counts[c][i]++] = ClipCandidate{
              f.source_index, A / B, B, need_qc ? QA / B : 1.0,
              need_q0 ? QA0 / B : 1.0, need_q1 ? QA1 / B : 1.0,
              need_qa ? QAA / B : 1.0, need_qa && QAF > 0.0};
        }
        s = e;
      }
    }

    // Priority 2 (§30.79): reused clip scratch; serial per window.
    DrizzleClipScratch clip_scratch;
    clip_scratch.reserve_for(frame_count, need_qa);
    const auto tp_reduce = cnow();
    for (int c = 0; c < channels; ++c)
      for (std::size_t i = 0; i < n; ++i) {
        if (!counts[c][i]) continue;
        const std::span<const ClipCandidate> pixel(
            cand[c].data() + i * frame_count, counts[c][i]);
        reduce_pixel_profiles(pixel, reduce_cfg, g_eff_for, reg_by_source, i,
                              up[c], rp[c], mb.emit_fine ? fp[c] : nullptr,
                              mb.emit_medium ? mp[c] : nullptr,
                              need_qa ? &ac_sep[i] : nullptr,
                              need_qa ? &ac_art[i] : nullptr,
                              need_qa ? &ac_reg[i] : nullptr, r.clipping,
                              &clip_scratch);
      }
    cadd(cp.reduce_s, tp_reduce);

    if (need_qa)
      for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ac_sep[i])) continue;
        r.a_separation[i] = static_cast<float>(ac_sep[i]);
        r.a_artifact[i] = static_cast<float>(ac_art[i]);
        r.a_registration[i] = static_cast<float>(ac_reg[i]);
        r.alpha_confidence_support[i] = 1u;
      }
    return r;
  };

  if (tiled) {
    // §30.81 step 3a: NO per-band record memo. B's memo held every frame's
    // sorted records for the whole band at once --- frame_count * source_width *
    // cells per internal row --- which reintroduced the §30.80 collapse-with-N
    // (the band shrank to ~2 rows at 600 frames). Instead each column tile is
    // reduced from a producer call SCOPED to that tile's target window: only
    // one frame's tile-window records are live at any time, so the band height
    // no longer carries a frame_count factor.
    //
    // Bit-identity: the producer keeps target_x keys ABSOLUTE (so the canonical
    // sort order matches the full-width records restricted to the window), and
    // reduce_window's per-frame loop appends candidates frame-ascending within
    // each target cell --- the same (channel, cell) candidate spans, same
    // order, that the pre-3a full-width tiled path produced.
    //
    // Cost: produce + sort now run once per (tile, frame) again, but each call
    // is scoped to ~tile_cols/W of the canvas (CPU: windowed enumeration; CUDA:
    // narrowed source band today, device X-window is step 3), so the total
    // record work stays ~flat in the tile count for well-behaved affines
    // (measured source-footprint duplication <= 1.03x at the resolved band
    // heights; shear / scale / edge / local-warp coverage is step 3/4).
    int cur_x0 = 0, cur_w = 0;
    std::vector<DrizzleContrib> scratch;
    auto from_window =
        [&](std::size_t fo) -> const std::vector<DrizzleContrib> & {
      scratch = produce_sorted(fo, cur_x0, cur_w);
      return scratch;
    };
    ForwardDrizzleUniformAndRawResult agg;  // planes empty; .clipping summed
    for (int wx0 = 0; wx0 < g.W; wx0 += tile_cols) {
      const int tw = std::min(tile_cols, g.W - wx0);
      cur_x0 = wx0;
      cur_w = tw;
      auto tile = reduce_window(wx0, tw, from_window);
      agg.clipping.pixel_channel_evaluations +=
          tile.clipping.pixel_channel_evaluations;
      agg.clipping.pixel_channel_rejected +=
          tile.clipping.pixel_channel_rejected;
      agg.clipping.candidate_contributions_clipped +=
          tile.clipping.candidate_contributions_clipped;
      (*tile_sink)(wx0, tw, tile);
    }
    return agg;
  }

  // Single-window path: produce + sort one frame at a time into a reused
  // scratch, so the peak is one frame's records + one window's candidate
  // buffer.
  const int xb = std::clamp(x_begin, 0, g.W);
  const int xe = cols < 0 ? g.W : std::clamp(x_begin + cols, xb, g.W);
  const int win_w = xe - xb;
  if (win_w <= 0)
    throw std::invalid_argument("DRIZZLE_EMPTY_TARGET_WINDOW");
  std::vector<DrizzleContrib> scratch;
  auto from_producer =
      [&](std::size_t fo) -> const std::vector<DrizzleContrib> & {
    scratch = produce_sorted(fo, xb, win_w);
    return scratch;
  };
  return reduce_window(xb, win_w, from_producer);
}

// The CPU record producer: the plan-19.6 reference rasterizer.
PairFrameRecordProducer cpu_pair_producer(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision, int y_begin, int rows,
    std::size_t mem_budget_bytes) {
  return [&plan, &source_of, &cfg, &subdivision, y_begin, rows,
          mem_budget_bytes](std::size_t fo,
                            const registration::FrameSamplingTransform &f,
                            const StripeGeom &g, int x_begin, int cols) {
    return build_frame_records(plan, f, static_cast<std::uint32_t>(fo),
                               source_of(f.source_index), cfg, g, y_begin, rows,
                               subdivision, mem_budget_bytes, x_begin, cols);
  };
}

// The CUDA record producer. Affine frames -> the device affine rasterizer.
// Local-warp frames -> the plan-19.6.2 hybrid path (CPU geometry, GPU raster).
// Throws ForwardDrizzleCudaError on any device failure --- the caller decides
// whether to fall back to the CPU producer for the whole stripe (plan 19.4: no
// mixed CPU/CUDA within a commit).
PairFrameRecordProducer cuda_pair_producer(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision, int y_begin, int rows,
    int max_cells_per_pixel, std::size_t mem_budget_bytes,
    std::size_t max_batch_items, HybridPathStats *hybrid_stats) {
  return [&plan, &source_of, &cfg, &subdivision, y_begin, rows,
          max_cells_per_pixel, mem_budget_bytes, max_batch_items, hybrid_stats](
             std::size_t fo, const registration::FrameSamplingTransform &f,
             const StripeGeom &g, int x_begin,
             int cols) -> std::vector<DrizzleContrib> {
    // §30.81 step 3a: the device kernel does not yet take an X-window (step 3),
    // so the affine path narrows the uploaded source-row band using the window
    // corners and drops out-of-window records after conversion; the hybrid
    // local-warp path (no safe inverse-affine bbox, §30.81 note) produces full
    // width and is filtered the same way. `cols < 0` => full width, unchanged.
    const int win_xb = std::clamp(x_begin, 0, g.W);
    const int win_xe = cols < 0 ? g.W : std::clamp(x_begin + cols, win_xb, g.W);
    const auto win_xb_u = static_cast<std::uint32_t>(win_xb);
    const auto win_xe_u = static_cast<std::uint32_t>(win_xe);
    const bool windowed = win_xb != 0 || win_xe != g.W;
    auto clip_to_window = [&](std::vector<DrizzleContrib> &v) {
      if (!windowed) return;
      v.erase(std::remove_if(v.begin(), v.end(),
                             [&](const DrizzleContrib &d) {
                               return d.key.target_x < win_xb_u ||
                                      d.key.target_x >= win_xe_u;
                             }),
              v.end());
    };
    if (f.has_smooth_local_model) {
      auto recs = build_frame_records_hybrid_local(
          plan, f, static_cast<std::uint32_t>(fo), source_of(f.source_index),
          cfg, g, y_begin, rows, subdivision, mem_budget_bytes, max_batch_items,
          hybrid_stats);
      clip_to_window(recs);
      return recs;
    }
    if (!f.source_to_canvas_affine_valid)
      throw ForwardDrizzleCudaError(
          "forward_drizzle CUDA: frame has no valid affine");
    const Matrix2Df &src = source_of(f.source_index);
    const int sw = plan.source_width, sh = plan.source_height;
    if (src.rows() != sh || src.cols() != sw)
      throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");

    // Source-row band for this stripe, derived exactly like the CPU path
    // (rasterize_drizzle_stripe): inverse-map the destination stripe corners.
    // §30.81 step 3a keeps this over the FULL canvas width even for a tile
    // window: the device kernel still enumerates [0, g.W), and narrowing the
    // band by the window's dest-x range would rest on a source-y margin
    // (droplet half-width + leaf-subdivision spread) that is not verified here.
    // The real per-window device saving (kernel X-window + narrowed upload) is
    // step 3; step 3a's CUDA producer only DROPS out-of-window records.
    const auto &s2c = f.source_to_canvas;
    WarpMatrix inv;
    if (!registration::invert_affine_2x3(s2c, 1e-12f, 1e12f, inv))
      throw ForwardDrizzleCudaError("forward_drizzle CUDA: singular affine");
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (double dx : {0.0, static_cast<double>(g.W) / g.scale})
      for (double dy : {static_cast<double>(y_begin) / g.scale,
                        static_cast<double>(y_begin + rows) / g.scale}) {
        const double sy = static_cast<double>(inv(1, 0)) * dx +
                          static_cast<double>(inv(1, 1)) * dy +
                          static_cast<double>(inv(1, 2));
        lo = std::min(lo, sy);
        hi = std::max(hi, sy);
      }
    int band0 = static_cast<int>(
        std::clamp(std::floor(lo - 1), 0.0, static_cast<double>(sh)));
    int band1 = static_cast<int>(
        std::clamp(std::ceil(hi + 1), 0.0, static_cast<double>(sh)));
    if (band1 <= band0) return {};

    // BAND-LOCAL source buffer only (row 0 == source row band0) --- the whole
    // image is never copied per frame.
    const int band_h = band1 - band0;
    std::vector<float> src_buf(static_cast<std::size_t>(band_h) * sw);
    for (int yy = 0; yy < band_h; ++yy)
      for (int xx = 0; xx < sw; ++xx)
        src_buf[static_cast<std::size_t>(yy) * sw + xx] = src(band0 + yy, xx);

    const double affine6[6] = {
        static_cast<double>(s2c(0, 0)), static_cast<double>(s2c(0, 1)),
        static_cast<double>(s2c(0, 2)), static_cast<double>(s2c(1, 0)),
        static_cast<double>(s2c(1, 1)), static_cast<double>(s2c(1, 2))};
    const double half = static_cast<double>(cfg.pixfrac) / 2.0;
    const bool mono = plan.color_mode == ColorMode::MONO;

    const long long band_rows = band1 - band0;
    // Generous capacity: a well-behaved affine leaf overlaps <= ~4 cells; size
    // for `max_cells_per_pixel` and let the device fail (-> CPU fallback) if a
    // degenerate frame exceeds it.
    const long long cap =
        band_rows * sw * static_cast<long long>(max_cells_per_pixel);
    std::vector<CudaDrizzleContribRecord> raw(static_cast<std::size_t>(cap));
    long long written = 0;
    if (!forward_drizzle_cuda_affine_frame_contributions(
            affine6, cfg.internal_scale, half, y_begin, rows, g.W, band0, band1,
            sw, sh, src_buf.data(), static_cast<int>(plan.bayer_pattern),
            plan.cfa_origin_x, plan.cfa_origin_y, mono, max_cells_per_pixel,
            raw.data(), cap, &written))
      throw ForwardDrizzleCudaError(
          "forward_drizzle CUDA: affine frame rasterization failed");

    std::vector<DrizzleContrib> out;
    out.reserve(static_cast<std::size_t>(written));
    for (long long t = 0; t < written; ++t) {
      const auto &r = raw[static_cast<std::size_t>(t)];
      DrizzleContrib d;
      d.key.frame_order = static_cast<std::uint32_t>(fo);
      d.key.channel = r.channel;
      d.key.target_y = r.target_y;
      d.key.target_x = r.target_x;
      d.key.source_y = r.source_y;
      d.key.source_x = r.source_x;
      d.key.leaf_order = 0;
      d.area = r.area;
      d.value = r.value;
      out.push_back(d);
    }
    clip_to_window(out);
    return out;
  };
}

}  // namespace

ForwardDrizzleUniformAndRawResult accumulate_pair_by_frame(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb,
    std::size_t mem_budget_bytes, int target_x_begin, int target_cols,
    const PairTileSink *tile_sink, int tile_cols) {
  return accumulate_pair_impl(
      plan, cfg, clip_cfg, y_begin, rows, subdivision, g_eff_by_source_index,
      quality_of, mb, mem_budget_bytes,
      cpu_pair_producer(plan, source_of, cfg, subdivision, y_begin, rows,
                        mem_budget_bytes),
      target_x_begin, target_cols, tile_sink, tile_cols);
}

ForwardDrizzleUniformAndRawResult accumulate_pair_by_frame_cuda(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb,
    std::size_t mem_budget_bytes, int max_cells_per_pixel,
    std::size_t max_batch_items, HybridPathStats *hybrid_stats,
    int target_x_begin, int target_cols, const PairTileSink *tile_sink,
    int tile_cols) {
  return accumulate_pair_impl(
      plan, cfg, clip_cfg, y_begin, rows, subdivision, g_eff_by_source_index,
      quality_of, mb, mem_budget_bytes,
      cuda_pair_producer(plan, source_of, cfg, subdivision, y_begin, rows,
                         max_cells_per_pixel, mem_budget_bytes, max_batch_items,
                         hybrid_stats),
      target_x_begin, target_cols, tile_sink, tile_cols);
}

}  // namespace tile_compile::reconstruction
