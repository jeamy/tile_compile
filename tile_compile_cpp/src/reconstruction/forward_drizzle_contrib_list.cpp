#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"

#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <algorithm>
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

// Materialise one frame's contributions for this stripe. `src` covers the
// source rectangle [src_y_origin, src_y_origin + src.rows()) x
// [src_x_origin, src_x_origin + src.cols()) --- a full frame for (0, 0)
// origins, or a banded rect-provider read (A1). It must contain the stripe's
// scan box. E1: single-pass fill --- the old two-pass form rasterized the
// whole stripe twice (count, then fill). The running bound enforces the same
// DRIZZLE_CONTRIB_LIST_BUDGET contract before `out` can exceed it, and every
// emitted record is stored, so the former predicted-count drift check is
// subsumed. NOT sorted here --- the caller sorts (once globally, or per
// frame).
std::vector<DrizzleContrib> build_frame_records(
    const RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &f, std::uint32_t frame_order,
    const Matrix2Df &src, int src_y_origin, int src_x_origin,
    const config::ReconstructionDrizzleConfig &cfg,
    const StripeGeom &g, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &sub, std::size_t mem_budget_bytes,
    int x_begin = 0, int cols = -1) {
  // The buffer must cover exactly what the rasterizer will scan. For a
  // full-frame buffer (origins 0) this is the historical shape check; for a
  // banded rect it is the coverage check.
  const auto scan_box = drizzle_source_scan_box(plan, f, g.scale, y_begin,
                                                rows, x_begin, cols);
  if (src_y_origin > scan_box.y0 || src_x_origin > scan_box.x0 ||
      src_y_origin + src.rows() < scan_box.y1 ||
      src_x_origin + src.cols() < scan_box.x1)
    throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");

  // §30.81 step 3a: the rasterizer sink indexes the window row-major (`win_w`
  // wide, origin `xb`); decode target_x/y back to ABSOLUTE canvas coords.
  const int xb = std::clamp(x_begin, 0, g.W);
  const int win_w = cols < 0 ? g.W : std::clamp(x_begin + cols, xb, g.W) - xb;
  if (win_w <= 0) return {};

  // E1: the two-pass budget contract, enforced incrementally. The old form
  // counted first and rejected when `count >= kCountCeiling ||
  // count * sizeof(DrizzleContrib) > mem_budget_bytes`; the single-pass form
  // therefore permits at most
  //   min(kCountCeiling - 1, mem_budget_bytes / sizeof(DrizzleContrib))
  // records and throws on the next one --- the identical bound.
  const std::size_t kRecordLimit = static_cast<std::size_t>(std::min(
      std::numeric_limits<unsigned long long>::max() /
          static_cast<unsigned long long>(sizeof(DrizzleContrib)) -
          1,
      static_cast<unsigned long long>(
          mem_budget_bytes / sizeof(DrizzleContrib))));

  std::vector<DrizzleContrib> out;
  const auto Wwin = static_cast<std::size_t>(win_w);
  const auto xb_u = static_cast<std::uint32_t>(xb);
  geomstats::ScopedVariant _v(geomstats::Variant::kContribFill, cfg.pixfrac);
  geomstats::ScopedGeometryTimer _t;
  rasterize_drizzle_stripe(
      plan, f, g.scale, cfg.pixfrac, y_begin, rows,
      [&](int sx, int sy, int c, int leaf, std::size_t i, double k) {
        const double v = static_cast<double>(
            src(sy - src_y_origin, sx - src_x_origin));
        if (!std::isfinite(v)) return;
        if (out.size() >= kRecordLimit)
          throw std::runtime_error("DRIZZLE_CONTRIB_LIST_BUDGET");
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
        plan, f, static_cast<std::uint32_t>(fo), source_of(f.source_index), 0,
        0, cfg, g, y_begin, rows, subdivision, mem_budget_bytes);
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
        plan, f, static_cast<std::uint32_t>(fo), source_of(f.source_index), 0,
        0, cfg, g, y_begin, rows, subdivision, per_frame_mem_budget_bytes);
    std::sort(recs.begin(), recs.end(),
              [](const DrizzleContrib &a, const DrizzleContrib &b) {
                return contrib_key_less(a.key, b.key);
              });
    if (!recs.empty())
      reduce_sorted_range(recs.data(), recs.data() + recs.size(), g.W, acc);
  }
  return acc;
}

}  // namespace tile_compile::reconstruction
