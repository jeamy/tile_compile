#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tile_compile::reconstruction {

void ProfilePlane::allocate(int w, int h) {
  width = w;
  height = h;
  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
  value.assign(n, std::numeric_limits<float>::quiet_NaN());
  weight_sum.assign(n, 0.0f);
  n_eff.assign(n, 0.0f);
  support.assign(n, 0);
}

namespace {

using registration::FrameSamplingTransform;
using registration::LocalInversionParams;
using registration::RegistrationSamplingPlan;

} // namespace

double shoelace_area(const double *x, const double *y, int n) {
  if (n < 3)
    return 0.0;
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1) % n;
    s += x[i] * y[j] - x[j] * y[i];
  }
  return std::abs(s) * 0.5;
}

namespace {

// Sutherland-Hodgman clip of a convex polygon against one axis-aligned
// half-plane, generic over the plane test/intersection so all four sides of
// a rectangle reuse it.
template <typename InsideFn, typename IntersectFn>
int clip_one_plane(const double *in_x, const double *in_y, int in_n,
                   InsideFn inside, IntersectFn intersect, double *out_x,
                   double *out_y) {
  if (in_n == 0)
    return 0;
  int out_n = 0;
  for (int i = 0; i < in_n; ++i) {
    const int j = (i + 1) % in_n;
    const bool in_i = inside(in_x[i], in_y[i]);
    const bool in_j = inside(in_x[j], in_y[j]);
    if (in_i) {
      out_x[out_n] = in_x[i];
      out_y[out_n] = in_y[i];
      ++out_n;
    }
    if (in_i != in_j) {
      double ix = 0.0, iy = 0.0;
      intersect(in_x[i], in_y[i], in_x[j], in_y[j], ix, iy);
      out_x[out_n] = ix;
      out_y[out_n] = iy;
      ++out_n;
    }
  }
  return out_n;
}

} // namespace

double polygon_rectangle_intersection_area(const double poly_x[4],
                                           const double poly_y[4], double rx0,
                                           double ry0, double rx1, double ry1) {
  // A convex quad clipped against 4 half-planes can gain at most 4 vertices
  // (one intersection point per edge crossing per plane); 4 + 4 = 8 is a
  // safe static bound used throughout.
  double bx[8], by[8];
  double cx[8], cy[8];
  int n = 4;
  for (int i = 0; i < 4; ++i) {
    bx[i] = poly_x[i];
    by[i] = poly_y[i];
  }

  n = clip_one_plane(
      bx, by, n, [&](double x, double /*y*/) { return x >= rx0; },
      [&](double x0, double y0, double x1, double y1, double &ix, double &iy) {
        const double t = (rx0 - x0) / (x1 - x0);
        ix = rx0;
        iy = y0 + t * (y1 - y0);
      },
      cx, cy);
  std::copy(cx, cx + n, bx);
  std::copy(cy, cy + n, by);

  n = clip_one_plane(
      bx, by, n, [&](double x, double /*y*/) { return x <= rx1; },
      [&](double x0, double y0, double x1, double y1, double &ix, double &iy) {
        const double t = (rx1 - x0) / (x1 - x0);
        ix = rx1;
        iy = y0 + t * (y1 - y0);
      },
      cx, cy);
  std::copy(cx, cx + n, bx);
  std::copy(cy, cy + n, by);

  n = clip_one_plane(
      bx, by, n, [&](double /*x*/, double y) { return y >= ry0; },
      [&](double x0, double y0, double x1, double y1, double &ix, double &iy) {
        const double t = (ry0 - y0) / (y1 - y0);
        ix = x0 + t * (x1 - x0);
        iy = ry0;
      },
      cx, cy);
  std::copy(cx, cx + n, bx);
  std::copy(cy, cy + n, by);

  n = clip_one_plane(
      bx, by, n, [&](double /*x*/, double y) { return y <= ry1; },
      [&](double x0, double y0, double x1, double y1, double &ix, double &iy) {
        const double t = (ry1 - y0) / (y1 - y0);
        ix = x0 + t * (x1 - x0);
        iy = ry1;
      },
      cx, cy);

  return shoelace_area(cx, cy, n);
}

// `Leaf` is declared in the header (the plan-11.14 geometry cache needs it).

namespace {

// Maps a native-canvas point to internal-canvas coordinates.
inline void to_internal(double nx, double ny, int internal_scale, double &ix,
                        double &iy) {
  ix = nx * static_cast<double>(internal_scale);
  iy = ny * static_cast<double>(internal_scale);
}

// Affine forward map (native canvas), using the frame's checked
// source_to_canvas (plan 7.2). Returns false if the affine inverse was never
// established for this frame.
bool affine_forward(const FrameSamplingTransform &f, double sx, double sy,
                    double &qx, double &qy) {
  if (!f.source_to_canvas_affine_valid)
    return false;
  const auto &s2c = f.source_to_canvas;
  qx = static_cast<double>(s2c(0, 0)) * sx +
       static_cast<double>(s2c(0, 1)) * sy + static_cast<double>(s2c(0, 2));
  qy = static_cast<double>(s2c(1, 0)) * sx +
       static_cast<double>(s2c(1, 1)) * sy + static_cast<double>(s2c(1, 2));
  return std::isfinite(qx) && std::isfinite(qy);
}

bool local_forward(const FrameSamplingTransform &f, double sx, double sy,
                   int canvas_w_native, int canvas_h_native,
                   const LocalInversionParams &params, double &qx, double &qy) {
  namespace gs = geomstats;
  if (gs::registry().enabled)
    ++gs::registry().cur().local_forward_calls;
  float fqx = 0.0f, fqy = 0.0f;
  if (!registration::invert_local_source_to_canvas(
          f, static_cast<float>(sx), static_cast<float>(sy), canvas_w_native,
          canvas_h_native, params, fqx, fqy)) {
    return false;
  }
  qx = fqx;
  qy = fqy;
  return true;
}

// Builds the exact affine droplet leaf for one source sample: the mapped
// parallelogram of the square [sx-h,sx+h] x [sy-h,sy+h] (native source
// space), in internal-canvas coordinates.
bool build_affine_leaf(const FrameSamplingTransform &f, double sx, double sy,
                       double half, int internal_scale, Leaf &out) {
  const double corner_sx[4] = {sx - half, sx + half, sx + half, sx - half};
  const double corner_sy[4] = {sy - half, sy - half, sy + half, sy + half};
  for (int i = 0; i < 4; ++i) {
    double qx = 0.0, qy = 0.0;
    if (!affine_forward(f, corner_sx[i], corner_sy[i], qx, qy))
      return false;
    to_internal(qx, qy, internal_scale, out.x[i], out.y[i]);
  }
  return true;
}

// Every accepted leaf passes both tests, including at maximum depth. The
// extra midpoint grid is a convergence probe, not another accepted level.
bool subdivide_local(const FrameSamplingTransform &f, double x0, double y0,
                     double x1, double y1, int depth, int cw, int ch, int scale,
                     const ForwardDrizzleSubdivisionParams &p,
                     const LocalInversionParams &inv,
                     std::vector<Leaf> &leaves) {
  if (geomstats::registry().enabled)
    ++geomstats::registry().cur().subdivide_local_calls;
  double x[3][3], y[3][3];
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      double qx, qy;
      if (!local_forward(f, x0 + (x1 - x0) * i / 2, y0 + (y1 - y0) * j / 2, cw,
                         ch, inv, qx, qy))
        return false;
      to_internal(qx, qy, scale, x[j][i], y[j][i]);
    }
  Leaf parent{{x[0][0], x[0][2], x[2][2], x[2][0]},
              {y[0][0], y[0][2], y[2][2], y[2][0]}};
  double error = 0, child_area = 0;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      const double u = i / 2.0, v = j / 2.0;
      const double bx = (1 - u) * (1 - v) * parent.x[0] +
                        u * (1 - v) * parent.x[1] + u * v * parent.x[2] +
                        (1 - u) * v * parent.x[3];
      const double by = (1 - u) * (1 - v) * parent.y[0] +
                        u * (1 - v) * parent.y[1] + u * v * parent.y[2] +
                        (1 - u) * v * parent.y[3];
      error = std::max(error, std::hypot(x[j][i] - bx, y[j][i] - by));
    }
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      double cx[] = {x[j][i], x[j][i + 1], x[j + 1][i + 1], x[j + 1][i]};
      double cy[] = {y[j][i], y[j][i + 1], y[j + 1][i + 1], y[j + 1][i]};
      child_area += shoelace_area(cx, cy, 4);
    }
  const double area = shoelace_area(parent.x, parent.y, 4);
  if (area > 0 && error <= p.position_epsilon_internal_px &&
      std::abs(child_area - area) / area <= p.area_relative_epsilon) {
    leaves.push_back(parent);
    return true;
  }
  if (depth >= p.max_subdivision_depth)
    return false;
  const size_t before = leaves.size();
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      if (!subdivide_local(f, x0 + (x1 - x0) * i / 2, y0 + (y1 - y0) * j / 2,
                           x0 + (x1 - x0) * (i + 1) / 2,
                           y0 + (y1 - y0) * (j + 1) / 2, depth + 1, cw, ch,
                           scale, p, inv, leaves)) {
        leaves.resize(before);
        return false; // reject the entire source sample
      }
    }
  return true;
}

size_t available_memory_headroom() {
  size_t available = std::numeric_limits<size_t>::max();
#ifdef __linux__
  std::ifstream info("/proc/meminfo");
  std::string line;
  while (std::getline(info, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      std::istringstream stream(line.substr(13));
      size_t kb = 0;
      if (stream >> kb)
        available = kb * 1024;
      break;
    }
  }
  // cgroup v2 limit is often tighter than host MemAvailable. Unknown/unlimited
  // values leave the configured allocation budget in charge.
  std::ifstream max_file("/sys/fs/cgroup/memory.max"),
      current_file("/sys/fs/cgroup/memory.current");
  size_t maximum = 0, current = 0;
  if ((max_file >> maximum) && (current_file >> current))
    available = std::min(available, maximum > current ? maximum - current : 0);
#endif
  return available;
}

size_t checked_product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  return a * b;
}

} // namespace

bool sample_leaves(const RegistrationSamplingPlan &plan,
                   const FrameSamplingTransform &f, int sx, int sy, int scale,
                   float pixfrac, const ForwardDrizzleSubdivisionParams &p,
                   std::vector<Leaf> &leaves) {
  namespace gs = geomstats;
  const bool instrument = gs::registry().enabled;
  if (instrument)
    ++gs::registry().cur().top_level_sample_leaves_calls;
  leaves.clear();
  const double x = sx + 0.5, y = sy + 0.5, h = pixfrac / 2.0;
  bool ok;
  if (f.has_smooth_local_model) {
    ok = subdivide_local(f, x - h, y - h, x + h, y + h, 0,
                         plan.canvas_width_native, plan.canvas_height_native,
                         scale, p, {}, leaves);
  } else {
    Leaf leaf;
    ok = build_affine_leaf(f, x, y, h, scale, leaf);
    if (ok)
      leaves.push_back(leaf);
  }
  if (instrument) {
    if (ok)
      gs::registry().cur().leaves_generated += leaves.size();
    else
      ++gs::registry().cur().sample_leaves_discarded;
  }
  return ok;
}

DrizzleMemoryPlan
plan_drizzle_memory(const RegistrationSamplingPlan &plan,
                    const config::ReconstructionDrizzleConfig &cfg,
                    size_t bytes_per_pixel, size_t retained_bytes,
                    bool loads_source) {
  if (plan.source_width <= 0 || plan.source_height <= 0 ||
      plan.canvas_width_native <= 0 || plan.canvas_height_native <= 0 ||
      cfg.internal_scale < 1 || cfg.internal_scale > 2 ||
      !std::isfinite(cfg.pixfrac) || cfg.pixfrac <= 0 || cfg.pixfrac > 1 ||
      cfg.kernel != "square" || cfg.chunk_rows < 0 ||
      cfg.chunk_halo_rows < -1 ||
      plan.canvas_width_native >
          std::numeric_limits<int>::max() / cfg.internal_scale ||
      plan.canvas_height_native >
          std::numeric_limits<int>::max() / cfg.internal_scale)
    throw std::invalid_argument("DRIZZLE_INVALID_GEOMETRY");
  if (plan.color_mode != ColorMode::MONO && plan.color_mode != ColorMode::OSC)
    throw std::invalid_argument("DRIZZLE_UNSUPPORTED_COLOR_MODE");
  if (plan.color_mode == ColorMode::OSC &&
      plan.bayer_pattern == BayerPattern::UNKNOWN)
    throw std::invalid_argument("DRIZZLE_UNKNOWN_CFA");
  DrizzleMemoryPlan m;
  m.width = plan.canvas_width_native * cfg.internal_scale;
  m.height = plan.canvas_height_native * cfg.internal_scale;
  m.budget_bytes = checked_product(
      cfg.memory_budget_mb ? cfg.memory_budget_mb : 512, 1024 * 1024);
  const size_t available = available_memory_headroom();
  if (available != std::numeric_limits<size_t>::max())
    m.budget_bytes = std::min(m.budget_bytes, available - available / 5);
  size_t fixed =
      checked_product(plan.frames.size(), sizeof(FrameSamplingTransform) + 128);
  const size_t source =
      loads_source ? checked_product(
                         checked_product(plan.source_width, plan.source_height),
                         sizeof(float) * 2)
                   : 0;
  // Include one source plus a transient load/normalization copy and 1 MiB for
  // bounded geometry scratch, callbacks and allocator overhead.
  if (retained_bytes > m.budget_bytes ||
      source > m.budget_bytes - retained_bytes ||
      fixed > m.budget_bytes - retained_bytes - source ||
      m.budget_bytes - retained_bytes - source - fixed < 1024 * 1024)
    throw std::runtime_error(
        "DRIZZLE_MEMORY_BUDGET: retained/source buffers exceed budget");
  fixed += retained_bytes + source + 1024 * 1024;
  const size_t per_row = checked_product(m.width, bytes_per_pixel);
  const size_t max_rows = per_row ? (m.budget_bytes - fixed) / per_row : 0;
  if (!max_rows)
    throw std::runtime_error(
        "DRIZZLE_MEMORY_BUDGET: one stripe row does not fit");
  m.rows = static_cast<int>(std::min<size_t>(m.height, max_rows));
  if (cfg.chunk_rows) {
    const int requested = std::min(cfg.chunk_rows, m.height);
    if (static_cast<size_t>(requested) > max_rows)
      throw std::runtime_error(
          "DRIZZLE_MEMORY_BUDGET: explicit chunk_rows exceeds budget");
    m.rows = requested;
  } else
    m.rows = std::min(m.rows, 256);
  m.estimated_peak_bytes = fixed + per_row * m.rows;
  return m;
}

PreparedDrizzleFrames
prepare_drizzle_frames(const RegistrationSamplingPlan &plan,
                       const config::ReconstructionDrizzleConfig &cfg,
                       const ForwardDrizzleSubdivisionParams &p) {
  if (p.max_subdivision_depth < 0 || p.max_subdivision_depth > 2 ||
      !(p.position_epsilon_internal_px > 0) ||
      !(p.area_relative_epsilon >= 0) ||
      !(p.per_frame_inversion_error_rate_max >= 0 &&
        p.per_frame_inversion_error_rate_max <= 1))
    throw std::invalid_argument("DRIZZLE_INVALID_SUBDIVISION");
  PreparedDrizzleFrames result;
  std::set<size_t> ids;
  std::set<std::string> names;
  std::vector<Leaf> leaves;
  leaves.reserve(16);
  for (const auto &f : plan.frames) {
    if (!ids.insert(f.source_index).second || !names.insert(f.frame_id).second)
      throw std::invalid_argument("DRIZZLE_DUPLICATE_FRAME_ID");
    if (!f.valid)
      continue;
    if (!f.source_to_canvas_affine_valid || !f.source_to_canvas.allFinite())
      throw std::invalid_argument("DRIZZLE_INVALID_TRANSFORM");
    if (f.has_smooth_local_model) {
      // Plan 11.14 P1: a published geometry cache has already run the full
      // per-sample sweep for this frame and finalised its exclusion rate.
      // Consume that instead of re-running sample_leaves (§11.14.3: exclusion
      // counting folds into the build; chunk height does not change the set).
      if (const auto *cache = active_geometry_cache()) {
        const auto st = cache->frame_stats(cfg.pixfrac, f.source_index);
        if (st.present) {
          result.diagnostics.local_model_samples_total +=
              static_cast<long long>(st.samples_total);
          result.diagnostics.local_model_samples_discarded +=
              static_cast<long long>(st.samples_discarded);
          if (st.excluded) {
            result.diagnostics.frames_excluded_subdivision_error_rate
                .emplace_back(f.frame_id, st.subdivision_error_rate);
            continue;
          }
          result.frames.push_back(&f);
          continue;
        }
      }
      geomstats::ScopedVariant _v(geomstats::Variant::kPrepareExclusionScan,
                                  cfg.pixfrac);
      geomstats::ScopedGeometryTimer _t;
      if (geomstats::registry().enabled) {
        ++geomstats::registry().cur().enumerate_calls;
        geomstats::registry().cur().source_rows_scanned += plan.source_height;
        geomstats::registry().cur().source_samples_visited +=
            static_cast<std::uint64_t>(plan.source_width) * plan.source_height;
      }
      long long total = static_cast<long long>(plan.source_width) *
                        plan.source_height,
                discarded = 0;
      for (int y = 0; y < plan.source_height; ++y)
        for (int x = 0; x < plan.source_width; ++x)
          if (!sample_leaves(plan, f, x, y, cfg.internal_scale, cfg.pixfrac, p,
                             leaves))
            ++discarded;
      result.diagnostics.local_model_samples_total += total;
      result.diagnostics.local_model_samples_discarded += discarded;
      const double rate = static_cast<double>(discarded) / total;
      if (rate > p.per_frame_inversion_error_rate_max) {
        result.diagnostics.frames_excluded_subdivision_error_rate.emplace_back(
            f.frame_id, rate);
        continue;
      }
    }
    result.frames.push_back(&f);
  }
  std::sort(result.frames.begin(), result.frames.end(),
            [](auto a, auto b) { return a->source_index < b->source_index; });
  return result;
}

void enumerate_drizzle_stripe_leaf_cells(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &f,
    int scale, float pixfrac, int y_begin, int rows,
    const DrizzleLeafCellSink &sink, const ForwardDrizzleSubdivisionParams &p) {
  namespace gs = geomstats;
  const bool instrument = gs::registry().enabled;

  // Plan 11.14 P1/P2: for a LOCAL-WARP frame that a published geometry cache
  // holds, replay the pre-built leaves for this stripe instead of re-running
  // sample_leaves over the whole source. Bit-identical to the scan below
  // (same corners, same bbox clamp, same canonical order); adds zero local
  // basis evaluations. Affine frames and cache misses fall through unchanged.
  if (f.has_smooth_local_model) {
    const auto *cache = active_geometry_cache();
    if (cache && cache->has_frame(pixfrac, f.source_index)) {
      if (instrument) ++gs::registry().cur().enumerate_calls;
      cache->enumerate_stripe(pixfrac, f.source_index, scale, y_begin, rows,
                              sink);
      return;
    }
  }

  const int W = plan.canvas_width_native * scale;
  int source_y0 = 0, source_y1 = plan.source_height;
  int source_x0 = 0, source_x1 = plan.source_width;
  if (!f.has_smooth_local_model) {
    // Inverse-map the destination-stripe rectangle back into the source, in
    // BOTH axes (plus a one-source-pixel droplet margin). A source pixel
    // outside this box maps entirely outside the stripe, so sample_leaves
    // there would only ever produce leaves whose y/x bbox clamps empty and the
    // cell sink is never called --- skipping it is bit-identical. This is the
    // same argument the Y bound already relied on, now extended to X so a frame
    // that covers only part of the canvas width (rotation, shear, a smaller
    // frame) does not rescan full source rows.
    WarpMatrix inverse;
    if (!registration::invert_affine_2x3(f.source_to_canvas, 1e-12f, 1e12f,
                                         inverse))
      throw std::invalid_argument("DRIZZLE_SINGULAR_TRANSFORM");
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    double xlo = lo, xhi = hi;
    for (double x : {0.0, static_cast<double>(W) / scale})
      for (double y : {static_cast<double>(y_begin) / scale,
                       static_cast<double>(y_begin + rows) / scale}) {
        const double sx = inverse(0, 0) * x + inverse(0, 1) * y + inverse(0, 2);
        const double sy = inverse(1, 0) * x + inverse(1, 1) * y + inverse(1, 2);
        lo = std::min(lo, sy);
        hi = std::max(hi, sy);
        xlo = std::min(xlo, sx);
        xhi = std::max(xhi, sx);
      }
    source_y0 = static_cast<int>(std::clamp(
        std::floor(lo - 1), 0.0, static_cast<double>(plan.source_height)));
    source_y1 = static_cast<int>(std::clamp(
        std::ceil(hi + 1), 0.0, static_cast<double>(plan.source_height)));
    source_x0 = static_cast<int>(std::clamp(
        std::floor(xlo - 1), 0.0, static_cast<double>(plan.source_width)));
    source_x1 = static_cast<int>(std::clamp(
        std::ceil(xhi + 1), 0.0, static_cast<double>(plan.source_width)));
  }
  if (instrument) {
    auto &c = gs::registry().cur();
    ++c.enumerate_calls;
    c.source_rows_scanned +=
        static_cast<std::uint64_t>(std::max(0, source_y1 - source_y0));
    c.source_samples_visited +=
        static_cast<std::uint64_t>(std::max(0, source_y1 - source_y0)) *
        static_cast<std::uint64_t>(std::max(0, source_x1 - source_x0));
  }
  std::vector<Leaf> leaves;
  leaves.reserve(16);
  for (int sy = source_y0; sy < source_y1; ++sy)
    for (int sx = source_x0; sx < source_x1; ++sx) {
      if (!sample_leaves(plan, f, sx, sy, scale, pixfrac, p, leaves))
        continue;
      int c = 0;
      if (plan.color_mode == ColorMode::OSC) {
        const auto channel = cfa_channel_for_source_pixel(
            sx, sy, plan.bayer_pattern, plan.cfa_origin_x, plan.cfa_origin_y);
        c = channel == CfaChannel::R ? 0 : channel == CfaChannel::G ? 1 : 2;
      }
      for (size_t li = 0; li < leaves.size(); ++li) {
        const auto &leaf = leaves[li];
        double xmin = *std::min_element(leaf.x, leaf.x + 4),
               xmax = *std::max_element(leaf.x, leaf.x + 4);
        double ymin = *std::min_element(leaf.y, leaf.y + 4),
               ymax = *std::max_element(leaf.y, leaf.y + 4);
        int x0 = static_cast<int>(
            std::clamp(std::floor(xmin), 0.0, static_cast<double>(W)));
        int x1 = static_cast<int>(
            std::clamp(std::ceil(xmax), 0.0, static_cast<double>(W)));
        int y0 = static_cast<int>(
            std::clamp(std::floor(ymin), static_cast<double>(y_begin),
                       static_cast<double>(y_begin + rows)));
        int y1 = static_cast<int>(
            std::clamp(std::ceil(ymax), static_cast<double>(y_begin),
                       static_cast<double>(y_begin + rows)));
        for (int y = y0; y < y1; ++y)
          for (int x = x0; x < x1; ++x) {
            if (instrument)
              ++gs::registry().cur().leaf_cells_emitted;
            sink(sx, sy, c, static_cast<int>(li), x, y, leaf.x, leaf.y);
          }
      }
    }
}

void rasterize_drizzle_stripe(const RegistrationSamplingPlan &plan,
                              const FrameSamplingTransform &f, int scale,
                              float pixfrac, int y_begin, int rows,
                              const DrizzleAreaSink &sink,
                              const ForwardDrizzleSubdivisionParams &p) {
  const int W = plan.canvas_width_native * scale;
  enumerate_drizzle_stripe_leaf_cells(
      plan, f, scale, pixfrac, y_begin, rows,
      [&](int sx, int sy, int c, int leaf_order, int x, int y,
          const double *lx, const double *ly) {
        const double k = polygon_rectangle_intersection_area(lx, ly, x, y,
                                                             x + 1.0, y + 1.0);
        if (k > 0)
          sink(sx, sy, c, leaf_order,
               static_cast<size_t>(y - y_begin) * W + x, k);
      },
      p);
}

namespace {

// x-bbox of a convex quad intersected with the horizontal slab [ylo, yhi].
// Returns false when the intersection is empty. Used only to bound which
// internal cells of a stripe row need explicit classification against the
// dense footprint parallelogram --- the classification itself is exact.
bool convex_quad_slab_x_range(const double *px, const double *py, double ylo,
                              double yhi, double &xlo, double &xhi) {
  xlo = std::numeric_limits<double>::infinity();
  xhi = -xlo;
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3;
    const double ax = px[i], ay = py[i], bx = px[j], by = py[j];
    if (ay >= ylo && ay <= yhi) {
      xlo = std::min(xlo, ax);
      xhi = std::max(xhi, ax);
    }
    for (double yc : {ylo, yhi}) {
      if ((ay < yc) != (by < yc)) {
        const double t = (yc - ay) / (by - ay);
        const double xc = ax + t * (bx - ax);
        xlo = std::min(xlo, xc);
        xhi = std::max(xhi, xc);
      }
    }
  }
  return xlo <= xhi;
}

} // namespace

void dense_footprint_touched_stripe(const RegistrationSamplingPlan &plan,
                                    const FrameSamplingTransform &f, int scale,
                                    int y_begin, int rows,
                                    std::vector<std::uint8_t> &touched,
                                    bool force_exact) {
  const int W = plan.canvas_width_native * scale;
  const size_t n = static_cast<size_t>(W) * static_cast<size_t>(std::max(0, rows));
  touched.assign(n, 0);
  if (rows <= 0 || W <= 0)
    return;

  const ForwardDrizzleSubdivisionParams p{};

  // The exact reference path, kept verbatim: it is the definition this routine
  // must reproduce byte-for-byte.
  auto exact_reference = [&] {
    rasterize_drizzle_stripe(
        plan, f, scale, 1.0f, y_begin, rows,
        [&](int, int, int, int, size_t i, double) { touched[i] = 1; }, p);
  };

  if (force_exact || f.has_smooth_local_model ||
      !f.source_to_canvas_affine_valid) {
    exact_reference();
    return;
  }

  // P = affine_f([0, W_src] x [0, H_src]) in internal-canvas coordinates: the
  // union of every unshrunk source-pixel square's image.
  const double src_x[4] = {0.0, static_cast<double>(plan.source_width),
                           static_cast<double>(plan.source_width), 0.0};
  const double src_y[4] = {0.0, 0.0, static_cast<double>(plan.source_height),
                           static_cast<double>(plan.source_height)};
  double Px[4], Py[4];
  for (int k = 0; k < 4; ++k) {
    double qx = 0.0, qy = 0.0;
    if (!affine_forward(f, src_x[k], src_y[k], qx, qy)) {
      exact_reference();
      return;
    }
    to_internal(qx, qy, scale, Px[k], Py[k]);
  }

  WarpMatrix inverse;
  if (!registration::invert_affine_2x3(f.source_to_canvas, 1e-12f, 1e12f,
                                       inverse)) {
    exact_reference();  // matches enumerate_drizzle_stripe_leaf_cells' throw
    return;             // path being taken by the reference rasterize
  }

  // Inward unit normals + a support vertex for each of P's four edges. Signed
  // distance dot(n_e, q - v_e) is >= 0 for points inside that edge's half-plane.
  const double cx = 0.25 * (Px[0] + Px[1] + Px[2] + Px[3]);
  const double cy = 0.25 * (Py[0] + Py[1] + Py[2] + Py[3]);
  double nx[4], ny[4];
  for (int e = 0; e < 4; ++e) {
    const int j = (e + 1) & 3;
    double ex = Px[j] - Px[e], ey = Py[j] - Py[e];
    double cand_x = -ey, cand_y = ex;
    const double len = std::hypot(cand_x, cand_y);
    if (!(len > 0.0)) {  // degenerate edge -> fall back wholesale
      exact_reference();
      return;
    }
    cand_x /= len;
    cand_y /= len;
    if (cand_x * (cx - Px[e]) + cand_y * (cy - Py[e]) < 0.0) {
      cand_x = -cand_x;
      cand_y = -cand_y;
    }
    nx[e] = cand_x;
    ny[e] = cand_y;
  }
  auto signed_dist = [&](int e, double x, double y) {
    return nx[e] * (x - Px[e]) + ny[e] * (y - Py[e]);
  };

  // Interior margin + soundness of the interior fast-fill.
  //
  // `ext` bounds one mapped source-pixel square's internal-canvas extent along
  // either axis: ext = scale * max(|a|+|b|, |c|+|d|). A cell whose four corners
  // are all >= `margin = 2*ext + 3` inside every edge of P is:
  //   (1) fully tiled by mapped source-pixel squares --- every point of the
  //       cell is farther from dP than one square's diameter, and the squares
  //       tile P (affine image of a partition), so the cell lies in the union;
  //   (2) touched by the reference: the unit cell's area (= 1) is partitioned
  //       among the tiles that meet it; at most (1+ext)^2 tiles can meet a unit
  //       cell (their bounding boxes are ext x ext), so some tile contributes
  //       >= 1/(1+ext)^2 of the cell. With ext <= 64 that share is >= ~2.4e-4,
  //       far above any double-precision rounding in
  //       polygon_rectangle_intersection_area --- it returns > 0, and the
  //       reference sets touched = 1.
  // This holds for any non-singular linear part (invert_affine_2x3 already
  // rejected singular ones above); it does not need a near-unit determinant.
  // ext > 64 (a large scale-up / extreme shear) is bounced to the exact path
  // so the 1/(1+ext)^2 argument always applies.
  const auto &s2c = f.source_to_canvas;
  const double lin_x = std::fabs(static_cast<double>(s2c(0, 0))) +
                       std::fabs(static_cast<double>(s2c(0, 1)));
  const double lin_y = std::fabs(static_cast<double>(s2c(1, 0))) +
                       std::fabs(static_cast<double>(s2c(1, 1)));
  const double ext = static_cast<double>(scale) * std::max(lin_x, lin_y);
  if (!(ext > 0.0) || ext > 64.0) {
    exact_reference();
    return;
  }
  const double margin = 2.0 * ext + 3.0;

  std::vector<Leaf> leaves;
  leaves.reserve(4);

  // Exact per-cell fallback for a boundary cell: replay the reference test
  // (sample_leaves + polygon_rectangle_intersection_area > 0) over exactly the
  // source pixels whose unshrunk square can reach this cell. The inverse-mapped
  // cell rectangle plus a one-source-pixel margin is the same superset rule
  // enumerate_drizzle_stripe_leaf_cells already relies on for its stripe bound.
  auto boundary_cell_touched = [&](int gx, int gy) -> bool {
    double slo_x = std::numeric_limits<double>::infinity(), shi_x = -slo_x;
    double slo_y = slo_x, shi_y = shi_x;
    for (double dx : {static_cast<double>(gx) - 1.0,
                      static_cast<double>(gx) + 2.0})
      for (double dy : {static_cast<double>(gy) - 1.0,
                        static_cast<double>(gy) + 2.0}) {
        const double xn = dx / scale, yn = dy / scale;
        const double sx = inverse(0, 0) * xn + inverse(0, 1) * yn + inverse(0, 2);
        const double sy = inverse(1, 0) * xn + inverse(1, 1) * yn + inverse(1, 2);
        slo_x = std::min(slo_x, sx);
        shi_x = std::max(shi_x, sx);
        slo_y = std::min(slo_y, sy);
        shi_y = std::max(shi_y, sy);
      }
    const int sx0 = static_cast<int>(std::clamp(
        std::floor(slo_x - 1.0), 0.0, static_cast<double>(plan.source_width)));
    const int sx1 = static_cast<int>(std::clamp(
        std::ceil(shi_x + 1.0), 0.0, static_cast<double>(plan.source_width)));
    const int sy0 = static_cast<int>(std::clamp(
        std::floor(slo_y - 1.0), 0.0, static_cast<double>(plan.source_height)));
    const int sy1 = static_cast<int>(std::clamp(
        std::ceil(shi_y + 1.0), 0.0, static_cast<double>(plan.source_height)));
    for (int sy = sy0; sy < sy1; ++sy)
      for (int sx = sx0; sx < sx1; ++sx) {
        if (!sample_leaves(plan, f, sx, sy, scale, 1.0f, p, leaves))
          continue;
        for (const auto &L : leaves)
          if (polygon_rectangle_intersection_area(L.x, L.y, gx, gy, gx + 1.0,
                                                  gy + 1.0) > 0)
            return true;
      }
    return false;
  };

  for (int gy = y_begin; gy < y_begin + rows; ++gy) {
    double xr_lo, xr_hi;
    if (!convex_quad_slab_x_range(Px, Py, static_cast<double>(gy),
                                  static_cast<double>(gy) + 1.0, xr_lo, xr_hi))
      continue;  // P does not reach this stripe row at all
    int gx0 = static_cast<int>(
        std::clamp(std::floor(xr_lo), 0.0, static_cast<double>(W)));
    int gx1 = static_cast<int>(
        std::clamp(std::ceil(xr_hi), 0.0, static_cast<double>(W)));
    const size_t row_off = static_cast<size_t>(gy - y_begin) * W;
    for (int gx = gx0; gx < gx1; ++gx) {
      const double corner_x[4] = {static_cast<double>(gx),
                                  static_cast<double>(gx) + 1.0,
                                  static_cast<double>(gx) + 1.0,
                                  static_cast<double>(gx)};
      const double corner_y[4] = {static_cast<double>(gy),
                                  static_cast<double>(gy),
                                  static_cast<double>(gy) + 1.0,
                                  static_cast<double>(gy) + 1.0};
      bool interior = true;
      bool exterior = false;
      for (int e = 0; e < 4; ++e) {
        double dmin = std::numeric_limits<double>::infinity();
        double dmax = -dmin;
        for (int c = 0; c < 4; ++c) {
          const double d = signed_dist(e, corner_x[c], corner_y[c]);
          dmin = std::min(dmin, d);
          dmax = std::max(dmax, d);
        }
        if (dmin < margin)
          interior = false;
        if (dmax <= 0.0)
          exterior = true;
      }
      if (exterior)
        continue;
      if (interior || boundary_cell_touched(gx, gy))
        touched[row_off + static_cast<size_t>(gx)] = 1;
    }
  }
}

ForwardDrizzleDiagnostics stream_forward_drizzle_uniform(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const UniformStripeSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision, size_t retained_bytes) {
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  const auto memory = plan_drizzle_memory(
      plan, cfg, channels * (5 * sizeof(double) + 13), retained_bytes);
  auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  auto diag = prepared.diagnostics;
  diag.estimated_peak_bytes = memory.estimated_peak_bytes;
  diag.resolved_chunk_rows = memory.rows;
  {
    int local_n = 0;
    for (const auto *pf : prepared.frames)
      if (pf->has_smooth_local_model)
        ++local_n;
    geomstats::stamp_context(plan.source_width, plan.source_height,
                             plan.canvas_width_native,
                             plan.canvas_height_native, cfg.internal_scale,
                             memory.rows, memory.height,
                             static_cast<int>(prepared.frames.size()), local_n);
  }
  for (int y = 0; y < memory.height; y += memory.rows) {
    const int rows = std::min(memory.rows, memory.height - y);
    const size_t n = static_cast<size_t>(memory.width) * rows;
    std::array<std::vector<double>, 3> wx, w, w2, A, B;
    for (int c = 0; c < channels; ++c) {
      wx[c].assign(n, 0);
      w[c].assign(n, 0);
      w2[c].assign(n, 0);
      A[c].assign(n, 0);
      B[c].assign(n, 0);
    }
    for (const auto *f : prepared.frames) {
      const Matrix2Df &source = source_of(f->source_index);
      if (source.rows() != plan.source_height ||
          source.cols() != plan.source_width)
        throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");
      for (int c = 0; c < channels; ++c) {
        std::fill(A[c].begin(), A[c].end(), 0);
        std::fill(B[c].begin(), B[c].end(), 0);
      }
      {
        geomstats::ScopedVariant _v(
            geomstats::Variant::kUniformDiagnostic, cfg.pixfrac);
        geomstats::ScopedGeometryTimer _t;
        rasterize_drizzle_stripe(
            plan, *f, cfg.internal_scale, cfg.pixfrac, y, rows,
            [&](int sx, int sy, int c, int /*leaf*/, size_t i, double k) {
              const double v = source(sy, sx);
              if (std::isfinite(v)) {
                A[c][i] += k * v;
                B[c][i] += k;
              }
            },
            subdivision);
      }
      for (int c = 0; c < channels; ++c)
        for (size_t i = 0; i < n; ++i)
          if (B[c][i] > 0) {
            wx[c][i] += A[c][i];
            w[c][i] += B[c][i];
            w2[c][i] += B[c][i] * B[c][i];
          }
    }
    ForwardDrizzleUniformResult stripe;
    stripe.color_mode = plan.color_mode;
    stripe.internal_width = memory.width;
    stripe.internal_height = rows;
    std::array<ProfilePlane *, 3> planes =
        channels == 1
            ? std::array<ProfilePlane *, 3>{&stripe.L, nullptr, nullptr}
            : std::array<ProfilePlane *, 3>{&stripe.R, &stripe.G, &stripe.B};
    for (int c = 0; c < channels; ++c) {
      auto &plane = *planes[c];
      plane.allocate(memory.width, rows);
      for (size_t i = 0; i < n; ++i)
        if (w[c][i] > 0) {
          plane.value[i] = wx[c][i] / w[c][i];
          plane.weight_sum[i] = w[c][i];
          plane.n_eff[i] = w[c][i] * w[c][i] / w2[c][i];
          plane.support[i] = 1;
        }
    }
    sink(y, stripe);
  }
  return diag;
}

ForwardDrizzleUniformResult compute_forward_drizzle_uniform(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision) {
  // Budget the materialized output before allocating it. Production diagnostics
  // use stream_forward_drizzle_uniform and retain no full profile planes.
  const auto initial = plan_drizzle_memory(plan, cfg, 1);
  const size_t retained =
      checked_product(checked_product(initial.width, initial.height),
                      (plan.color_mode == ColorMode::MONO ? 1 : 3) * 13);
  const auto memory = plan_drizzle_memory(
      plan, cfg, (plan.color_mode == ColorMode::MONO ? 1 : 3) * 53, retained);
  ForwardDrizzleUniformResult result;
  result.color_mode = plan.color_mode;
  result.internal_width = memory.width;
  result.internal_height = memory.height;
  if (plan.color_mode == ColorMode::MONO)
    result.L.allocate(memory.width, memory.height);
  else {
    result.R.allocate(memory.width, memory.height);
    result.G.allocate(memory.width, memory.height);
    result.B.allocate(memory.width, memory.height);
  }
  result.diagnostics = stream_forward_drizzle_uniform(
      plan, source_of, cfg,
      [&](int y, const ForwardDrizzleUniformResult &stripe) {
        auto copy = [&](ProfilePlane &dst, const ProfilePlane &src) {
          if (src.empty())
            return;
          const size_t offset = static_cast<size_t>(y) * memory.width;
          std::copy(src.value.begin(), src.value.end(),
                    dst.value.begin() + offset);
          std::copy(src.weight_sum.begin(), src.weight_sum.end(),
                    dst.weight_sum.begin() + offset);
          std::copy(src.n_eff.begin(), src.n_eff.end(),
                    dst.n_eff.begin() + offset);
          std::copy(src.support.begin(), src.support.end(),
                    dst.support.begin() + offset);
        };
        copy(result.R, stripe.R);
        copy(result.G, stripe.G);
        copy(result.B, stripe.B);
        copy(result.L, stripe.L);
      },
      subdivision, retained);
  return result;
}

// M3 (plan section 11.8): shared robust clipping. See the header for the
// integration status note --- this is the reviewed 8-step algorithm, not yet
// wired into the streaming Uniform-Control computation.
void DrizzleClipScratch::reserve_for(std::size_t capacity, bool with_alpha) {
  bool grew = false;
  if (accepted.capacity() < capacity) { accepted.reserve(capacity); grew = true; }
  if (order.capacity() < capacity) { order.reserve(capacity); grew = true; }
  if (active.capacity() < capacity) { active.reserve(capacity); grew = true; }
  if (dev_order.capacity() < capacity) { dev_order.reserve(capacity); grew = true; }
  if (with_alpha && alpha_contribs.capacity() < capacity) {
    alpha_contribs.reserve(capacity);
    grew = true;
  }
  if (grew) ++growth_count;
}

namespace {

// Core of apply_robust_clipping (plan 0.2 priority 2, §30.79) on caller-
// provided reusable scratch instead of per-call heap allocations. Steps,
// value/deviation orders, tie-breaks, summation order, weighted median/MAD
// and the abort condition are IDENTICAL to the reviewed 8-step procedure;
// only the buffer ownership changed. Returns `pixel_rejected`; the used
// `accepted` prefix (size == candidates.size()) lives in the scratch.
bool robust_clip_core(std::span<const ClipCandidate> candidates,
                      int min_clip_contributors, int robust_passes,
                      float clip_sigma_low, float clip_sigma_high,
                      float min_fraction, float min_n_eff,
                      DrizzleClipScratch &scratch) {
  if (min_clip_contributors < 1 || robust_passes < 0 ||
      !std::isfinite(clip_sigma_low) || clip_sigma_low < 0 ||
      !std::isfinite(clip_sigma_high) || clip_sigma_high < 0 ||
      !std::isfinite(min_fraction) || min_fraction < 0 || min_fraction > 1 ||
      !std::isfinite(min_n_eff) || min_n_eff < 0)
    throw std::invalid_argument("DRIZZLE_INVALID_CLIPPING_CONFIG");
  for (const auto &c : candidates)
    if (!std::isfinite(c.x) || !std::isfinite(c.b) || c.b <= 0)
      throw std::invalid_argument("DRIZZLE_INVALID_CLIP_CANDIDATE");
  const size_t n = candidates.size();
  auto &accepted = scratch.accepted;
  accepted.assign(n, std::uint8_t{1});
  if (n == 0) {
    return true;
  }

  // Step 2: below min_clip_contributors, skip straight to step 8 with every
  // candidate still valid (protects thin R/B channels at low frame counts).
  if (n >= static_cast<size_t>(min_clip_contributors)) {
    // Step 3: one fixed, deterministic value order for the whole procedure.
    auto &order = scratch.order;
    order.resize(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if (candidates[a].x != candidates[b].x) return candidates[a].x < candidates[b].x;
      return candidates[a].frame_index < candidates[b].frame_index;
    });

    for (int pass = 0; pass < robust_passes; ++pass) {
      auto &active = scratch.active;
      active.clear();
      for (size_t idx : order)
        if (accepted[idx]) active.push_back(idx);
      if (active.empty()) break;

      double total_w = 0.0;
      for (size_t idx : active) total_w += candidates[idx].b;

      // Step 4: weighted median (active is already value-sorted, since it is
      // a stable subsequence of `order`).
      double median = candidates[active.back()].x;
      if (total_w > 0.0) {
        double cum = 0.0;
        for (size_t idx : active) {
          cum += candidates[idx].b;
          if (cum >= total_w / 2.0) {
            median = candidates[idx].x;
            break;
          }
        }
      }

      // Weighted MAD: same weighted-median construction, over |x - median|,
      // with the same deterministic tie-break re-applied for the new order.
      auto &dev_order = scratch.dev_order;
      dev_order = active;
      std::sort(dev_order.begin(), dev_order.end(), [&](size_t a, size_t b) {
        const double da = std::abs(candidates[a].x - median);
        const double db = std::abs(candidates[b].x - median);
        if (da != db) return da < db;
        return candidates[a].frame_index < candidates[b].frame_index;
      });
      double mad = std::abs(candidates[dev_order.back()].x - median);
      if (total_w > 0.0) {
        double cum = 0.0;
        for (size_t idx : dev_order) {
          cum += candidates[idx].b;
          if (cum >= total_w / 2.0) {
            mad = std::abs(candidates[idx].x - median);
            break;
          }
        }
      }

      // Step 5/6: asymmetric bounds; degenerate MAD == 0 is used literally
      // (no invented epsilon) --- identical values all equal the median and
      // therefore stay inside [median, median], per the plan's explicit
      // "kein willkürliches epsilonbasiertes Wegclippen" guard.
      const double lower = median - static_cast<double>(clip_sigma_low) * mad;
      const double upper = median + static_cast<double>(clip_sigma_high) * mad;
      bool changed = false;
      for (size_t idx : active) {
        const double x = candidates[idx].x;
        if (!(x >= lower && x <= upper)) {
          accepted[idx] = 0;
          changed = true;
        }
      }
      // Step 7: stop once the mask stops changing.
      if (!changed) break;
    }
  }

  // Step 8: min_fraction / min_n_eff against the geometrically possible
  // frame support (== n, since candidates are only constructed for
  // B_f,c(q) > 0, plan 11.8's exact denominator).
  size_t accepted_count = 0;
  double sum_w = 0.0, sum_w2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (!accepted[i]) continue;
    ++accepted_count;
    sum_w += candidates[i].b;
    sum_w2 += candidates[i].b * candidates[i].b;
  }
  const double fraction = static_cast<double>(accepted_count) / static_cast<double>(n);
  const double n_eff = sum_w2 > 0.0 ? (sum_w * sum_w) / sum_w2 : 0.0;
  return fraction < static_cast<double>(min_fraction) ||
         n_eff < static_cast<double>(min_n_eff);
}

} // namespace

// M3 (plan section 11.8): shared robust clipping. Thin wrapper over
// robust_clip_core with a per-call scratch: identical results, kept for the
// standalone callers (config validation, unit tests). Hot paths call
// reduce_pixel_profiles with a reused scratch instead.
ClipResult apply_robust_clipping(std::span<const ClipCandidate> candidates,
                                 int min_clip_contributors, int robust_passes,
                                 float clip_sigma_low, float clip_sigma_high,
                                 float min_fraction, float min_n_eff) {
  DrizzleClipScratch scratch;
  ClipResult result;
  result.pixel_rejected =
      robust_clip_core(candidates, min_clip_contributors, robust_passes,
                       clip_sigma_low, clip_sigma_high, min_fraction,
                       min_n_eff, scratch);
  result.accepted.assign(scratch.accepted.begin(), scratch.accepted.end());
  return result;
}

// Plan 11.8 / 11.9 / 14.4: the per-(channel, target cell) clip + profile +
// alpha reduction, factored out so the streaming path below and the plan-19.6
// contribution-list path (forward_drizzle_contrib_list.cpp) run the IDENTICAL
// code and are therefore bit-identical. `candidates` must be frame-ordered.
void reduce_pixel_profiles(
    std::span<const ClipCandidate> pixel, const DrizzleProfileReduceConfig &cfg,
    const std::function<double(std::size_t)> &g_eff_for,
    const std::vector<std::pair<std::uint8_t, float>> &reg_by_source,
    std::size_t gi, ProfilePlane *uniform_c, ProfilePlane *raw_c,
    ProfilePlane *fine_c, ProfilePlane *medium_c, double *ac_sep, double *ac_art,
    double *ac_reg, ForwardDrizzleClippingDiagnostics &diag,
    DrizzleClipScratch *scratch) {
  if (pixel.empty()) return;
  ++diag.pixel_channel_evaluations;
  // Reusable caller scratch when provided (one reduce at a time per
  // instance); a per-call fallback otherwise. Identical results either way.
  DrizzleClipScratch fallback;
  DrizzleClipScratch &clip_scratch = scratch ? *scratch : fallback;
  clip_scratch.reserve_for(pixel.size(), cfg.emit_alpha);
  const auto &accepted = clip_scratch.accepted;
  const bool pixel_rejected =
      robust_clip_core(pixel, cfg.min_clip_contributors, cfg.robust_passes,
                       cfg.clip_sigma_low, cfg.clip_sigma_high,
                       cfg.min_fraction, cfg.min_n_eff, clip_scratch);
  for (std::uint8_t a : accepted)
    if (!a) ++diag.candidate_contributions_clipped;
  if (pixel_rejected) {
    ++diag.pixel_channel_rejected;
    return;
  }
  // Uniform: w = B. Raw/Fine/Medium: w = B * G_eff(f) * Q^e. The per-candidate
  // geometric K-averages q / q0 / q1 never entered the clip decision above.
  struct Accum { double wx = 0, w = 0, w2 = 0; };
  Accum au, ar, af, am;
  auto add = [](Accum &a, double w, double x) {
    a.wx += w * x; a.w += w; a.w2 += w * w;
  };
  for (size_t k = 0; k < pixel.size(); ++k) {
    if (!accepted[k]) continue;
    const auto &cd = pixel[k];
    const double g = g_eff_for(cd.frame_index);
    add(au, cd.b, cd.x);
    add(ar, cd.b * g * cd.q, cd.x);
    if (cfg.emit_fine)
      add(af, cd.b * g * std::pow(cd.q0, cfg.fine_quality_exponent), cd.x);
    if (cfg.emit_medium)
      add(am, cd.b * g * std::pow(cd.q1, cfg.medium_quality_exponent), cd.x);
  }
  auto write = [&](ProfilePlane *p, const Accum &a) {
    if (!p || a.w <= 0.0) return;
    p->value[gi] = static_cast<float>(a.wx / a.w);
    p->weight_sum[gi] = static_cast<float>(a.w);
    p->n_eff[gi] = static_cast<float>(a.w2 > 0.0 ? (a.w * a.w) / a.w2 : 0.0);
    p->support[gi] = 1;
  };
  write(uniform_c, au);
  write(raw_c, ar);
  if (cfg.emit_fine) write(fine_c, af);
  if (cfg.emit_medium) write(medium_c, am);

  if (cfg.emit_alpha && ac_sep && ac_art && ac_reg) {
    // Plan 14.4: A_separation / A_artifact / A_registration from the accepted
    // frame contributions for this channel; the frame result takes the
    // conservative min over active channels.
    auto &contribs = clip_scratch.alpha_contribs;
    contribs.clear();
    for (size_t k = 0; k < pixel.size(); ++k) {
      if (!accepted[k]) continue;
      const auto &cd = pixel[k];
      const auto &rg = reg_by_source[cd.frame_index];
      const double art_conf = cd.qa_has_data
                                  ? cd.qa
                                  : std::numeric_limits<double>::quiet_NaN();
      contribs.push_back({cd.b, cd.q, art_conf, rg.first != 0u,
                          static_cast<double>(rg.second)});
    }
    const auto fac =
        compute_alpha_confidence_channel(contribs, cfg.alpha_confidence);
    *ac_sep = std::min(*ac_sep, fac.a_separation);
    *ac_art = std::min(*ac_art, fac.a_artifact);
    *ac_reg = std::min(*ac_reg, fac.a_registration);
  }
}

// M3/M6 (plan 11.8/11.9): Uniform (clipped), Raw and --- when requested ---
// Fine and Medium computed together, all sharing one clipping decision per
// pixel/channel. G_eff is supplied per frame; Q_composite / Q_scale0 /
// Q_scale1 come from `quality_of` (each stream optional).
ForwardDrizzlePairDiagnostics stream_forward_drizzle_uniform_and_raw(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg,
    const UniformAndRawStripeSink &sink,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index, size_t retained_bytes,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb,
    int workers) {
  if ((mb.emit_fine || mb.emit_medium || mb.emit_alpha_confidence) && !quality_of)
    throw std::invalid_argument("DRIZZLE_MULTIBAND_REQUIRES_QUALITY_PROVIDER");
  const bool need_q0 = mb.emit_fine;
  const bool need_q1 = mb.emit_medium;
  const bool need_qa = mb.emit_alpha_confidence;  // artifact_confidence K-avg
  apply_robust_clipping({}, cfg.min_clip_contributors, cfg.robust_passes,
                        clip_cfg.clip_sigma_low, clip_cfg.clip_sigma_high,
                        clip_cfg.min_fraction, clip_cfg.min_n_eff);
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  size_t source_count = 0;
  for (const auto &f : plan.frames) {
    if (f.source_index == std::numeric_limits<size_t>::max())
      throw std::invalid_argument("DRIZZLE_INVALID_SOURCE_INDEX");
    source_count = std::max(source_count, f.source_index + 1);
  }
  // Raw applies Q_composite only if the provider actually supplies a composite
  // map for at least one frame (a caller that supplies only scale0/scale1
  // wants Fine/Medium weighted but Raw left as B*G_eff). This pre-scan only
  // null-checks the returned pointers --- it never retains or dereferences
  // one, so it does not clash with the "valid until the next call" contract.
  bool need_qc = false;
  if (quality_of)
    for (const auto &f : plan.frames)
      if (quality_of(f.source_index).composite) { need_qc = true; break; }
  if (need_qa) {
    if (!need_qc)
      throw std::invalid_argument(
          "DRIZZLE_ALPHA_CONFIDENCE_REQUIRES_COMPOSITE_MAP");
    bool any_artifact = false;
    for (const auto &f : plan.frames)
      if (quality_of(f.source_index).artifact) { any_artifact = true; break; }
    if (!any_artifact)
      throw std::invalid_argument(
          "DRIZZLE_ALPHA_CONFIDENCE_REQUIRES_ARTIFACT_MAP");
  }
  // Per-frame registration inputs for A_registration (plan 14.4): a frame is
  // "direct" when it was directly measured / astrometrically rescued (plan
  // 11.9) --- i.e. the provenance flag says so AND the folded weight factor is
  // still exactly 1.0. Requiring both keeps the check robust if the 11.9
  // factor derivation ever stops emitting a literal 1.0 for direct frames.
  std::vector<std::pair<uint8_t, float>> reg_by_source;
  if (need_qa) {
    reg_by_source.assign(source_count, {0u, 1.0f});
    for (const auto &f : plan.frames) {
      const bool direct =
          !f.model_predicted && f.model_prediction_factor == 1.0f;
      reg_by_source[f.source_index] = {direct ? uint8_t{1} : uint8_t{0},
                                       f.registration_residual_factor};
    }
  }
  if (!g_eff_by_source_index.empty() && g_eff_by_source_index.size() != source_count)
    throw std::invalid_argument("DRIZZLE_GEFF_SIZE_MISMATCH");
  if (!sink) throw std::invalid_argument("DRIZZLE_MISSING_SINK");
  for (const auto &f : plan.frames) {
    if (!g_eff_by_source_index.empty() &&
        (f.source_index >= g_eff_by_source_index.size() ||
         !std::isfinite(g_eff_by_source_index[f.source_index]) ||
         g_eff_by_source_index[f.source_index] < 0 ||
         g_eff_by_source_index[f.source_index] > 1))
      throw std::invalid_argument("DRIZZLE_INVALID_GEFF");
  }
  auto g_eff_for = [&](std::size_t source_index) -> double {
    if (g_eff_by_source_index.empty()) return 1.0;
    return static_cast<double>(g_eff_by_source_index[source_index]);
  };
  const DrizzleProfileReduceConfig reduce_cfg{
      cfg.min_clip_contributors,   cfg.robust_passes,
      clip_cfg.clip_sigma_low,     clip_cfg.clip_sigma_high,
      clip_cfg.min_fraction,       clip_cfg.min_n_eff,
      mb.emit_fine,                mb.emit_medium,
      need_qa,                     mb.fine_quality_exponent,
      mb.medium_quality_exponent,  mb.alpha_confidence};
  // Worst case: every frame contributes at every pixel. Use flat, exactly
  // sized storage; no vector growth or per-pixel heap allocations.
  const size_t quality_bytes = checked_product(g_eff_by_source_index.size(), sizeof(float));
  if (retained_bytes > std::numeric_limits<size_t>::max() - quality_bytes)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  retained_bytes += quality_bytes;
  {
    // Up to four source-sized float buffers (composite / scale0 / scale1 /
    // artifact Q-maps) can be held concurrently with the decoded source frame.
    const int qmap_streams = (need_qc ? 1 : 0) + (need_q0 ? 1 : 0) +
                             (need_q1 ? 1 : 0) + (need_qa ? 1 : 0);
    const size_t qmap_bytes = checked_product(
        checked_product(plan.source_width, plan.source_height),
        static_cast<size_t>(qmap_streams) * sizeof(float));
    if (retained_bytes > std::numeric_limits<size_t>::max() - qmap_bytes)
      throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
    retained_bytes += qmap_bytes;
  }
  const size_t frame_count = plan.frames.size();
  const auto *geometry_reader = active_geometry_cache();
  // Concurrent per-pixel clipping and alpha statistics own frame-sized vectors.
  // Reader resize may briefly retain its old block as well as the new block.
  const size_t reader_scratch = geometry_reader
      ? checked_product(geometry_reader->max_row_record_count(), 2 * 72) : 0;
  // Priority 2 (§30.79): exact per-worker clipping scratch — the reusable
  // DrizzleClipScratch buffers sized to the maximum candidate span (==
  // frame_count), including alpha contributions only when emitted. This
  // REPLACES the previous blanket statistic estimate; nothing is charged to
  // retained_bytes on top.
  const size_t clip_scratch_bytes = checked_product(
      frame_count, sizeof(std::uint8_t) + 3 * sizeof(size_t) +
                       (need_qa ? sizeof(AlphaFactorContribution) : 0));
  if (clip_scratch_bytes > std::numeric_limits<size_t>::max() - 262144 ||
      reader_scratch >
          std::numeric_limits<size_t>::max() - clip_scratch_bytes - 262144)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: worker scratch overflow");
  const size_t worker_scratch = reader_scratch + clip_scratch_bytes + 262144;
  if (geometry_reader) {
    const auto resident = geometry_reader->resident_bytes();
    if (resident > std::numeric_limits<size_t>::max() - retained_bytes)
      throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: geometry index overflow");
    retained_bytes += resident;
  }
  int req_workers = std::max(1, std::min(workers,
      plan.canvas_height_native * cfg.internal_scale));
  size_t scratch = checked_product(static_cast<size_t>(req_workers), worker_scratch);
  if (retained_bytes > std::numeric_limits<size_t>::max() - scratch)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  const size_t per_channel = checked_product(frame_count, sizeof(ClipCandidate));
  // Up to 6 doubles/pixel/channel of stripe accumulators (A, B, QA, QA0, QA1,
  // QAA) plus up to 4 output profile planes and the 3 alpha-confidence maps.
  constexpr size_t fixed_pixel = 6 * sizeof(double) + sizeof(size_t) + 100;
  if (per_channel > std::numeric_limits<size_t>::max() - fixed_pixel)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  DrizzleMemoryPlan memory;
  for (;;) {
    try {
      memory = plan_drizzle_memory(
          plan, cfg, checked_product(channels, per_channel + fixed_pixel),
          retained_bytes + scratch);
      break;
    } catch (const std::runtime_error &e) {
      if (req_workers == 1 || std::string(e.what()).find("DRIZZLE_MEMORY_BUDGET") != 0)
        throw;
      --req_workers;
      scratch = checked_product(static_cast<size_t>(req_workers), worker_scratch);
    }
  }
  // Measurement-only coarse phase timing (P6). Zero effect unless TC_FD_PROFILE
  // is set; never touches compute, order or bounds.
  const bool fd_profile = std::getenv("TC_FD_PROFILE") != nullptr;
  static std::atomic<std::uint64_t> g_fd_cells_emitted{0};
  if (fd_profile) g_fd_cells_emitted.store(0, std::memory_order_relaxed);
  double prof_prep = 0, prof_alloc = 0, prof_raster = 0, prof_reduce = 0,
         prof_sink = 0;
  auto prof_now = [] { return std::chrono::steady_clock::now(); };
  auto prof_add = [&](double &acc, std::chrono::steady_clock::time_point s) {
    if (fd_profile)
      acc += std::chrono::duration<double>(prof_now() - s).count();
  };
  auto prof_t0 = prof_now();
  auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  prof_add(prof_prep, prof_t0);
  ForwardDrizzlePairDiagnostics summary;
  summary.diagnostics = prepared.diagnostics;
  summary.diagnostics.estimated_peak_bytes = memory.estimated_peak_bytes;
  summary.diagnostics.resolved_chunk_rows = memory.rows;
  summary.diagnostics.workers_requested = std::max(1, workers);
  summary.diagnostics.workers_budgeted = req_workers;
  summary.diagnostics.worker_scratch_bytes = worker_scratch;
  {
    int local_n = 0;
    for (const auto *pf : prepared.frames)
      if (pf->has_smooth_local_model)
        ++local_n;
    geomstats::stamp_context(plan.source_width, plan.source_height,
                             plan.canvas_width_native,
                             plan.canvas_height_native, cfg.internal_scale,
                             memory.rows, memory.height,
                             static_cast<int>(prepared.frames.size()), local_n);
  }

  // Plan 11.14.5 P3 (Teil 2): per-stripe output-row-band parallelism. Each band
  // is a disjoint slice of the stripe's canvas rows; the frame loop stays
  // outer + serial (one `source_of` load per frame on this thread) and the
  // reduce is band-partitioned on the SAME `i` ranges, so every canvas cell is
  // written by exactly one worker and every source contribution is added in the
  // unchanged canonical order --- bit-identical to `req_workers == 1`.
  // The process-global geometry-stats registry is not concurrency-safe; only
  // collect it on the serial path. Restored on exit.
  const bool geom_stats_was_enabled = geomstats::registry().enabled;
  const bool suppress_geom_stats = req_workers > 1 && geom_stats_was_enabled;
  struct GeomStatsRestore {
    bool value;
    ~GeomStatsRestore() { geomstats::registry().enabled = value; }
  } geom_stats_restore{geom_stats_was_enabled};
  if (suppress_geom_stats) geomstats::registry().enabled = false;
  summary.diagnostics.reduction_stats_suppressed = suppress_geom_stats;
  // The active geometry cache reader lives in a thread_local guard that OpenMP
  // worker threads do not inherit; capture it here and re-publish it per band.
  const DrizzleGeometryCacheReader *const geom_cache = active_geometry_cache();

  // P6: the per-stripe accumulators and the flat candidate buffer are sized to
  // the LARGEST stripe (`memory.rows`) and reused across stripes. A fresh
  // allocation + value-initialisation of the ~channels*width*rows*frame_count
  // ClipCandidate buffer per stripe was ~24% of the phase and fully serial
  // (measured, §30.76). `candidates` is never zeroed --- every entry read by
  // the reduce was written by the gather this stripe, bounded by `counts` ---
  // so only the `counts` prefix is reset. Bit-identical: same values, same
  // per-pixel frame order, same reduce spans.
  const size_t max_n =
      static_cast<size_t>(memory.width) * static_cast<size_t>(memory.rows);
  std::array<std::vector<ClipCandidate>, 3> candidates;
  std::array<std::vector<size_t>, 3> counts;
  std::array<std::vector<double>, 3> A, B, QA, QA0, QA1, QAA, QAF;
  std::vector<double> ac_sep, ac_art, ac_reg;
  for (int c = 0; c < channels; ++c) {
    candidates[c].resize(checked_product(max_n, frame_count));
    counts[c].resize(max_n);
    A[c].resize(max_n);
    B[c].resize(max_n);
    if (need_qc) QA[c].resize(max_n);
    if (need_q0) QA0[c].resize(max_n);
    if (need_q1) QA1[c].resize(max_n);
    if (need_qa) { QAA[c].resize(max_n); QAF[c].resize(max_n); }
  }
  if (need_qa) {
    ac_sep.resize(max_n);
    ac_art.resize(max_n);
    ac_reg.resize(max_n);
  }

  // Priority 2 (§30.79): one reusable budgeted clip scratch per band worker,
  // sized once to the maximum candidate span (frame_count) and reused for
  // every pixel of every band/stripe. Band b's instance is only touched by
  // fn(b) of a single for_each_band invocation at a time, so concurrent
  // reduce calls never share one instance. Cover charge: worker_scratch.
  std::vector<DrizzleClipScratch> band_clip_scratch(
      static_cast<size_t>(req_workers));

  for (int y = 0; y < memory.height;) {
    auto prof_ts = prof_now();
    const int rows = std::min(memory.rows, memory.height - y);
    const size_t n = static_cast<size_t>(memory.width) * rows;
    ForwardDrizzleUniformAndRawResult result;
    auto init_profile = [&](ForwardDrizzleUniformResult &p) {
      p.color_mode = plan.color_mode;
      p.internal_width = memory.width;
      p.internal_height = rows;
      if (plan.color_mode == ColorMode::MONO) {
        p.L.allocate(memory.width, rows);
      } else {
        p.R.allocate(memory.width, rows);
        p.G.allocate(memory.width, rows);
        p.B.allocate(memory.width, rows);
      }
    };
    init_profile(result.uniform);
    init_profile(result.raw);
    if (mb.emit_fine) init_profile(result.fine);
    if (mb.emit_medium) init_profile(result.medium);
    // Alpha-confidence stripe maps (channel-min): NaN until at least one
    // channel writes a value. `ac_*` are hoisted; reset the used prefix.
    if (need_qa) {
      result.a_separation.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.a_artifact.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.a_registration.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.alpha_confidence_support.assign(n, 0u);
      std::fill_n(ac_sep.begin(), n, std::numeric_limits<double>::infinity());
      std::fill_n(ac_art.begin(), n, std::numeric_limits<double>::infinity());
      std::fill_n(ac_reg.begin(), n, std::numeric_limits<double>::infinity());
    }
    auto planes_of = [&](ForwardDrizzleUniformResult &p) {
      return channels == 1
                 ? std::array<ProfilePlane *, 3>{&p.L, nullptr, nullptr}
                 : std::array<ProfilePlane *, 3>{&p.R, &p.G, &p.B};
    };
    const auto uniform_planes = planes_of(result.uniform);
    const auto raw_planes = planes_of(result.raw);
    const auto fine_planes = planes_of(result.fine);
    const auto medium_planes = planes_of(result.medium);

    // Hoisted buffers: reset only the `[0, n)` prefix this stripe uses. `A`/`B`
    // (and the QA* streams) are additionally re-zeroed per band per frame
    // below; the prefix wipe here keeps the stripe-entry state identical to the
    // old per-stripe `assign`. `candidates` is deliberately NOT touched.
    // QAF = K-weight of contributions whose artifact sample was FINITE (plan
    // 14.4's "< 8 gueltige Framebeiträge" counts contributions with real
    // artifact data).
    for (int c = 0; c < channels; ++c) {
      std::fill_n(counts[c].begin(), n, size_t{0});
      std::fill_n(A[c].begin(), n, 0.0);
      std::fill_n(B[c].begin(), n, 0.0);
      if (need_qc) std::fill_n(QA[c].begin(), n, 0.0);
      if (need_q0) std::fill_n(QA0[c].begin(), n, 0.0);
      if (need_q1) std::fill_n(QA1[c].begin(), n, 0.0);
      if (need_qa) {
        std::fill_n(QAA[c].begin(), n, 0.0);
        std::fill_n(QAF[c].begin(), n, 0.0);
      }
    }

    // Output-row-band partition of this stripe: band b covers canvas rows
    // [band_lo(b), band_lo(b + 1)) of the stripe (0-based within the stripe).
    // The tiling is gap-free and non-overlapping, so unioning per-band
    // enumerate/reduce over the bands reproduces the whole-stripe result
    // exactly (the leaf-cell y-clamp already restricts each call to its
    // window). `nb == 1` on the default serial path.
    const int nb = std::max(1, std::min(req_workers, rows));
    const int stripe_w = memory.width;
    auto band_lo = [&](int b) {
      return static_cast<int>((static_cast<long long>(b) * rows) / nb);
    };
    // Run `fn(b)` for every band in [0, nb), concurrently when nb > 1. Band
    // coverage is guaranteed by the worksharing loop, not by the team size:
    // `num_threads` is an upper bound the runtime may reduce, so keying off
    // `omp_get_thread_num()` could leave bands unprocessed.
    auto for_each_band = [&](auto &&fn) {
#ifdef _OPENMP
      if (nb > 1) {
        std::exception_ptr eptr;
#pragma omp parallel num_threads(nb)
        {
#pragma omp single
          summary.diagnostics.workers_used = std::max(
              summary.diagnostics.workers_used, omp_get_num_threads());
#pragma omp for schedule(static, 1)
          for (int b = 0; b < nb; ++b) {
            try {
              fn(b);
            } catch (...) {
#pragma omp critical
              if (!eptr) eptr = std::current_exception();
            }
          }
        }
        if (eptr) std::rethrow_exception(eptr);
        return;
      }
#endif
      for (int b = 0; b < nb; ++b) fn(b);
    };

    prof_add(prof_alloc, prof_ts);
    prof_ts = prof_now();
    for (const auto *f : prepared.frames) {
      const Matrix2Df &source = source_of(f->source_index);
      if (source.rows() != plan.source_height || source.cols() != plan.source_width)
        throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");
      FrameQualityMaps qm;
      if (quality_of) {
        qm = quality_of(f->source_index);
        auto check_shape = [&](const Matrix2Df *m) {
          if (m && (m->rows() != plan.source_height ||
                    m->cols() != plan.source_width))
            throw std::invalid_argument("DRIZZLE_QUALITY_SHAPE_MISMATCH");
        };
        check_shape(qm.composite);
        check_shape(qm.scale0);
        check_shape(qm.scale1);
        check_shape(qm.artifact);
      }
      const Matrix2Df *qc = need_qc ? qm.composite : nullptr;
      const Matrix2Df *q0 = need_q0 ? qm.scale0 : nullptr;
      const Matrix2Df *q1 = need_q1 ? qm.scale1 : nullptr;
      const Matrix2Df *qa = need_qa ? qm.artifact : nullptr;

      for_each_band([&](int b) {
        const int r0 = band_lo(b), r1 = band_lo(b + 1);
        if (r1 <= r0) return;
        // OpenMP workers do not inherit the thread_local active-cache guard;
        // re-publish it for this worker so local-warp frames still replay from
        // the geometry cache instead of re-running sample_leaves.
        std::optional<ScopedActiveGeometryCache> band_cache_guard;
        if (geom_cache) band_cache_guard.emplace(geom_cache);
        const size_t bi0 = static_cast<size_t>(r0) * stripe_w;
        const size_t bn = static_cast<size_t>(r1 - r0) * stripe_w;
        for (int c = 0; c < channels; ++c) {
          std::fill_n(A[c].begin() + bi0, bn, 0.0);
          std::fill_n(B[c].begin() + bi0, bn, 0.0);
          if (need_qc) std::fill_n(QA[c].begin() + bi0, bn, 0.0);
          if (need_q0) std::fill_n(QA0[c].begin() + bi0, bn, 0.0);
          if (need_q1) std::fill_n(QA1[c].begin() + bi0, bn, 0.0);
          if (need_qa) {
            std::fill_n(QAA[c].begin() + bi0, bn, 0.0);
            std::fill_n(QAF[c].begin() + bi0, bn, 0.0);
          }
        }
        // Geometry-stats instrumentation is serial-path only (registry is not
        // concurrency-safe); on the default nb == 1 path this is the identical
        // scope as before.
        std::optional<geomstats::ScopedVariant> gs_variant;
        std::optional<geomstats::ScopedGeometryTimer> gs_timer;
        if (nb == 1) {
          gs_variant.emplace(geomstats::Variant::kProductionUniformRaw,
                             cfg.pixfrac);
          gs_timer.emplace();
        }
        std::uint64_t band_cells = 0;
        rasterize_drizzle_stripe(
            plan, *f, cfg.internal_scale, cfg.pixfrac, y + r0, r1 - r0,
            [&](int sx, int sy, int c, int /*leaf*/, size_t i, double k) {
              if (fd_profile) ++band_cells;
              // `i` is relative to the (y + r0) window origin; re-base it into
              // the stripe-wide accumulators.
              const size_t gi = i + bi0;
              const double v = source(sy, sx);
              if (!std::isfinite(v)) return;
              A[c][gi] += k * v;
              B[c][gi] += k;
              // Plan 11.9: a NaN / <= 0 source Q contributes 0 to the K-average
              // (a missing Q-map is not an unweighted fallback; Q=0 is an
              // explicit per-sample veto).
              auto acc = [&](const Matrix2Df *m, std::vector<double> &dst) {
                if (!m) return;
                const double qv = (*m)(sy, sx);
                dst[gi] += k * (std::isfinite(qv) && qv > 0.0 ? qv : 0.0);
              };
              acc(qc, QA[c]);
              acc(q0, QA0[c]);
              acc(q1, QA1[c]);
              acc(qa, QAA[c]);
              if (qa) {
                const double av = (*qa)(sy, sx);
                if (std::isfinite(av)) QAF[c][gi] += k;  // real artifact datum
              }
            },
            subdivision);
        if (fd_profile)
          g_fd_cells_emitted.fetch_add(band_cells, std::memory_order_relaxed);
        for (int c = 0; c < channels; ++c)
          for (size_t i = bi0; i < bi0 + bn; ++i)
            if (B[c][i] > 0)
              candidates[c][i * frame_count + counts[c][i]++] = {
                  f->source_index, A[c][i] / B[c][i], B[c][i],
                  need_qc ? QA[c][i] / B[c][i] : 1.0,
                  need_q0 ? QA0[c][i] / B[c][i] : 1.0,
                  need_q1 ? QA1[c][i] / B[c][i] : 1.0,
                  need_qa ? QAA[c][i] / B[c][i] : 1.0,
                  need_qa && QAF[c][i] > 0.0};
      });
    }

    prof_add(prof_raster, prof_ts);
    prof_ts = prof_now();
    std::vector<ForwardDrizzleClippingDiagnostics> band_clip(nb);
    for_each_band([&](int b) {
      const int r0 = band_lo(b), r1 = band_lo(b + 1);
      if (r1 <= r0) return;
      const size_t bi0 = static_cast<size_t>(r0) * stripe_w;
      const size_t bn = static_cast<size_t>(r1 - r0) * stripe_w;
      ForwardDrizzleClippingDiagnostics lc;
      auto &clip_scratch = band_clip_scratch[static_cast<size_t>(b)];
      clip_scratch.reserve_for(frame_count, need_qa);
      for (int c = 0; c < channels; ++c) {
        for (size_t i = bi0; i < bi0 + bn; ++i) {
          if (!counts[c][i]) continue;
          // The per-frame accumulation above pushes candidates in
          // prepared-frame order, so this slice is already frame-ordered
          // (plan 19.6 step 4).
          const std::span<const ClipCandidate> pixel(
              candidates[c].data() + i * frame_count, counts[c][i]);
          reduce_pixel_profiles(
              pixel, reduce_cfg, g_eff_for, reg_by_source, i, uniform_planes[c],
              raw_planes[c], mb.emit_fine ? fine_planes[c] : nullptr,
              mb.emit_medium ? medium_planes[c] : nullptr,
              need_qa ? &ac_sep[i] : nullptr, need_qa ? &ac_art[i] : nullptr,
              need_qa ? &ac_reg[i] : nullptr, lc, &clip_scratch);
        }
      }
      if (need_qa) {
        for (size_t i = bi0; i < bi0 + bn; ++i) {
          if (!std::isfinite(ac_sep[i])) continue;  // no active channel
          result.a_separation[i] = static_cast<float>(ac_sep[i]);
          result.a_artifact[i] = static_cast<float>(ac_art[i]);
          result.a_registration[i] = static_cast<float>(ac_reg[i]);
          result.alpha_confidence_support[i] = 1u;
        }
      }
      band_clip[b] = lc;
    });
    // Integer counters --- order-independent, so the summed result is identical
    // to the serial single-accumulator path.
    for (const auto &lc : band_clip) {
      result.clipping.pixel_channel_evaluations += lc.pixel_channel_evaluations;
      result.clipping.pixel_channel_rejected += lc.pixel_channel_rejected;
      result.clipping.candidate_contributions_clipped +=
          lc.candidate_contributions_clipped;
    }
    prof_add(prof_reduce, prof_ts);
    prof_ts = prof_now();
    result.diagnostics = summary.diagnostics;
    sink(y, result);
    prof_add(prof_sink, prof_ts);
    summary.clipping.pixel_channel_evaluations += result.clipping.pixel_channel_evaluations;
    summary.clipping.pixel_channel_rejected += result.clipping.pixel_channel_rejected;
    summary.clipping.candidate_contributions_clipped += result.clipping.candidate_contributions_clipped;
    y += rows;
  }
  if (fd_profile) {
    // Capacity-growing scratch reserves; ≈ one per worker on the first
    // stripe, zero while warm. Counts allocations removed from the per-pixel
    // hot path by the priority-2 scratch.
    std::uint64_t clip_grows = 0;
    for (const auto &s : band_clip_scratch) clip_grows += s.growth_count;
    std::fprintf(stderr,
                 "[TC_FD_PROFILE] prep=%.3f alloc=%.3f raster=%.3f "
                 "reduce=%.3f sink=%.3f (s)  cells_emitted=%llu "
                 "clip_scratch_grows=%llu\n",
                 prof_prep, prof_alloc, prof_raster, prof_reduce, prof_sink,
                 (unsigned long long)g_fd_cells_emitted.load(
                     std::memory_order_relaxed),
                 (unsigned long long)clip_grows);
  }
  return summary;
}

ForwardDrizzleUniformAndRawResult compute_forward_drizzle_uniform_and_raw(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb,
    int workers) {
  const auto dimensions = plan_drizzle_memory(plan, cfg, 1);
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  const size_t retained = checked_product(
      checked_product(dimensions.width, dimensions.height), channels * 100);
  ForwardDrizzleUniformAndRawResult result;
  const size_t full_n =
      static_cast<size_t>(dimensions.width) * dimensions.height;
  if (mb.emit_alpha_confidence) {
    result.a_separation.assign(full_n, std::numeric_limits<float>::quiet_NaN());
    result.a_artifact.assign(full_n, std::numeric_limits<float>::quiet_NaN());
    result.a_registration.assign(full_n,
                                 std::numeric_limits<float>::quiet_NaN());
    result.alpha_confidence_support.assign(full_n, 0u);
  }
  auto copy = [&](ForwardDrizzleUniformResult &dst,
                  const ForwardDrizzleUniformResult &stripe, int y) {
    if (stripe.internal_height <= 0) return;  // profile not emitted
    dst.color_mode = plan.color_mode;
    dst.internal_width = dimensions.width;
    dst.internal_height = dimensions.height;
    auto plane = [&](ProfilePlane &d, const ProfilePlane &p) {
      if (p.empty()) return;
      if (d.empty()) d.allocate(dimensions.width, dimensions.height);
      const size_t offset = static_cast<size_t>(y) * dimensions.width;
      std::copy(p.value.begin(), p.value.end(), d.value.begin() + offset);
      std::copy(p.weight_sum.begin(), p.weight_sum.end(), d.weight_sum.begin() + offset);
      std::copy(p.n_eff.begin(), p.n_eff.end(), d.n_eff.begin() + offset);
      std::copy(p.support.begin(), p.support.end(), d.support.begin() + offset);
    };
    if (channels == 1) plane(dst.L, stripe.L);
    else { plane(dst.R, stripe.R); plane(dst.G, stripe.G); plane(dst.B, stripe.B); }
  };
  const auto summary = stream_forward_drizzle_uniform_and_raw(
      plan, source_of, cfg, clip_cfg,
      [&](int y, const ForwardDrizzleUniformAndRawResult &stripe) {
        copy(result.uniform, stripe.uniform, y);
        copy(result.raw, stripe.raw, y);
        copy(result.fine, stripe.fine, y);
        copy(result.medium, stripe.medium, y);
        if (mb.emit_alpha_confidence && !stripe.a_separation.empty()) {
          const size_t off = static_cast<size_t>(y) * dimensions.width;
          std::copy(stripe.a_separation.begin(), stripe.a_separation.end(),
                    result.a_separation.begin() + off);
          std::copy(stripe.a_artifact.begin(), stripe.a_artifact.end(),
                    result.a_artifact.begin() + off);
          std::copy(stripe.a_registration.begin(), stripe.a_registration.end(),
                    result.a_registration.begin() + off);
          std::copy(stripe.alpha_confidence_support.begin(),
                    stripe.alpha_confidence_support.end(),
                    result.alpha_confidence_support.begin() + off);
        }
      }, subdivision, g_eff_by_source_index, retained, quality_of, mb, workers);
  result.diagnostics = summary.diagnostics;
  result.uniform.diagnostics = summary.diagnostics;
  result.raw.diagnostics = summary.diagnostics;
  result.fine.diagnostics = summary.diagnostics;
  result.medium.diagnostics = summary.diagnostics;
  result.clipping = summary.clipping;
  return result;
}

} // namespace tile_compile::reconstruction
