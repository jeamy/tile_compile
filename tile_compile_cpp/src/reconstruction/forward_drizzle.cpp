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
      m.budget_bytes - retained_bytes - source - fixed < 1024 * 1024) {
    std::ostringstream msg;
    msg << "DRIZZLE_MEMORY_BUDGET: retained/source buffers exceed budget "
        << "(retained=" << retained_bytes / (1024 * 1024) << " MiB, source="
        << source / (1024 * 1024) << " MiB, fixed=" << fixed / (1024 * 1024)
        << " MiB, budget=" << m.budget_bytes / (1024 * 1024) << " MiB)";
    throw std::runtime_error(msg.str());
  }
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

DrizzleMemoryPlan plan_drizzle_memory_autogrow(
    const registration::RegistrationSamplingPlan &plan,
    config::ReconstructionDrizzleConfig &cfg, size_t bytes_per_pixel,
    size_t retained_bytes, bool loads_source,
    const std::function<void(const std::string &)> &warn) {
  size_t mb = cfg.memory_budget_mb ? cfg.memory_budget_mb : 512;
  for (;;) {
    config::ReconstructionDrizzleConfig attempt = cfg;
    attempt.memory_budget_mb = mb;
    try {
      const auto result = plan_drizzle_memory(plan, attempt, bytes_per_pixel,
                                              retained_bytes, loads_source);
      if (mb != cfg.memory_budget_mb) cfg.memory_budget_mb = mb;
      return result;
    } catch (const std::runtime_error &e) {
      if (std::string(e.what()).rfind("DRIZZLE_MEMORY_BUDGET", 0) != 0)
        throw;
      // The effective budget is min(configured, ~80% of the live memory
      // headroom); once the headroom cap binds, further +1 GiB steps cannot
      // grow it, so the working set genuinely does not fit.
      const size_t available = available_memory_headroom();
      const size_t cap = available == std::numeric_limits<size_t>::max()
                             ? std::numeric_limits<size_t>::max()
                             : available - available / 5;
      if (mb > std::numeric_limits<size_t>::max() / (1024 * 1024) - 1024)
        throw;
      const size_t next_mb = mb + 1024;
      if (std::min(next_mb * 1024 * 1024, cap) <=
          std::min(mb * 1024 * 1024, cap))
        throw;
      if (warn)
        warn(std::string(e.what()) +
             " — raising drizzle memory budget to " +
             std::to_string(next_mb) + " MiB");
      mb = next_mb;
    }
  }
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

DrizzleSourceScanBox drizzle_source_scan_box(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &f,
    int scale, int y_begin, int rows, int x_begin, int cols) {
  const int W_full = plan.canvas_width_native * scale;
  const int xb = std::clamp(x_begin, 0, W_full);
  const int xe = cols < 0 ? W_full : std::clamp(x_begin + cols, xb, W_full);
  DrizzleSourceScanBox box{0, plan.source_height, 0, plan.source_width};
  if (f.has_smooth_local_model) return box;
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
  // §30.81: inverse-map the TARGET-COLUMN-WINDOW rectangle (not the full
  // canvas width) so a narrow tile only scans the source pixels that can
  // reach it. xb/xe collapse to 0/W for the default window.
  for (double x : {static_cast<double>(xb) / scale,
                   static_cast<double>(xe) / scale})
    for (double y : {static_cast<double>(y_begin) / scale,
                     static_cast<double>(y_begin + rows) / scale}) {
      const double sx = inverse(0, 0) * x + inverse(0, 1) * y + inverse(0, 2);
      const double sy = inverse(1, 0) * x + inverse(1, 1) * y + inverse(1, 2);
      lo = std::min(lo, sy);
      hi = std::max(hi, sy);
      xlo = std::min(xlo, sx);
      xhi = std::max(xhi, sx);
    }
  box.y0 = static_cast<int>(std::clamp(
      std::floor(lo - 1), 0.0, static_cast<double>(plan.source_height)));
  box.y1 = static_cast<int>(std::clamp(
      std::ceil(hi + 1), 0.0, static_cast<double>(plan.source_height)));
  box.x0 = static_cast<int>(std::clamp(
      std::floor(xlo - 1), 0.0, static_cast<double>(plan.source_width)));
  box.x1 = static_cast<int>(std::clamp(
      std::ceil(xhi + 1), 0.0, static_cast<double>(plan.source_width)));
  return box;
}

void drizzle_affine_source_spans_into(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &f,
    float pixfrac, int band_y_begin_native, int band_rows_native,
    std::vector<DrizzleAffineSourceSpan> &out) {
  out.clear();
  if (!f.valid || f.has_smooth_local_model ||
      !f.source_to_canvas_affine_valid || !std::isfinite(pixfrac) ||
      pixfrac <= 0.0f || band_rows_native <= 0 || band_y_begin_native < 0 ||
      band_y_begin_native + band_rows_native > plan.canvas_height_native)
    return;
  const int sw = plan.source_width, sh = plan.source_height;
  const double W = static_cast<double>(plan.canvas_width_native);
  const double y0 = static_cast<double>(band_y_begin_native);
  const double y1 = y0 + static_cast<double>(band_rows_native);
  const double a0 = f.source_to_canvas(0, 0), a1 = f.source_to_canvas(0, 1);
  const double a2 = f.source_to_canvas(0, 2), a3 = f.source_to_canvas(1, 0);
  const double a4 = f.source_to_canvas(1, 1), a5 = f.source_to_canvas(1, 2);
  const double half = 0.5 * static_cast<double>(pixfrac);
  const double rx = half * (std::abs(a0) + std::abs(a1));
  const double ry = half * (std::abs(a3) + std::abs(a4));
  const double inf = std::numeric_limits<double>::infinity();
  for (int sy = 0; sy < sh; ++sy) {
    const double syc = static_cast<double>(sy) + 0.5;
    // covers() is the exact per-pixel membership predicate: the four strict
    // droplet-bbox inequalities evaluated on integer source coordinates.
    auto covers = [&](int sx) {
      const double sxc = static_cast<double>(sx) + 0.5;
      const double qx = a0 * sxc + a1 * syc + a2;
      const double qy = a3 * sxc + a4 * syc + a5;
      return qx + rx > 0.0 && qx - rx < W && qy + ry > y0 && qy - ry < y1;
    };
    // Analytic interval: each inequality is linear in sx' = sx + 0.5 and is
    // rewritten as  coef * sx' > rhs.
    double lo = -inf, hi = inf;
    bool impossible = false;
    auto add = [&](double coef, double rhs) {
      if (coef > 0.0)
        lo = std::max(lo, rhs / coef);
      else if (coef < 0.0)
        hi = std::min(hi, rhs / coef);
      else if (!(0.0 > rhs))
        impossible = true;
    };
    add(a0, -(a1 * syc + a2 + rx));      // qx_max > 0
    add(-a0, a1 * syc + a2 - rx - W);    // qx_min < W
    add(a3, y0 - ry - a4 * syc - a5);    // qy_max > y0
    add(-a3, a4 * syc + a5 - ry - y1);   // qy_min < y1
    if (impossible) continue;
    // sx+0.5 in (lo, hi) -> sx in (lo-.5, hi-.5); pad 3 px against rounding,
    // then covers() decides the true edges (shrink failing boundary pixels,
    // extend while membership holds).
    int x0 = std::isinf(lo) ? 0
                            : static_cast<int>(std::floor(lo - 0.5)) - 3;
    int x1 = std::isinf(hi) ? sw
                            : static_cast<int>(std::ceil(hi - 0.5)) + 3;
    x0 = std::clamp(x0, 0, sw);
    x1 = std::clamp(x1, 0, sw);
    if (x0 >= x1) {
      // Degenerate (near-)empty analytic interval: probe the bound positions
      // and midpoint before declaring the row inactive.
      for (double px : {lo, hi, 0.5 * (lo + hi)}) {
        if (!std::isfinite(px)) continue;
        const int p = std::clamp(
            static_cast<int>(std::lround(px - 0.5)), 0, sw - 1);
        if (covers(p)) {
          x0 = p;
          x1 = p + 1;
          break;
        }
      }
      if (x0 >= x1) continue;
    }
    while (x0 < x1 && !covers(x0)) ++x0;
    while (x1 > x0 && !covers(x1 - 1)) --x1;
    while (x0 > 0 && covers(x0 - 1)) --x0;
    while (x1 < sw && covers(x1)) ++x1;
    if (x0 < x1) out.push_back({sy, x0, x1});
  }
}

std::vector<DrizzleAffineSourceSpan> drizzle_affine_source_spans(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &f,
    float pixfrac, int band_y_begin_native, int band_rows_native) {
  std::vector<DrizzleAffineSourceSpan> out;
  drizzle_affine_source_spans_into(plan, f, pixfrac, band_y_begin_native,
                                   band_rows_native, out);
  return out;
}

void enumerate_drizzle_stripe_leaf_cells(
    const RegistrationSamplingPlan &plan, const FrameSamplingTransform &f,
    int scale, float pixfrac, int y_begin, int rows,
    const DrizzleLeafCellSink &sink, const ForwardDrizzleSubdivisionParams &p,
    int x_begin, int cols) {
  namespace gs = geomstats;
  const bool instrument = gs::registry().enabled;
  // §30.81 target-column window. `xb == 0 && xe == W` reproduces the historical
  // full-width scan exactly (same source box, same bbox clamp, same order).
  const int W_full = plan.canvas_width_native * scale;
  const int xb = std::clamp(x_begin, 0, W_full);
  const int xe = cols < 0 ? W_full : std::clamp(x_begin + cols, xb, W_full);

  // Plan 11.14 P1/P2: for a LOCAL-WARP frame that a published geometry cache
  // holds, replay the pre-built leaves for this stripe instead of re-running
  // sample_leaves over the whole source. Bit-identical to the scan below
  // (same corners, same bbox clamp, same canonical order); adds zero local
  // basis evaluations. Affine frames and cache misses fall through unchanged.
  if (f.has_smooth_local_model) {
    const auto *cache = active_geometry_cache();
    if (cache && cache->has_frame(pixfrac, f.source_index)) {
      if (instrument) ++gs::registry().cur().enumerate_calls;
      if (xb == 0 && xe == W_full) {
        cache->enumerate_stripe(pixfrac, f.source_index, scale, y_begin, rows,
                                sink);
      } else {
        // §30.81: the cache replay contract is full-width; a narrower window
        // drops the out-of-window cells here. Order and per-cell corners are
        // untouched, so the retained subset is byte-identical to the windowed
        // affine scan. (Local-warp frames do not reach the tiled CUDA store
        // path; this keeps the CPU streaming build correct under a window.)
        cache->enumerate_stripe(
            pixfrac, f.source_index, scale, y_begin, rows,
            [&](int sx, int sy, int c, int lo, int cx, int cy, const double *lx,
                const double *ly) {
              if (cx >= xb && cx < xe)
                sink(sx, sy, c, lo, cx, cy, lx, ly);
            });
      }
      return;
    }
  }

  // The scanned source box is shared with every banded source/quality read
  // (A1): drizzle_source_scan_box reproduces the exact inverse-mapped box the
  // scan below uses (full extent for local-warp frames).
  const auto scan_box =
      drizzle_source_scan_box(plan, f, scale, y_begin, rows, x_begin, cols);
  const int source_y0 = scan_box.y0, source_y1 = scan_box.y1;
  const int source_x0 = scan_box.x0, source_x1 = scan_box.x1;
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
  // Hoisted out of the per-source-pixel loop (redundant-reload analysis C5):
  // invariant for the whole call. `sink` is an opaque std::function the
  // compiler cannot assume leaves `plan` (a const-ref alias) unmodified
  // across the call, so without this it must re-read every iteration.
  const bool is_osc = plan.color_mode == ColorMode::OSC;
  const BayerPattern bayer_pattern = plan.bayer_pattern;
  const int cfa_origin_x = plan.cfa_origin_x;
  const int cfa_origin_y = plan.cfa_origin_y;
  for (int sy = source_y0; sy < source_y1; ++sy)
    for (int sx = source_x0; sx < source_x1; ++sx) {
      if (!sample_leaves(plan, f, sx, sy, scale, pixfrac, p, leaves))
        continue;
      int c = 0;
      if (is_osc) {
        const auto channel = cfa_channel_for_source_pixel(
            sx, sy, bayer_pattern, cfa_origin_x, cfa_origin_y);
        c = channel == CfaChannel::R ? 0 : channel == CfaChannel::G ? 1 : 2;
      }
      for (size_t li = 0; li < leaves.size(); ++li) {
        const auto &leaf = leaves[li];
        double xmin = *std::min_element(leaf.x, leaf.x + 4),
               xmax = *std::max_element(leaf.x, leaf.x + 4);
        double ymin = *std::min_element(leaf.y, leaf.y + 4),
               ymax = *std::max_element(leaf.y, leaf.y + 4);
        int x0 = static_cast<int>(std::clamp(
            std::floor(xmin), static_cast<double>(xb), static_cast<double>(xe)));
        int x1 = static_cast<int>(std::clamp(
            std::ceil(xmax), static_cast<double>(xb), static_cast<double>(xe)));
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
                              const ForwardDrizzleSubdivisionParams &p,
                              int x_begin, int cols) {
  const int W_full = plan.canvas_width_native * scale;
  const int xb = std::clamp(x_begin, 0, W_full);
  const int xe = cols < 0 ? W_full : std::clamp(x_begin + cols, xb, W_full);
  const int win_w = xe - xb;
  enumerate_drizzle_stripe_leaf_cells(
      plan, f, scale, pixfrac, y_begin, rows,
      [&](int sx, int sy, int c, int leaf_order, int x, int y,
          const double *lx, const double *ly) {
        const double k = polygon_rectangle_intersection_area(lx, ly, x, y,
                                                             x + 1.0, y + 1.0);
        if (k > 0)
          sink(sx, sy, c, leaf_order,
               static_cast<size_t>(y - y_begin) * win_w + (x - xb), k);
      },
      p, xb, win_w);
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
    const ForwardDrizzleSubdivisionParams &subdivision, size_t retained_bytes,
    const SourceImageRectProvider &source_rect_of) {
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
  // Hoisted out of the stripe loop (redundant-reload analysis D2): these were
  // previously declared fresh every stripe iteration, forcing a malloc/free
  // per stripe per array. `n` shrinks only for the final (possibly partial)
  // stripe, so hoisting the std::array<vector> declarations lets .assign(n,0)
  // below reuse each vector's already-grown capacity across stripes instead
  // of reallocating. Bit-identical: same values assigned, same n per stripe.
  std::array<std::vector<double>, 3> wx, w, w2, A, B;
  // D3: the stripe result object is reused across stripes (sinks consume it
  // synchronously and may not retain it); ProfilePlane::allocate assigns, so
  // the plane vectors keep their grown capacity instead of reallocating per
  // stripe.
  ForwardDrizzleUniformResult stripe;
  for (int y = 0; y < memory.height; y += memory.rows) {
    const int rows = std::min(memory.rows, memory.height - y);
    const size_t n = static_cast<size_t>(memory.width) * rows;
    for (int c = 0; c < channels; ++c) {
      wx[c].assign(n, 0);
      w[c].assign(n, 0);
      w2[c].assign(n, 0);
      A[c].assign(n, 0);
      B[c].assign(n, 0);
    }
    for (const auto *f : prepared.frames) {
      // A3: with a banded source provider only the stripe's inverse-mapped
      // source box is read; without it the historical full-frame load runs.
      Matrix2Df src_rect;
      const Matrix2Df *src_p = nullptr;
      int src_yo = 0, src_xo = 0;
      if (source_rect_of) {
        const auto box = drizzle_source_scan_box(
            plan, *f, cfg.internal_scale, y, rows, 0, -1);
        if (box.y1 <= box.y0 || box.x1 <= box.x0)
          continue;  // no scanned source pixels -> no contributions
        src_rect = source_rect_of(f->source_index, box.y0, box.y1, box.x0,
                                  box.x1);
        if (src_rect.rows() != box.y1 - box.y0 ||
            src_rect.cols() != box.x1 - box.x0)
          throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");
        src_yo = box.y0;
        src_xo = box.x0;
        src_p = &src_rect;
      } else {
        src_p = &source_of(f->source_index);
        if (src_p->rows() != plan.source_height ||
            src_p->cols() != plan.source_width)
          throw std::invalid_argument("DRIZZLE_SOURCE_SHAPE_MISMATCH");
      }
      const Matrix2Df &source = *src_p;
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
              const double v = source(sy - src_yo, sx - src_xo);
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
    const ForwardDrizzleSubdivisionParams &subdivision,
    const SourceImageRectProvider &source_rect_of) {
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
      subdivision, retained, source_rect_of);
  return result;
}
} // namespace tile_compile::reconstruction
