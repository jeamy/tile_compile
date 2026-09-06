#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

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

namespace {

// One accepted leaf of the (possibly subdivided) droplet: a convex
// quadrilateral in internal-canvas coordinates.
struct Leaf {
  double x[4];
  double y[4];
};

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

bool sample_leaves(const RegistrationSamplingPlan &plan,
                   const FrameSamplingTransform &f, int sx, int sy, int scale,
                   float pixfrac, const ForwardDrizzleSubdivisionParams &p,
                   std::vector<Leaf> &leaves) {
  leaves.clear();
  const double x = sx + 0.5, y = sy + 0.5, h = pixfrac / 2.0;
  if (f.has_smooth_local_model)
    return subdivide_local(f, x - h, y - h, x + h, y + h, 0,
                           plan.canvas_width_native, plan.canvas_height_native,
                           scale, p, {}, leaves);
  Leaf leaf;
  if (!build_affine_leaf(f, x, y, h, scale, leaf))
    return false;
  leaves.push_back(leaf);
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

void rasterize_drizzle_stripe(const RegistrationSamplingPlan &plan,
                              const FrameSamplingTransform &f, int scale,
                              float pixfrac, int y_begin, int rows,
                              const DrizzleAreaSink &sink,
                              const ForwardDrizzleSubdivisionParams &p) {
  const int W = plan.canvas_width_native * scale;
  int source_y0 = 0, source_y1 = plan.source_height;
  if (!f.has_smooth_local_model) {
    // Inverse-map the entire destination stripe (plus source half-pixel).
    // This includes every intersecting droplet, including across stripe edges;
    // no value halo or duplicate output rows are necessary.
    WarpMatrix inverse;
    if (!registration::invert_affine_2x3(f.source_to_canvas, 1e-12f, 1e12f,
                                         inverse))
      throw std::invalid_argument("DRIZZLE_SINGULAR_TRANSFORM");
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (double x : {0.0, static_cast<double>(W) / scale})
      for (double y : {static_cast<double>(y_begin) / scale,
                       static_cast<double>(y_begin + rows) / scale}) {
        const double sy = inverse(1, 0) * x + inverse(1, 1) * y + inverse(1, 2);
        lo = std::min(lo, sy);
        hi = std::max(hi, sy);
      }
    source_y0 = static_cast<int>(std::clamp(
        std::floor(lo - 1), 0.0, static_cast<double>(plan.source_height)));
    source_y1 = static_cast<int>(std::clamp(
        std::ceil(hi + 1), 0.0, static_cast<double>(plan.source_height)));
  }
  std::vector<Leaf> leaves;
  leaves.reserve(16);
  for (int sy = source_y0; sy < source_y1; ++sy)
    for (int sx = 0; sx < plan.source_width; ++sx) {
      if (!sample_leaves(plan, f, sx, sy, scale, pixfrac, p, leaves))
        continue;
      int c = 0;
      if (plan.color_mode == ColorMode::OSC) {
        const auto channel = cfa_channel_for_source_pixel(
            sx, sy, plan.bayer_pattern, plan.cfa_origin_x, plan.cfa_origin_y);
        c = channel == CfaChannel::R ? 0 : channel == CfaChannel::G ? 1 : 2;
      }
      for (const auto &leaf : leaves) {
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
            const double k = polygon_rectangle_intersection_area(
                leaf.x, leaf.y, x, y, x + 1.0, y + 1.0);
            if (k > 0)
              sink(sx, sy, c, static_cast<size_t>(y - y_begin) * W + x, k);
          }
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
      rasterize_drizzle_stripe(
          plan, *f, cfg.internal_scale, cfg.pixfrac, y, rows,
          [&](int sx, int sy, int c, size_t i, double k) {
            const double v = source(sy, sx);
            if (std::isfinite(v)) {
              A[c][i] += k * v;
              B[c][i] += k;
            }
          },
          subdivision);
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
ClipResult apply_robust_clipping(std::span<const ClipCandidate> candidates,
                                 int min_clip_contributors, int robust_passes,
                                 float clip_sigma_low, float clip_sigma_high,
                                 float min_fraction, float min_n_eff) {
  if (min_clip_contributors < 1 || robust_passes < 0 ||
      !std::isfinite(clip_sigma_low) || clip_sigma_low < 0 ||
      !std::isfinite(clip_sigma_high) || clip_sigma_high < 0 ||
      !std::isfinite(min_fraction) || min_fraction < 0 || min_fraction > 1 ||
      !std::isfinite(min_n_eff) || min_n_eff < 0)
    throw std::invalid_argument("DRIZZLE_INVALID_CLIPPING_CONFIG");
  for (const auto &c : candidates)
    if (!std::isfinite(c.x) || !std::isfinite(c.b) || c.b <= 0)
      throw std::invalid_argument("DRIZZLE_INVALID_CLIP_CANDIDATE");
  ClipResult result;
  const size_t n = candidates.size();
  result.accepted.assign(n, true);
  if (n == 0) {
    result.pixel_rejected = true;
    return result;
  }

  // Step 2: below min_clip_contributors, skip straight to step 8 with every
  // candidate still valid (protects thin R/B channels at low frame counts).
  if (n >= static_cast<size_t>(min_clip_contributors)) {
    // Step 3: one fixed, deterministic value order for the whole procedure.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if (candidates[a].x != candidates[b].x) return candidates[a].x < candidates[b].x;
      return candidates[a].frame_index < candidates[b].frame_index;
    });

    for (int pass = 0; pass < robust_passes; ++pass) {
      std::vector<size_t> active;
      active.reserve(n);
      for (size_t idx : order)
        if (result.accepted[idx]) active.push_back(idx);
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
      std::vector<size_t> dev_order = active;
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
          result.accepted[idx] = false;
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
    if (!result.accepted[i]) continue;
    ++accepted_count;
    sum_w += candidates[i].b;
    sum_w2 += candidates[i].b * candidates[i].b;
  }
  const double fraction = static_cast<double>(accepted_count) / static_cast<double>(n);
  const double n_eff = sum_w2 > 0.0 ? (sum_w * sum_w) / sum_w2 : 0.0;
  if (fraction < static_cast<double>(min_fraction) || n_eff < static_cast<double>(min_n_eff)) {
    result.pixel_rejected = true;
  }
  return result;
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
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb) {
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
  const size_t scratch = checked_product(frame_count, 3 * sizeof(size_t) + 8);
  if (retained_bytes > std::numeric_limits<size_t>::max() - scratch)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  const size_t per_channel = checked_product(frame_count, sizeof(ClipCandidate));
  // Up to 6 doubles/pixel/channel of stripe accumulators (A, B, QA, QA0, QA1,
  // QAA) plus up to 4 output profile planes and the 3 alpha-confidence maps.
  constexpr size_t fixed_pixel = 6 * sizeof(double) + sizeof(size_t) + 100;
  if (per_channel > std::numeric_limits<size_t>::max() - fixed_pixel)
    throw std::runtime_error("DRIZZLE_MEMORY_BUDGET: size overflow");
  const auto memory = plan_drizzle_memory(
      plan, cfg, checked_product(channels, per_channel + fixed_pixel),
      retained_bytes + scratch);
  auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  ForwardDrizzlePairDiagnostics summary;
  summary.diagnostics = prepared.diagnostics;
  summary.diagnostics.estimated_peak_bytes = memory.estimated_peak_bytes;
  summary.diagnostics.resolved_chunk_rows = memory.rows;

  for (int y = 0; y < memory.height;) {
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
    // channel writes a value.
    std::vector<double> ac_sep, ac_art, ac_reg;
    if (need_qa) {
      result.a_separation.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.a_artifact.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.a_registration.assign(n, std::numeric_limits<float>::quiet_NaN());
      result.alpha_confidence_support.assign(n, 0u);
      ac_sep.assign(n, std::numeric_limits<double>::infinity());
      ac_art.assign(n, std::numeric_limits<double>::infinity());
      ac_reg.assign(n, std::numeric_limits<double>::infinity());
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

    std::array<std::vector<ClipCandidate>, 3> candidates;
    std::array<std::vector<size_t>, 3> counts;
    for (int c = 0; c < channels; ++c) {
      candidates[c].resize(checked_product(n, frame_count));
      counts[c].assign(n, 0);
    }

    // QAF = K-weight of contributions whose artifact sample was FINITE. Plan
    // 14.4's "< 8 gueltige Framebeiträge" counts contributions with real
    // artifact data, so QAA (which folds NaN -> 0 like every other Q stream)
    // cannot distinguish "no artifact map here" from "artifact_conf == 0".
    std::array<std::vector<double>, 3> A, B, QA, QA0, QA1, QAA, QAF;
    for (int c = 0; c < channels; ++c) {
      A[c].assign(n, 0);
      B[c].assign(n, 0);
      if (need_qc) QA[c].assign(n, 0);
      if (need_q0) QA0[c].assign(n, 0);
      if (need_q1) QA1[c].assign(n, 0);
      if (need_qa) { QAA[c].assign(n, 0); QAF[c].assign(n, 0); }
    }

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
      for (int c = 0; c < channels; ++c) {
        std::fill(A[c].begin(), A[c].end(), 0);
        std::fill(B[c].begin(), B[c].end(), 0);
        if (need_qc) std::fill(QA[c].begin(), QA[c].end(), 0);
        if (need_q0) std::fill(QA0[c].begin(), QA0[c].end(), 0);
        if (need_q1) std::fill(QA1[c].begin(), QA1[c].end(), 0);
        if (need_qa) {
          std::fill(QAA[c].begin(), QAA[c].end(), 0);
          std::fill(QAF[c].begin(), QAF[c].end(), 0);
        }
      }
      rasterize_drizzle_stripe(
          plan, *f, cfg.internal_scale, cfg.pixfrac, y, rows,
          [&](int sx, int sy, int c, size_t i, double k) {
            const double v = source(sy, sx);
            if (!std::isfinite(v)) return;
            A[c][i] += k * v;
            B[c][i] += k;
            // Plan 11.9: a NaN / <= 0 source Q contributes 0 to the K-average
            // (a missing Q-map is not an unweighted fallback; Q=0 is an
            // explicit per-sample veto).
            auto acc = [&](const Matrix2Df *m, std::vector<double> &dst) {
              if (!m) return;
              const double qv = (*m)(sy, sx);
              dst[i] += k * (std::isfinite(qv) && qv > 0.0 ? qv : 0.0);
            };
            acc(qc, QA[c]);
            acc(q0, QA0[c]);
            acc(q1, QA1[c]);
            acc(qa, QAA[c]);
            if (qa) {
              const double av = (*qa)(sy, sx);
              if (std::isfinite(av)) QAF[c][i] += k;  // real artifact datum
            }
          },
          subdivision);
      for (int c = 0; c < channels; ++c)
        for (size_t i = 0; i < n; ++i)
          if (B[c][i] > 0)
            candidates[c][i * frame_count + counts[c][i]++] = {
                f->source_index, A[c][i] / B[c][i], B[c][i],
                need_qc ? QA[c][i] / B[c][i] : 1.0,
                need_q0 ? QA0[c][i] / B[c][i] : 1.0,
                need_q1 ? QA1[c][i] / B[c][i] : 1.0,
                need_qa ? QAA[c][i] / B[c][i] : 1.0,
                need_qa && QAF[c][i] > 0.0};
    }

    for (int c = 0; c < channels; ++c) {
      for (size_t i = 0; i < n; ++i) {
        if (!counts[c][i]) continue;
        const std::span<const ClipCandidate> pixel(
            candidates[c].data() + i * frame_count, counts[c][i]);
        ++result.clipping.pixel_channel_evaluations;
        auto clip = apply_robust_clipping(
            pixel, cfg.min_clip_contributors, cfg.robust_passes,
            clip_cfg.clip_sigma_low, clip_cfg.clip_sigma_high, clip_cfg.min_fraction,
            clip_cfg.min_n_eff);
        for (bool accepted : clip.accepted)
          if (!accepted) ++result.clipping.candidate_contributions_clipped;
        if (clip.pixel_rejected) {
          ++result.clipping.pixel_channel_rejected;
          continue;
        }
        // Uniform: w = B. Raw/Fine/Medium: w = B * G_eff(f) * Q^e, where the
        // per-candidate geometric K-averages are q (composite, e=1),
        // q0 (scale0, e=fine_quality_exponent) and q1 (scale1,
        // e=medium_quality_exponent). All share the clip mask above (plan
        // 11.8); none of q/q0/q1 entered it.
        struct Accum { double wx = 0, w = 0, w2 = 0; };
        Accum au, ar, af, am;
        auto add = [](Accum &a, double w, double x) {
          a.wx += w * x; a.w += w; a.w2 += w * w;
        };
        for (size_t k = 0; k < pixel.size(); ++k) {
          if (!clip.accepted[k]) continue;
          const auto &cd = pixel[k];
          const double g = g_eff_for(cd.frame_index);
          add(au, cd.b, cd.x);
          add(ar, cd.b * g * cd.q, cd.x);
          if (mb.emit_fine)
            add(af, cd.b * g * std::pow(cd.q0, mb.fine_quality_exponent), cd.x);
          if (mb.emit_medium)
            add(am, cd.b * g * std::pow(cd.q1, mb.medium_quality_exponent),
                cd.x);
        }
        const size_t gi = i;
        auto write = [&](const std::array<ProfilePlane *, 3> &pl,
                         const Accum &a) {
          if (a.w <= 0.0) return;
          ProfilePlane &p = *pl[c];
          p.value[gi] = static_cast<float>(a.wx / a.w);
          p.weight_sum[gi] = static_cast<float>(a.w);
          p.n_eff[gi] =
              static_cast<float>(a.w2 > 0.0 ? (a.w * a.w) / a.w2 : 0.0);
          p.support[gi] = 1;
        };
        write(uniform_planes, au);
        write(raw_planes, ar);
        if (mb.emit_fine) write(fine_planes, af);
        if (mb.emit_medium) write(medium_planes, am);

        if (need_qa) {
          // Plan 14.4: A_separation / A_artifact / A_registration from the
          // accepted frame contributions for this channel; the frame result
          // takes the conservative min over active channels.
          std::vector<AlphaFactorContribution> contribs;
          contribs.reserve(pixel.size());
          for (size_t k = 0; k < pixel.size(); ++k) {
            if (!clip.accepted[k]) continue;
            const auto &cd = pixel[k];
            const auto &rg = reg_by_source[cd.frame_index];
            // No real artifact datum for this contribution => exclude it from
            // the robust artifact statistic (NaN, not a fabricated value).
            const double art_conf =
                cd.qa_has_data ? cd.qa
                               : std::numeric_limits<double>::quiet_NaN();
            contribs.push_back({cd.b, cd.q, art_conf, rg.first != 0u,
                                static_cast<double>(rg.second)});
          }
          const auto fac =
              compute_alpha_confidence_channel(contribs, mb.alpha_confidence);
          ac_sep[gi] = std::min(ac_sep[gi], fac.a_separation);
          ac_art[gi] = std::min(ac_art[gi], fac.a_artifact);
          ac_reg[gi] = std::min(ac_reg[gi], fac.a_registration);
        }
      }
    }
    if (need_qa) {
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ac_sep[i])) continue;  // no active channel
        result.a_separation[i] = static_cast<float>(ac_sep[i]);
        result.a_artifact[i] = static_cast<float>(ac_art[i]);
        result.a_registration[i] = static_cast<float>(ac_reg[i]);
        result.alpha_confidence_support[i] = 1u;
      }
    }
    result.diagnostics = summary.diagnostics;
    sink(y, result);
    summary.clipping.pixel_channel_evaluations += result.clipping.pixel_channel_evaluations;
    summary.clipping.pixel_channel_rejected += result.clipping.pixel_channel_rejected;
    summary.clipping.candidate_contributions_clipped += result.clipping.candidate_contributions_clipped;
    y += rows;
  }
  return summary;
}

ForwardDrizzleUniformAndRawResult compute_forward_drizzle_uniform_and_raw(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clip_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff_by_source_index,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb) {
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
      }, subdivision, g_eff_by_source_index, retained, quality_of, mb);
  result.diagnostics = summary.diagnostics;
  result.uniform.diagnostics = summary.diagnostics;
  result.raw.diagnostics = summary.diagnostics;
  result.fine.diagnostics = summary.diagnostics;
  result.medium.diagnostics = summary.diagnostics;
  result.clipping = summary.clipping;
  return result;
}

} // namespace tile_compile::reconstruction
