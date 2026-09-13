#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

struct AffineInverse {
  double a = 0.0, b = 0.0, c = 0.0;
  double d = 0.0, e = 0.0, f = 0.0;
};

AffineInverse invert_source_to_canvas(
    const registration::FrameSamplingTransform &frame) {
  const auto &m = frame.source_to_canvas;
  const double a = m(0, 0), b = m(0, 1), c = m(0, 2);
  const double d = m(1, 0), e = m(1, 1), f = m(1, 2);
  const double det = a * e - b * d;
  if (!frame.source_to_canvas_affine_valid || !std::isfinite(det) ||
      std::abs(det) <= std::numeric_limits<double>::epsilon())
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_AFFINE");
  const double id = 1.0 / det;
  AffineInverse out;
  out.a = e * id;
  out.b = -b * id;
  out.d = -d * id;
  out.e = a * id;
  out.c = -(out.a * c + out.b * f);
  out.f = -(out.d * c + out.e * f);
  return out;
}

std::array<double, 2> map_to_source(const AffineInverse &m, double x,
                                    double y) {
  return {m.a * x + m.b * y + m.c, m.d * x + m.e * y + m.f};
}

int channel_for(const registration::RegistrationSamplingPlan &plan, int sx,
                int sy) {
  if (plan.color_mode == ColorMode::MONO) return 0;
  const auto ch = cfa_channel_for_source_pixel(
      sx, sy, plan.bayer_pattern, plan.cfa_origin_x, plan.cfa_origin_y);
  return ch == CfaChannel::R ? 0 : ch == CfaChannel::G ? 1 : 2;
}

DrizzleUniformAccum make_accum(int width, int rows, int channels) {
  DrizzleUniformAccum out;
  out.width = width;
  out.rows = rows;
  out.channels = channels;
  const std::size_t n = static_cast<std::size_t>(width) * rows;
  for (int c = 0; c < channels; ++c) {
    out.wx[c].assign(n, 0.0);
    out.w[c].assign(n, 0.0);
    out.w2[c].assign(n, 0.0);
  }
  return out;
}

}  // namespace

ForwardDrizzleV2UniformResult gather_affine_uniform_v2(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &subdivision) {
  if (cfg.internal_scale <= 0 || rows <= 0 || y_begin < 0)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_STRIPE");
  if (plan.color_mode != ColorMode::MONO &&
      plan.color_mode != ColorMode::OSC)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_UNSUPPORTED_COLOR_MODE");

  const int scale = cfg.internal_scale;
  const int width = plan.canvas_width_native * scale;
  const int height = plan.canvas_height_native * scale;
  if (width <= 0 || height <= 0 || y_begin + rows > height)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_STRIPE");

  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  ForwardDrizzleV2UniformResult result;
  result.accum = make_accum(width, rows, channels);
  result.stats.target_cells = static_cast<std::uint64_t>(width) * rows;
  result.stats.workspace_bytes =
      static_cast<std::size_t>(channels) * width * rows * 3 * sizeof(double);

  const auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  const double half = 0.5 * static_cast<double>(cfg.pixfrac);
  std::vector<Leaf> leaves;

  for (const auto *frame : prepared.frames) {
    if (frame->has_smooth_local_model)
      throw std::invalid_argument("FORWARD_DRIZZLE_V2_LOCAL_WARP_NOT_READY");
    const AffineInverse inv = invert_source_to_canvas(*frame);
    const Matrix2Df &src = source_of(frame->source_index);
    if (src.rows() != plan.source_height || src.cols() != plan.source_width)
      throw std::invalid_argument("FORWARD_DRIZZLE_V2_SOURCE_SHAPE");

    for (int ly = 0; ly < rows; ++ly) {
      const int ty = y_begin + ly;
      for (int tx = 0; tx < width; ++tx) {
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (int cy = 0; cy < 2; ++cy)
          for (int cx = 0; cx < 2; ++cx) {
            const auto p = map_to_source(
                inv, static_cast<double>(tx + cx) / scale,
                static_cast<double>(ty + cy) / scale);
            min_x = std::min(min_x, p[0]);
            max_x = std::max(max_x, p[0]);
            min_y = std::min(min_y, p[1]);
            max_y = std::max(max_y, p[1]);
          }

        // Source samples are centred at (sx+0.5, sy+0.5).  Expand by one
        // additional integer on both sides: the exact polygon test removes
        // false positives, while roundoff at a tangency cannot create a false
        // negative in this search bound.
        const int sx0 = std::max(
            0, static_cast<int>(std::floor(min_x - half - 0.5)) - 1);
        const int sx1 = std::min(
            plan.source_width,
            static_cast<int>(std::ceil(max_x + half - 0.5)) + 2);
        const int sy0 = std::max(
            0, static_cast<int>(std::floor(min_y - half - 0.5)) - 1);
        const int sy1 = std::min(
            plan.source_height,
            static_cast<int>(std::ceil(max_y + half - 0.5)) + 2);

        const std::size_t i = static_cast<std::size_t>(ly) * width + tx;
        std::array<double, 3> A{0.0, 0.0, 0.0};
        std::array<double, 3> B{0.0, 0.0, 0.0};
        for (int sy = sy0; sy < sy1; ++sy) {
          for (int sx = sx0; sx < sx1; ++sx) {
            ++result.stats.source_candidates;
            const double v = static_cast<double>(src(sy, sx));
            if (!std::isfinite(v)) continue;
            if (!sample_leaves(plan, *frame, sx, sy, scale, cfg.pixfrac,
                               subdivision, leaves))
              continue;
            const int c = channel_for(plan, sx, sy);
            for (const auto &leaf : leaves) {
              ++result.stats.leaves_tested;
              const double k = polygon_rectangle_intersection_area(
                  leaf.x, leaf.y, tx, ty, tx + 1.0, ty + 1.0);
              if (!(k > 0.0)) continue;
              ++result.stats.positive_overlaps;
              A[c] += k * v;
              B[c] += k;
            }
          }
        }
        for (int c = 0; c < channels; ++c) {
          if (!(B[c] > 0.0)) continue;
          result.accum.wx[c][i] += A[c];
          result.accum.w[c][i] += B[c];
          result.accum.w2[c][i] += B[c] * B[c];
        }
      }
    }
  }
  return result;
}

bool gather_affine_uniform_v2_cuda(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    ForwardDrizzleV2UniformResult &out,
    const ForwardDrizzleSubdivisionParams &subdivision) {
  if (!forward_drizzle_cuda_runtime_available()) return false;
  if (cfg.internal_scale <= 0 || rows <= 0 || y_begin < 0)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_STRIPE");
  if (plan.color_mode != ColorMode::MONO && plan.color_mode != ColorMode::OSC)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_UNSUPPORTED_COLOR_MODE");
  const int scale = cfg.internal_scale;
  const int width = plan.canvas_width_native * scale;
  const int height = plan.canvas_height_native * scale;
  if (width <= 0 || height <= 0 || y_begin + rows > height)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_STRIPE");
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  out = {};
  out.accum = make_accum(width, rows, channels);
  out.stats.target_cells = static_cast<std::uint64_t>(width) * rows;
  out.stats.workspace_bytes =
      static_cast<std::size_t>(channels) * width * rows * 5 * sizeof(double);
  const std::size_t n = static_cast<std::size_t>(width) * rows;
  std::vector<double> frame_a(static_cast<std::size_t>(channels) * n);
  std::vector<double> frame_b(static_cast<std::size_t>(channels) * n);
  const auto prepared = prepare_drizzle_frames(plan, cfg, subdivision);
  for (const auto *frame : prepared.frames) {
    if (frame->has_smooth_local_model)
      throw std::invalid_argument("FORWARD_DRIZZLE_V2_LOCAL_WARP_NOT_READY");
    const AffineInverse inv = invert_source_to_canvas(*frame);
    const auto &m = frame->source_to_canvas;
    const double affine6[6] = {m(0, 0), m(0, 1), m(0, 2),
                               m(1, 0), m(1, 1), m(1, 2)};
    const double inverse6[6] = {inv.a, inv.b, inv.c, inv.d, inv.e, inv.f};
    const Matrix2Df &src = source_of(frame->source_index);
    if (src.rows() != plan.source_height || src.cols() != plan.source_width)
      throw std::invalid_argument("FORWARD_DRIZZLE_V2_SOURCE_SHAPE");
    unsigned long long candidates = 0, overlaps = 0;
    if (!forward_drizzle_cuda_affine_target_gather(
            affine6, inverse6, scale, 0.5 * static_cast<double>(cfg.pixfrac),
            0, y_begin, width, rows, plan.source_width, plan.source_height,
            src.data(), static_cast<int>(plan.bayer_pattern), plan.cfa_origin_x,
            plan.cfa_origin_y, channels == 1, frame_a.data(), frame_b.data(),
            &candidates, &overlaps))
      return false;
    out.stats.source_candidates += candidates;
    out.stats.positive_overlaps += overlaps;
    out.stats.leaves_tested += candidates;
    for (int c = 0; c < channels; ++c) {
      const std::size_t base = static_cast<std::size_t>(c) * n;
      for (std::size_t i = 0; i < n; ++i) {
        const double b = frame_b[base + i];
        if (!(b > 0.0)) continue;
        out.accum.wx[c][i] += frame_a[base + i];
        out.accum.w[c][i] += b;
        out.accum.w2[c][i] += b * b;
      }
    }
  }
  return true;
}

ForwardDrizzleV2FoldResult fold_native_pixel_v2(
    std::span<const ForwardDrizzleV2FrameSubpixel> entries,
    std::size_t frame_count, std::span<const double> area) {
  if (frame_count == 0 || area.empty() ||
      frame_count > std::numeric_limits<std::size_t>::max() / area.size() ||
      entries.size() != frame_count * area.size())
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_FOLD_SHAPE");
  double total_area = 0.0;
  for (double a : area) {
    if (!std::isfinite(a) || a < 0.0)
      throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_FOLD_AREA");
    total_area += a;
  }
  if (!(total_area > 0.0))
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_FOLD_AREA");

  ForwardDrizzleV2FoldResult out;
  std::vector<std::uint8_t> geometry_supported(area.size(), 0u);
  std::vector<std::uint8_t> source_supported(area.size(), 0u);
  std::vector<std::uint8_t> estimator_supported(area.size(), 0u);
  std::vector<std::uint8_t> profile_supported(area.size(), 0u);
  for (std::size_t f = 0; f < frame_count; ++f) {
    double af = 0.0;
    double bf = 0.0;
    for (std::size_t j = 0; j < area.size(); ++j) {
      const auto &v = entries[f * area.size() + j];
      if (v.frame_order != f)
        throw std::invalid_argument("FORWARD_DRIZZLE_V2_FOLD_FRAME_ORDER");
      const auto valid_weight = [](double w) {
        return std::isfinite(w) && w >= 0.0;
      };
      if (!valid_weight(v.geometry_b) || !valid_weight(v.source_b) ||
          !valid_weight(v.estimator_b) || !valid_weight(v.b))
        throw std::invalid_argument("FORWARD_DRIZZLE_V2_FOLD_INVALID_WEIGHT");
      if ((v.source_b > 0.0 && !(v.geometry_b > 0.0)) ||
          (v.estimator_b > 0.0 && !(v.source_b > 0.0)) ||
          (v.b > 0.0 && !(v.estimator_b > 0.0)))
        throw std::invalid_argument("FORWARD_DRIZZLE_V2_FOLD_SUPPORT_ORDER");
      if (v.b > 0.0 && !std::isfinite(v.a))
        throw std::invalid_argument("FORWARD_DRIZZLE_V2_FOLD_NONFINITE_VALUE");
      geometry_supported[j] |= static_cast<std::uint8_t>(v.geometry_b > 0.0);
      source_supported[j] |= static_cast<std::uint8_t>(v.source_b > 0.0);
      estimator_supported[j] |=
          static_cast<std::uint8_t>(v.estimator_b > 0.0);
      profile_supported[j] |= static_cast<std::uint8_t>(v.b > 0.0);
      if (!(v.b > 0.0)) continue;
      af += area[j] * v.a;
      bf += area[j] * v.b;
    }
    if (bf > 0.0) {
      out.a += af;
      out.b += bf;
      out.b2 += bf * bf;
    }
  }
  auto area_fraction = [&](const std::vector<std::uint8_t> &supported) {
    double sum = 0.0;
    for (std::size_t j = 0; j < area.size(); ++j)
      if (supported[j]) sum += area[j];
    return std::clamp(sum / total_area, 0.0, 1.0);
  };
  out.geometry_area_fraction = area_fraction(geometry_supported);
  out.source_area_fraction = area_fraction(source_supported);
  out.estimator_area_fraction = area_fraction(estimator_supported);
  out.profile_area_fraction = area_fraction(profile_supported);
  out.geometry_support = out.geometry_area_fraction > 0.0;
  out.source_support = out.source_area_fraction > 0.0;
  out.estimator_support = out.estimator_area_fraction > 0.0;
  out.profile_support = out.b > 0.0 && std::isfinite(out.a) &&
                        std::isfinite(out.b) && std::isfinite(out.b2);
  if (out.profile_support) {
    out.value = out.a / out.b;
    out.n_eff = out.b2 > 0.0 ? out.b * out.b / out.b2 : 0.0;
  }
  return out;
}

#if !TILE_COMPILE_WITH_CUDA
// CUDA-free build: the device oracle is defined in
// forward_drizzle_cuda_device.cu when CUDA is compiled in. Here it always
// reports unavailable and never touches `out`.
bool fold_native_pixel_v2_cuda(
    std::span<const ForwardDrizzleV2FrameSubpixel>, std::size_t,
    std::span<const double>, ForwardDrizzleV2FoldResult &) {
  return false;
}
#endif

namespace {

double median_sorted(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t m = values.size() / 2;
  if (values.size() & 1U) return values[m];
  return 0.5 * (values[m - 1] + values[m]);
}

void finish_uniform_fallback(ForwardDrizzleV2RobustResult &out, double a,
                             double b, double b2) {
  out.a = a;
  out.b = b;
  out.b2 = b2;
  out.value = b > 0.0 ? a / b : 0.0;
  out.n_eff = b2 > 0.0 ? b * b / b2 : 0.0;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t &out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t &out) {
  if (b > std::numeric_limits<std::size_t>::max() - a) return false;
  out = a + b;
  return true;
}

// Gate-4: sigma2 for candidate i; absent or invalid input contributes 0 and
// is counted as degraded. Never throws.
double conf_sigma2_of(std::span<const double> sigma2, std::size_t i,
                      std::uint64_t &degraded) {
  if (sigma2.empty()) return 0.0;
  const double v = sigma2[i];
  if (!std::isfinite(v) || v < 0.0) {
    ++degraded;
    return 0.0;
  }
  return v;
}

void finalize_confidence(ForwardDrizzleV2RobustResult &out) {
  if (!(out.conf_b > 0.0)) {
    out.confidence = 0.0;
    out.conf_state = ForwardDrizzleV2ConfidenceState::no_source_support;
    return;
  }
  if (out.conf_c > 0.0 && std::isfinite(out.conf_c)) {
    out.confidence = (out.conf_s * out.conf_s) /
                     (out.conf_s * out.conf_s + out.conf_c);
    out.conf_state = ForwardDrizzleV2ConfidenceState::modeled;
    return;
  }
  out.confidence = out.n_eff > 0.0 ? out.n_eff / (out.n_eff + 1.0) : 0.0;
  out.conf_state = ForwardDrizzleV2ConfidenceState::fallback_n_eff;
}

}  // namespace

ForwardDrizzleV2RobustResult robust_reduce_v2(
    const ForwardDrizzleV2CandidateReplay &replay,
    const ForwardDrizzleV2RobustConfig &cfg) {
  if (!replay || cfg.groups < 3 || (cfg.groups % 2) == 0 ||
      cfg.min_candidates < 1 || cfg.min_groups < 1 ||
      cfg.min_groups > cfg.groups || !std::isfinite(cfg.winsor_sigma) ||
      cfg.winsor_sigma <= 0.0)
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_ROBUST_CONFIG");

  struct Group {
    double a = 0.0;
    double b = 0.0;
  };
  std::vector<Group> groups(static_cast<std::size_t>(cfg.groups));
  ForwardDrizzleV2RobustResult out;
  double uniform_a = 0.0, uniform_b = 0.0, uniform_b2 = 0.0;
  replay([&](const ForwardDrizzleV2RobustCandidate &v) {
    if (!std::isfinite(v.x) || !std::isfinite(v.b) || !(v.b > 0.0)) return;
    ++out.candidates;
    uniform_a += v.b * v.x;
    uniform_b += v.b;
    uniform_b2 += v.b * v.b;
    Group &g = groups[v.frame_order % groups.size()];
    g.a += v.b * v.x;
    g.b += v.b;
  });

  if (!(uniform_b > 0.0)) return out;
  if (out.candidates < static_cast<std::uint64_t>(cfg.min_candidates)) {
    out.state = ForwardDrizzleV2RobustState::too_few_candidates_fallback;
    finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
    out.conf_b = uniform_b;
    finalize_confidence(out);
    return out;
  }

  std::vector<double> means;
  means.reserve(groups.size());
  for (const auto &g : groups)
    if (g.b > 0.0) means.push_back(g.a / g.b);
  out.groups_used = means.size();
  if (means.size() < static_cast<std::size_t>(cfg.min_groups)) {
    out.state = ForwardDrizzleV2RobustState::too_few_groups_fallback;
    finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
    out.conf_b = uniform_b;
    finalize_confidence(out);
    return out;
  }

  out.center = median_sorted(means);
  std::vector<double> deviations;
  deviations.reserve(means.size());
  for (double x : means) deviations.push_back(std::abs(x - out.center));
  out.scale = 1.482602218505602 * median_sorted(std::move(deviations));
  if (!(out.scale > 0.0) || !std::isfinite(out.scale)) {
    out.state = ForwardDrizzleV2RobustState::degenerate_scale;
    // A zero group-MAD means at least half of the valid groups agree exactly.
    // Returning Uniform here would give an arbitrarily large isolated sample
    // full influence.  The mathematically defined degenerate solution is the
    // common robust centre with the original geometric support weights.
    finish_uniform_fallback(out, out.center * uniform_b, uniform_b, uniform_b2);
    out.conf_b = uniform_b;
    finalize_confidence(out);
    return out;
  }

  const double lo = out.center - cfg.winsor_sigma * out.scale;
  const double hi = out.center + cfg.winsor_sigma * out.scale;
  replay([&](const ForwardDrizzleV2RobustCandidate &v) {
    if (!std::isfinite(v.x) || !std::isfinite(v.b) || !(v.b > 0.0)) return;
    const double x = std::clamp(v.x, lo, hi);
    out.a += v.b * x;
    out.b += v.b;
    out.b2 += v.b * v.b;
  });
  out.value = out.a / out.b;
  out.n_eff = out.b2 > 0.0 ? out.b * out.b / out.b2 : 0.0;
  out.state = ForwardDrizzleV2RobustState::primary_winsorized_mom;
  out.conf_b = out.b;
  finalize_confidence(out);
  return out;
}

namespace {

struct GroupSlot {
  double a = 0.0;
  double b = 0.0;
  double b2 = 0.0;
  double s = 0.0;  // Gate-4: sum b*sigma
  double c = 0.0;  // Gate-4: sum b^2*sigma2
};

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double weighted_median_of(std::span<const double> values,
                          std::span<const double> weights) {
  // Deterministic (value, index) order; cumulative weight reaching half of
  // the total selects the value, matching the production clip semantics.
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
    if (values[i] != values[j]) return values[i] < values[j];
    return i < j;
  });
  double total_w = 0.0;
  for (double w : weights) total_w += w;
  if (!(total_w > 0.0)) return values.empty() ? 0.0 : values[order.back()];
  double cum = 0.0;
  for (std::size_t i : order) {
    cum += weights[i];
    if (cum >= total_w / 2.0) return values[i];
  }
  return values[order.back()];
}

}  // namespace

double forward_drizzle_v2_sigma2_model(double sigma_noise, double grad_x,
                                       double grad_y, double sigma_reg_px,
                                       double droplet_half) {
  if (!std::isfinite(sigma_noise) || sigma_noise < 0.0 ||
      !std::isfinite(grad_x) || !std::isfinite(grad_y) ||
      !std::isfinite(sigma_reg_px) || sigma_reg_px < 0.0 ||
      !std::isfinite(droplet_half) || !(droplet_half > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  const double g2 = grad_x * grad_x + grad_y * grad_y;
  return sigma_noise * sigma_noise + g2 * sigma_reg_px * sigma_reg_px +
         droplet_half * droplet_half / 3.0;
}

ForwardDrizzleV2RobustResult robust_frame_oracle_v2(
    std::span<const ForwardDrizzleV2RobustCandidate> candidates,
    int min_clip_contributors, int robust_passes, double sigma_low,
    double sigma_high, std::span<const double> candidate_sigma2) {
  if (min_clip_contributors < 1 || robust_passes < 1 ||
      !std::isfinite(sigma_low) || !std::isfinite(sigma_high) ||
      sigma_low <= 0.0 || sigma_high <= 0.0 ||
      (!candidate_sigma2.empty() &&
       candidate_sigma2.size() != candidates.size()))
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_ORACLE_CONFIG");
  ForwardDrizzleV2RobustResult out;
  std::vector<ForwardDrizzleV2RobustCandidate> valid;
  std::vector<std::size_t> valid_index;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto &v = candidates[i];
    if (!std::isfinite(v.x) || !std::isfinite(v.b) || !(v.b > 0.0)) continue;
    valid.push_back(v);
    valid_index.push_back(i);
  }
  out.candidates = valid.size();
  if (valid.empty()) return out;

  const std::size_t n = valid.size();
  std::vector<std::uint8_t> accepted(n, std::uint8_t{1});
  if (n >= static_cast<std::size_t>(min_clip_contributors)) {
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
      if (valid[i].x != valid[j].x) return valid[i].x < valid[j].x;
      return valid[i].frame_order < valid[j].frame_order;
    });
    std::vector<std::size_t> active;
    std::vector<std::size_t> dev_order;
    for (int pass = 0; pass < robust_passes; ++pass) {
      active.clear();
      for (std::size_t idx : order)
        if (accepted[idx]) active.push_back(idx);
      if (active.empty()) break;
      double total_w = 0.0;
      for (std::size_t idx : active) total_w += valid[idx].b;
      double median = valid[active.back()].x;
      if (total_w > 0.0) {
        double cum = 0.0;
        for (std::size_t idx : active) {
          cum += valid[idx].b;
          if (cum >= total_w / 2.0) {
            median = valid[idx].x;
            break;
          }
        }
      }
      dev_order = active;
      std::sort(dev_order.begin(), dev_order.end(),
                [&](std::size_t i, std::size_t j) {
                  const double di = std::abs(valid[i].x - median);
                  const double dj = std::abs(valid[j].x - median);
                  if (di != dj) return di < dj;
                  return valid[i].frame_order < valid[j].frame_order;
                });
      double mad = std::abs(valid[dev_order.back()].x - median);
      if (total_w > 0.0) {
        double cum = 0.0;
        for (std::size_t idx : dev_order) {
          cum += valid[idx].b;
          if (cum >= total_w / 2.0) {
            mad = std::abs(valid[idx].x - median);
            break;
          }
        }
      }
      const double lower = median - sigma_low * mad;
      const double upper = median + sigma_high * mad;
      bool changed = false;
      for (std::size_t idx : active) {
        const double x = valid[idx].x;
        if (!(x >= lower && x <= upper)) {
          accepted[idx] = 0;
          changed = true;
        }
      }
      if (!changed) break;
    }
  }

  double a = 0.0, b = 0.0, b2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!accepted[i]) continue;
    a += valid[i].b * valid[i].x;
    b += valid[i].b;
    b2 += valid[i].b * valid[i].b;
    out.conf_b += valid[i].b;
    const double s2 =
        conf_sigma2_of(candidate_sigma2, valid_index[i], out.conf_degraded);
    if (s2 > 0.0) {
      out.conf_s += valid[i].b * std::sqrt(s2);
      out.conf_c += valid[i].b * valid[i].b * s2;
    }
  }
  out.state = ForwardDrizzleV2RobustState::oracle_sigma_clip;
  out.center = b > 0.0 ? a / b : 0.0;
  finish_uniform_fallback(out, a, b, b2);
  out.state = ForwardDrizzleV2RobustState::oracle_sigma_clip;
  finalize_confidence(out);
  return out;
}

ForwardDrizzleV2RobustResult robust_reduce_candidates_v2(
    std::span<const ForwardDrizzleV2RobustCandidate> candidates,
    ForwardDrizzleV2Estimator estimator, const ForwardDrizzleV2RobustConfig &cfg,
    std::uint64_t stream_length, std::span<const double> candidate_sigma2) {
  if (estimator == ForwardDrizzleV2Estimator::two_pass_winsorized_frames ||
      cfg.groups < 3 || (cfg.groups % 2) == 0 || cfg.min_candidates < 1 ||
      cfg.min_groups < 1 || cfg.min_groups > cfg.groups ||
      !std::isfinite(cfg.winsor_sigma) || cfg.winsor_sigma <= 0.0 ||
      cfg.reservoir_size < 1 ||
      (!candidate_sigma2.empty() &&
       candidate_sigma2.size() != candidates.size()))
    throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_ROBUST_CONFIG");

  if (estimator == ForwardDrizzleV2Estimator::reservoir_sigma_clip) {
    // Single pass: the hash keep predicate only needs the known stream
    // length, so a production stream can push kept frames into the reservoir
    // while accumulating the uniform totals; no replay is required.
    const std::uint64_t n_stream =
        stream_length > 0 ? stream_length
                          : static_cast<std::uint64_t>(candidates.size());
    const bool keep_all =
        n_stream <= static_cast<std::uint64_t>(cfg.reservoir_size);
    const std::uint64_t threshold = keep_all ? 0 : static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(cfg.reservoir_size) << 64) / n_stream);
    ForwardDrizzleV2RobustResult out;
    std::vector<ForwardDrizzleV2RobustCandidate> reservoir;
    std::vector<double> reservoir_sigma2;
    if (keep_all) reservoir.reserve(candidates.size());
    else reservoir.reserve(static_cast<std::size_t>(cfg.reservoir_size) +
                           static_cast<std::size_t>(cfg.reservoir_size) / 4);
    double uniform_a = 0.0, uniform_b = 0.0, uniform_b2 = 0.0;
    double stream_s = 0.0, stream_c = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const auto &v = candidates[i];
      if (!std::isfinite(v.x) || !std::isfinite(v.b) || !(v.b > 0.0)) continue;
      ++out.candidates;
      uniform_a += v.b * v.x;
      uniform_b += v.b;
      uniform_b2 += v.b * v.b;
      const double s2 =
          conf_sigma2_of(candidate_sigma2, i, out.conf_degraded);
      if (s2 > 0.0) {
        stream_s += v.b * std::sqrt(s2);
        stream_c += v.b * v.b * s2;
      }
      if (keep_all || splitmix64(v.frame_order ^ cfg.reservoir_seed) <
                          threshold) {
        reservoir.push_back(v);
        reservoir_sigma2.push_back(s2);
      }
    }
    if (!(uniform_b > 0.0)) return out;
    if (out.candidates < static_cast<std::uint64_t>(cfg.min_candidates) ||
        reservoir.size() <
            static_cast<std::size_t>(cfg.oracle_min_clip_contributors)) {
      out.state = ForwardDrizzleV2RobustState::too_few_candidates_fallback;
      finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
      out.conf_b = uniform_b;
      out.conf_s = stream_s;
      out.conf_c = stream_c;
      finalize_confidence(out);
      return out;
    }
    const auto clipped = robust_frame_oracle_v2(
        reservoir, cfg.oracle_min_clip_contributors, cfg.oracle_passes,
        cfg.oracle_sigma_low, cfg.oracle_sigma_high, reservoir_sigma2);
    // The reservoir decides the value; B/B2 cover the full stream so support
    // and effective-N reflect every contributing frame. Confidence uses the
    // clip-accepted reservoir members (its sigma inputs were already
    // validated above, so the oracle's degraded count stays zero there).
    out.state = ForwardDrizzleV2RobustState::primary_reservoir_sigma_clip;
    out.center = clipped.center;
    out.scale = clipped.scale;
    out.groups_used = reservoir.size();
    finish_uniform_fallback(out, clipped.value * uniform_b, uniform_b,
                            uniform_b2);
    out.conf_b = clipped.conf_b;
    out.conf_s = clipped.conf_s;
    out.conf_c = clipped.conf_c;
    finalize_confidence(out);
    return out;
  }

  const std::size_t k = static_cast<std::size_t>(cfg.groups);
  std::vector<GroupSlot> groups(k);
  ForwardDrizzleV2RobustResult out;
  double uniform_a = 0.0, uniform_b = 0.0, uniform_b2 = 0.0;
  double stream_s = 0.0, stream_c = 0.0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto &v = candidates[i];
    if (!std::isfinite(v.x) || !std::isfinite(v.b) || !(v.b > 0.0)) continue;
    ++out.candidates;
    uniform_a += v.b * v.x;
    uniform_b += v.b;
    uniform_b2 += v.b * v.b;
    const double s2 = conf_sigma2_of(candidate_sigma2, i, out.conf_degraded);
    const double s = s2 > 0.0 ? std::sqrt(s2) : 0.0;
    stream_s += v.b * s;
    stream_c += v.b * v.b * s2;
    GroupSlot &g = groups[v.frame_order % k];
    g.a += v.b * v.x;
    g.b += v.b;
    g.b2 += v.b * v.b;
    g.s += v.b * s;
    g.c += v.b * v.b * s2;
  }

  if (!(uniform_b > 0.0)) return out;
  if (out.candidates < static_cast<std::uint64_t>(cfg.min_candidates)) {
    out.state = ForwardDrizzleV2RobustState::too_few_candidates_fallback;
    finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
    out.conf_b = uniform_b;
    out.conf_s = stream_s;
    out.conf_c = stream_c;
    finalize_confidence(out);
    return out;
  }

  std::vector<double> means, weights, group_b2, group_s, group_c;
  means.reserve(k);
  weights.reserve(k);
  group_b2.reserve(k);
  group_s.reserve(k);
  group_c.reserve(k);
  for (const auto &g : groups) {
    if (!(g.b > 0.0)) continue;
    means.push_back(g.a / g.b);
    weights.push_back(g.b);
    group_b2.push_back(g.b2);
    group_s.push_back(g.s);
    group_c.push_back(g.c);
  }
  out.groups_used = means.size();
  if (means.size() < static_cast<std::size_t>(cfg.min_groups)) {
    out.state = ForwardDrizzleV2RobustState::too_few_groups_fallback;
    finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
    out.conf_b = uniform_b;
    out.conf_s = stream_s;
    out.conf_c = stream_c;
    finalize_confidence(out);
    return out;
  }
  out.center = median_sorted(means);
  std::vector<double> deviations;
  deviations.reserve(means.size());
  for (double x : means) deviations.push_back(std::abs(x - out.center));
  out.scale = 1.482602218505602 * median_sorted(std::move(deviations));

  switch (estimator) {
    case ForwardDrizzleV2Estimator::uniform:
      out.state = ForwardDrizzleV2RobustState::primary_uniform;
      finish_uniform_fallback(out, uniform_a, uniform_b, uniform_b2);
      out.conf_b = uniform_b;
      out.conf_s = stream_s;
      out.conf_c = stream_c;
      finalize_confidence(out);
      return out;

    case ForwardDrizzleV2Estimator::mom_median: {
      out.state = ForwardDrizzleV2RobustState::primary_mom_median;
      const double value = weighted_median_of(means, weights);
      // No frame weight is dropped: B/B2 keep the uniform totals while the
      // value is the weighted median of the group means.
      finish_uniform_fallback(out, value * uniform_b, uniform_b, uniform_b2);
      out.center = value;
      out.conf_b = uniform_b;
      out.conf_s = stream_s;
      out.conf_c = stream_c;
      finalize_confidence(out);
      return out;
    }

    case ForwardDrizzleV2Estimator::mom_winsorized_groups:
    case ForwardDrizzleV2Estimator::mom_trimmed_groups: {
      if (!(out.scale > 0.0) || !std::isfinite(out.scale)) {
        out.state = ForwardDrizzleV2RobustState::degenerate_scale;
        finish_uniform_fallback(out, out.center * uniform_b, uniform_b,
                                uniform_b2);
        out.conf_b = uniform_b;
        out.conf_s = stream_s;
        out.conf_c = stream_c;
        finalize_confidence(out);
        return out;
      }
      const double lo = out.center - cfg.winsor_sigma * out.scale;
      const double hi = out.center + cfg.winsor_sigma * out.scale;
      const bool trimmed =
          estimator == ForwardDrizzleV2Estimator::mom_trimmed_groups;
      double a = 0.0, b = 0.0, b2 = 0.0, cb = 0.0, cs = 0.0, cc = 0.0;
      for (std::size_t g = 0; g < means.size(); ++g) {
        if (trimmed) {
          if (!(means[g] >= lo && means[g] <= hi)) continue;
          a += means[g] * weights[g];
        } else {
          a += std::clamp(means[g], lo, hi) * weights[g];
        }
        b += weights[g];
        b2 += group_b2[g];
        cb += weights[g];
        cs += group_s[g];
        cc += group_c[g];
      }
      if (!(b > 0.0)) {
        out.state = ForwardDrizzleV2RobustState::degenerate_scale;
        finish_uniform_fallback(out, out.center * uniform_b, uniform_b,
                                uniform_b2);
        out.conf_b = uniform_b;
        out.conf_s = stream_s;
        out.conf_c = stream_c;
        finalize_confidence(out);
        return out;
      }
      out.state = trimmed
                      ? ForwardDrizzleV2RobustState::primary_mom_trimmed_groups
                      : ForwardDrizzleV2RobustState::primary_mom_winsorized_groups;
      finish_uniform_fallback(out, a, b, b2);
      out.conf_b = cb;
      out.conf_s = cs;
      out.conf_c = cc;
      finalize_confidence(out);
      return out;
    }

    case ForwardDrizzleV2Estimator::two_pass_winsorized_frames:
      break;
  }
  throw std::invalid_argument("FORWARD_DRIZZLE_V2_INVALID_ROBUST_CONFIG");
}

ForwardDrizzleV2MemoryPlan plan_forward_drizzle_v2_memory(
    const ForwardDrizzleV2MemoryInputs &in) {
  ForwardDrizzleV2MemoryPlan out;
  const auto &r = in.roles;
  if (in.target_width <= 0 || in.target_height <= 0 ||
      (in.channels != 1 && in.channels != 3) || in.frame_count <= 0 ||
      in.internal_scale < 1 ||
      r.reservoir_size < 1 || r.reservoir_slot_factor < 1 ||
      r.reservoir_candidate_bytes == 0 ||
      r.frame_plane_doubles < 0 || r.robust_doubles < 0 ||
      r.confidence_doubles < 0 || r.centre_scale_doubles < 0 ||
      r.fold_staging_doubles < 0 || in.source_device_slots < 0 ||
      in.quality_device_slots < 0 || in.pinned_source_slots < 0 ||
      in.pinned_quality_slots < 0 || in.halo_rows_per_band_edge < 0 ||
      !std::isfinite(in.max_source_read_amplification) ||
      in.max_source_read_amplification < 1.0 ||
      !std::isfinite(in.max_quality_read_amplification) ||
      in.max_quality_read_amplification < 1.0)
    return out;
  // A nonzero per-row byte bound requires at least one resident slot.
  if ((in.pinned_source_bytes_per_frame_row > 0 &&
       in.pinned_source_slots < 1) ||
      (in.pinned_quality_bytes_per_frame_row > 0 &&
       in.pinned_quality_slots < 1) ||
      (in.source_slot_device_bytes > 0 && in.source_device_slots < 1) ||
      (in.quality_slot_device_bytes > 0 && in.quality_device_slots < 1))
    return out;

  // Per-native-pixel-per-channel accumulator bytes from the frozen role
  // table, plus frame planes at internal resolution (scale^2 subpixels).
  std::size_t per_channel = 0, tmp = 0, pixel_bytes = 0;
  const std::size_t acc_doubles =
      static_cast<std::size_t>(r.robust_doubles) +
      static_cast<std::size_t>(r.confidence_doubles) +
      static_cast<std::size_t>(r.centre_scale_doubles) +
      static_cast<std::size_t>(r.fold_staging_doubles);
  const std::size_t frame_planes =
      static_cast<std::size_t>(r.frame_plane_doubles) +
      (r.sigma2_plane ? std::size_t{1} : std::size_t{0});
  std::size_t internal_bytes = 0;
  const std::size_t scale_sq =
      static_cast<std::size_t>(in.internal_scale) *
      static_cast<std::size_t>(in.internal_scale);
  std::size_t reservoir_slots = 0;
  if (!checked_mul(static_cast<std::size_t>(r.reservoir_size),
                   static_cast<std::size_t>(r.reservoir_slot_factor),
                   reservoir_slots) ||
      !checked_mul(reservoir_slots,
                   r.reservoir_candidate_bytes, per_channel) ||
      !checked_add(per_channel, r.reservoir_kept_bytes, per_channel) ||
      !checked_mul(acc_doubles, sizeof(double), tmp) ||
      !checked_add(per_channel, tmp, per_channel) ||
      !checked_add(per_channel, r.confidence_counter_bytes, per_channel) ||
      !checked_add(per_channel, r.coverage_bytes_per_channel, per_channel) ||
      !checked_add(per_channel, r.result_record_bytes, per_channel) ||
      !checked_mul(per_channel, static_cast<std::size_t>(in.channels),
                   pixel_bytes) ||
      !checked_add(pixel_bytes, r.support_bytes_per_pixel, pixel_bytes) ||
      !checked_mul(frame_planes, sizeof(double), internal_bytes) ||
      !checked_mul(internal_bytes, static_cast<std::size_t>(in.channels),
                   internal_bytes) ||
      !checked_mul(internal_bytes, scale_sq, internal_bytes) ||
      !checked_add(pixel_bytes, internal_bytes, pixel_bytes))
    return out;
  out.device_bytes_per_target_pixel = pixel_bytes;

  // Device fixed roles: pipeline slots and reserve.
  std::size_t device_fixed = 0;
  if (!checked_mul(static_cast<std::size_t>(in.source_device_slots),
                   in.source_slot_device_bytes, device_fixed) ||
      !checked_mul(static_cast<std::size_t>(in.quality_device_slots),
                   in.quality_slot_device_bytes, tmp) ||
      !checked_add(device_fixed, tmp, device_fixed) ||
      !checked_add(device_fixed, in.device_reserve_bytes, device_fixed) ||
      device_fixed >= in.device_budget_bytes)
    return out;
  const std::size_t device_usable = in.device_budget_bytes - device_fixed;
  const std::size_t max_pixels = device_usable / pixel_bytes;

  // Minimum padded band height: one core row plus halo on both edges.
  std::size_t min_padded = 0;
  if (!checked_mul(static_cast<std::size_t>(in.halo_rows_per_band_edge), 2,
                   min_padded) ||
      !checked_add(min_padded, std::size_t{1}, min_padded))
    return out;

  // Full-width first; X-tiling only when one padded full-width row does not
  // fit. tile_cols never depends on frame_count.
  const std::size_t max_cols_for_band = max_pixels / min_padded;
  out.tile_cols = static_cast<int>(std::min<std::size_t>(
      static_cast<std::size_t>(in.target_width), max_cols_for_band));
  if (out.tile_cols <= 0) return out;
  out.x_tiled = out.tile_cols < in.target_width;
  const std::size_t device_rows = max_pixels / static_cast<std::size_t>(out.tile_cols);

  // Host pinned bytes per padded band row, bounded by slot counts.
  std::size_t pinned_per_row = 0;
  if (!checked_mul(static_cast<std::size_t>(in.pinned_source_slots),
                   in.pinned_source_bytes_per_frame_row, pinned_per_row) ||
      !checked_mul(static_cast<std::size_t>(in.pinned_quality_slots),
                   in.pinned_quality_bytes_per_frame_row, tmp) ||
      !checked_add(pinned_per_row, tmp, pinned_per_row))
    return out;
  std::size_t host_rows = std::numeric_limits<std::size_t>::max();
  if (pinned_per_row > 0) {
    if (in.host_fixed_bytes >= in.host_budget_bytes) return out;
    host_rows = (in.host_budget_bytes - in.host_fixed_bytes) / pinned_per_row;
  } else if (in.host_fixed_bytes > in.host_budget_bytes) {
    return out;
  }

  const std::size_t halo2 =
      2 * static_cast<std::size_t>(in.halo_rows_per_band_edge);
  std::size_t needed_rows = 0;
  if (!checked_add(static_cast<std::size_t>(in.target_height), halo2,
                   needed_rows))
    return out;
  const std::size_t padded_rows =
      std::min({device_rows, host_rows, needed_rows});
  if (padded_rows < min_padded ||
      padded_rows > static_cast<std::size_t>(
                        std::numeric_limits<int>::max()))
    return out;
  out.band_rows = static_cast<int>(padded_rows);
  out.band_core_rows = static_cast<int>(padded_rows - halo2);
  out.band_count = static_cast<int>(
      (static_cast<std::size_t>(in.target_height) +
       static_cast<std::size_t>(out.band_core_rows) - 1) /
      static_cast<std::size_t>(out.band_core_rows));
  // Conservative bound: every band is charged its full halo even though the
  // lifetime contract reuses shared edge rows.
  out.predicted_read_amplification =
      (static_cast<double>(in.target_height) +
       2.0 * static_cast<double>(in.halo_rows_per_band_edge) *
           static_cast<double>(out.band_count)) /
      static_cast<double>(in.target_height);

  std::size_t pixels = 0;
  if (!checked_mul(static_cast<std::size_t>(out.tile_cols), padded_rows,
                   pixels) ||
      !checked_mul(pixels, pixel_bytes, out.device_dynamic_bytes) ||
      !checked_add(device_fixed, out.device_dynamic_bytes,
                   out.device_peak_bytes) ||
      !checked_mul(pinned_per_row, padded_rows, out.host_pinned_bytes) ||
      !checked_add(in.host_fixed_bytes, out.host_pinned_bytes,
                   out.host_peak_bytes))
    return ForwardDrizzleV2MemoryPlan{};
  out.feasible = out.device_peak_bytes <= in.device_budget_bytes &&
                 out.host_peak_bytes <= in.host_budget_bytes &&
                 out.predicted_read_amplification <=
                     in.max_source_read_amplification &&
                 out.predicted_read_amplification <=
                     in.max_quality_read_amplification;
  return out;
}

}  // namespace tile_compile::reconstruction
