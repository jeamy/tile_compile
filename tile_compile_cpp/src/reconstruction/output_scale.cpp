#include "tile_compile/reconstruction/output_scale.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tile_compile::reconstruction {

ProfilePlane downsample_profile_plane_2x2(const ProfilePlane &in) {
  ProfilePlane out;
  if (in.empty()) return out;
  const int ow = in.width / 2;
  const int oh = in.height / 2;
  if (ow <= 0 || oh <= 0) return out;
  out.allocate(ow, oh);

  const int iw = in.width;
  auto at = [&](const std::vector<float> &v, int x, int y) -> double {
    return static_cast<double>(v[static_cast<size_t>(y) * iw + x]);
  };
  for (int oy = 0; oy < oh; ++oy) {
    for (int ox = 0; ox < ow; ++ox) {
      const int x0 = ox * 2, y0 = oy * 2;
      const size_t si[4] = {
          static_cast<size_t>(y0) * iw + x0, static_cast<size_t>(y0) * iw + x0 + 1,
          static_cast<size_t>(y0 + 1) * iw + x0, static_cast<size_t>(y0 + 1) * iw + x0 + 1};
      const bool all_valid = in.support[si[0]] && in.support[si[1]] &&
                             in.support[si[2]] && in.support[si[3]];
      const size_t oi = static_cast<size_t>(oy) * ow + ox;
      if (!all_valid) {
        // allocate() already left value=NaN, weight_sum/n_eff=0, support=0.
        continue;
      }
      const double v = 0.25 * (at(in.value, x0, y0) + at(in.value, x0 + 1, y0) +
                               at(in.value, x0, y0 + 1) + at(in.value, x0 + 1, y0 + 1));
      const double w = 0.25 * (at(in.weight_sum, x0, y0) + at(in.weight_sum, x0 + 1, y0) +
                               at(in.weight_sum, x0, y0 + 1) + at(in.weight_sum, x0 + 1, y0 + 1));
      const double ne = std::min(std::min(at(in.n_eff, x0, y0), at(in.n_eff, x0 + 1, y0)),
                                 std::min(at(in.n_eff, x0, y0 + 1), at(in.n_eff, x0 + 1, y0 + 1)));
      out.value[oi] = static_cast<float>(v);
      out.weight_sum[oi] = static_cast<float>(w);
      out.n_eff[oi] = static_cast<float>(ne);
      out.support[oi] = 1;
    }
  }
  return out;
}

namespace {
void copy_meta(ForwardDrizzleUniformResult &dst, const ForwardDrizzleUniformResult &src,
               int w, int h) {
  dst.color_mode = src.color_mode;
  dst.internal_width = w;
  dst.internal_height = h;
  dst.diagnostics = src.diagnostics;
}
ProfilePlane maybe_down(const ProfilePlane &p) {
  return p.empty() ? ProfilePlane{} : downsample_profile_plane_2x2(p);
}
}  // namespace

ForwardDrizzleUniformAndRawResult
downsample_uniform_and_raw_2x2(const ForwardDrizzleUniformAndRawResult &in) {
  ForwardDrizzleUniformAndRawResult out;
  out.diagnostics = in.diagnostics;
  out.clipping = in.clipping;

  const int ow = in.uniform.internal_width / 2;
  const int oh = in.uniform.internal_height / 2;

  auto down_profile = [&](ForwardDrizzleUniformResult &dst,
                          const ForwardDrizzleUniformResult &src) {
    if (src.internal_height <= 0) return;  // profile not emitted
    copy_meta(dst, src, ow, oh);
    dst.R = maybe_down(src.R);
    dst.G = maybe_down(src.G);
    dst.B = maybe_down(src.B);
    dst.L = maybe_down(src.L);
  };
  down_profile(out.uniform, in.uniform);
  down_profile(out.raw, in.raw);
  down_profile(out.fine, in.fine);
  down_profile(out.medium, in.medium);

  // Channel-min alpha-confidence maps (plan 14.4): a 2x2 block is supported
  // only if all four internal cells are (AND, like n_eff takes min), and the
  // block value is the MIN of the four --- 2x2-mean would not commute with the
  // channel-min these maps already encode, and min preserves the plan's
  // "confidence is never lifted" direction.
  if (!in.alpha_confidence_support.empty()) {
    const std::size_t on = static_cast<std::size_t>(ow) * oh;
    out.a_separation.assign(on, std::numeric_limits<float>::quiet_NaN());
    out.a_artifact.assign(on, std::numeric_limits<float>::quiet_NaN());
    out.a_registration.assign(on, std::numeric_limits<float>::quiet_NaN());
    out.alpha_confidence_support.assign(on, 0u);
    const int iw = in.uniform.internal_width;
    auto cell = [&](const std::vector<float> &m, int x, int y) {
      return static_cast<double>(m[static_cast<std::size_t>(y) * iw + x]);
    };
    for (int oy = 0; oy < oh; ++oy)
      for (int ox = 0; ox < ow; ++ox) {
        const int x0 = ox * 2, y0 = oy * 2;
        const std::size_t s00 = static_cast<std::size_t>(y0) * iw + x0;
        const std::size_t s01 = s00 + 1;
        const std::size_t s10 = s00 + iw;
        const std::size_t s11 = s10 + 1;
        if (!(in.alpha_confidence_support[s00] && in.alpha_confidence_support[s01] &&
              in.alpha_confidence_support[s10] && in.alpha_confidence_support[s11]))
          continue;
        const std::size_t oi = static_cast<std::size_t>(oy) * ow + ox;
        auto mn = [&](const std::vector<float> &m) {
          return static_cast<float>(std::min(
              std::min(cell(m, x0, y0), cell(m, x0 + 1, y0)),
              std::min(cell(m, x0, y0 + 1), cell(m, x0 + 1, y0 + 1))));
        };
        out.a_separation[oi] = mn(in.a_separation);
        out.a_artifact[oi] = mn(in.a_artifact);
        out.a_registration[oi] = mn(in.a_registration);
        out.alpha_confidence_support[oi] = 1u;
      }
  }
  return out;
}

namespace {
// overlap of interval [a, b] with [0, 1]
double overlap_unit(double a, double b) {
  return std::max(0.0, std::min(b, 1.0) - std::max(a, 0.0));
}
// K(0, x0) for a top-hat of side d centred at x0, integrated over cell [0,1]
double kernel_overlap(double x0, double d) {
  return overlap_unit(x0 - d / 2.0, x0 + d / 2.0);
}
}  // namespace

std::vector<double> kernel_noise_autocorrelation_1d(float pixfrac, int internal_scale,
                                                    int max_lag) {
  if (!(pixfrac > 0.0f) || pixfrac > 1.0f || (internal_scale != 1 && internal_scale != 2) ||
      max_lag < 0)
    throw std::invalid_argument("KERNEL_NOISE_INVALID_ARGS");
  const double d = static_cast<double>(pixfrac) * internal_scale;
  // Input samples at j + 0.5; a top-hat of side d reaches at most
  // ceil(d/2)+1 cells either way, so this range is generous.
  const int jr = static_cast<int>(std::ceil(d)) + max_lag + 3;

  auto s_at_lag = [&](int lag) {
    double s = 0.0;
    for (int j = -jr; j <= jr; ++j) {
      const double x0 = j + 0.5;
      s += kernel_overlap(x0, d) * kernel_overlap(x0 - lag, d);
    }
    return s;
  };
  const double s0 = s_at_lag(0);
  std::vector<double> rho(static_cast<size_t>(max_lag) + 1, 0.0);
  if (s0 <= 0.0) {
    rho[0] = 1.0;
    return rho;
  }
  for (int lag = 0; lag <= max_lag; ++lag) rho[static_cast<size_t>(lag)] = s_at_lag(lag) / s0;
  return rho;
}

double kernel_noise_correlation_sigma_factor(float pixfrac, int internal_scale) {
  if (!(pixfrac > 0.0f) || pixfrac > 1.0f || (internal_scale != 1 && internal_scale != 2))
    throw std::invalid_argument("KERNEL_NOISE_INVALID_ARGS");
  const double d = static_cast<double>(pixfrac) * internal_scale;
  const int jr = static_cast<int>(std::ceil(d)) + 4;
  double s0 = 0.0;
  for (int j = -jr; j <= jr; ++j) {
    const double k = kernel_overlap(j + 0.5, d);
    s0 += k * k;
  }
  // W = sum_k K(k, x0) = d for a top-hat of side d.
  return (s0 > 0.0) ? d / std::sqrt(s0) : 1.0;
}

namespace {

// Row-buffered 2x2 -> 1x adapter over a stripe stream. Internal-row stripes
// arrive contiguously in order; this buffers whole internal rows per plane
// field and flushes one output row for every complete even-aligned pair.
// Handles uniform/raw and, when present, the multiband fine/medium profiles
// (same 2x2 area-average, plan 12.1) plus the channel-min alpha-confidence
// maps (2x2 MIN + AND support, plan 14.4 --- 2x2-mean would not commute with
// the channel-min these already encode).
class Downsample2x2Adapter {
  const UniformAndRawStripeSink &out_;
  const int iw_;            // internal width
  const int ow_;            // output width = iw_/2
  const bool mono_;
  int next_internal_row_ = 0;  // index of the first buffered internal row
  int emitted_output_rows_ = 0;

  struct PlaneBuf {
    std::vector<float> value, weight_sum, n_eff;
    std::vector<uint8_t> support;
  };
  // profile*4 + ch: 0 uniform, 1 raw, 2 fine, 3 medium; ch 0=R 1=G 2=B 3=L.
  std::array<PlaneBuf, 16> buf_{};
  // alpha-confidence maps: separation/artifact/registration values + support.
  std::array<std::vector<float>, 3> conf_{};
  std::vector<uint8_t> conf_support_;
  bool have_fine_ = false, have_medium_ = false, have_conf_ = false;
  int buffered_rows_ = 0;

  static int slot(int profile, int ch) { return profile * 4 + ch; }
  static const ProfilePlane &pick(const ForwardDrizzleUniformResult &r, int ch) {
    return ch == 0 ? r.R : ch == 1 ? r.G : ch == 2 ? r.B : r.L;
  }
  static ProfilePlane &pick(ForwardDrizzleUniformResult &r, int ch) {
    return ch == 0 ? r.R : ch == 1 ? r.G : ch == 2 ? r.B : r.L;
  }

  void append_plane(int s, const ProfilePlane &p) {
    auto &b = buf_[static_cast<size_t>(s)];
    b.value.insert(b.value.end(), p.value.begin(), p.value.end());
    b.weight_sum.insert(b.weight_sum.end(), p.weight_sum.begin(), p.weight_sum.end());
    b.n_eff.insert(b.n_eff.end(), p.n_eff.begin(), p.n_eff.end());
    b.support.insert(b.support.end(), p.support.begin(), p.support.end());
  }
  void append_profile(int profile, const ForwardDrizzleUniformResult &src) {
    for (int ch = 0; ch < 4; ++ch)
      if ((ch == 3) == mono_) append_plane(slot(profile, ch), pick(src, ch));
  }
  static void erase_two(std::vector<float> &v, int iw) {
    v.erase(v.begin(), v.begin() + static_cast<size_t>(iw) * 2);
  }
  void erase_two_rows(int s) {
    auto &b = buf_[static_cast<size_t>(s)];
    erase_two(b.value, iw_);
    erase_two(b.weight_sum, iw_);
    erase_two(b.n_eff, iw_);
    b.support.erase(b.support.begin(), b.support.begin() + static_cast<size_t>(iw_) * 2);
  }
  void down_row(int s, ProfilePlane &dst) {
    const auto &b = buf_[static_cast<size_t>(s)];
    dst.allocate(ow_, 1);
    for (int ox = 0; ox < ow_; ++ox) {
      const int x0 = ox * 2;
      const size_t p00 = x0, p01 = x0 + 1, p10 = static_cast<size_t>(iw_) + x0,
                   p11 = static_cast<size_t>(iw_) + x0 + 1;
      if (!(b.support[p00] && b.support[p01] && b.support[p10] && b.support[p11])) continue;
      // Double intermediates + (00,01,10,11) order --- bit-identical to the
      // non-streaming downsample_profile_plane_2x2().
      dst.value[ox] = static_cast<float>(
          0.25 * (static_cast<double>(b.value[p00]) + static_cast<double>(b.value[p01]) +
                  static_cast<double>(b.value[p10]) + static_cast<double>(b.value[p11])));
      dst.weight_sum[ox] = static_cast<float>(
          0.25 * (static_cast<double>(b.weight_sum[p00]) + static_cast<double>(b.weight_sum[p01]) +
                  static_cast<double>(b.weight_sum[p10]) + static_cast<double>(b.weight_sum[p11])));
      dst.n_eff[ox] = static_cast<float>(
          std::min(std::min(static_cast<double>(b.n_eff[p00]), static_cast<double>(b.n_eff[p01])),
                   std::min(static_cast<double>(b.n_eff[p10]), static_cast<double>(b.n_eff[p11]))));
      dst.support[ox] = 1;
    }
  }

 public:
  Downsample2x2Adapter(const UniformAndRawStripeSink &out, int internal_width, bool mono)
      : out_(out), iw_(internal_width), ow_(internal_width / 2), mono_(mono) {}

  void feed(int y_begin, const ForwardDrizzleUniformAndRawResult &stripe) {
    if (y_begin != next_internal_row_ + buffered_rows_)
      throw std::runtime_error("DRIZZLE_2X2_NONCONTIGUOUS_STRIPE");
    const int rows = stripe.uniform.internal_height;
    append_profile(0, stripe.uniform);
    append_profile(1, stripe.raw);
    // The contiguity check above forces the first feed to have y_begin == 0
    // (initial next_internal_row_ + buffered_rows_ == 0), so this
    // which-planes latch always sees the first stripe.
    if (y_begin == 0) {
      have_fine_ = stripe.fine.internal_height > 0;
      have_medium_ = stripe.medium.internal_height > 0;
      have_conf_ = !stripe.alpha_confidence_support.empty();
    }
    if (have_fine_) append_profile(2, stripe.fine);
    if (have_medium_) append_profile(3, stripe.medium);
    if (have_conf_) {
      conf_[0].insert(conf_[0].end(), stripe.a_separation.begin(), stripe.a_separation.end());
      conf_[1].insert(conf_[1].end(), stripe.a_artifact.begin(), stripe.a_artifact.end());
      conf_[2].insert(conf_[2].end(), stripe.a_registration.begin(), stripe.a_registration.end());
      conf_support_.insert(conf_support_.end(), stripe.alpha_confidence_support.begin(),
                           stripe.alpha_confidence_support.end());
    }
    buffered_rows_ += rows;

    while (buffered_rows_ >= 2) {
      ForwardDrizzleUniformAndRawResult o;
      auto set_meta = [&](ForwardDrizzleUniformResult &r) {
        r.color_mode = stripe.uniform.color_mode;
        r.internal_width = ow_;
        r.internal_height = 1;
      };
      set_meta(o.uniform);
      set_meta(o.raw);
      if (have_fine_) set_meta(o.fine);
      if (have_medium_) set_meta(o.medium);
      for (int ch = 0; ch < 4; ++ch) {
        if ((ch == 3) != mono_) continue;
        down_row(slot(0, ch), pick(o.uniform, ch));
        down_row(slot(1, ch), pick(o.raw, ch));
        erase_two_rows(slot(0, ch));
        erase_two_rows(slot(1, ch));
        if (have_fine_) { down_row(slot(2, ch), pick(o.fine, ch)); erase_two_rows(slot(2, ch)); }
        if (have_medium_) { down_row(slot(3, ch), pick(o.medium, ch)); erase_two_rows(slot(3, ch)); }
      }
      if (have_conf_) {
        o.a_separation.assign(ow_, std::numeric_limits<float>::quiet_NaN());
        o.a_artifact.assign(ow_, std::numeric_limits<float>::quiet_NaN());
        o.a_registration.assign(ow_, std::numeric_limits<float>::quiet_NaN());
        o.alpha_confidence_support.assign(ow_, 0u);
        for (int ox = 0; ox < ow_; ++ox) {
          const int x0 = ox * 2;
          const size_t q00 = x0, q01 = x0 + 1, q10 = static_cast<size_t>(iw_) + x0,
                       q11 = static_cast<size_t>(iw_) + x0 + 1;
          if (!(conf_support_[q00] && conf_support_[q01] && conf_support_[q10] &&
                conf_support_[q11]))
            continue;
          for (int k = 0; k < 3; ++k) {
            const auto &c = conf_[static_cast<size_t>(k)];
            const double m = std::min(std::min(static_cast<double>(c[q00]),
                                               static_cast<double>(c[q01])),
                                      std::min(static_cast<double>(c[q10]),
                                               static_cast<double>(c[q11])));
            (k == 0 ? o.a_separation : k == 1 ? o.a_artifact : o.a_registration)[ox] =
                static_cast<float>(m);
          }
          o.alpha_confidence_support[ox] = 1u;
        }
        for (auto &c : conf_) erase_two(c, iw_);
        conf_support_.erase(conf_support_.begin(),
                            conf_support_.begin() + static_cast<size_t>(iw_) * 2);
      }
      out_(emitted_output_rows_, o);
      ++emitted_output_rows_;
      next_internal_row_ += 2;
      buffered_rows_ -= 2;
    }
  }
  void finish() {
    if (buffered_rows_ != 0)
      throw std::runtime_error("DRIZZLE_2X2_ODD_INTERNAL_HEIGHT");
  }
};

}  // namespace

ForwardDrizzlePairDiagnostics stream_forward_drizzle_uniform_and_raw_2x2(
    const registration::RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const UniformAndRawStripeSink &output_sink,
    const ForwardDrizzleSubdivisionParams &subdivision_params,
    const std::vector<float> &g_eff_by_source_index, size_t retained_bytes,
    const FrameQualityProvider &quality_of, const MultibandProfileParams &mb) {
  if (drizzle_cfg.internal_scale != 2)
    throw std::invalid_argument("DRIZZLE_2X2_REQUIRES_INTERNAL_SCALE_2");
  const int internal_width = plan.canvas_width_native * 2;
  const bool mono = plan.color_mode == ColorMode::MONO;
  Downsample2x2Adapter adapter(output_sink, internal_width, mono);

  auto diag = stream_forward_drizzle_uniform_and_raw(
      plan, source_of, drizzle_cfg, clipping_cfg,
      [&](int y, const ForwardDrizzleUniformAndRawResult &stripe) { adapter.feed(y, stripe); },
      subdivision_params, g_eff_by_source_index, retained_bytes, quality_of, mb);
  adapter.finish();
  return diag;
}

astrometry::WCS scale_wcs_to_output(const astrometry::WCS &in, const OutputWcsParams &p) {
  const double S = static_cast<double>(p.output_scale);
  astrometry::WCS out = in;

  const double crpix1_canvas = in.crpix1 + p.canvas_offset_x_native;
  const double crpix2_canvas = in.crpix2 + p.canvas_offset_y_native;
  out.crpix1 = S * (crpix1_canvas - 0.5) + 0.5 - p.crop_origin_x_out;
  out.crpix2 = S * (crpix2_canvas - 0.5) + 0.5 - p.crop_origin_y_out;

  out.cd1_1 = in.cd1_1 / S;
  out.cd1_2 = in.cd1_2 / S;
  out.cd2_1 = in.cd2_1 / S;
  out.cd2_2 = in.cd2_2 / S;
  return out;
}

}  // namespace tile_compile::reconstruction
