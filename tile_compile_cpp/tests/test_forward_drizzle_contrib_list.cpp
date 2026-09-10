// Plan 19.6 --- the deterministic forward-contribution-list rasterizer is the
// CPU reference for the ordering the CUDA path must obey. These tests pin that
// it reproduces the streaming Uniform reference
// (stream_forward_drizzle_uniform's wx/w/w2 accumulation) BIT-FOR-BIT across the
// plan-19.5 warp/CFA/chunk matrix, and that its pre-count / budget / canonical
// sort contracts hold.

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

WarpMatrix s2c(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

FrameSamplingTransform affine_frame(const std::string &id, size_t idx,
                                    const WarpMatrix &m) {
  FrameSamplingTransform f;
  f.frame_id = id;
  f.source_index = idx;
  f.valid = true;
  f.source_to_canvas = m;
  f.source_to_canvas_affine_valid = true;
  return f;
}

// Inline mirror of stream_forward_drizzle_uniform's per-stripe wx/w/w2
// accumulation (src/reconstruction/forward_drizzle.cpp) over one stripe.
DrizzleUniformAccum streaming_reference(
    const RegistrationSamplingPlan &plan, const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg, int y_begin, int rows,
    const ForwardDrizzleSubdivisionParams &sub) {
  const int scale = cfg.internal_scale;
  const int W = plan.canvas_width_native * scale;
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  const size_t n = static_cast<size_t>(W) * rows;
  DrizzleUniformAccum a;
  a.width = W;
  a.rows = rows;
  a.channels = channels;
  for (int c = 0; c < channels; ++c) {
    a.wx[c].assign(n, 0.0);
    a.w[c].assign(n, 0.0);
    a.w2[c].assign(n, 0.0);
  }
  const auto prepared = prepare_drizzle_frames(plan, cfg, sub);
  std::array<std::vector<double>, 3> A, B;
  for (int c = 0; c < channels; ++c) {
    A[c].assign(n, 0.0);
    B[c].assign(n, 0.0);
  }
  for (const auto *f : prepared.frames) {
    const Matrix2Df &src = source_of(f->source_index);
    for (int c = 0; c < channels; ++c) {
      std::fill(A[c].begin(), A[c].end(), 0.0);
      std::fill(B[c].begin(), B[c].end(), 0.0);
    }
    rasterize_drizzle_stripe(
        plan, *f, scale, cfg.pixfrac, y_begin, rows,
        [&](int sx, int sy, int c, int /*leaf*/, size_t i, double k) {
          const double v = static_cast<double>(src(sy, sx));
          if (std::isfinite(v)) {
            A[c][i] += k * v;
            B[c][i] += k;
          }
        },
        sub);
    for (int c = 0; c < channels; ++c)
      for (size_t i = 0; i < n; ++i)
        if (B[c][i] > 0.0) {
          a.wx[c][i] += A[c][i];
          a.w[c][i] += B[c][i];
          a.w2[c][i] += B[c][i] * B[c][i];
        }
  }
  return a;
}

void require_bit_identical(const DrizzleUniformAccum &ref,
                           const DrizzleUniformAccum &got) {
  REQUIRE(got.width == ref.width);
  REQUIRE(got.rows == ref.rows);
  REQUIRE(got.channels == ref.channels);
  size_t nonzero = 0;
  for (int c = 0; c < ref.channels; ++c) {
    REQUIRE(got.wx[c].size() == ref.wx[c].size());
    for (size_t i = 0; i < ref.wx[c].size(); ++i) {
      REQUIRE(got.wx[c][i] == ref.wx[c][i]);  // exact double compare
      REQUIRE(got.w[c][i] == ref.w[c][i]);
      REQUIRE(got.w2[c][i] == ref.w2[c][i]);
      if (ref.w[c][i] > 0.0) ++nonzero;
    }
  }
  REQUIRE(nonzero > 0);  // the case actually exercised some contributions
}

struct Case {
  RegistrationSamplingPlan plan;
  std::vector<Matrix2Df> imgs;
  config::ReconstructionDrizzleConfig cfg;
  ForwardDrizzleSubdivisionParams sub;
  SourceImageProvider provider() const {
    return [this](size_t i) -> const Matrix2Df & { return imgs.at(i); };
  }
};

Case make_case(ColorMode mode, bool with_local_warp,
               BayerPattern bayer = BayerPattern::RGGB,
               bool clip_at_edges = false) {
  Case k;
  auto &plan = k.plan;
  plan.source_width = 6;
  plan.source_height = 6;
  plan.canvas_width_native = 16;
  plan.canvas_height_native = 16;
  plan.color_mode = mode;
  if (mode == ColorMode::OSC) {
    plan.bayer_pattern = bayer;
    plan.cfa_origin_x = 0;
    plan.cfa_origin_y = 0;
  }
  // Three frames. Default: integer shift, sub-pixel shift, small rotation about
  // centre, all well inside the canvas. `clip_at_edges`: push two frames so the
  // droplets straddle x=0 / y=0 and x=W / y=H (exercises the clamp() paths in
  // rasterize_drizzle_stripe).
  const double base = clip_at_edges ? -1.5 : 5.0;
  const double far = clip_at_edges ? 13.5 : 5.0;  // 16 - ~2.5 for a 6px source
  plan.frames.push_back(affine_frame("f0", 0, s2c(1, 0, base, 0, 1, base)));
  plan.frames.push_back(affine_frame("f1", 1, s2c(1, 0, far + 0.37, 0, 1, far + 0.81)));
  const double ca = std::cos(0.06), sa = std::sin(0.06);
  const double rc = clip_at_edges ? 0.0 : 5.2;
  FrameSamplingTransform f2 =
      affine_frame("f2", 2, s2c(ca, -sa, rc - 3 * ca + 3 * sa, sa, ca,
                                rc + 0.1 - 3 * sa - 3 * ca));
  if (with_local_warp) {
    // A convergent zero-displacement local model: exercises the subdivision
    // leaf path (leaf_order > 0 possible) without changing the geometry
    // (coeff_x / coeff_y default to zero -> d(q) == 0, i.e. the affine seed).
    f2.has_smooth_local_model = true;
    f2.smooth_local_model.valid = true;
    f2.smooth_local_model.image_rows = plan.canvas_height_native;
    f2.smooth_local_model.image_cols = plan.canvas_width_native;
    f2.model_coordinate_scale = 1.0f;
  }
  plan.frames.push_back(f2);

  k.cfg.internal_scale = 1;
  k.cfg.pixfrac = 0.9f;
  k.cfg.kernel = "square";

  for (int fr = 0; fr < 3; ++fr) {
    Matrix2Df img(6, 6);
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 6; ++x)
        img(y, x) = 10.0f + fr * 3.0f + 2.0f * x + 1.5f * y +
                    7.0f * std::sin(0.5f * (x + y));
    k.imgs.push_back(std::move(img));
  }
  return k;
}

}  // namespace

TEST_CASE("plan-19.6 contrib list: bit-identical to the streaming Uniform "
          "reference across warp / CFA / edge cases",
          "[forward-drizzle][contrib-list]") {
  struct Variant { ColorMode mode; bool local; BayerPattern bayer; bool edge; };
  std::vector<Variant> variants;
  for (bool local : {false, true}) {
    variants.push_back({ColorMode::MONO, local, BayerPattern::RGGB, false});
    variants.push_back({ColorMode::MONO, local, BayerPattern::RGGB, true});
    // Plan 19.5 "alle Bayer-Pattern".
    for (BayerPattern bp : {BayerPattern::RGGB, BayerPattern::BGGR,
                            BayerPattern::GRBG, BayerPattern::GBRG}) {
      variants.push_back({ColorMode::OSC, local, bp, false});
      variants.push_back({ColorMode::OSC, local, bp, true});
    }
  }

  for (const auto &v : variants) {
    const Case k = make_case(v.mode, v.local, v.bayer, v.edge);
    const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
    const auto ref =
        streaming_reference(k.plan, k.provider(), k.cfg, 0, H, k.sub);
    const auto list =
        build_uniform_contrib_list(k.plan, k.provider(), k.cfg, 0, H, k.sub);
    REQUIRE(list.records.size() == list.predicted_count);
    REQUIRE(std::is_sorted(list.records.begin(), list.records.end(),
                           [](const DrizzleContrib &a, const DrizzleContrib &b) {
                             return contrib_key_less(a.key, b.key);
                           }));
    for (const auto &r : list.records) {
      REQUIRE(r.area > 0.0);
      REQUIRE(std::isfinite(r.value));
    }
    require_bit_identical(ref, reduce_uniform_contrib_list(list));

    // The per-frame production accumulator is bit-identical to the whole-stripe
    // build+sort+reduce (frame_order leads the key -> segments never span
    // frames).
    const auto by_frame =
        accumulate_uniform_by_frame(k.plan, k.provider(), k.cfg, 0, H, k.sub);
    require_bit_identical(ref, by_frame);
  }
}

TEST_CASE("plan-19.6 contrib list: splitting the canvas into stripes yields "
          "bit-identical accumulators", "[forward-drizzle][contrib-list]") {
  const Case k = make_case(ColorMode::OSC, /*with_local_warp=*/true);
  const int W = k.plan.canvas_width_native * k.cfg.internal_scale;
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
  const auto whole = reduce_uniform_contrib_list(
      build_uniform_contrib_list(k.plan, k.provider(), k.cfg, 0, H, k.sub));

  // Reassemble from three uneven stripes.
  DrizzleUniformAccum stitched;
  stitched.width = W;
  stitched.rows = H;
  stitched.channels = 3;
  for (int c = 0; c < 3; ++c) {
    stitched.wx[c].assign(static_cast<size_t>(W) * H, 0.0);
    stitched.w[c].assign(static_cast<size_t>(W) * H, 0.0);
    stitched.w2[c].assign(static_cast<size_t>(W) * H, 0.0);
  }
  for (auto [y0, rows] : std::vector<std::pair<int, int>>{{0, 5}, {5, 7}, {12, 4}}) {
    const auto part = reduce_uniform_contrib_list(build_uniform_contrib_list(
        k.plan, k.provider(), k.cfg, y0, rows, k.sub));
    for (int c = 0; c < 3; ++c)
      for (int ty = 0; ty < rows; ++ty)
        for (int x = 0; x < W; ++x) {
          const size_t src = static_cast<size_t>(ty) * W + x;
          const size_t dst = static_cast<size_t>(y0 + ty) * W + x;
          stitched.wx[c][dst] += part.wx[c][src];
          stitched.w[c][dst] += part.w[c][src];
          stitched.w2[c][dst] += part.w2[c][src];
        }
  }
  require_bit_identical(whole, stitched);
}

namespace {

void require_plane_identical(const ProfilePlane &ref, const ProfilePlane &got) {
  REQUIRE(got.width == ref.width);
  REQUIRE(got.height == ref.height);
  REQUIRE(got.value.size() == ref.value.size());
  for (size_t i = 0; i < ref.value.size(); ++i) {
    const bool nr = std::isnan(ref.value[i]), ng = std::isnan(got.value[i]);
    REQUIRE(nr == ng);
    if (!nr) REQUIRE(got.value[i] == ref.value[i]);
    REQUIRE(got.weight_sum[i] == ref.weight_sum[i]);
    REQUIRE(got.n_eff[i] == ref.n_eff[i]);
    REQUIRE(got.support[i] == ref.support[i]);
  }
}

void require_float_vec_identical(const std::vector<float> &ref,
                                 const std::vector<float> &got) {
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    const bool nr = std::isnan(ref[i]), ng = std::isnan(got[i]);
    REQUIRE(nr == ng);
    if (!nr) REQUIRE(got[i] == ref[i]);
  }
}

// A pair case: adds per-frame quality maps, a G_eff vector and an outlier frame
// so robust clipping actually engages.
struct PairCase {
  RegistrationSamplingPlan plan;
  std::vector<Matrix2Df> imgs, comp, s0, s1, art;
  std::vector<float> g_eff;
  config::ReconstructionDrizzleConfig cfg;
  config::ReconstructionClippingConfig clip;
  MultibandProfileParams mb;
  ForwardDrizzleSubdivisionParams sub;
  SourceImageProvider src() const {
    return [this](size_t i) -> const Matrix2Df & { return imgs.at(i); };
  }
  FrameQualityProvider quality() const {
    return [this](size_t i) -> FrameQualityMaps {
      return {&comp.at(i), &s0.at(i), &s1.at(i),
              i == 2 ? nullptr : &art.at(i)};  // frame 2 has no artifact map
    };
  }
};

PairCase make_pair_case(ColorMode mode, bool local, BayerPattern bayer,
                        bool edge) {
  PairCase k;
  auto &plan = k.plan;
  plan.source_width = 6;
  plan.source_height = 6;
  plan.canvas_width_native = 16;
  plan.canvas_height_native = 16;
  plan.color_mode = mode;
  if (mode == ColorMode::OSC) {
    plan.bayer_pattern = bayer;
    plan.cfa_origin_x = 0;
    plan.cfa_origin_y = 0;
  }
  const int nf = 5;
  const double b = edge ? -1.4 : 4.7;
  for (int i = 0; i < nf; ++i) {
    const double dx = b + ((i * 7) % 5) / 5.0;
    const double dy = b + ((i * 3) % 5) / 5.0;
    const double a = 0.015 * std::sin(0.4 * i);
    FrameSamplingTransform f = affine_frame(
        "f" + std::to_string(i), i,
        s2c(std::cos(a), -std::sin(a), dx, std::sin(a), std::cos(a), dy));
    if (local && i == nf - 1) {
      f.has_smooth_local_model = true;
      f.smooth_local_model.valid = true;
      f.smooth_local_model.image_rows = plan.canvas_height_native;
      f.smooth_local_model.image_cols = plan.canvas_width_native;
      f.model_coordinate_scale = 1.0f;
    }
    plan.frames.push_back(f);
  }

  for (int f = 0; f < nf; ++f) {
    Matrix2Df img(6, 6), c(6, 6), a0(6, 6), a1(6, 6), ar(6, 6);
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 6; ++x) {
        // Frame 3 is a bright outlier -> forces robust-clip rejections.
        const float outlier = (f == 3) ? 400.0f : 0.0f;
        img(y, x) = 30.0f + outlier + 3.0f * x + 2.0f * y +
                    9.0f * std::sin(0.6f * (x + y)) + 0.5f * f;
        c(y, x) = 0.4f + 0.1f * f;
        a0(y, x) = 0.55f + 0.07f * f;
        a1(y, x) = 0.6f + 0.05f * f;
        ar(y, x) = 0.9f;
      }
    k.imgs.push_back(std::move(img));
    k.comp.push_back(std::move(c));
    k.s0.push_back(std::move(a0));
    k.s1.push_back(std::move(a1));
    k.art.push_back(std::move(ar));
  }
  k.g_eff.assign(nf, 0.0f);
  for (int f = 0; f < nf; ++f) k.g_eff[f] = 0.6f + 0.08f * f;

  k.cfg.internal_scale = 1;
  k.cfg.pixfrac = 0.85f;
  k.cfg.kernel = "square";
  k.cfg.robust_passes = 2;
  k.cfg.min_clip_contributors = 3;
  k.clip.clip_sigma_low = 2.5f;
  k.clip.clip_sigma_high = 2.5f;
  k.clip.min_fraction = 0.1f;
  k.clip.min_n_eff = 1.0f;

  k.mb.emit_fine = true;
  k.mb.emit_medium = true;
  k.mb.emit_alpha_confidence = true;
  return k;
}

}  // namespace

TEST_CASE("plan-19.6 pair list: Uniform+Raw+Fine+Medium+alpha bit-identical to "
          "compute_forward_drizzle_uniform_and_raw (clipping engaged)",
          "[forward-drizzle][contrib-list]") {
  for (bool local : {false, true})
    for (bool edge : {false, true})
      for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
        for (BayerPattern bp : {BayerPattern::RGGB, BayerPattern::BGGR,
                                BayerPattern::GRBG, BayerPattern::GBRG}) {
          if (mode == ColorMode::MONO && bp != BayerPattern::RGGB) continue;
          const PairCase k = make_pair_case(mode, local, bp, edge);
          const int H = k.plan.canvas_height_native * k.cfg.internal_scale;

          const auto ref = compute_forward_drizzle_uniform_and_raw(
              k.plan, k.src(), k.cfg, k.clip, k.sub, k.g_eff, k.quality(), k.mb);
          const auto got = accumulate_pair_by_frame(
              k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
              k.mb);

          auto planes = [](const ForwardDrizzleUniformResult &r) {
            return r.color_mode == ColorMode::MONO
                       ? std::vector<const ProfilePlane *>{&r.L}
                       : std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
          };
          for (auto pr : {std::pair{&ref.uniform, &got.uniform},
                          std::pair{&ref.raw, &got.raw},
                          std::pair{&ref.fine, &got.fine},
                          std::pair{&ref.medium, &got.medium}}) {
            const auto rp = planes(*pr.first), gp = planes(*pr.second);
            REQUIRE(rp.size() == gp.size());
            for (size_t c = 0; c < rp.size(); ++c)
              require_plane_identical(*rp[c], *gp[c]);
          }
          require_float_vec_identical(ref.a_separation, got.a_separation);
          require_float_vec_identical(ref.a_artifact, got.a_artifact);
          require_float_vec_identical(ref.a_registration, got.a_registration);
          REQUIRE(got.alpha_confidence_support == ref.alpha_confidence_support);

          // The clipping counters are the most ordering-sensitive outputs.
          REQUIRE(got.clipping.pixel_channel_evaluations ==
                  ref.clipping.pixel_channel_evaluations);
          REQUIRE(got.clipping.pixel_channel_rejected ==
                  ref.clipping.pixel_channel_rejected);
          REQUIRE(got.clipping.candidate_contributions_clipped ==
                  ref.clipping.candidate_contributions_clipped);
          // Clipping actually engaged somewhere in this matrix.
          REQUIRE(ref.clipping.candidate_contributions_clipped > 0);
        }
}

TEST_CASE("plan-19.6 pair list: stripe splitting leaves the clipped profiles, "
          "alpha maps and clipping counters bit-identical",
          "[forward-drizzle][contrib-list]") {
  const PairCase k = make_pair_case(ColorMode::OSC, /*local=*/true,
                                    BayerPattern::GRBG, /*edge=*/true);
  const int W = k.plan.canvas_width_native * k.cfg.internal_scale;
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;

  const auto whole = accumulate_pair_by_frame(
      k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb);

  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  auto mut_planes = [](ForwardDrizzleUniformResult &r) {
    return std::vector<ProfilePlane *>{&r.R, &r.G, &r.B};
  };

  // Stitch three uneven stripes into a full-size result.
  ForwardDrizzleUniformAndRawResult stitched;
  auto init = [&](ForwardDrizzleUniformResult &p, bool on) {
    p.color_mode = ColorMode::OSC;
    if (!on) return;
    p.R.allocate(W, H); p.G.allocate(W, H); p.B.allocate(W, H);
  };
  init(stitched.uniform, true); init(stitched.raw, true);
  init(stitched.fine, true); init(stitched.medium, true);
  const size_t N = static_cast<size_t>(W) * H;
  stitched.a_separation.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_artifact.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_registration.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.alpha_confidence_support.assign(N, 0u);

  for (auto [y0, rows] : std::vector<std::pair<int, int>>{
           {0, 5}, {5, 7}, {12, H - 12}}) {
    const auto part = accumulate_pair_by_frame(k.plan, k.src(), k.cfg, k.clip,
                                               y0, rows, k.sub, k.g_eff,
                                               k.quality(), k.mb);
    stitched.clipping.pixel_channel_evaluations +=
        part.clipping.pixel_channel_evaluations;
    stitched.clipping.pixel_channel_rejected +=
        part.clipping.pixel_channel_rejected;
    stitched.clipping.candidate_contributions_clipped +=
        part.clipping.candidate_contributions_clipped;
    for (auto pr : {std::pair{&stitched.uniform, &part.uniform},
                    std::pair{&stitched.raw, &part.raw},
                    std::pair{&stitched.fine, &part.fine},
                    std::pair{&stitched.medium, &part.medium}}) {
      auto dp = mut_planes(*pr.first);
      auto sp = planes(*pr.second);
      for (int c = 0; c < 3; ++c)
        for (int ty = 0; ty < rows; ++ty)
          for (int x = 0; x < W; ++x) {
            const size_t s = static_cast<size_t>(ty) * W + x;
            const size_t d = static_cast<size_t>(y0 + ty) * W + x;
            dp[c]->value[d] = sp[c]->value[s];
            dp[c]->weight_sum[d] = sp[c]->weight_sum[s];
            dp[c]->n_eff[d] = sp[c]->n_eff[s];
            dp[c]->support[d] = sp[c]->support[s];
          }
    }
    for (int ty = 0; ty < rows; ++ty)
      for (int x = 0; x < W; ++x) {
        const size_t s = static_cast<size_t>(ty) * W + x;
        const size_t d = static_cast<size_t>(y0 + ty) * W + x;
        stitched.a_separation[d] = part.a_separation[s];
        stitched.a_artifact[d] = part.a_artifact[s];
        stitched.a_registration[d] = part.a_registration[s];
        stitched.alpha_confidence_support[d] = part.alpha_confidence_support[s];
      }
  }

  for (auto pr : {std::pair{&whole.uniform, &stitched.uniform},
                  std::pair{&whole.raw, &stitched.raw},
                  std::pair{&whole.fine, &stitched.fine},
                  std::pair{&whole.medium, &stitched.medium}}) {
    const auto wp = planes(*pr.first), sp2 = planes(*pr.second);
    for (int c = 0; c < 3; ++c) require_plane_identical(*wp[c], *sp2[c]);
  }
  require_float_vec_identical(whole.a_separation, stitched.a_separation);
  require_float_vec_identical(whole.a_artifact, stitched.a_artifact);
  require_float_vec_identical(whole.a_registration, stitched.a_registration);
  REQUIRE(whole.alpha_confidence_support == stitched.alpha_confidence_support);
  REQUIRE(whole.clipping.pixel_channel_evaluations ==
          stitched.clipping.pixel_channel_evaluations);
  REQUIRE(whole.clipping.pixel_channel_rejected ==
          stitched.clipping.pixel_channel_rejected);
  REQUIRE(whole.clipping.candidate_contributions_clipped ==
          stitched.clipping.candidate_contributions_clipped);
}

TEST_CASE("§30.81 pair list: 2D row-band x column-tile splitting leaves the "
          "clipped profiles, alpha maps and clipping counters bit-identical",
          "[forward-drizzle][contrib-list][fd-tile-window]") {
  const PairCase k = make_pair_case(ColorMode::OSC, /*local=*/true,
                                    BayerPattern::GRBG, /*edge=*/true);
  const int W = k.plan.canvas_width_native * k.cfg.internal_scale;  // 16
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;  // 16

  const auto whole = accumulate_pair_by_frame(
      k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb);

  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  auto mut_planes = [](ForwardDrizzleUniformResult &r) {
    return std::vector<ProfilePlane *>{&r.R, &r.G, &r.B};
  };

  ForwardDrizzleUniformAndRawResult stitched;
  auto init = [&](ForwardDrizzleUniformResult &p, bool on) {
    p.color_mode = ColorMode::OSC;
    if (!on) return;
    p.R.allocate(W, H); p.G.allocate(W, H); p.B.allocate(W, H);
  };
  init(stitched.uniform, true); init(stitched.raw, true);
  init(stitched.fine, true); init(stitched.medium, true);
  const size_t N = static_cast<size_t>(W) * H;
  stitched.a_separation.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_artifact.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_registration.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.alpha_confidence_support.assign(N, 0u);

  // ragged row bands x ragged column tiles (last tile is width 1: W % 15 == 1)
  const std::vector<std::pair<int, int>> ybands{{0, 5}, {5, 7}, {12, H - 12}};
  const std::vector<std::pair<int, int>> xtiles{{0, 3}, {3, 8}, {11, 4}, {15, 1}};
  for (auto [y0, rows] : ybands)
    for (auto [x0, cols] : xtiles) {
      const auto part = accumulate_pair_by_frame(
          k.plan, k.src(), k.cfg, k.clip, y0, rows, k.sub, k.g_eff, k.quality(),
          k.mb, static_cast<std::size_t>(1) << 32, x0, cols);
      REQUIRE(part.uniform.internal_width == cols);
      // counters sum over the disjoint 2D partition
      stitched.clipping.pixel_channel_evaluations +=
          part.clipping.pixel_channel_evaluations;
      stitched.clipping.pixel_channel_rejected +=
          part.clipping.pixel_channel_rejected;
      stitched.clipping.candidate_contributions_clipped +=
          part.clipping.candidate_contributions_clipped;
      for (auto pr : {std::pair{&stitched.uniform, &part.uniform},
                      std::pair{&stitched.raw, &part.raw},
                      std::pair{&stitched.fine, &part.fine},
                      std::pair{&stitched.medium, &part.medium}}) {
        auto dp = mut_planes(*pr.first);
        auto sp = planes(*pr.second);
        for (int c = 0; c < 3; ++c)
          for (int ty = 0; ty < rows; ++ty)
            for (int x = 0; x < cols; ++x) {
              const size_t s = static_cast<size_t>(ty) * cols + x;
              const size_t d = static_cast<size_t>(y0 + ty) * W + (x0 + x);
              dp[c]->value[d] = sp[c]->value[s];
              dp[c]->weight_sum[d] = sp[c]->weight_sum[s];
              dp[c]->n_eff[d] = sp[c]->n_eff[s];
              dp[c]->support[d] = sp[c]->support[s];
            }
      }
      for (int ty = 0; ty < rows; ++ty)
        for (int x = 0; x < cols; ++x) {
          const size_t s = static_cast<size_t>(ty) * cols + x;
          const size_t d = static_cast<size_t>(y0 + ty) * W + (x0 + x);
          stitched.a_separation[d] = part.a_separation[s];
          stitched.a_artifact[d] = part.a_artifact[s];
          stitched.a_registration[d] = part.a_registration[s];
          stitched.alpha_confidence_support[d] =
              part.alpha_confidence_support[s];
        }
    }

  for (auto pr : {std::pair{&whole.uniform, &stitched.uniform},
                  std::pair{&whole.raw, &stitched.raw},
                  std::pair{&whole.fine, &stitched.fine},
                  std::pair{&whole.medium, &stitched.medium}}) {
    const auto wp = planes(*pr.first), sp2 = planes(*pr.second);
    for (int c = 0; c < 3; ++c) require_plane_identical(*wp[c], *sp2[c]);
  }
  require_float_vec_identical(whole.a_separation, stitched.a_separation);
  require_float_vec_identical(whole.a_artifact, stitched.a_artifact);
  require_float_vec_identical(whole.a_registration, stitched.a_registration);
  REQUIRE(whole.alpha_confidence_support == stitched.alpha_confidence_support);
  REQUIRE(whole.clipping.pixel_channel_evaluations ==
          stitched.clipping.pixel_channel_evaluations);
  REQUIRE(whole.clipping.pixel_channel_rejected ==
          stitched.clipping.pixel_channel_rejected);
  REQUIRE(whole.clipping.candidate_contributions_clipped ==
          stitched.clipping.candidate_contributions_clipped);
}

TEST_CASE("§30.81 step-5 (B): the tiled driver (produce+sort each frame once "
          "per band, replay through column tiles) is bit-identical to the "
          "whole-width build",
          "[forward-drizzle][contrib-list][fd-tile-window]") {
  const PairCase k = make_pair_case(ColorMode::OSC, /*local=*/false,
                                    BayerPattern::RGGB, /*edge=*/false);
  const int W = k.plan.canvas_width_native * k.cfg.internal_scale;
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;

  const auto whole = accumulate_pair_by_frame(
      k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb);

  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  auto mut_planes = [](ForwardDrizzleUniformResult &r) {
    return std::vector<ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  ForwardDrizzleUniformAndRawResult stitched;
  for (auto *p : {&stitched.uniform, &stitched.raw, &stitched.fine,
                  &stitched.medium}) {
    p->R.allocate(W, H); p->G.allocate(W, H); p->B.allocate(W, H);
  }
  const size_t N = static_cast<size_t>(W) * H;
  stitched.a_separation.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_artifact.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.a_registration.assign(N, std::numeric_limits<float>::quiet_NaN());
  stitched.alpha_confidence_support.assign(N, 0u);

  int tiles_seen = 0;
  PairTileSink sink = [&](int xb, int tw,
                          const ForwardDrizzleUniformAndRawResult &part) {
    ++tiles_seen;
    REQUIRE(part.uniform.internal_width == tw);
    for (auto pr : {std::pair{&stitched.uniform, &part.uniform},
                    std::pair{&stitched.raw, &part.raw},
                    std::pair{&stitched.fine, &part.fine},
                    std::pair{&stitched.medium, &part.medium}}) {
      auto dp = mut_planes(*pr.first);
      auto sp = planes(*pr.second);
      for (int c = 0; c < 3; ++c)
        for (int ty = 0; ty < H; ++ty)
          for (int x = 0; x < tw; ++x) {
            const size_t s = static_cast<size_t>(ty) * tw + x;
            const size_t d = static_cast<size_t>(ty) * W + (xb + x);
            dp[c]->value[d] = sp[c]->value[s];
            dp[c]->weight_sum[d] = sp[c]->weight_sum[s];
            dp[c]->n_eff[d] = sp[c]->n_eff[s];
            dp[c]->support[d] = sp[c]->support[s];
          }
    }
    for (int ty = 0; ty < H; ++ty)
      for (int x = 0; x < tw; ++x) {
        const size_t s = static_cast<size_t>(ty) * tw + x;
        const size_t d = static_cast<size_t>(ty) * W + (xb + x);
        stitched.a_separation[d] = part.a_separation[s];
        stitched.a_artifact[d] = part.a_artifact[s];
        stitched.a_registration[d] = part.a_registration[s];
        stitched.alpha_confidence_support[d] = part.alpha_confidence_support[s];
      }
  };

  const int tile_cols = 5;  // W=16 -> tiles [0,5)[5,10)[10,15)[15,16)
  const auto agg = accumulate_pair_by_frame(
      k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb,
      static_cast<std::size_t>(1) << 32, 0, -1, &sink, tile_cols);
  REQUIRE(tiles_seen == 4);
  REQUIRE(agg.uniform.R.value.empty());  // aggregate carries only .clipping

  for (auto pr : {std::pair{&whole.uniform, &stitched.uniform},
                  std::pair{&whole.raw, &stitched.raw},
                  std::pair{&whole.fine, &stitched.fine},
                  std::pair{&whole.medium, &stitched.medium}}) {
    const auto wp = planes(*pr.first), sp2 = planes(*pr.second);
    for (int c = 0; c < 3; ++c) require_plane_identical(*wp[c], *sp2[c]);
  }
  require_float_vec_identical(whole.a_separation, stitched.a_separation);
  require_float_vec_identical(whole.a_artifact, stitched.a_artifact);
  require_float_vec_identical(whole.a_registration, stitched.a_registration);
  REQUIRE(whole.alpha_confidence_support == stitched.alpha_confidence_support);
  REQUIRE(agg.clipping.pixel_channel_evaluations ==
          whole.clipping.pixel_channel_evaluations);
  REQUIRE(agg.clipping.pixel_channel_rejected ==
          whole.clipping.pixel_channel_rejected);
  REQUIRE(agg.clipping.candidate_contributions_clipped ==
          whole.clipping.candidate_contributions_clipped);
}

TEST_CASE("plan-19.5 pair parity: the CUDA affine rasterizer path is "
          "bit-identical to the CPU pair reference",
          "[forward-drizzle][contrib-list][cuda-parity]") {
  if (forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- parity check skipped");
    return;
  }
  int checked_variants = 0;
  for (bool edge : {false, true})
    for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
      for (BayerPattern bp : {BayerPattern::RGGB, BayerPattern::BGGR,
                              BayerPattern::GRBG, BayerPattern::GBRG}) {
        if (mode == ColorMode::MONO && bp != BayerPattern::RGGB) continue;
        // Affine frames only on the CUDA path.
        const PairCase k = make_pair_case(mode, /*local=*/false, bp, edge);
        const int H = k.plan.canvas_height_native * k.cfg.internal_scale;

        const auto cpu = accumulate_pair_by_frame(
            k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
            k.mb);
        const auto gpu = accumulate_pair_by_frame_cuda(
            k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
            k.mb);

        auto planes = [](const ForwardDrizzleUniformResult &r) {
          return r.color_mode == ColorMode::MONO
                     ? std::vector<const ProfilePlane *>{&r.L}
                     : std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
        };
        for (auto pr : {std::pair{&cpu.uniform, &gpu.uniform},
                        std::pair{&cpu.raw, &gpu.raw},
                        std::pair{&cpu.fine, &gpu.fine},
                        std::pair{&cpu.medium, &gpu.medium}}) {
          const auto cp = planes(*pr.first), gp = planes(*pr.second);
          REQUIRE(cp.size() == gp.size());
          for (size_t c = 0; c < cp.size(); ++c)
            require_plane_identical(*cp[c], *gp[c]);
        }
        require_float_vec_identical(cpu.a_separation, gpu.a_separation);
        require_float_vec_identical(cpu.a_artifact, gpu.a_artifact);
        require_float_vec_identical(cpu.a_registration, gpu.a_registration);
        REQUIRE(cpu.alpha_confidence_support == gpu.alpha_confidence_support);
        // The clipping counters would diverge first on any ClipCandidate or
        // frame-order mismatch (reduce_pixel_profiles is shared, so identical
        // counters + profiles == identical candidates).
        REQUIRE(gpu.clipping.pixel_channel_evaluations ==
                cpu.clipping.pixel_channel_evaluations);
        REQUIRE(gpu.clipping.pixel_channel_rejected ==
                cpu.clipping.pixel_channel_rejected);
        REQUIRE(gpu.clipping.candidate_contributions_clipped ==
                cpu.clipping.candidate_contributions_clipped);
        REQUIRE(cpu.clipping.candidate_contributions_clipped > 0);
        ++checked_variants;
      }
  REQUIRE(checked_variants == 10);
}

TEST_CASE("plan-19.6.2 hybrid: a local-warp frame on the CUDA path (CPU "
          "geometry -> GPU rasterization) is bit-identical to the CPU pair "
          "reference, at any leaf-batch size and any stripe split",
          "[forward-drizzle][contrib-list][cuda-parity]") {
  if (forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- skipped");
    return;
  }
  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return r.color_mode == ColorMode::MONO
               ? std::vector<const ProfilePlane *>{&r.L}
               : std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  auto require_same = [&](const ForwardDrizzleUniformAndRawResult &cpu,
                          const ForwardDrizzleUniformAndRawResult &gpu) {
    for (auto pr : {std::pair{&cpu.uniform, &gpu.uniform},
                    std::pair{&cpu.raw, &gpu.raw},
                    std::pair{&cpu.fine, &gpu.fine},
                    std::pair{&cpu.medium, &gpu.medium}}) {
      const auto cp = planes(*pr.first), gp = planes(*pr.second);
      REQUIRE(cp.size() == gp.size());
      for (size_t c = 0; c < cp.size(); ++c)
        require_plane_identical(*cp[c], *gp[c]);
    }
    require_float_vec_identical(cpu.a_separation, gpu.a_separation);
    require_float_vec_identical(cpu.a_artifact, gpu.a_artifact);
    require_float_vec_identical(cpu.a_registration, gpu.a_registration);
    REQUIRE(cpu.alpha_confidence_support == gpu.alpha_confidence_support);
    REQUIRE(gpu.clipping.pixel_channel_evaluations ==
            cpu.clipping.pixel_channel_evaluations);
    REQUIRE(gpu.clipping.pixel_channel_rejected ==
            cpu.clipping.pixel_channel_rejected);
    REQUIRE(gpu.clipping.candidate_contributions_clipped ==
            cpu.clipping.candidate_contributions_clipped);
  };

  int checked = 0;
  for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC})
    for (bool edge : {false, true}) {
      const PairCase k = make_pair_case(mode, /*local=*/true,
                                        BayerPattern::RGGB, edge);
      const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
      const auto cpu = accumulate_pair_by_frame(
          k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
          k.mb);

      // Whole stripe, several leaf-batch caps. A cap of 1/7/64 forces a GPU
      // polygon-area flush mid-frame, mid-leaf; the records are order-sorted
      // downstream so the result must not move.
      for (std::size_t batch : {std::size_t{1}, std::size_t{7}, std::size_t{64},
                                std::size_t{1} << 20}) {
        const auto gpu = accumulate_pair_by_frame_cuda(
            k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
            k.mb, static_cast<std::size_t>(1) << 32, 32, batch);
        require_same(cpu, gpu);
        ++checked;
      }
      REQUIRE(cpu.clipping.candidate_contributions_clipped > 0);  // non-vacuous

      // Uneven stripe split with a tiny batch cap: both new order-dependence
      // sources (flush boundaries, band boundaries) exercised at once.
      const int y0 = H / 3, y1 = H - H / 4;
      ForwardDrizzleClippingDiagnostics split_clip;
      for (auto seg : {std::pair{0, y0}, std::pair{y0, y1 - y0},
                       std::pair{y1, H - y1}}) {
        const auto part = accumulate_pair_by_frame_cuda(
            k.plan, k.src(), k.cfg, k.clip, seg.first, seg.second, k.sub,
            k.g_eff, k.quality(), k.mb, static_cast<std::size_t>(1) << 32, 32,
            std::size_t{5});
        split_clip.pixel_channel_evaluations +=
            part.clipping.pixel_channel_evaluations;
        split_clip.pixel_channel_rejected +=
            part.clipping.pixel_channel_rejected;
        split_clip.candidate_contributions_clipped +=
            part.clipping.candidate_contributions_clipped;
      }
      REQUIRE(split_clip.pixel_channel_evaluations ==
              cpu.clipping.pixel_channel_evaluations);
      REQUIRE(split_clip.pixel_channel_rejected ==
              cpu.clipping.pixel_channel_rejected);
      REQUIRE(split_clip.candidate_contributions_clipped ==
              cpu.clipping.candidate_contributions_clipped);
    }
  REQUIRE(checked == 2 * 2 * 4);  // modes x edge x batch caps
}

TEST_CASE("plan 11.14 P2 x CUDA: a published geometry cache keeps the "
          "CPU-vs-CUDA-hybrid pair parity byte-identical for a local-warp frame",
          "[forward-drizzle][contrib-list][cuda-parity][geometry-cache]") {
  if (forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- skipped");
    return;
  }
  namespace fs = std::filesystem;
  using namespace tile_compile::reconstruction;

  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return r.color_mode == ColorMode::MONO
               ? std::vector<const ProfilePlane *>{&r.L}
               : std::vector<const ProfilePlane *>{&r.R, &r.G, &r.B};
  };
  auto require_same = [&](const ForwardDrizzleUniformAndRawResult &a,
                          const ForwardDrizzleUniformAndRawResult &b) {
    for (auto pr : {std::pair{&a.uniform, &b.uniform},
                    std::pair{&a.raw, &b.raw}, std::pair{&a.fine, &b.fine},
                    std::pair{&a.medium, &b.medium}}) {
      const auto ap = planes(*pr.first), bp = planes(*pr.second);
      REQUIRE(ap.size() == bp.size());
      for (size_t c = 0; c < ap.size(); ++c)
        require_plane_identical(*ap[c], *bp[c]);
    }
    require_float_vec_identical(a.a_separation, b.a_separation);
    require_float_vec_identical(a.a_artifact, b.a_artifact);
    require_float_vec_identical(a.a_registration, b.a_registration);
    REQUIRE(a.alpha_confidence_support == b.alpha_confidence_support);
    REQUIRE(a.clipping.pixel_channel_evaluations ==
            b.clipping.pixel_channel_evaluations);
    REQUIRE(a.clipping.candidate_contributions_clipped ==
            b.clipping.candidate_contributions_clipped);
  };

  for (ColorMode mode : {ColorMode::MONO, ColorMode::OSC}) {
    const PairCase k =
        make_pair_case(mode, /*local=*/true, BayerPattern::RGGB, /*edge=*/false);
    const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
    const std::size_t batch = 7;

    // No-cache baselines (this is the existing [cuda-parity] combination).
    const auto cpu0 = accumulate_pair_by_frame(k.plan, k.src(), k.cfg, k.clip, 0,
                                               H, k.sub, k.g_eff, k.quality(),
                                               k.mb);
    const auto gpu0 = accumulate_pair_by_frame_cuda(
        k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb,
        static_cast<std::size_t>(1) << 32, 32, batch);

    // Build + publish the geometry cache for this plan's local-warp frame.
    const fs::path root =
        fs::temp_directory_path() /
        ("tc_gc_cudaparity_" + std::to_string(std::random_device{}()));
    fs::remove_all(root);
    std::vector<GeometryVariant> vars{{k.cfg.pixfrac}};
    std::vector<std::size_t> local_idx;
    for (const auto &fr : k.plan.frames)
      if (fr.has_smooth_local_model) local_idx.push_back(fr.source_index);
    const auto built = build_drizzle_geometry_cache(root, k.plan, k.cfg, vars,
                                                    k.sub, 64ull << 20);
    DrizzleGeometryCacheReader reader(root, built.identities, local_idx);
    {
      ScopedActiveGeometryCache guard(&reader);
      const auto cpu1 = accumulate_pair_by_frame(k.plan, k.src(), k.cfg, k.clip,
                                                 0, H, k.sub, k.g_eff,
                                                 k.quality(), k.mb);
      const auto gpu1 = accumulate_pair_by_frame_cuda(
          k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(),
          k.mb, static_cast<std::size_t>(1) << 32, 32, batch);
      require_same(cpu0, cpu1);  // cache does not move the CPU reference
      require_same(gpu0, gpu1);  // cache does not move the CUDA-hybrid result
      require_same(cpu1, gpu1);  // parity still holds WITH the cache active
    }
    fs::remove_all(root);
  }
}

TEST_CASE("plan-19.6.2 hybrid: a subdivided local-warp frame (leaf_order > 0) "
          "is bit-identical CPU vs the CUDA hybrid path",
          "[forward-drizzle][contrib-list][cuda-parity]") {
  if (forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device -- skipped");
    return;
  }
  PairCase k = make_pair_case(ColorMode::MONO, /*local=*/true,
                              BayerPattern::RGGB, /*edge=*/false);
  // Give the local-warp frame a curved (second-order) displacement: coeffs
  // that vary quadratically across the 4x4 Gauss grid so the warp deviates
  // from bilinear over a droplet -> adaptive subdivision splits it -> the
  // (sx, sy) sample yields leaves 0..n-1, not a constant 0.
  auto &lf = k.plan.frames.back();
  REQUIRE(lf.has_smooth_local_model);
  for (int gy = 0; gy < 4; ++gy)
    for (int gx = 0; gx < 4; ++gx) {
      const float ux = gx / 3.0f - 0.5f, uy = gy / 3.0f - 0.5f;
      lf.smooth_local_model.coeff_x[gy * 4 + gx] = 60.0f * (ux * ux - uy * uy);
      lf.smooth_local_model.coeff_y[gy * 4 + gx] = 60.0f * (ux * uy);
    }
  k.sub.max_subdivision_depth = 2;
  k.sub.position_epsilon_internal_px = 0.05f;
  k.sub.area_relative_epsilon = 0.005f;
  k.sub.per_frame_inversion_error_rate_max = 0.95f;

  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;

  int max_leaf = 0;
  bool local_seen = false;
  for (const auto &f : k.plan.frames)
    if (f.has_smooth_local_model)
      rasterize_drizzle_stripe(
          k.plan, f, k.cfg.internal_scale, k.cfg.pixfrac, 0, H,
          [&](int, int, int, int leaf, size_t, double) {
            local_seen = true;
            max_leaf = std::max(max_leaf, leaf);
          },
          k.sub);
  REQUIRE(local_seen);       // the frame was not excluded whole
  REQUIRE(max_leaf > 0);     // it actually subdivided

  const auto cpu = accumulate_pair_by_frame(
      k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb);
  auto planes = [](const ForwardDrizzleUniformResult &r) {
    return std::vector<const ProfilePlane *>{&r.L};
  };
  for (std::size_t batch : {std::size_t{1}, std::size_t{9},
                            std::size_t{1} << 20}) {
    const auto gpu = accumulate_pair_by_frame_cuda(
        k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub, k.g_eff, k.quality(), k.mb,
        static_cast<std::size_t>(1) << 32, 32, batch);
    for (auto pr : {std::pair{&cpu.uniform, &gpu.uniform},
                    std::pair{&cpu.raw, &gpu.raw},
                    std::pair{&cpu.fine, &gpu.fine},
                    std::pair{&cpu.medium, &gpu.medium}})
      require_plane_identical(*planes(*pr.first)[0], *planes(*pr.second)[0]);
    REQUIRE(gpu.clipping.candidate_contributions_clipped ==
            cpu.clipping.candidate_contributions_clipped);
    REQUIRE(gpu.clipping.pixel_channel_rejected ==
            cpu.clipping.pixel_channel_rejected);
  }
}

TEST_CASE("plan-19.6 pair list: the candidate buffer is budgeted before "
          "allocation", "[forward-drizzle][contrib-list]") {
  const PairCase k = make_pair_case(ColorMode::OSC, false, BayerPattern::RGGB,
                                    false);
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
  REQUIRE_THROWS_WITH(
      accumulate_pair_by_frame(k.plan, k.src(), k.cfg, k.clip, 0, H, k.sub,
                               k.g_eff, k.quality(), k.mb,
                               /*mem_budget_bytes=*/512),
      Catch::Matchers::ContainsSubstring("DRIZZLE_CONTRIB_LIST_BUDGET"));
  REQUIRE_NOTHROW(accumulate_pair_by_frame(k.plan, k.src(), k.cfg, k.clip, 0, H,
                                           k.sub, k.g_eff, k.quality(), k.mb,
                                           static_cast<size_t>(1) << 30));
}

TEST_CASE("plan-19.6 contrib list: pre-plan rejects an oversized list before "
          "materialising", "[forward-drizzle][contrib-list]") {
  const Case k = make_case(ColorMode::MONO, /*with_local_warp=*/false);
  const int H = k.plan.canvas_height_native * k.cfg.internal_scale;
  REQUIRE_THROWS_WITH(
      build_uniform_contrib_list(k.plan, k.provider(), k.cfg, 0, H, k.sub,
                                 /*mem_budget_bytes=*/64),
      Catch::Matchers::ContainsSubstring("DRIZZLE_CONTRIB_LIST_BUDGET"));
  // A generous budget is fine.
  REQUIRE_NOTHROW(build_uniform_contrib_list(k.plan, k.provider(), k.cfg, 0, H,
                                             k.sub,
                                             static_cast<size_t>(1) << 30));
}
