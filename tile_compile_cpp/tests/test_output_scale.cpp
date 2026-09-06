// M4 tests: internal 2x raster -> output geometry (plan section 12).

#include "tile_compile/reconstruction/output_scale.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using Catch::Approx;

namespace {
ProfilePlane make_plane(int w, int h) {
  ProfilePlane p;
  p.allocate(w, h);
  return p;
}
void set_px(ProfilePlane &p, int x, int y, float value, float weight, float n_eff) {
  const size_t i = static_cast<size_t>(y) * p.width + x;
  p.value[i] = value;
  p.weight_sum[i] = weight;
  p.n_eff[i] = n_eff;
  p.support[i] = 1;
}
}  // namespace

TEST_CASE("output_scale 2x2: a fully-valid quad averages value/weight and "
          "takes the min n_eff (plan 12.1)") {
  ProfilePlane in = make_plane(2, 2);
  set_px(in, 0, 0, 10.0f, 4.0f, 8.0f);
  set_px(in, 1, 0, 20.0f, 2.0f, 3.0f);
  set_px(in, 0, 1, 30.0f, 6.0f, 5.0f);
  set_px(in, 1, 1, 40.0f, 8.0f, 2.0f);

  ProfilePlane out = downsample_profile_plane_2x2(in);
  REQUIRE(out.width == 1);
  REQUIRE(out.height == 1);
  REQUIRE(out.support[0] == 1);
  REQUIRE(out.value[0] == Approx(0.25 * (10 + 20 + 30 + 40)));  // 25
  REQUIRE(out.weight_sum[0] == Approx(0.25 * (4 + 2 + 6 + 8)));  // 5
  REQUIRE(out.n_eff[0] == Approx(2.0));                          // min(8,3,5,2)
}

TEST_CASE("output_scale 2x2: any invalid subpixel makes the 1x pixel "
          "invalid --- never a partial mean or a zero (plan 12.1)") {
  ProfilePlane in = make_plane(2, 2);
  set_px(in, 0, 0, 10.0f, 1.0f, 5.0f);
  set_px(in, 1, 0, 20.0f, 1.0f, 5.0f);
  set_px(in, 0, 1, 30.0f, 1.0f, 5.0f);
  // (1,1) left invalid (support 0, value NaN from allocate)

  ProfilePlane out = downsample_profile_plane_2x2(in);
  REQUIRE(out.support[0] == 0);
  REQUIRE(std::isnan(out.value[0]));
  REQUIRE(out.weight_sum[0] == 0.0f);
  REQUIRE(out.n_eff[0] == 0.0f);
}

TEST_CASE("output_scale 2x2: an odd internal dimension drops its last "
          "row/column, operator stays exact 2x2") {
  ProfilePlane in = make_plane(5, 3);  // -> 2 x 1
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 5; ++x) set_px(in, x, y, 1.0f, 1.0f, 4.0f);

  ProfilePlane out = downsample_profile_plane_2x2(in);
  REQUIRE(out.width == 2);
  REQUIRE(out.height == 1);
  for (int i = 0; i < 2; ++i) {
    REQUIRE(out.support[i] == 1);
    REQUIRE(out.value[i] == Approx(1.0));
  }
}

TEST_CASE("output_scale 2x2: surface brightness is preserved (a constant "
          "field downsamples to the same constant, plan 12.1)") {
  ProfilePlane in = make_plane(8, 8);
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) set_px(in, x, y, 123.5f, 2.0f, 6.0f);
  ProfilePlane out = downsample_profile_plane_2x2(in);
  REQUIRE(out.width == 4);
  REQUIRE(out.height == 4);
  for (size_t i = 0; i < out.value.size(); ++i) {
    REQUIRE(out.support[i] == 1);
    REQUIRE(out.value[i] == Approx(123.5));
  }
}

TEST_CASE("output_scale mode: 2/1 needs a downsample; 1/1 and 2/2 do not") {
  REQUIRE(OutputScaleMode{2, 1}.needs_2x2_downsample());
  REQUIRE_FALSE(OutputScaleMode{1, 1}.needs_2x2_downsample());
  REQUIRE_FALSE(OutputScaleMode{2, 2}.needs_2x2_downsample());
  REQUIRE(OutputScaleMode{2, 1}.valid());
  REQUIRE_FALSE(OutputScaleMode{1, 2}.valid());  // output > internal
  REQUIRE_FALSE(OutputScaleMode{3, 1}.valid());
}

TEST_CASE("WCS scaling: S=1 with a canvas offset and native crop matches the "
          "hand-computed plan 12.2 formula") {
  astrometry::WCS in;
  in.crpix1 = 100.0;
  in.crpix2 = 200.0;
  in.cd1_1 = 1e-4;
  in.cd1_2 = 2e-5;
  in.cd2_1 = -3e-5;
  in.cd2_2 = 1e-4;

  OutputWcsParams p;
  p.output_scale = 1;
  p.canvas_offset_x_native = 12.0;
  p.canvas_offset_y_native = 8.0;
  p.crop_origin_x_out = 5.0;   // == output_scale * crop_origin_native, S=1
  p.crop_origin_y_out = 3.0;

  auto out = scale_wcs_to_output(in, p);
  // S=1: CRPIX_out = (CRPIX_in + offset - 0.5) + 0.5 - crop = CRPIX_in + offset - crop
  REQUIRE(out.crpix1 == Approx(100.0 + 12.0 - 5.0));
  REQUIRE(out.crpix2 == Approx(200.0 + 8.0 - 3.0));
  REQUIRE(out.cd1_1 == Approx(1e-4));
  REQUIRE(out.cd2_1 == Approx(-3e-5));
}

TEST_CASE("WCS scaling: S=2 rebins CRPIX by the standard FITS form and "
          "halves the CD matrix (plan 12.2, mode 2/2)") {
  astrometry::WCS in;
  in.crpix1 = 50.0;
  in.crpix2 = 60.0;
  in.cd1_1 = 4e-4;
  in.cd1_2 = 0.0;
  in.cd2_1 = 0.0;
  in.cd2_2 = 4e-4;

  OutputWcsParams p;
  p.output_scale = 2;
  // no canvas offset, no crop
  auto out = scale_wcs_to_output(in, p);
  // CRPIX_out = 2*(CRPIX - 0.5) + 0.5 = 2*CRPIX - 0.5  (== S*CRPIX - (S-1)/2)
  REQUIRE(out.crpix1 == Approx(2.0 * 50.0 - 0.5));
  REQUIRE(out.crpix2 == Approx(2.0 * 60.0 - 0.5));
  REQUIRE(out.cd1_1 == Approx(2e-4));
  REQUIRE(out.cd2_2 == Approx(2e-4));
}

TEST_CASE("output_scale 2x2 on a uniform+raw result: both profiles land at "
          "1x, shared support preserved") {
  ForwardDrizzleUniformAndRawResult in;
  in.uniform.color_mode = ColorMode::MONO;
  in.uniform.internal_width = 4;
  in.uniform.internal_height = 4;
  in.raw.color_mode = ColorMode::MONO;
  in.raw.internal_width = 4;
  in.raw.internal_height = 4;
  in.uniform.L = make_plane(4, 4);
  in.raw.L = make_plane(4, 4);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      set_px(in.uniform.L, x, y, 10.0f, 1.0f, 3.0f);
      set_px(in.raw.L, x, y, 7.0f, 0.5f, 3.0f);
    }

  auto out = downsample_uniform_and_raw_2x2(in);
  REQUIRE(out.uniform.L.width == 2);
  REQUIRE(out.raw.L.width == 2);
  REQUIRE(out.uniform.L.support == out.raw.L.support);
  REQUIRE(out.uniform.L.value[0] == Approx(10.0));
  REQUIRE(out.raw.L.value[0] == Approx(7.0));
  REQUIRE(out.uniform.R.empty());  // MONO: R/G/B stay absent
}

// --- plan 12.4: kernel-induced noise correlation --------------------------

TEST_CASE("kernel noise: pixfrac*internal_scale == 1 gives no correlation "
          "(factor exactly 1, plan 12.4)") {
  REQUIRE(kernel_noise_correlation_sigma_factor(1.0f, 1) == Approx(1.0).epsilon(1e-9));
  REQUIRE(kernel_noise_correlation_sigma_factor(0.5f, 2) == Approx(1.0).epsilon(1e-9));
  auto rho = kernel_noise_autocorrelation_1d(1.0f, 1, 4);
  REQUIRE(rho[0] == Approx(1.0));
  for (size_t i = 1; i < rho.size(); ++i) REQUIRE(rho[i] == Approx(0.0).margin(1e-12));
}

TEST_CASE("kernel noise: pixfrac=1, internal_scale=2 (d=2) matches the "
          "hand-computed value W/sqrt(S0) = 2/sqrt(1.5) (plan 12.4)") {
  // d = 2: overlaps of [j-0.5, j+1.5] with [0,1] over j = -1,0,1 are
  // 0.5, 1.0, 0.5 -> S0 = 0.25 + 1 + 0.25 = 1.5; W = d = 2.
  REQUIRE(kernel_noise_correlation_sigma_factor(1.0f, 2) ==
          Approx(2.0 / std::sqrt(1.5)).epsilon(1e-9));
  auto rho = kernel_noise_autocorrelation_1d(1.0f, 2, 3);
  // S_1 = sum over j of K(0,j+0.5)*K(1,j+0.5):
  //   j=-1: 0.5 * (overlap of [-1.5,0.5] shifted by -1 = [-0.5,1.5]&[0,1]=1.0) ... compute:
  //   K(0, x0) uses x0=j+0.5; K(1, x0) = kernel_overlap(x0 - 1, d).
  //   j=-1: K0=overlap([-1.5,0.5])=0.5 ; K1=overlap([-2.5,-0.5])=0
  //   j=0 : K0=overlap([-0.5,1.5])=1.0 ; K1=overlap([-1.5,0.5])=0.5
  //   j=1 : K0=overlap([0.5,2.5])=0.5  ; K1=overlap([-0.5,1.5])=1.0
  //   j=2 : K0=0                        ; K1=overlap([0.5,2.5])=0.5
  //   S_1 = 0 + 0.5 + 0.5 + 0 = 1.0 -> rho_1 = 1.0 / 1.5
  REQUIRE(rho[1] == Approx(1.0 / 1.5).epsilon(1e-9));
  // sqrt(sum of rho over ALL lags, both signs) must equal the scalar factor.
  double sum = rho[0];
  for (size_t i = 1; i < rho.size(); ++i) sum += 2.0 * rho[i];
  REQUIRE(std::sqrt(sum) == Approx(kernel_noise_correlation_sigma_factor(1.0f, 2)).epsilon(1e-9));
}

TEST_CASE("kernel noise: production default pixfrac=0.8, internal_scale=2 "
          "(d=1.6) matches 1.6/sqrt(1.18) (plan 12.4)") {
  // d = 1.6: overlaps of [j+0.5-0.8, j+0.5+0.8] with [0,1] over j=-1,0,1
  //   j=-1: [-1.3,0.3]&[0,1] = 0.3
  //   j=0 : [-0.3,1.3]&[0,1] = 1.0
  //   j=1 : [ 0.7,2.3]&[0,1] = 0.3
  //   S0 = 0.09 + 1 + 0.09 = 1.18; W = 1.6
  REQUIRE(kernel_noise_correlation_sigma_factor(0.8f, 2) ==
          Approx(1.6 / std::sqrt(1.18)).epsilon(1e-6));
  REQUIRE(kernel_noise_correlation_sigma_factor(0.8f, 2) > 1.0);
}

TEST_CASE("kernel noise: invalid arguments are rejected") {
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(0.0f, 2));
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(1.5f, 2));
  REQUIRE_THROWS(kernel_noise_correlation_sigma_factor(0.8f, 3));
}

// --- streaming 2/1 must match the non-streaming reference ----------------

#include "tile_compile/reconstruction/forward_drizzle.hpp"

namespace {
WarpMatrix affine_s2c(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a); m(0, 1) = static_cast<float>(b); m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c); m(1, 1) = static_cast<float>(d); m(1, 2) = static_cast<float>(ty);
  return m;
}
registration::FrameSamplingTransform frame(const std::string &id, size_t idx,
                                           const WarpMatrix &s2c) {
  registration::FrameSamplingTransform f;
  f.frame_id = id; f.source_index = idx; f.valid = true;
  f.source_to_canvas = s2c; f.source_to_canvas_affine_valid = true;
  return f;
}
void compare_plane(const ProfilePlane &a, const ProfilePlane &b) {
  REQUIRE(a.width == b.width);
  REQUIRE(a.height == b.height);
  REQUIRE(a.support == b.support);
  REQUIRE(a.weight_sum == b.weight_sum);
  REQUIRE(a.n_eff == b.n_eff);
  for (size_t i = 0; i < a.value.size(); ++i)
    if (a.support[i]) REQUIRE(a.value[i] == b.value[i]);
}
}  // namespace

TEST_CASE("streaming 2/1 == non-streaming compute + downsample, independent "
          "of internal chunk height (plan 12.1)") {
  registration::RegistrationSamplingPlan plan;
  plan.source_width = 20;
  plan.source_height = 16;
  plan.canvas_width_native = 24;
  plan.canvas_height_native = 20;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::RGGB;
  const double ang = 0.05;
  for (size_t i = 0; i < 4; ++i)
    plan.frames.push_back(frame("f" + std::to_string(i), i,
        affine_s2c(std::cos(ang), -std::sin(ang), 8.0 + 0.3 * i,
                   std::sin(ang), std::cos(ang), 6.0 + 0.2 * i)));

  Matrix2Df src(16, 20);
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 20; ++x) src(y, x) = 1.0f + 0.5f * x - 0.2f * y;
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & { return src; };

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.pixfrac = 0.8f;
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  const std::vector<float> g_eff = {1.0f, 0.7f, 0.4f, 0.9f};

  auto reference = downsample_uniform_and_raw_2x2(
      compute_forward_drizzle_uniform_and_raw(plan, source_of, cfg, clip, {}, g_eff));

  for (int chunk : {1, 3, 7, 1000}) {
    config::ReconstructionDrizzleConfig c = cfg;
    c.chunk_rows = chunk;
    ForwardDrizzleUniformAndRawResult acc;
    acc.uniform.internal_width = reference.uniform.internal_width;
    acc.uniform.internal_height = reference.uniform.internal_height;
    acc.raw.internal_width = reference.raw.internal_width;
    acc.raw.internal_height = reference.raw.internal_height;
    acc.uniform.R.allocate(reference.uniform.R.width, reference.uniform.R.height);
    acc.uniform.G.allocate(reference.uniform.G.width, reference.uniform.G.height);
    acc.uniform.B.allocate(reference.uniform.B.width, reference.uniform.B.height);
    acc.raw.R.allocate(reference.raw.R.width, reference.raw.R.height);
    acc.raw.G.allocate(reference.raw.G.width, reference.raw.G.height);
    acc.raw.B.allocate(reference.raw.B.width, reference.raw.B.height);

    stream_forward_drizzle_uniform_and_raw_2x2(
        plan, source_of, c, clip,
        [&](int y, const ForwardDrizzleUniformAndRawResult &row) {
          auto put = [&](ProfilePlane &dst, const ProfilePlane &s) {
            const size_t off = static_cast<size_t>(y) * dst.width;
            for (int x = 0; x < s.width; ++x) {
              dst.value[off + x] = s.value[x];
              dst.weight_sum[off + x] = s.weight_sum[x];
              dst.n_eff[off + x] = s.n_eff[x];
              dst.support[off + x] = s.support[x];
            }
          };
          put(acc.uniform.R, row.uniform.R); put(acc.uniform.G, row.uniform.G);
          put(acc.uniform.B, row.uniform.B);
          put(acc.raw.R, row.raw.R); put(acc.raw.G, row.raw.G); put(acc.raw.B, row.raw.B);
        },
        {}, g_eff);

    compare_plane(acc.uniform.R, reference.uniform.R);
    compare_plane(acc.uniform.G, reference.uniform.G);
    compare_plane(acc.uniform.B, reference.uniform.B);
    compare_plane(acc.raw.R, reference.raw.R);
    compare_plane(acc.raw.G, reference.raw.G);
    compare_plane(acc.raw.B, reference.raw.B);
  }
}

TEST_CASE("streaming 2/1 rejects internal_scale != 2") {
  registration::RegistrationSamplingPlan plan;
  plan.source_width = 2; plan.source_height = 2;
  plan.canvas_width_native = 8; plan.canvas_height_native = 8;
  plan.color_mode = ColorMode::MONO;
  Matrix2Df s = Matrix2Df::Zero(2, 2);
  SourceImageProvider src = [&](std::size_t) -> const Matrix2Df & { return s; };
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  config::ReconstructionClippingConfig clip;
  REQUIRE_THROWS(stream_forward_drizzle_uniform_and_raw_2x2(
      plan, src, cfg, clip, [](int, const ForwardDrizzleUniformAndRawResult &) {}));
}
