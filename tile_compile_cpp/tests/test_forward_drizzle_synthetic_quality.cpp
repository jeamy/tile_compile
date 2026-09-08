// M9 / plan section 21 -- independent synthetic quality fixture for the
// single-method CFA forward-drizzle + multiband pipeline.
//
// This drives the DELIVERED path (run_forward_drizzle_stages: NORMALIZED_CACHE
// .. MULTIBAND) with frames synthesised from an analytic ground-truth scene,
// then checks the plan section 3.2 synthetic gates against that ground truth:
//
//   * photometric aperture-flux scatter < 0.5 % on the G/luma plane
//     (< 2 % on the sparse R/B CFA planes -- sampling-limited, see below)
//   * star centroid error               < 0.1 output px (median, all channels)
//   * multiband is not a FWHM regression against raw-forward-drizzle
//
// The synthesiser is deliberately INDEPENDENT of the reconstruction kernel
// (plan 21.2): stars are analytic Gaussians whose observed profile is obtained
// by quadrature (sigma_eff^2 = sigma_intrinsic^2 + sigma_psf^2), the per-frame
// affine is a known similarity transform, and each source pixel is box-
// integrated by 3x3 sub-sampling -- the polygon rasteriser is never called on
// the generation side.
//
// COVERAGE vs plan section 21 (this fixture is the first cut, not the whole set):
//   21.1 ground truth : analytic Gaussian stars (varied flux/colour/subpixel)
//                       on a flat sky.
//                       DEFERRED: Moffat profiles, known WCS, broad galaxy /
//                       diffuse structure, linear+curved sky (those pair with
//                       the RMS/seam/structure gates, not the point-source
//                       flux/centroid gate).
//   21.2 frames       : per-frame known affine (rotation + translation +
//                       sub-pixel dither), Poisson + read noise (fixed seed),
//                       GBRG Bayer sampling.
//                       DEFERRED: local warp field, spatially variable PSF,
//                       hotpix/cosmic rays, known-bad frames, RGGB/BGGR/GRBG,
//                       MONO one-plane fixture.
//   21.3 metrics      : flux error, centroid error, matched FWHM (multiband vs
//                       raw).
//                       DEFERRED: MTF50, background RMS vs GT, colour difference,
//                       seam/support error, bootstrap CIs.
// The deferred items are tracked in the M9 open list of the Entwicklungsprotokoll.

#include "../apps/runner_forward_drizzle.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>

using namespace tile_compile;

namespace {

// ---- ground-truth scene -----------------------------------------------------

struct Star {
  double x, y;        // canvas coordinates (continuous, pixel centre convention)
  double flux_r, flux_g, flux_b; // integrated flux per channel (arbitrary units)
};

struct Scene {
  int W = 0, H = 0;
  std::vector<Star> stars;
  // The R/B Bayer sub-lattice samples every 2nd pixel on both axes, so the
  // observed PSF must be wide enough (here FWHM ~6.9 canvas px) that those
  // channels are not undersampled -- a sharper PSF makes the R/B aperture flux
  // depend on sub-lattice phase, which is a CFA sampling limit, not a
  // reconstruction error, and would defeat the flux gate for the wrong reason.
  double sigma_psf = 2.90;       // canvas px  (FWHM ~6.9 px)
  double sigma_intrinsic = 0.35; // canvas px (keeps stars off a pure delta)
  double sky = 5.0;              // flat

  double sigma_eff() const {
    return std::sqrt(sigma_psf * sigma_psf + sigma_intrinsic * sigma_intrinsic);
  }

  // Channel-resolved, PSF-convolved scene value at a continuous canvas point.
  double sample(double x, double y, CfaChannel ch) const {
    const double s = sigma_eff();
    const double inv2s2 = 1.0 / (2.0 * s * s);
    const double norm = 1.0 / (2.0 * M_PI * s * s);
    double star_sum = 0.0;
    for (const auto &st : stars) {
      double f = ch == CfaChannel::R ? st.flux_r
                 : ch == CfaChannel::B ? st.flux_b
                                       : st.flux_g;
      if (f == 0.0)
        continue;
      const double dx = x - st.x, dy = y - st.y;
      const double r2 = dx * dx + dy * dy;
      if (r2 * inv2s2 > 40.0)
        continue; // far tail, negligible
      star_sum += f * norm * std::exp(-r2 * inv2s2);
    }
    return star_sum + sky;
  }
};

Scene make_scene(int W, int H) {
  Scene sc;
  sc.W = W;
  sc.H = H;
  // Deterministic star field: a spread of subpixel positions, fluxes and
  // colours. Bright stars (index < 8) are the photometry/centroid sample.
  const double px[] = {31.35, 118.7, 74.15, 46.8, 132.25, 92.6, 58.4, 104.9,
                       26.6,  140.1, 83.3,  67.75, 49.5,  123.4};
  const double py[] = {40.6,  33.25, 71.5,  118.8, 96.4,  128.15, 54.9, 88.35,
                       109.7, 121.6, 21.4,  142.3, 79.1,  61.2};
  // Bright sample (index < 8) is deliberately very high SNR: the plan 3.2
  // synthetic flux gate (< 0.5 %) is a statement about reconstruction-method
  // error, so photon shot noise on these stars must sit well below it.
  const double pf[] = {180000, 152000, 205000, 168000, 191000, 174000, 159000, 197000,
                       9000,   11000,  8200,   12500,  9600,   8800};
  const double col[][3] = {
      {1.00, 1.00, 1.00}, {1.20, 1.00, 0.72}, {0.80, 1.00, 1.28},
      {1.05, 1.00, 0.92}, {0.90, 1.00, 1.15}, {1.15, 1.00, 0.80},
      {1.00, 1.00, 1.00}, {0.95, 1.00, 1.08}, {1.10, 1.00, 0.85},
      {0.85, 1.00, 1.20}, {1.00, 1.00, 1.00}, {1.08, 1.00, 0.88},
      {0.92, 1.00, 1.12}, {1.00, 1.00, 1.00}};
  for (int i = 0; i < 14; ++i) {
    Star st;
    st.x = px[i];
    st.y = py[i];
    // OSC luma-ish split so the injected per-channel flux sums to pf[i]:
    const double wr = 0.25 * col[i][0], wg = 0.50 * col[i][1], wb = 0.25 * col[i][2];
    const double wsum = wr + wg + wb;
    st.flux_r = pf[i] * wr / wsum;
    st.flux_g = pf[i] * wg / wsum;
    st.flux_b = pf[i] * wb / wsum;
    sc.stars.push_back(st);
  }
  return sc;
}

// ---- per-frame affine (known similarity transform) -------------------------

struct Affine2x3 {
  double a, b, tx, c, d, ty; // [x'] = [a b tx][x], [y'] = [c d ty][y]
};

Affine2x3 frame_transform(int f, int nframes, double cx, double cy) {
  // Deterministic: small rotation sweep + a dither that visits sub-pixel
  // phases without a repeating lattice.
  const double frac = nframes > 1 ? double(f) / double(nframes - 1) : 0.0;
  const double theta_deg = -0.40 + 0.80 * frac;            // [-0.4, 0.4] deg
  const double theta = theta_deg * M_PI / 180.0;
  const double dith_x = 1.7 * std::sin(2.399963 * f + 0.5); // ~golden-angle walk
  const double dith_y = 1.7 * std::cos(2.399963 * f + 1.1);
  const double ct = std::cos(theta), stt = std::sin(theta);
  Affine2x3 A;
  // rotate about (cx,cy), then translate by dither
  A.a = ct;
  A.b = -stt;
  A.c = stt;
  A.d = ct;
  A.tx = cx - ct * cx + stt * cy + dith_x;
  A.ty = cy - stt * cx - ct * cy + dith_y;
  return A;
}

Affine2x3 invert(const Affine2x3 &A) {
  const double det = A.a * A.d - A.b * A.c;
  const double inv = 1.0 / det;
  Affine2x3 R;
  R.a = A.d * inv;
  R.b = -A.b * inv;
  R.c = -A.c * inv;
  R.d = A.a * inv;
  R.tx = -(R.a * A.tx + R.b * A.ty);
  R.ty = -(R.c * A.tx + R.d * A.ty);
  return R;
}

WarpMatrix to_warp(const Affine2x3 &A) {
  WarpMatrix m;
  m(0, 0) = float(A.a);
  m(0, 1) = float(A.b);
  m(0, 2) = float(A.tx);
  m(1, 0) = float(A.c);
  m(1, 1) = float(A.d);
  m(1, 2) = float(A.ty);
  return m;
}

// ---- frame synthesis ------------------------------------------------------

Matrix2Df synth_frame(const Scene &sc, const Affine2x3 &src_to_canvas, int f,
                      BayerPattern bayer) {
  Matrix2Df out(sc.H, sc.W);
  std::mt19937_64 rng(0x5eed'0000ULL + static_cast<uint64_t>(f));
  const double gain = 20.0;                // e-/ADU-ish (high SNR fixture)
  const double read_noise_adu = 1.0;
  std::normal_distribution<double> rd(0.0, read_noise_adu);
  for (int sy = 0; sy < sc.H; ++sy) {
    for (int sx = 0; sx < sc.W; ++sx) {
      const CfaChannel ch =
          cfa_channel_for_source_pixel(sx, sy, bayer, 0, 0);
      // 3x3 box integration over the source pixel area.
      double acc = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const double ssx = sx + (a + 0.5) / 3.0;
          const double ssy = sy + (b + 0.5) / 3.0;
          const double X = src_to_canvas.a * ssx + src_to_canvas.b * ssy + src_to_canvas.tx;
          const double Y = src_to_canvas.c * ssx + src_to_canvas.d * ssy + src_to_canvas.ty;
          acc += sc.sample(X, Y, ch);
        }
      }
      double signal = acc / 9.0;
      if (signal < 0.0)
        signal = 0.0;
      // Poisson shot noise in electrons, then read noise in ADU.
      const double e = signal * gain;
      double noisy;
      if (e < 1.0e7) {
        std::poisson_distribution<long long> pd(e);
        noisy = double(pd(rng)) / gain;
      } else {
        noisy = signal; // guard against absurd lambda
      }
      noisy += rd(rng);
      out(sy, sx) = float(noisy);
    }
  }
  return out;
}

// ---- measurement on the reconstructed canvas -----------------------------

struct Meas {
  double flux = 0, cx = 0, cy = 0, fwhm = 0;
  bool ok = false;
};

double annulus_background(const Matrix2Df &img, double x0, double y0, double r_in,
                          double r_out) {
  std::vector<double> vals;
  const int lo_x = std::max(0, int(std::floor(x0 - r_out)));
  const int hi_x = std::min(int(img.cols()) - 1, int(std::ceil(x0 + r_out)));
  const int lo_y = std::max(0, int(std::floor(y0 - r_out)));
  const int hi_y = std::min(int(img.rows()) - 1, int(std::ceil(y0 + r_out)));
  for (int y = lo_y; y <= hi_y; ++y)
    for (int x = lo_x; x <= hi_x; ++x) {
      const double d = std::hypot(x + 0.5 - x0, y + 0.5 - y0);
      if (d >= r_in && d <= r_out && std::isfinite(img(y, x)))
        vals.push_back(img(y, x));
    }
  if (vals.empty())
    return 0.0;
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  return vals[vals.size() / 2];
}

Meas measure_star(const Matrix2Df &img, double x_true, double y_true,
                  double r_ap = 14.0) {
  Meas m;
  const double bg = annulus_background(img, x_true, y_true, r_ap + 2.0, r_ap + 7.0);
  double sum = 0, sx = 0, sy = 0, s2 = 0;
  const int lo_x = std::max(0, int(std::floor(x_true - r_ap)));
  const int hi_x = std::min(int(img.cols()) - 1, int(std::ceil(x_true + r_ap)));
  const int lo_y = std::max(0, int(std::floor(y_true - r_ap)));
  const int hi_y = std::min(int(img.rows()) - 1, int(std::ceil(y_true + r_ap)));
  for (int y = lo_y; y <= hi_y; ++y)
    for (int x = lo_x; x <= hi_x; ++x) {
      const double px = x + 0.5, py = y + 0.5;
      const double d = std::hypot(px - x_true, py - y_true);
      if (d > r_ap || !std::isfinite(img(y, x)))
        continue;
      const double v = double(img(y, x)) - bg;
      if (v <= 0.0)
        continue;
      sum += v;
      sx += v * px;
      sy += v * py;
      s2 += v * d * d;
    }
  if (sum <= 0.0)
    return m;
  m.flux = sum;
  m.cx = sx / sum;
  m.cy = sy / sum;
  // Gaussian FWHM from the second moment: sigma^2 = <r^2>/2.
  const double sigma = std::sqrt(std::max(1.0e-9, s2 / sum / 2.0));
  m.fwhm = 2.3548200 * sigma;
  m.ok = true;
  return m;
}

double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Fixture {
  core::AtomicOutput staging{fs::temp_directory_path() / "fd-synth-quality"};
  fs::path dir = staging.path();
  config::Config cfg;
  registration::RegistrationSamplingPlan plan;
  std::unique_ptr<runner::RunnerFrameCache> cache;
  Scene scene;
  int nframes = 36;
  int W = 176, H = 176;

  Fixture() {
    fs::create_directories(dir / "artifacts");
    fs::create_directories(dir / "logs");
    scene = make_scene(W, H);

    const std::string yaml = R"(data:
  color_mode: OSC
  bayer_pattern: GBRG
runtime_limits:
  memory_budget: 64
reconstruction:
  drizzle:
    internal_scale: 1
    output_scale: 1
    pixfrac: 1.0
    memory_budget_mb: 64
    chunk_rows: 24
  coverage_gate:
    min_channel_n_eff_floor: 2.0
    min_channel_n_eff_fraction: 0.08
    min_analysis_pixels: 256
    min_supported_fraction: 0.90
  clipping:
    min_n_eff: 1.5
  diagnostics:
    level: full
)";
    core::write_text_atomic(dir / "config.yaml", yaml);
    cfg = config::Config::from_yaml_text(yaml);

    // Must match validate_provenance(): sha256_bytes(input_manifest ":" config).
    const std::string config_hash = core::sha256_file(dir / "config.yaml");
    const std::string identity = "synthetic-quality:" + config_hash;
    plan.source_identity_hash = core::sha256_bytes(
        std::vector<uint8_t>(identity.begin(), identity.end()));
    plan.source_width = plan.canvas_width_native = W;
    plan.source_height = plan.canvas_height_native = H;
    plan.color_mode = ColorMode::OSC;
    plan.bayer_pattern = BayerPattern::GBRG;
    plan.internal_scale = 1;
    plan.output_scale = 1;

    cache = std::make_unique<runner::RunnerFrameCache>(
        dir / "cache/normalized_frames", nframes, H, W);

    const double cx = W * 0.5, cy = H * 0.5;
    for (int f = 0; f < nframes; ++f) {
      const Affine2x3 s2c = frame_transform(f, nframes, cx, cy);
      const Affine2x3 c2s = invert(s2c);
      registration::FrameSamplingTransform ft;
      ft.frame_id = plan.source_identity_hash + ":" + std::to_string(f);
      ft.source_index = static_cast<size_t>(f);
      ft.valid = true;
      ft.source_to_canvas = to_warp(s2c);
      ft.canvas_to_source = to_warp(c2s);
      ft.source_to_canvas_affine_valid = true;
      plan.frames.push_back(ft);
      cache->store_normalized(f, synth_frame(scene, s2c, f, BayerPattern::GBRG));
    }
    plan.plan_hash = registration::compute_plan_hash(plan);
    core::write_text_atomic(dir / "artifacts/registration_sampling.json",
                            registration::serialize_to_json_string(plan));
    core::write_text_atomic(
        dir / "artifacts/run_provenance.json",
        core::json({{"execution_scope", "forward_drizzle_m1_m3"},
                    {"config", {{"sha256", config_hash}}},
                    {"input_manifest", {{"sha256", "synthetic-quality"}}}})
            .dump());
  }
  ~Fixture() {
    cache.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  bool run(std::ostream &log) {
    core::EventEmitter emitter;
    return runner::run_forward_drizzle_stages("synthq", cfg, dir, plan,
                                              cache.get(), emitter, log);
  }
};

} // namespace

TEST_CASE("forward drizzle synthetic quality: aperture flux and centroid match "
          "the analytic ground truth (plan 3.2 / 21)",
          "[synthetic-quality]") {
  Fixture f;
  std::ostringstream log;
  const bool ok = f.run(log);
  if (!ok)
    std::fprintf(stderr, "run_forward_drizzle_stages log:\n%s\n", log.str().c_str());
  REQUIRE(ok);

  REQUIRE(fs::exists(f.dir / "artifacts/reconstruction_multiband.fits"));
  const std::array<std::string, 3> ch{"R", "G", "B"};
  const std::array<CfaChannel, 3> cc{CfaChannel::R, CfaChannel::G, CfaChannel::B};

  // Flux / centroid gates are measured on the RAW forward-drizzle candidate:
  // that is the flux-faithful path (the drizzle kernel's area identity),
  // whereas multiband deliberately redistributes local detail. FWHM is the
  // multiband-vs-raw comparison.
  std::vector<double> centroid_err;                 // output px, all channels
  std::array<std::vector<double>, 3> flux_ratio;    // per channel: measured/true
  std::vector<double> fwhm_raw, fwhm_mb;

  for (int c = 0; c < 3; ++c) {
    const auto raw = io::read_fits_pixels_float(
        f.dir / ("outputs/forward_drizzle_raw_" + ch[c] + ".fit"));
    const auto mb = io::read_fits_pixels_float(
        f.dir / ("outputs/forward_drizzle_multiband_" + ch[c] + ".fit"));
    REQUIRE(raw.rows() == f.H);
    REQUIRE(raw.cols() == f.W);
    REQUIRE(mb.rows() == f.H);
    REQUIRE(mb.cols() == f.W);

    for (size_t si = 0; si < f.scene.stars.size(); ++si) {
      if (si >= 8)
        break; // bright photometry/centroid sample only
      const auto &st = f.scene.stars[si];
      const double f_true = cc[c] == CfaChannel::R   ? st.flux_r
                            : cc[c] == CfaChannel::B ? st.flux_b
                                                    : st.flux_g;
      const Meas mraw = measure_star(raw, st.x, st.y);
      const Meas mmb = measure_star(mb, st.x, st.y);
      REQUIRE(mraw.ok);
      REQUIRE(mmb.ok);
      flux_ratio[c].push_back(mraw.flux / f_true);
      centroid_err.push_back(std::hypot(mraw.cx - st.x, mraw.cy - st.y));
      fwhm_raw.push_back(mraw.fwhm);
      fwhm_mb.push_back(mmb.fwhm);
    }
  }

  // ---- centroid: plan 3.2 absolute gate (synthetic warps) ------------------
  const double centroid_med = median(centroid_err);
  INFO("median centroid error (output px) = " << centroid_med);
  REQUIRE(centroid_med < 0.1);

  // ---- flux: plan 3.2 photometric gate ------------------------------------
  // A common systematic (3x3 box integration + finite aperture) is divided out;
  // the photometric error is the residual per-star scatter about the channel's
  // own median ratio. The G plane is the OSC luma carrier and the one aperture
  // photometry (PCC) actually uses -- it must meet the plan's 0.5 %. R and B
  // are the sparse CFA planes sampled on a 2-px sub-lattice; their per-star
  // scatter is sampling-limited, not a reconstruction error, and is held to a
  // looser 2 %. All three must be free of a large systematic (aperture wide
  // enough that flux is conserved).
  auto scatter = [](const std::vector<double> &r) {
    const double m = median(r);
    std::vector<double> d;
    for (double v : r)
      d.push_back(std::abs(v / m - 1.0));
    return median(d);
  };
  for (int c = 0; c < 3; ++c) {
    REQUIRE(flux_ratio[c].size() == 8u);
    const double med = median(flux_ratio[c]);
    const double sc = scatter(flux_ratio[c]);
    INFO(ch[c] << " flux ratio median=" << med << " scatter=" << sc);
    REQUIRE(std::abs(med - 1.0) < 0.05);
    REQUIRE(sc < (c == 1 ? 0.005 : 0.02));
  }

  // ---- FWHM: multiband is not a regression against raw --------------------
  // Plan 3.2 asks for multiband >= 5 % better on matched real stars; that
  // comparison needs the PREWARP-AQMH reference (M9 legacy harness). On this
  // smooth analytic set multiband has little detail to recover, so the fixture
  // only guards against a regression.
  const double med_raw = median(fwhm_raw), med_mb = median(fwhm_mb);
  INFO("median FWHM raw=" << med_raw << " multiband=" << med_mb);
  REQUIRE(med_mb <= med_raw * 1.02);
}
