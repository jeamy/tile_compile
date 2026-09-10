#include "tile_compile/reconstruction/drizzle_profile_store.hpp"
#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/reconstruction/profile_store_manifest.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using json = nlohmann::json;

namespace {
struct Fixture {
  core::AtomicOutput staging{fs::temp_directory_path() / "drizzle-store-test"};
  fs::path root = staging.path();
  Fixture() { fs::create_directory(root); }
  ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }
};

// The forward-drizzle CUDA fault injection is process-global; a leaked value
// silently perturbs every later test. Scope every use through this guard so a
// failing REQUIRE inside a SECTION still disarms it.
struct CudaFaultGuard {
  explicit CudaFaultGuard(int n) {
    reconstruction::set_forward_drizzle_cuda_fault_after_chunks(n);
  }
  ~CudaFaultGuard() {
    reconstruction::set_forward_drizzle_cuda_fault_after_chunks(-1);
  }
  CudaFaultGuard(const CudaFaultGuard &) = delete;
  CudaFaultGuard &operator=(const CudaFaultGuard &) = delete;
};
registration::RegistrationSamplingPlan plan_for(int size = 16, bool osc = false) {
  registration::RegistrationSamplingPlan plan;
  plan.source_width = plan.source_height = size;
  plan.canvas_width_native = plan.canvas_height_native = size;
  plan.source_identity_hash = "synthetic-normalized-source";
  plan.color_mode = osc ? ColorMode::OSC : ColorMode::MONO;
  plan.bayer_pattern = osc ? BayerPattern::RGGB : BayerPattern::UNKNOWN;
  registration::FrameSamplingTransform f;
  f.valid = f.source_to_canvas_affine_valid = true;
  f.frame_id = "synthetic:0";
  plan.frames = {f};
  return plan;
}
config::ReconstructionDrizzleConfig config_for() {
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 1;
  cfg.chunk_rows = 3;
  cfg.memory_budget_mb = 32;
  return cfg;
}
json read_json(const fs::path &p) { std::ifstream f(p); json j; f >> j; return j; }
void rehash_commit(json &commit) {
  commit.erase("commit_hash");
  const auto s = commit.dump();
  commit["commit_hash"] = core::sha256_bytes(std::vector<uint8_t>(s.begin(), s.end()));
}
void replace_commit(const Fixture &fixture, json commit) {
  rehash_commit(commit);
  core::write_text_atomic(fixture.root / commit.at("generation").get<std::string>() / "commit.json", commit.dump());
  core::write_text_atomic(fixture.root / "current.json", commit.dump());
}
}

TEST_CASE("drizzle store: streamed planes and required context round trip", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for();
  auto cfg = config_for();
  Matrix2Df source = Matrix2Df::Constant(16,16,-7.25f);
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { return source; };
  const auto result = persist_forward_drizzle_uniform(fixture.root, plan, provider, cfg);
  const auto identity = make_drizzle_store_identity(plan,cfg);
  const auto verified = verify_drizzle_profile_store(fixture.root, identity);
  REQUIRE(verified.usable);
  REQUIRE(verified.generation_dir == result.generation_dir);
  const auto values = io::read_fits_pixels_float(result.generation_dir / "uniform_L_value.fits");
  REQUIRE(values == source);
  REQUIRE(io::read_fits_pixels_float(result.generation_dir / "uniform_L_support.fits").minCoeff() == 1.0f);
  auto other = identity;
  other.source_identity_hash = "different calibration";
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,other).usable);
  other = identity;
  other.mode = "uniform_raw_clipped";
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,other).usable);
  auto changed = cfg;
  changed.chunk_rows = 1;
  changed.memory_budget_mb = 64;
  REQUIRE(make_drizzle_store_identity(plan,changed) == identity);
  changed.pixfrac = 0.8f;
  REQUIRE_FALSE(make_drizzle_store_identity(plan,changed) == identity);
}

TEST_CASE("drizzle store: interrupted next generation preserves prior commit", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for();
  auto cfg = config_for();
  Matrix2Df source = Matrix2Df::Ones(16,16);
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { return source; };
  const auto old = persist_forward_drizzle_uniform(fixture.root,plan,provider,cfg);
  const auto before = read_json(fixture.root / "current.json");
  int calls = 0;
  SourceImageProvider failing = [&](size_t) -> const Matrix2Df & {
    if (++calls == 2) throw std::runtime_error("injected source failure after first stripe");
    return source;
  };
  REQUIRE_THROWS(persist_forward_drizzle_uniform(fixture.root,plan,failing,cfg));
  REQUIRE(calls == 2);
  REQUIRE(read_json(fixture.root / "current.json") == before);
  REQUIRE(verify_drizzle_profile_store(fixture.root,make_drizzle_store_identity(plan,cfg)).usable);
  size_t dirs = 0;
  for (const auto &e : fs::directory_iterator(fixture.root)) if (e.is_directory()) ++dirs;
  REQUIRE(dirs == 1);
  source.setConstant(2.0f);
  const auto next = persist_forward_drizzle_uniform(fixture.root,plan,provider,cfg);
  REQUIRE(next.generation_dir != old.generation_dir);
  REQUIRE(fs::exists(old.generation_dir / "uniform_L_value.fits"));
  REQUIRE(io::read_fits_pixels_float(old.generation_dir / "uniform_L_value.fits").maxCoeff() == 1.0f);
  REQUIRE(io::read_fits_pixels_float(next.generation_dir / "uniform_L_value.fits").minCoeff() == 2.0f);
}

TEST_CASE("drizzle store: incomplete or forged planes fail despite rehashed manifest", "[drizzle-store]") {
  for (int kind = 0; kind < 4; ++kind) {
    Fixture fixture;
    auto plan = plan_for();
    auto cfg = config_for();
    Matrix2Df source = Matrix2Df::Ones(16,16);
    const auto store = persist_forward_drizzle_uniform(fixture.root,plan,
        [&](size_t) -> const Matrix2Df & { return source; },cfg);
    auto commit = read_json(fixture.root / "current.json");
    auto &manifest = commit["planes"];
    if (kind == 0) manifest["planes"].erase(0);
    if (kind == 1) {
      Matrix2Df wrong = Matrix2Df::Ones(2,2);
      io::write_fits_float(store.generation_dir / "uniform_L_value.fits",wrong,{});
      for (auto &p : manifest["planes"])
        if (p["name"] == "uniform_L_value")
          p["sha256"] = core::sha256_file(store.generation_dir / "uniform_L_value.fits");
    }
    if (kind == 2) fs::remove(store.generation_dir / "uniform_L_support.fits");
    if (kind == 3) {
      std::ofstream(store.generation_dir / "uniform_L_value.fits",std::ios::binary) << "broken FITS";
      for (auto &p : manifest["planes"])
        if (p["name"] == "uniform_L_value")
          p["sha256"] = core::sha256_file(store.generation_dir / "uniform_L_value.fits");
    }
    // Rehash the generic manifest so completeness and FITS checks must act.
    ProfileStoreManifest parsed;
    parsed.profile = manifest["profile"];
    parsed.internal_width = 16; parsed.internal_height = 16;
    for (const auto &p : manifest["planes"])
      parsed.planes.push_back({p["name"], p["sha256"], p["width"], p["height"]});
    manifest["manifest_hash"] = compute_profile_store_manifest_hash(parsed);
    replace_commit(fixture,commit);
    REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,make_drizzle_store_identity(plan,cfg)).usable);
  }
}

TEST_CASE("drizzle store: OSC pair streams within budget that rejects full outputs", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(512,true);
  auto cfg = config_for();
  cfg.memory_budget_mb = 16;
  cfg.chunk_rows = 0;
  config::ReconstructionClippingConfig clipping;
  clipping.min_n_eff = 1;
  Matrix2Df source = Matrix2Df::Constant(512,512,13.0f);
  size_t calls = 0;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { ++calls; return source; };
  REQUIRE_THROWS(compute_forward_drizzle_uniform_and_raw(plan,provider,cfg,clipping));
  REQUIRE(calls == 0);
  const auto store = persist_forward_drizzle_uniform_and_raw(fixture.root,plan,provider,cfg,clipping,{}, {0.5f});
  REQUIRE(store.diagnostics.estimated_peak_bytes <= 16 * 1024 * 1024);
  REQUIRE(verify_drizzle_profile_store(fixture.root,
      make_drizzle_store_identity(plan,cfg,{},&clipping,{0.5f})).usable);
  auto weights = io::read_fits_pixels_float(store.generation_dir / "raw_R_weight_sum.fits");
  REQUIRE(weights(0,0) == 0.5f);
  auto values = io::read_fits_pixels_float(store.generation_dir / "raw_R_value.fits");
  REQUIRE(values(0,0) == 13.0f);
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,
      make_drizzle_store_identity(plan,cfg,{},&clipping,{1.0f})).usable);
}

TEST_CASE("drizzle store: memory rejection precedes IO and publication", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(512,true);
  auto cfg = config_for();
  cfg.memory_budget_mb = 8;
  size_t calls = 0;
  Matrix2Df source;
  REQUIRE_THROWS(persist_forward_drizzle_uniform(fixture.root,plan,
      [&](size_t) -> const Matrix2Df & { ++calls; return source; },cfg));
  REQUIRE(calls == 0);
  REQUIRE(fs::is_empty(fixture.root));
}

TEST_CASE("drizzle store: bounded region reads preserve layout and reject invalid requests", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for();
  auto cfg = config_for();
  Matrix2Df source(16,16);
  for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) source(y,x) = y * 100 + x;
  persist_forward_drizzle_uniform(fixture.root,plan,
      [&](size_t) -> const Matrix2Df & { return source; },cfg);
  const auto identity = make_drizzle_store_identity(plan,cfg);
  const auto roi = read_drizzle_profile_region(fixture.root,identity,"uniform","L",3,5,4,6,16);
  REQUIRE(roi.width == 4);
  REQUIRE(roi.height == 6);
  for (int y = 0; y < 6; ++y) for (int x = 0; x < 4; ++x) {
    REQUIRE(roi.value[y * 4 + x] == source(y + 5,x + 3));
    REQUIRE(roi.weight_sum[y * 4 + x] == 1.0f);
    REQUIRE(roi.n_eff[y * 4 + x] == 1.0f);
    REQUIRE(roi.support[y * 4 + x] == 1);
  }
  REQUIRE_THROWS(read_drizzle_profile_region(fixture.root,identity,"uniform","L",15,0,2,1));
  REQUIRE_THROWS(read_drizzle_profile_region(fixture.root,identity,"raw","L",0,0,1,1));
  REQUIRE_THROWS(read_drizzle_profile_region(fixture.root,identity,"uniform","L",0,0,16,16,8));
  fs::remove(verify_drizzle_profile_store(fixture.root,identity).generation_dir / "uniform_L_support.fits");
  REQUIRE_THROWS(read_drizzle_profile_region(fixture.root,identity,"uniform","L",0,0,1,1));
}

TEST_CASE("drizzle store: malformed commit never selects an unchecked generation", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for();
  auto cfg = config_for();
  Matrix2Df source = Matrix2Df::Ones(16,16);
  persist_forward_drizzle_uniform(fixture.root,plan,
      [&](size_t) -> const Matrix2Df & { return source; },cfg);
  const auto identity = make_drizzle_store_identity(plan,cfg);
  auto commit = read_json(fixture.root / "current.json");
  { std::ofstream f(fixture.root / "current.json",std::ios::app); f << " trailing garbage"; }
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,identity).usable);
  commit["generation"] = "../escape";
  rehash_commit(commit);
  core::write_text_atomic(fixture.root / "current.json",commit.dump());
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,identity).usable);
}

TEST_CASE("drizzle store: 2/1 mode persists at output (1x) resolution, "
          "bit-identical to the non-streaming reference (plan 12.1)", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(/*size=*/12, /*osc=*/true);
  registration::FrameSamplingTransform f0 = plan.frames[0], f1 = f0;
  f0.frame_id = "synthetic:0"; f0.source_index = 0;
  f1.frame_id = "synthetic:1"; f1.source_index = 1;
  plan.frames = {f0, f1};

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.output_scale = 1;   // 2/1
  cfg.pixfrac = 0.8f;
  cfg.chunk_rows = 3;
  cfg.memory_budget_mb = 64;
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;

  Matrix2Df src(12, 12);
  for (int y = 0; y < 12; ++y)
    for (int x = 0; x < 12; ++x) src(y, x) = 3.0f + 0.4f * x - 0.1f * y;
  SourceImageProvider provider = [&](size_t) -> const Matrix2Df & { return src; };

  const auto result = persist_forward_drizzle_uniform_and_raw(fixture.root, plan, provider, cfg, clip);
  const auto identity = make_drizzle_store_identity(plan, cfg, {}, &clip, {}, {});
  // canvas_native = 12, internal = 24, output(2/1) = 12.
  REQUIRE(identity.width == 12);
  REQUIRE(identity.height == 12);
  REQUIRE(verify_drizzle_profile_store(fixture.root, identity).usable);

  // Non-streaming reference at the same 1x geometry.
  auto reference = downsample_uniform_and_raw_2x2(
      compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clip));

  auto stored = io::read_fits_pixels_float(result.generation_dir / "uniform_G_value.fits");
  REQUIRE(stored.rows() == 12);
  REQUIRE(stored.cols() == 12);
  auto sup = io::read_fits_pixels_float(result.generation_dir / "uniform_G_support.fits");
  for (int y = 0; y < 12; ++y)
    for (int x = 0; x < 12; ++x)
      if (sup(y, x) == 1.0f)
        REQUIRE(stored(y, x) == reference.uniform.G.value[static_cast<size_t>(y) * 12 + x]);

  // The 2/1 identity hash differs from the same config left at 2/2.
  auto cfg22 = cfg; cfg22.output_scale = 2;
  REQUIRE_FALSE(make_drizzle_store_identity(plan, cfg22, {}, &clip, {}, {}) == identity);
}

namespace {
// Reassemble a full-frame ForwardDrizzleUniformResult (MONO) from the stored
// value/weight/n_eff/support planes of one profile.
ForwardDrizzleUniformResult read_mono_profile(const fs::path &root,
                                              const DrizzleStoreIdentity &id,
                                              const std::string &profile) {
  ForwardDrizzleUniformResult r;
  r.color_mode = ColorMode::MONO;
  r.internal_width = id.width;
  r.internal_height = id.height;
  r.L = read_drizzle_profile_region(root, id, profile, "L", 0, 0, id.width,
                                    id.height, 256);
  return r;
}
std::vector<float> read_alpha_map(const fs::path &root,
                                  const DrizzleStoreIdentity &id,
                                  const std::string &name) {
  return read_drizzle_profile_region(root, id, name, "X", 0, 0, id.width,
                                     id.height, 256)
      .value;
}
}  // namespace

TEST_CASE("drizzle store: multiband store round-trips U/R/F/M + the four "
          "alpha-confidence maps, and fuse_multiband_streamed on the read-back "
          "planes is bit-identical to the in-memory reference", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(/*size=*/20);
  plan.canvas_width_native = plan.canvas_height_native = 20;
  plan.frames.clear();
  for (int i = 0; i < 6; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    plan.frames.push_back(f);
  }

  config::ReconstructionDrizzleConfig cfg = config_for();  // 1/1, pixfrac 1
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  clip.min_fraction = 0.1f;
  cfg.min_clip_contributors = 7;  // > 6 frames: no clipping

  std::vector<Matrix2Df> imgs(6, Matrix2Df(20, 20));
  for (int i = 0; i < 6; ++i)
    for (int y = 0; y < 20; ++y)
      for (int x = 0; x < 20; ++x)
        imgs[i](y, x) = 100.0f + 7.0f * std::sin(0.2f * x) + 4.0f * std::cos(0.15f * y);
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & { return imgs[i]; };

  Matrix2Df comp = Matrix2Df::Constant(20, 20, 0.7f);
  Matrix2Df s0 = Matrix2Df::Constant(20, 20, 0.6f);
  Matrix2Df s1 = Matrix2Df::Constant(20, 20, 0.65f);
  Matrix2Df art = Matrix2Df::Constant(20, 20, 0.9f);
  FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
    return {&comp, &s0, &s1, &art};
  };

  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 3;

  const auto result = persist_forward_drizzle_multiband(
      fixture.root, plan, provider, cfg, clip, mbc, quality_of);
  const auto identity = make_drizzle_store_identity(plan, cfg, {}, &clip, {}, {}, mbc);
  REQUIRE(identity.mode == "uniform_raw_multiband_clipped");
  REQUIRE(identity.multiband_levels == 3);
  REQUIRE(verify_drizzle_profile_store(fixture.root, identity).usable);

  // A non-multiband expectation must NOT validate this generation.
  REQUIRE_FALSE(verify_drizzle_profile_store(
                    fixture.root, make_drizzle_store_identity(plan, cfg, {}, &clip))
                    .usable);
  // Wrong band count must not validate either.
  auto id2 = identity; id2.multiband_levels = 2;
  REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root, id2).usable);

  // Read every plane back and fuse.
  const auto U = read_mono_profile(fixture.root, identity, "uniform");
  const auto R = read_mono_profile(fixture.root, identity, "raw");
  const auto F = read_mono_profile(fixture.root, identity, "fine");
  const auto M = read_mono_profile(fixture.root, identity, "medium");
  const auto a_sep = read_alpha_map(fixture.root, identity, "alpha_separation");
  const auto a_art = read_alpha_map(fixture.root, identity, "alpha_artifact");
  const auto a_reg = read_alpha_map(fixture.root, identity, "alpha_registration");

  config::ReconstructionMultibandConfig fcfg;
  fcfg.levels = 3;
  const auto stored_xout =
      fuse_multiband_streamed(U, R, F, M, ColorMode::MONO, identity.width,
                              identity.height, fcfg, /*chunk_rows=*/5, {}, {},
                              a_sep, a_art, a_reg, {});

  // In-memory reference: same drizzle + same fuse.
  MultibandProfileParams mb;
  mb.emit_fine = true; mb.emit_medium = true; mb.emit_alpha_confidence = true;
  mb.fine_quality_exponent = mbc.fine_quality_exponent;
  mb.medium_quality_exponent = mbc.medium_quality_exponent;
  const auto ref_dz = compute_forward_drizzle_uniform_and_raw(
      plan, provider, cfg, clip, {}, {}, quality_of, mb);
  const auto ref_xout = fuse_multiband(
      ref_dz.uniform, ref_dz.raw, ref_dz.fine, ref_dz.medium, ColorMode::MONO,
      identity.width, identity.height, fcfg, {}, {}, ref_dz.a_separation,
      ref_dz.a_artifact, ref_dz.a_registration, {});

  int checked = 0;
  for (int i = 0; i < identity.width * identity.height; ++i) {
    REQUIRE(stored_xout.support_L[i] == ref_xout.support_L[i]);
    if (stored_xout.support_L[i]) {
      REQUIRE(stored_xout.L[i] == ref_xout.L[i]);  // bit-exact
      ++checked;
    }
  }
  REQUIRE(checked > 0);

  // fuse_multiband_store_to_image also fills the plan-15 candidate fields in
  // the same pass. For MONO the working luminance IS the L channel, so the
  // multiband candidate must be bit-identical to the fused X_out where the
  // fusion is supported (and NaN elsewhere); the uniform candidate carries the
  // UNIFORM profile's own support (not the fused one); alpha_final_by_band has
  // one entry per fused band with D3 (Raw-sourced) empty; none of it depends
  // on chunk.
  reconstruction::MultibandCandidateLuma cand;
  reconstruction::fuse_multiband_store_to_image(
      fixture.root, identity, fixture.root / "x_cand.fits", fcfg,
      /*chunk_rows=*/7, 64, &cand);
  REQUIRE(cand.width == identity.width);
  REQUIRE(cand.height == identity.height);
  REQUIRE(cand.alpha_final_by_band.size() == 3u);
  REQUIRE(cand.alpha_final_by_band[2].empty());        // D3 <- Raw
  REQUIRE_FALSE(cand.alpha_final_by_band[0].empty());  // D1 <- Fine
  int luma_checked = 0;
  for (int i = 0; i < identity.width * identity.height; ++i) {
    REQUIRE(cand.uniform_support[i] == U.L.support[i]);  // uniform's own support
    if (ref_xout.support_L[i]) {
      REQUIRE(cand.multiband_luma[i] == ref_xout.L[i]);  // bit-exact
      ++luma_checked;
    } else {
      REQUIRE(std::isnan(cand.multiband_luma[i]));
    }
  }
  REQUIRE(luma_checked == checked);
  reconstruction::MultibandCandidateLuma cand2;
  reconstruction::fuse_multiband_store_to_image(
      fixture.root, identity, fixture.root / "x_cand2.fits", fcfg,
      /*chunk_rows=*/2, 64, &cand2);
  auto same = [](const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      const bool na = std::isnan(a[i]), nb = std::isnan(b[i]);
      if (na != nb || (!na && a[i] != b[i])) return false;
    }
    return true;
  };
  REQUIRE(same(cand2.multiband_luma, cand.multiband_luma));
  REQUIRE(same(cand2.uniform_luma, cand.uniform_luma));
  REQUIRE(same(cand2.raw_luma, cand.raw_luma));
  REQUIRE(same(cand2.alpha_final_by_band[0], cand.alpha_final_by_band[0]));
}

TEST_CASE("drizzle store: OSC multiband store + fuse_multiband_store_to_image "
          "writes an RGB X_out bit-identical to the in-memory reference "
          "(the object-agnostic OSC path, e.g. M42)", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(/*size=*/18, /*osc=*/true);
  plan.canvas_width_native = plan.canvas_height_native = 18;
  plan.frames.clear();
  for (int i = 0; i < 8; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    plan.frames.push_back(f);
  }
  config::ReconstructionDrizzleConfig cfg = config_for();  // 1/1
  cfg.min_clip_contributors = 9;  // > 8 frames: no clipping
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  clip.min_fraction = 0.1f;

  // A high-dynamic-range field (a bright compact core over a faint gradient) --
  // the M42 regime, not M31's smooth extended structure.
  std::vector<Matrix2Df> imgs(8, Matrix2Df(18, 18));
  for (int i = 0; i < 8; ++i)
    for (int y = 0; y < 18; ++y)
      for (int x = 0; x < 18; ++x) {
        const double r2 = (x - 9.0) * (x - 9.0) + (y - 8.0) * (y - 8.0);
        imgs[i](y, x) = 30.0f + 0.6f * x + 0.4f * y +
                        900.0f * std::exp(-r2 / 6.0);  // bright core
      }
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & { return imgs[i]; };

  Matrix2Df comp = Matrix2Df::Constant(18, 18, 0.7f);
  Matrix2Df s0 = Matrix2Df::Constant(18, 18, 0.6f);
  Matrix2Df s1 = Matrix2Df::Constant(18, 18, 0.65f);
  Matrix2Df art = Matrix2Df::Constant(18, 18, 0.9f);
  FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
    return {&comp, &s0, &s1, &art};
  };

  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 3;

  const auto result = persist_forward_drizzle_multiband(
      fixture.root, plan, provider, cfg, clip, mbc, quality_of);
  const auto identity = result.identity;
  REQUIRE(identity.color_mode == ColorMode::OSC);
  REQUIRE(verify_drizzle_profile_store(fixture.root, identity).usable);

  const auto out_fits = fixture.root / "reconstruction_multiband.fits";
  config::ReconstructionMultibandConfig fcfg;
  fcfg.levels = 3;
  const long long pixels = reconstruction::fuse_multiband_store_to_image(
      fixture.root, identity, out_fits, fcfg, /*chunk_rows=*/4, /*mb=*/64);
  REQUIRE(pixels > 0);

  const auto rgb = io::read_fits_rgb(out_fits);
  REQUIRE(rgb.R.rows() == identity.height);
  REQUIRE(rgb.R.cols() == identity.width);

  // In-memory reference.
  MultibandProfileParams mb;
  mb.emit_fine = true; mb.emit_medium = true; mb.emit_alpha_confidence = true;
  mb.fine_quality_exponent = mbc.fine_quality_exponent;
  mb.medium_quality_exponent = mbc.medium_quality_exponent;
  const auto ref_dz = compute_forward_drizzle_uniform_and_raw(
      plan, provider, cfg, clip, {}, {}, quality_of, mb);
  const auto ref = fuse_multiband(ref_dz.uniform, ref_dz.raw, ref_dz.fine,
                                  ref_dz.medium, ColorMode::OSC, identity.width,
                                  identity.height, fcfg, {}, {},
                                  ref_dz.a_separation, ref_dz.a_artifact,
                                  ref_dz.a_registration, {});
  REQUIRE(pixels == ref.pixels_supported);  // exact, not just > 0
  int checked = 0;
  for (int y = 0; y < identity.height; ++y)
    for (int x = 0; x < identity.width; ++x) {
      const size_t i = static_cast<size_t>(y) * identity.width + x;
      auto chk = [&](float got, float exp, const std::vector<uint8_t> &sup) {
        if (!sup[i]) return;
        if (std::isfinite(exp)) { REQUIRE(got == exp); ++checked; }
      };
      chk(rgb.R(y, x), ref.R[i], ref.support_R);
      chk(rgb.G(y, x), ref.G[i], ref.support_G);
      chk(rgb.B(y, x), ref.B[i], ref.support_B);
    }
  REQUIRE(checked > 0);

  // The striped store-read fusion must not depend on chunk height (guards the
  // halo/seam logic on the store path, not just the in-memory one).
  const std::string base_sha = core::sha256_file(out_fits);
  for (int cr : {2, 7, identity.height}) {
    const auto alt = fixture.root / ("x_" + std::to_string(cr) + ".fits");
    reconstruction::fuse_multiband_store_to_image(fixture.root, identity, alt,
                                                  fcfg, cr, 64);
    REQUIRE(core::sha256_file(alt) == base_sha);
  }

  // The plan-15 candidate fields assembled in the same pass (OSC path).
  // alpha_final_by_band carries one entry per fused band with D3 (Raw) empty;
  // none of it depends on chunk height (checked below).
  //
  // The OSC working luminance needs R AND G AND B co-support in the SAME output
  // cell. This non-dithered CFA fixture never co-locates all three colours, so
  // luma support is legitimately empty everywhere -- what stays non-vacuous
  // here is that `combine_luma` does NOT over-claim: uniform_support must match
  // the tri-channel predicate exactly (i.e. be all-zero) and multiband_luma
  // must be all-NaN. The bit-exact OSC luma combine is exercised by the real
  // M42/OSC registration run (a dithered geometry); the accumulation/striping
  // maths is covered bit-exact by the MONO store test above.
  reconstruction::MultibandCandidateLuma cand;
  reconstruction::MultibandCandidateSpool spool;
  spool.dir = fixture.root / "cand_spool";
  fs::create_directories(spool.dir);
  reconstruction::MultibandFusionMemoryPlan mem_plan;
  reconstruction::fuse_multiband_store_to_image(fixture.root, identity,
      fixture.root / "x_cand.fits", fcfg, /*chunk_rows=*/5, 64, &cand, &spool,
      &mem_plan);
  REQUIRE(cand.width == identity.width);
  REQUIRE(cand.height == identity.height);
  REQUIRE(cand.alpha_final_by_band.size() == 3u);
  REQUIRE(cand.alpha_final_by_band[2].empty());          // D3 <- Raw
  REQUIRE(mem_plan.fits);
  REQUIRE(mem_plan.estimated_peak_bytes > 0u);

  // Plan 11.13(2) / 16.1: the per-channel candidate capture, now streamed to a
  // spool. The read-back "multiband" plane is the fused X_out per channel --
  // bit-exact to the non-streaming reference where the channel is supported,
  // NaN elsewhere.
  REQUIRE(spool.nch == 3);
  REQUIRE_FALSE(spool.mono);
  REQUIRE(spool.width == identity.width);
  REQUIRE(spool.height == identity.height);
  REQUIRE(spool.populated);
  {
    const std::vector<float> *rref[3] = {&ref.R, &ref.G, &ref.B};
    const std::vector<uint8_t> *sref[3] = {&ref.support_R, &ref.support_G,
                                           &ref.support_B};
    for (int c = 0; c < 3; ++c) {
      const auto plane =
          reconstruction::read_candidate_spool_plane(spool, "multiband", c);
      REQUIRE(plane.size() == rref[c]->size());
      for (size_t i = 0; i < plane.size(); ++i) {
        if ((*sref[c])[i] && std::isfinite((*rref[c])[i]))
          REQUIRE(plane[i] == (*rref[c])[i]);
        else
          REQUIRE(std::isnan(plane[i]));
      }
    }
  }
  int tri_cells = 0, luma_cells = 0;
  for (int y = 0; y < identity.height; ++y)
    for (int x = 0; x < identity.width; ++x) {
      const size_t i = static_cast<size_t>(y) * identity.width + x;
      const bool tri = ref.support_R[i] && ref.support_G[i] && ref.support_B[i] &&
                       std::isfinite(ref.R[i]) && std::isfinite(ref.G[i]) &&
                       std::isfinite(ref.B[i]);
      REQUIRE(cand.uniform_support[i] == (tri ? 1u : 0u));
      if (tri) {
        const double exp = 0.25 * ref.R[i] + 0.5 * ref.G[i] + 0.25 * ref.B[i];
        REQUIRE(cand.multiband_luma[i] == static_cast<float>(exp));  // bit-exact
        ++tri_cells;
      } else {
        REQUIRE(std::isnan(cand.multiband_luma[i]));
      }
      if (cand.uniform_support[i]) ++luma_cells;
    }
  INFO("OSC tri-channel co-supported luma cells = " << tri_cells);
  REQUIRE(luma_cells == tri_cells);  // combine_luma AND-support is exact
  reconstruction::MultibandCandidateLuma cand2;
  reconstruction::fuse_multiband_store_to_image(fixture.root, identity,
      fixture.root / "x_cand2.fits", fcfg, /*chunk_rows=*/2, 64, &cand2);
  auto same = [](const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      const bool na = std::isnan(a[i]), nb = std::isnan(b[i]);
      if (na != nb || (!na && a[i] != b[i])) return false;
    }
    return true;
  };
  REQUIRE(same(cand2.multiband_luma, cand.multiband_luma));
  REQUIRE(same(cand2.uniform_luma, cand.uniform_luma));
  REQUIRE(same(cand2.raw_luma, cand.raw_luma));
  REQUIRE(cand2.uniform_support == cand.uniform_support);
  REQUIRE(cand2.alpha_final_by_band.size() == cand.alpha_final_by_band.size());
  for (size_t b = 0; b < cand.alpha_final_by_band.size(); ++b)
    REQUIRE(same(cand2.alpha_final_by_band[b], cand.alpha_final_by_band[b]));
}

// ---------------------------------------------------------------------------
// Plan 11.13: the MULTIBAND fuse / validation / export resource contract.
// ---------------------------------------------------------------------------

TEST_CASE("plan-11.13 multiband working-set planner: exact terms, monotonicity "
          "and the fail-closed budget boundary", "[drizzle-store]") {
  using reconstruction::plan_multiband_fusion_memory;

  // Degenerate inputs never "fit".
  REQUIRE_FALSE(plan_multiband_fusion_memory(0, 10, 1, 3, 8, 1, false, false,
                                             std::size_t(1) << 30)
                    .fits);
  REQUIRE_FALSE(plan_multiband_fusion_memory(10, 10, 3, 0, 8, 1, false, false,
                                             std::size_t(1) << 30)
                    .fits);
  REQUIRE_FALSE(plan_multiband_fusion_memory(10, 10, 3, 3, 0, 1, false, false,
                                             std::size_t(1) << 30)
                    .fits);

  const int W = 4000, H = 3000, LV = 3;
  const std::size_t N = static_cast<std::size_t>(W) * H;
  const std::size_t big = std::size_t(64) << 30;  // 64 GiB: never the limiter

  // MONO (nch 1) and OSC (nch 3): final image is exactly nch * N * 4.
  const auto m = plan_multiband_fusion_memory(W, H, 1, LV, 64, 1, true, true, big);
  const auto o = plan_multiband_fusion_memory(W, H, 3, LV, 64, 1, true, true, big);
  REQUIRE(m.final_image_bytes == N * 4);
  REQUIRE(o.final_image_bytes == 3 * N * 4);
  // Candidate working-luminance buffers: N * (3*4 + 1 + levels*4) when captured,
  // 0 when not.
  REQUIRE(o.candidate_luma_bytes == N * (13 + LV * 4));
  const auto o_noluma =
      plan_multiband_fusion_memory(W, H, 3, LV, 64, 1, false, true, big);
  REQUIRE(o_noluma.candidate_luma_bytes == 0u);
  // The spool term appears only when candidate channels are spooled.
  REQUIRE(o.spool_stripe_bytes > 0u);
  REQUIRE(o.delivery_readback_bytes == N * 4);
  // Plan 11.11 temp space: spooled bytes are 3 * nch * N * 4; required free temp
  // is spool*1.2 + at least the 2 GiB floor.
  REQUIRE(o.spool_temp_bytes == 3u * 3u * N * 4u);
  REQUIRE(o.required_free_temp_bytes >
          o.spool_temp_bytes + (std::size_t(2) << 30) - 1u);
  const auto o_nospool =
      plan_multiband_fusion_memory(W, H, 3, LV, 64, 1, true, false, big);
  REQUIRE(o_nospool.spool_stripe_bytes == 0u);
  REQUIRE(o_nospool.delivery_readback_bytes == 0u);
  REQUIRE(o_nospool.spool_temp_bytes == 0u);
  // estimated_peak == sum of the six sub-terms.
  REQUIRE(o.estimated_peak_bytes ==
          o.final_image_bytes + o.stripe_working_bytes + o.candidate_luma_bytes +
              o.spool_stripe_bytes + o.delivery_readback_bytes + o.margin_bytes);
  // OSC needs strictly more than MONO for the same geometry.
  REQUIRE(o.estimated_peak_bytes > m.estimated_peak_bytes);
  // Bigger canvas -> bigger estimate (monotone in N).
  const auto small =
      plan_multiband_fusion_memory(1000, 1000, 3, LV, 64, 1, true, true, big);
  REQUIRE(small.estimated_peak_bytes < o.estimated_peak_bytes);

  // Fail-closed boundary: an explicit tiny budget does not fit; a budget just
  // above the estimate does. No silent raising -- the planner honours the value.
  const auto tight = plan_multiband_fusion_memory(W, H, 3, LV, 64, 1, true, true,
                                                  std::size_t(8) << 20);
  REQUIRE_FALSE(tight.fits);
  REQUIRE(tight.budget_bytes == (std::size_t(8) << 20));
  const auto exact = plan_multiband_fusion_memory(W, H, 3, LV, 64, 1, true, true,
                                                  o.estimated_peak_bytes);
  REQUIRE(exact.fits);
  const auto under = plan_multiband_fusion_memory(
      W, H, 3, LV, 64, 1, true, true, o.estimated_peak_bytes - 1);
  REQUIRE_FALSE(under.fits);
}

namespace {
// A compact MONO multiband store fixture shared by the resource-contract tests.
struct MbFixture {
  Fixture fixture;
  reconstruction::DrizzleStoreIdentity identity;
  config::ReconstructionMultibandConfig fcfg;
  // Members (not statics): they only need to outlive the constructor body, where
  // persist_forward_drizzle_multiband consumes the provider/quality lambdas
  // synchronously. Statics would cross-contaminate a reordered/parallel run.
  std::vector<Matrix2Df> imgs;
  Matrix2Df comp, s0, s1, art;
  explicit MbFixture(int S = 16) {
    auto plan = plan_for(S);
    plan.canvas_width_native = plan.canvas_height_native = S;
    plan.frames.clear();
    for (int i = 0; i < 5; ++i) {
      registration::FrameSamplingTransform f;
      f.valid = f.source_to_canvas_affine_valid = true;
      f.frame_id = "synthetic:" + std::to_string(i);
      f.source_index = static_cast<size_t>(i);
      f.model_prediction_factor = 1.0f;
      f.registration_residual_factor = 1.0f;
      plan.frames.push_back(f);
    }
    config::ReconstructionDrizzleConfig cfg = config_for();
    cfg.min_clip_contributors = 6;  // > 5 frames: no clipping
    config::ReconstructionClippingConfig clip;
    clip.min_n_eff = 1.0f;
    clip.min_fraction = 0.1f;
    imgs.assign(5, Matrix2Df(S, S));
    for (int i = 0; i < 5; ++i)
      for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
          imgs[i](y, x) = 100.0f + 6.0f * std::sin(0.3f * x) +
                          3.0f * std::cos(0.2f * y) +
                          40.0f * std::exp(-((x - S / 2.0) * (x - S / 2.0) +
                                             (y - S / 2.0) * (y - S / 2.0)) /
                                           5.0);
    SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & {
      return imgs[i];
    };
    comp = Matrix2Df::Constant(S, S, 0.7f);
    s0 = Matrix2Df::Constant(S, S, 0.6f);
    s1 = Matrix2Df::Constant(S, S, 0.65f);
    art = Matrix2Df::Constant(S, S, 0.9f);
    FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
      return {&comp, &s0, &s1, &art};
    };
    MultibandStoreContract mbc;
    mbc.enabled = true;
    mbc.levels = 3;
    const auto result = persist_forward_drizzle_multiband(
        fixture.root, plan, provider, cfg, clip, mbc, quality_of);
    identity = result.identity;
    fcfg.levels = 3;
    REQUIRE(verify_drizzle_profile_store(fixture.root, identity).usable);
  }
};

std::string run_and_catch(const std::function<void()> &fn) {
  try {
    fn();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "<no throw>";
}
}  // namespace

TEST_CASE("plan-11.13(1)(3): an explicit small budget is rejected before any "
          "large allocation and the prior generation is untouched",
          "[drizzle-store]") {
  MbFixture mb(200);  // large enough that a 1 MiB budget cannot hold the plan
  const auto prior_current = read_json(mb.fixture.root / "current.json");
  const auto target = mb.fixture.root / "x_budget.fits";

  reconstruction::MultibandCandidateLuma cand;
  reconstruction::MultibandCandidateSpool spool;
  spool.dir = mb.fixture.root / "spool_budget";
  fs::create_directories(spool.dir);
  reconstruction::MultibandFusionMemoryPlan plan;

  const auto msg = run_and_catch([&] {
    reconstruction::fuse_multiband_store_to_image(
        mb.fixture.root, mb.identity, target, mb.fcfg, /*chunk_rows=*/4,
        /*mb=*/1, &cand, &spool, &plan);
  });
  REQUIRE(msg.find("MULTIBAND_MEMORY_BUDGET") != std::string::npos);
  REQUIRE_FALSE(plan.fits);
  REQUIRE(plan.budget_bytes == (std::size_t(1) << 20));
  // Nothing was produced; nothing prior was disturbed.
  REQUIRE_FALSE(fs::exists(target));
  REQUIRE_FALSE(spool.populated);
  REQUIRE(read_json(mb.fixture.root / "current.json") == prior_current);
  REQUIRE(verify_drizzle_profile_store(mb.fixture.root, mb.identity).usable);

  // A non-zero budget is honoured verbatim (no silent raise to the old 256
  // floor): the same call under a generous explicit budget now succeeds.
  reconstruction::MultibandCandidateSpool spool_ok;
  spool_ok.dir = mb.fixture.root / "spool_ok";
  fs::create_directories(spool_ok.dir);
  REQUIRE_NOTHROW(reconstruction::fuse_multiband_store_to_image(
      mb.fixture.root, mb.identity, target, mb.fcfg, /*chunk_rows=*/4,
      /*mb=*/256, &cand, &spool_ok, &plan));
  REQUIRE(plan.fits);
  REQUIRE(spool_ok.populated);
  REQUIRE(fs::exists(target));
}

TEST_CASE("plan-11.13(5): injected spool failure preserves the prior generation "
          "and writes no final image", "[drizzle-store]") {
  MbFixture mb;
  const auto prior_current = read_json(mb.fixture.root / "current.json");
  const auto target = mb.fixture.root / "x_injfail.fits";

  reconstruction::MultibandCandidateSpool spool;
  spool.dir = mb.fixture.root / "does_not_exist";  // never created
  reconstruction::MultibandFusionMemoryPlan plan;
  const auto msg = run_and_catch([&] {
    reconstruction::fuse_multiband_store_to_image(
        mb.fixture.root, mb.identity, target, mb.fcfg, /*chunk_rows=*/4,
        /*mb=*/256, nullptr, &spool, &plan);
  });
  REQUIRE(msg.find("SPOOL_DIR_MISSING") != std::string::npos);
  REQUIRE_FALSE(fs::exists(target));
  REQUIRE_FALSE(spool.populated);
  REQUIRE(read_json(mb.fixture.root / "current.json") == prior_current);
  REQUIRE(verify_drizzle_profile_store(mb.fixture.root, mb.identity).usable);
}

TEST_CASE("plan-11.13(5): chunk-height variation does not change the spooled "
          "candidates, the luma masks or the plan-15 selection outcome",
          "[drizzle-store]") {
  MbFixture mb;
  const int H = mb.identity.height;

  auto to_mat = [](const std::vector<float> &v, int w, int h) {
    Matrix2Df m(h, w);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        m(y, x) = v[static_cast<std::size_t>(y) * w + x];
    return m;
  };

  struct Run {
    reconstruction::MultibandCandidateLuma cand;
    reconstruction::MultibandCandidateSpool spool;
    reconstruction::SelectedCandidate selected{};
    std::array<std::string, 3> plane_sha;  // sha of "uniform"/"raw"/"multiband"
    std::vector<bool> applicable;          // per-metric applicability bitset
  };
  static constexpr const char *kNames[3] = {"uniform", "raw", "multiband"};

  auto do_run = [&](int chunk) {
    Run r;
    r.spool.dir = mb.fixture.root / ("spool_cr" + std::to_string(chunk));
    fs::create_directories(r.spool.dir);
    reconstruction::MultibandFusionMemoryPlan plan;
    reconstruction::fuse_multiband_store_to_image(
        mb.fixture.root, mb.identity,
        mb.fixture.root / ("x_cr" + std::to_string(chunk) + ".fits"), mb.fcfg,
        chunk, /*mb=*/256, &r.cand, &r.spool, &plan);
    REQUIRE(plan.fits);
    const int W = r.cand.width;
    const auto uni = to_mat(r.cand.uniform_luma, W, r.cand.height);
    const auto raw = to_mat(r.cand.raw_luma, W, r.cand.height);
    const auto mbm = to_mat(r.cand.multiband_luma, W, r.cand.height);
    const auto stars = reconstruction::prepare_validation_samples(
        uni, W, r.cand.height, r.cand.uniform_support,
        r.cand.alpha_final_by_band);
    const reconstruction::MultibandValidationConfig vcfg{};
    const auto sel = reconstruction::select_reconstruction_candidate(
        uni, raw, mbm, W, r.cand.height, stars, vcfg, r.cand.uniform_support);
    r.selected = sel.selected;
    for (const auto *cm : {&sel.uniform, &sel.raw, &sel.multiband}) {
      for (bool a : {cm->median_fwhm.applicable, cm->p90_fwhm.applicable,
                     cm->tail.applicable, cm->elongation.applicable,
                     cm->background_rms.applicable, cm->seam_score.applicable})
        r.applicable.push_back(a);
    }
    for (int k = 0; k < 3; ++k)
      r.plane_sha[static_cast<std::size_t>(k)] = core::sha256_bytes([&] {
        const auto p =
            reconstruction::read_candidate_spool_plane(r.spool, kNames[k], 0);
        const auto *raw_bytes = reinterpret_cast<const uint8_t *>(p.data());
        return std::vector<uint8_t>(raw_bytes,
                                    raw_bytes + p.size() * sizeof(float));
      }());
    return r;
  };

  const auto a = do_run(1);
  const auto b = do_run(3);
  const auto c = do_run(H);  // single chunk, no halo split

  REQUIRE(a.selected == b.selected);
  REQUIRE(b.selected == c.selected);
  REQUIRE(a.applicable == b.applicable);
  REQUIRE(b.applicable == c.applicable);
  REQUIRE(a.plane_sha == b.plane_sha);
  REQUIRE(b.plane_sha == c.plane_sha);
}

TEST_CASE("drizzle store: multiband 2/1 store persists at output (1x) resolution "
          "and matches downsample_uniform_and_raw_2x2 of the non-streaming "
          "reference (fine/medium + the channel-min alpha maps)", "[drizzle-store]") {
  Fixture fixture;
  auto plan = plan_for(/*size=*/16);
  plan.canvas_width_native = plan.canvas_height_native = 16;
  plan.frames.clear();
  for (int i = 0; i < 6; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    plan.frames.push_back(f);
  }
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 2;
  cfg.output_scale = 1;   // 2/1
  cfg.pixfrac = 0.8f;
  cfg.chunk_rows = 4;
  cfg.memory_budget_mb = 64;
  cfg.min_clip_contributors = 7;
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  clip.min_fraction = 0.1f;

  std::vector<Matrix2Df> imgs(6, Matrix2Df(16, 16));
  for (int i = 0; i < 6; ++i)
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x)
        imgs[i](y, x) = 80.0f + 5.0f * std::sin(0.3f * x + i) + 3.0f * std::cos(0.2f * y);
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & { return imgs[i]; };

  Matrix2Df comp = Matrix2Df::Constant(16, 16, 0.7f);
  Matrix2Df s0 = Matrix2Df::Constant(16, 16, 0.6f);
  Matrix2Df s1 = Matrix2Df::Constant(16, 16, 0.65f);
  Matrix2Df art = Matrix2Df::Constant(16, 16, 0.9f);
  FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
    return {&comp, &s0, &s1, &art};
  };

  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 3;

  const auto result = persist_forward_drizzle_multiband(
      fixture.root, plan, provider, cfg, clip, mbc, quality_of);
  const auto identity = result.identity;
  REQUIRE(identity.mode == "uniform_raw_multiband_clipped");
  // canvas 16 -> internal 32 -> output (2/1) 16.
  REQUIRE(identity.width == 16);
  REQUIRE(identity.height == 16);
  REQUIRE(verify_drizzle_profile_store(fixture.root, identity).usable);

  MultibandProfileParams mb;
  mb.emit_fine = true; mb.emit_medium = true; mb.emit_alpha_confidence = true;
  mb.fine_quality_exponent = mbc.fine_quality_exponent;
  mb.medium_quality_exponent = mbc.medium_quality_exponent;
  const auto reference = downsample_uniform_and_raw_2x2(
      compute_forward_drizzle_uniform_and_raw(plan, provider, cfg, clip, {}, {},
                                              quality_of, mb));
  // The reference confidence maps must be non-empty at halved geometry, or the
  // comparison below is vacuous.
  REQUIRE(reference.alpha_confidence_support.size() ==
          static_cast<size_t>(16) * 16);
  bool any_conf = false;
  for (auto s : reference.alpha_confidence_support) any_conf |= (s != 0u);
  REQUIRE(any_conf);

  auto cmp_plane = [&](const std::string &profile, const ProfilePlane &ref) {
    auto stored = io::read_fits_pixels_float(
        result.generation_dir / (profile + "_L_value.fits"));
    auto sup = io::read_fits_pixels_float(
        result.generation_dir / (profile + "_L_support.fits"));
    int checked = 0;
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x)
        if (sup(y, x) == 1.0f) {
          REQUIRE(stored(y, x) == ref.value[static_cast<size_t>(y) * 16 + x]);
          ++checked;
        }
    REQUIRE(checked > 0);
  };
  cmp_plane("uniform", reference.uniform.L);
  cmp_plane("raw", reference.raw.L);
  cmp_plane("fine", reference.fine.L);
  cmp_plane("medium", reference.medium.L);

  auto cmp_alpha = [&](const std::string &name, const std::vector<float> &ref) {
    auto stored = io::read_fits_pixels_float(
        result.generation_dir / (name + "_X_value.fits"));
    int checked = 0;
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) {
        const float r = ref[static_cast<size_t>(y) * 16 + x];
        if (std::isfinite(r)) { REQUIRE(stored(y, x) == r); ++checked; }
      }
    REQUIRE(checked > 0);
  };
  cmp_alpha("alpha_separation", reference.a_separation);
  cmp_alpha("alpha_artifact", reference.a_artifact);
  cmp_alpha("alpha_registration", reference.a_registration);

  // Chunk-height independence of the streamed 2x2 multiband path.
  auto digest_planes = [&](const fs::path &gen) {
    std::string d;
    for (const char *p : {"fine_L_value", "medium_L_value", "alpha_separation_X_value",
                          "alpha_artifact_X_value", "alpha_registration_X_value"})
      d += core::sha256_file(gen / (std::string(p) + ".fits"));
    return d;
  };
  const auto base_digest = digest_planes(result.generation_dir);
  for (int cr : {2, 16}) {
    Fixture alt;
    auto c = cfg; c.chunk_rows = cr;
    const auto r2 = persist_forward_drizzle_multiband(alt.root, plan, provider, c,
                                                     clip, mbc, quality_of);
    REQUIRE(digest_planes(r2.generation_dir) == base_digest);
  }
}

TEST_CASE("drizzle store: plan-19.4 CUDA transactional restart -- an injected "
          "CUDA fault discards the uncommitted generation, and the CPU restart "
          "commits a store bit-identical to a pure CPU build", "[drizzle-store]") {
  // A MONO levels-3 multiband fixture tall enough for several stripes so the
  // "fault after N stripes" hook lands mid-stream.
  auto plan = plan_for(/*size=*/48);
  plan.canvas_width_native = plan.canvas_height_native = 48;
  plan.frames.clear();
  for (int i = 0; i < 6; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    plan.frames.push_back(f);
  }
  config::ReconstructionDrizzleConfig cfg = config_for();  // 1/1
  cfg.chunk_rows = 8;                                      // -> 6 stripes
  cfg.min_clip_contributors = 7;
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  clip.min_fraction = 0.1f;

  std::vector<Matrix2Df> imgs(6, Matrix2Df(48, 48));
  for (int i = 0; i < 6; ++i)
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 48; ++x)
        imgs[i](y, x) = 100.0f + 7.0f * std::sin(0.2f * x) + 4.0f * std::cos(0.15f * y);
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & { return imgs[i]; };
  Matrix2Df comp = Matrix2Df::Constant(48, 48, 0.7f);
  Matrix2Df s0 = Matrix2Df::Constant(48, 48, 0.6f);
  Matrix2Df s1 = Matrix2Df::Constant(48, 48, 0.65f);
  Matrix2Df art = Matrix2Df::Constant(48, 48, 0.9f);
  FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
    return {&comp, &s0, &s1, &art};
  };
  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 3;
  const auto identity = make_drizzle_store_identity(plan, cfg, {}, &clip, {}, {}, mbc);

  auto digest = [&](const fs::path &gen) {
    std::string d;
    for (const char *pr : {"uniform", "raw", "fine", "medium"})
      for (const char *fld : {"value", "weight_sum", "n_eff", "support"})
        d += core::sha256_file(gen / (std::string(pr) + "_L_" + fld + ".fits"));
    for (const char *a : {"alpha_separation", "alpha_artifact",
                          "alpha_registration", "alpha_support"})
      d += core::sha256_file(gen / (std::string(a) + "_X_value.fits"));
    return d;
  };
  auto count_dirs = [](const fs::path &root) {
    int n = 0;
    for (const auto &e : fs::directory_iterator(root))
      if (e.is_directory()) ++n;
    return n;
  };

  // Pure CPU baseline.
  Fixture cpu_fx;
  const auto cpu = persist_forward_drizzle_multiband(cpu_fx.root, plan, provider,
                                                     cfg, clip, mbc, quality_of);
  const std::string cpu_digest = digest(cpu.generation_dir);

  reconstruction::ForwardDrizzleCudaOptions attempt;
  attempt.attempt = true;

  SECTION("attempt with no fault armed -> the CUDA per-stripe path commits a "
          "store bit-identical to the pure CPU build (affine frames)") {
    CudaFaultGuard guard(-1);
    if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
      // No device: the wiring throws before any generation directory, and the
      // caller (persist_multiband_store_from_predecessors) restarts on CPU.
      Fixture fx;
      REQUIRE_THROWS_AS(
          persist_forward_drizzle_multiband(fx.root, plan, provider, cfg, clip,
                                            mbc, quality_of, {}, {}, {}, attempt),
          reconstruction::ForwardDrizzleCudaError);
      REQUIRE(count_dirs(fx.root) == 0);
    } else {
      Fixture fx;
      const auto gpu = persist_forward_drizzle_multiband(
          fx.root, plan, provider, cfg, clip, mbc, quality_of, {}, {}, {},
          attempt);
      REQUIRE(gpu.cuda_timing.used);
      REQUIRE(gpu.cuda_timing.bands >= 1);
      REQUIRE(digest(gpu.generation_dir) == cpu_digest);
      REQUIRE(verify_drizzle_profile_store(fx.root, identity).usable);
    }
  }

  SECTION("fault after 3 stripes -> throw, uncommitted generation discarded") {
    CudaFaultGuard guard(3);
    Fixture fx;
    REQUIRE_THROWS_AS(
        persist_forward_drizzle_multiband(fx.root, plan, provider, cfg, clip, mbc,
                                          quality_of, {}, {}, {}, attempt),
        reconstruction::ForwardDrizzleCudaError);
    REQUIRE_FALSE(fs::exists(fx.root / "current.json"));
    REQUIRE(count_dirs(fx.root) == 0);  // StoreWriter dtor removed it
  }

  SECTION("fault then CPU restart in the same root -> committed store is "
          "bit-identical to the pure CPU build") {
    Fixture fx;
    {
      CudaFaultGuard guard(2);
      REQUIRE_THROWS_AS(
          persist_forward_drizzle_multiband(fx.root, plan, provider, cfg, clip,
                                            mbc, quality_of, {}, {}, {}, attempt),
          reconstruction::ForwardDrizzleCudaError);
    }
    const auto restart = persist_forward_drizzle_multiband(
        fx.root, plan, provider, cfg, clip, mbc, quality_of);  // cuda = {}
    REQUIRE(digest(restart.generation_dir) == cpu_digest);
    REQUIRE(verify_drizzle_profile_store(fx.root, identity).usable);
  }
}

TEST_CASE("drizzle store: plan-19.6 CUDA per-stripe path commits a store "
          "byte-identical to the CPU reference build",
          "[drizzle-store][cuda-parity]") {
  if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device");
    return;
  }
  // Digest of the committed SCIENTIFIC artifact --- every plane FITS file,
  // keyed by name. The generation directory name embeds a timestamp/counter
  // and commit.json embeds that name, so neither is part of the bit-exactness
  // claim (the existing chunk-invariance tests digest planes only for the same
  // reason).
  auto plane_digest = [](const fs::path &gen) {
    std::map<std::string, std::string> h;
    for (const auto &e : fs::directory_iterator(gen)) {
      const std::string n = e.path().filename().string();
      if (e.is_regular_file() && n.size() > 5 &&
          n.substr(n.size() - 5) == ".fits")
        h[n] = core::sha256_file(e.path());
    }
    REQUIRE(h.size() >= 12);  // uniform/raw/fine/medium x 4 fields + alpha maps
    return h;
  };

  for (bool osc : {false, true}) {
    CAPTURE(osc);
    const int S = 24;           // source
    const int C = 32;           // native canvas
    auto plan = plan_for(S, osc);
    plan.canvas_width_native = plan.canvas_height_native = C;
    plan.frames.clear();
    const int nf = 5;
    for (int i = 0; i < nf; ++i) {
      registration::FrameSamplingTransform f;
      f.valid = f.source_to_canvas_affine_valid = true;
      f.frame_id = "synthetic:" + std::to_string(i);
      f.source_index = static_cast<size_t>(i);
      f.model_prediction_factor = 1.0f;
      f.registration_residual_factor = 1.0f;
      // Sub-pixel rotation + translation so the device rasterizer does real
      // multi-cell work (identity frames would hide most of the geometry).
      const double ang = 0.012 * std::sin(0.5 * i);
      const double dx = 3.0 + ((i * 7) % 5) / 5.0;
      const double dy = 3.0 + ((i * 3) % 5) / 5.0;
      WarpMatrix m;
      m(0, 0) = static_cast<float>(std::cos(ang));
      m(0, 1) = static_cast<float>(-std::sin(ang));
      m(0, 2) = static_cast<float>(dx);
      m(1, 0) = static_cast<float>(std::sin(ang));
      m(1, 1) = static_cast<float>(std::cos(ang));
      m(1, 2) = static_cast<float>(dy);
      f.source_to_canvas = m;
      plan.frames.push_back(f);
    }

    config::ReconstructionDrizzleConfig cfg = config_for();  // 1/1
    cfg.pixfrac = 0.85f;
    cfg.chunk_rows = 8;                 // CPU: 4 stripes; CUDA: 8-row bands
    cfg.min_clip_contributors = 3;      // < nf -> robust clip engaged
    config::ReconstructionClippingConfig clip;
    clip.clip_sigma_low = clip.clip_sigma_high = 2.5f;
    clip.min_fraction = 0.1f;
    clip.min_n_eff = 1.0f;

    std::vector<Matrix2Df> imgs(nf, Matrix2Df(S, S));
    for (int i = 0; i < nf; ++i)
      for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
          const float outlier = (i == 3) ? 500.0f : 0.0f;  // forces clipping
          imgs[i](y, x) = 40.0f + outlier + 3.0f * x + 2.0f * y +
                          11.0f * std::sin(0.55f * (x + y)) + 0.7f * i;
        }
    SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & {
      return imgs[i];
    };
    Matrix2Df comp = Matrix2Df::Constant(S, S, 0.7f);
    Matrix2Df q0 = Matrix2Df::Constant(S, S, 0.6f);
    Matrix2Df q1 = Matrix2Df::Constant(S, S, 0.65f);
    Matrix2Df art = Matrix2Df::Constant(S, S, 0.9f);
    FrameQualityProvider quality_of = [&](size_t i) -> FrameQualityMaps {
      return {&comp, &q0, &q1, i == 2 ? nullptr : &art};  // frame 2: no artifact
    };

    MultibandStoreContract mbc;
    mbc.enabled = true;
    mbc.levels = 3;

    Fixture cpu_fx;
    const auto cpu = persist_forward_drizzle_multiband(
        cpu_fx.root, plan, provider, cfg, clip, mbc, quality_of);  // cuda = {}
    REQUIRE_FALSE(cpu.cuda_timing.used);

    reconstruction::ForwardDrizzleCudaOptions attempt;
    attempt.attempt = true;
    Fixture cuda_fx;
    const auto gpu = persist_forward_drizzle_multiband(
        cuda_fx.root, plan, provider, cfg, clip, mbc, quality_of, {}, {}, {},
        attempt);

    REQUIRE(gpu.cuda_timing.used);
    REQUIRE(gpu.cuda_timing.bands >= 2);        // 32 canvas rows / 8-row bands
    REQUIRE(gpu.cuda_timing.resolved_chunk_rows == 8);
    REQUIRE(gpu.cuda_timing.total_seconds >= 0.0);
    // Same clipping totals -> the shared acceptance mask matched exactly.
    REQUIRE(gpu.clipping.pixel_channel_evaluations ==
            cpu.clipping.pixel_channel_evaluations);
    REQUIRE(gpu.clipping.pixel_channel_rejected ==
            cpu.clipping.pixel_channel_rejected);
    REQUIRE(gpu.clipping.candidate_contributions_clipped ==
            cpu.clipping.candidate_contributions_clipped);
    REQUIRE(gpu.clipping.candidate_contributions_clipped > 0);  // clip engaged

    REQUIRE(verify_drizzle_profile_store(cuda_fx.root, gpu.identity).usable);

    // The scientific planes are bit-identical to the CPU streaming build...
    REQUIRE(plane_digest(gpu.generation_dir) == plane_digest(cpu.generation_dir));

    // ...and also to a whole-canvas CPU build --- so the CUDA band boundaries
    // (8 rows) introduce no seam of their own (plan 19.6 chunk invariance).
    Fixture cpu1_fx;
    auto cfg1 = cfg;
    cfg1.chunk_rows = C;
    const auto cpu1 = persist_forward_drizzle_multiband(
        cpu1_fx.root, plan, provider, cfg1, clip, mbc, quality_of);
    REQUIRE(plane_digest(gpu.generation_dir) == plane_digest(cpu1.generation_dir));

    // The token the wiring's DRIZZLE_CONTRIB_LIST_BUDGET -> CudaAllocFailure
    // translation matches on must actually be what a too-small host budget
    // raises (else run_cuda_chunked's halving ladder is unreachable, plan 19.4).
    if (!osc) {
      MultibandProfileParams mbp;
      mbp.emit_fine = mbp.emit_medium = mbp.emit_alpha_confidence = true;
      const int Hc = plan.canvas_height_native * cfg.internal_scale;
      REQUIRE_THROWS_WITH(
          accumulate_pair_by_frame_cuda(plan, provider, cfg, clip, 0, Hc, {}, {},
                                        to_rect_provider(quality_of), mbp,
                                        /*mem_budget=*/512),
          Catch::Matchers::ContainsSubstring("DRIZZLE_CONTRIB_LIST_BUDGET"));
    }
  }
}

TEST_CASE("drizzle store: §30.81 the CUDA path splits each band into column "
          "tiles when the host ClipCandidate buffer would exceed the budget, "
          "and the committed store is byte-identical to the whole-canvas CPU "
          "build",
          "[drizzle-store][cuda-parity][fd-tile-window]") {
  if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
    SUCCEED("no CUDA device");
    return;
  }
  auto plane_digest = [](const fs::path &gen) {
    std::map<std::string, std::string> h;
    for (const auto &e : fs::directory_iterator(gen)) {
      const std::string n = e.path().filename().string();
      if (e.is_regular_file() && n.size() > 5 &&
          n.substr(n.size() - 5) == ".fits")
        h[n] = core::sha256_file(e.path());
    }
    REQUIRE(h.size() >= 12);
    return h;
  };

  // §30.81 step 4: the canvas must be wider than the tile-width floor (64) for
  // column tiling to be the lever the planner picks (a narrower canvas can only
  // move the band height). nf * S drives the per-band memo; channels * nf * W
  // drives the full-width ClipCandidate buffer --- sized here so that, at a
  // band the device permits in full, the candidate buffer overflows the host
  // budget and the per-band column tiling engages.
  const int S = 40, C = 160, nf = 24;
  auto plan = plan_for(S, /*osc=*/true);
  plan.canvas_width_native = plan.canvas_height_native = C;
  plan.frames.clear();
  for (int i = 0; i < nf; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    const double ang = 0.010 * std::sin(0.5 * i);
    const double dx = 3.0 + ((i * 7) % 5) / 5.0;
    const double dy = 3.0 + ((i * 3) % 5) / 5.0;
    WarpMatrix m;
    m(0, 0) = static_cast<float>(std::cos(ang));
    m(0, 1) = static_cast<float>(-std::sin(ang));
    m(0, 2) = static_cast<float>(dx);
    m(1, 0) = static_cast<float>(std::sin(ang));
    m(1, 1) = static_cast<float>(std::cos(ang));
    m(1, 2) = static_cast<float>(dy);
    f.source_to_canvas = m;
    plan.frames.push_back(f);
  }

  config::ReconstructionDrizzleConfig cfg = config_for();  // 1/1
  cfg.pixfrac = 0.85f;
  cfg.min_clip_contributors = 3;
  config::ReconstructionClippingConfig clip;
  clip.clip_sigma_low = clip.clip_sigma_high = 2.5f;
  clip.min_fraction = 0.1f;
  clip.min_n_eff = 1.0f;

  std::vector<Matrix2Df> imgs(nf, Matrix2Df(S, S));
  for (int i = 0; i < nf; ++i)
    for (int y = 0; y < S; ++y)
      for (int x = 0; x < S; ++x) {
        const float outlier = (i == 7) ? 500.0f : 0.0f;
        imgs[i](y, x) = 40.0f + outlier + 3.0f * x + 2.0f * y +
                        11.0f * std::sin(0.55f * (x + y)) + 0.7f * i;
      }
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & {
    return imgs[i];
  };
  Matrix2Df comp = Matrix2Df::Constant(S, S, 0.7f);
  Matrix2Df q0 = Matrix2Df::Constant(S, S, 0.6f);
  Matrix2Df q1 = Matrix2Df::Constant(S, S, 0.65f);
  Matrix2Df art = Matrix2Df::Constant(S, S, 0.9f);
  FrameQualityProvider quality_of = [&](size_t i) -> FrameQualityMaps {
    return {&comp, &q0, &q1, i == 2 ? nullptr : &art};
  };

  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 3;

  // Whole-canvas CPU reference.
  auto cfg_cpu = cfg;
  cfg_cpu.chunk_rows = C;
  cfg_cpu.memory_budget_mb = 256;
  Fixture cpu_fx;
  const auto cpu = persist_forward_drizzle_multiband(
      cpu_fx.root, plan, provider, cfg_cpu, clip, mbc, quality_of);
  REQUIRE_FALSE(cpu.cuda_timing.used);

  // CUDA build with a host budget that admits a full-height band's memo +
  // reassembly stripe, but NOT a full-width ClipCandidate buffer at that band
  // height -> the per-band column tiling engages (§30.81 step 4: band height
  // and tile width are derived jointly from this one ceiling).
  auto cfg_gpu = cfg;
  cfg_gpu.chunk_rows = C;           // one band over the whole canvas
  cfg_gpu.memory_budget_mb = 100;   // ~100 MiB host ceiling
  reconstruction::ForwardDrizzleCudaOptions attempt;
  attempt.attempt = true;
  Fixture gpu_fx;
  const auto gpu = persist_forward_drizzle_multiband(
      gpu_fx.root, plan, provider, cfg_gpu, clip, mbc, quality_of, {}, {}, {},
      attempt);

  REQUIRE(gpu.cuda_timing.used);
  REQUIRE(gpu.cuda_timing.max_tiles_per_band >= 2);   // tiling actually engaged
  REQUIRE(gpu.cuda_timing.min_tile_w < C);
  REQUIRE(gpu.cuda_timing.resolved_tile_w >= 1);

  // Same clip acceptance mask.
  REQUIRE(gpu.clipping.candidate_contributions_clipped ==
          cpu.clipping.candidate_contributions_clipped);
  REQUIRE(gpu.clipping.candidate_contributions_clipped > 0);

  REQUIRE(verify_drizzle_profile_store(gpu_fx.root, gpu.identity).usable);
  // The scientific planes are bit-identical to the whole-canvas CPU build ---
  // the column tile seams introduce nothing (the store identity carries no
  // tile/band terms).
  REQUIRE(plane_digest(gpu.generation_dir) == plane_digest(cpu.generation_dir));
}

TEST_CASE("drizzle store: the CUDA per-stripe path takes local-warp frames "
          "through the plan-19.6.2 hybrid path (store byte-identical to CPU) "
          "and still declines mode 2/1 to the CPU reference",
          "[drizzle-store][cuda-parity]") {
  auto plan = plan_for(/*size=*/16, /*osc=*/false);
  plan.canvas_width_native = plan.canvas_height_native = 16;
  plan.frames.clear();
  const int nfr = 5;
  for (int i = 0; i < nfr; ++i) {
    registration::FrameSamplingTransform f;
    f.valid = f.source_to_canvas_affine_valid = true;
    f.frame_id = "synthetic:" + std::to_string(i);
    f.source_index = static_cast<size_t>(i);
    f.model_prediction_factor = 1.0f;
    f.registration_residual_factor = 1.0f;
    // Sub-pixel rotation + translation per frame, so droplets straddle cell
    // boundaries and every contribution carries a distinct polygon area --- a
    // digest over identity transforms + a flat source could not catch a
    // leaf_order / target-cell mix-up.
    const double ang = 0.02 * std::sin(0.5 * i);
    const double tx = 0.37 + 0.21 * ((i * 7) % 5) / 5.0;
    const double ty = 0.11 + 0.19 * ((i * 3) % 5) / 5.0;
    f.source_to_canvas(0, 0) = static_cast<float>(std::cos(ang));
    f.source_to_canvas(0, 1) = static_cast<float>(-std::sin(ang));
    f.source_to_canvas(0, 2) = static_cast<float>(tx);
    f.source_to_canvas(1, 0) = static_cast<float>(std::sin(ang));
    f.source_to_canvas(1, 1) = static_cast<float>(std::cos(ang));
    f.source_to_canvas(1, 2) = static_cast<float>(ty);
    plan.frames.push_back(f);
  }
  config::ReconstructionDrizzleConfig cfg = config_for();
  config::ReconstructionClippingConfig clip;
  clip.min_n_eff = 1.0f;
  clip.min_fraction = 0.1f;
  std::vector<Matrix2Df> imgs;
  for (int i = 0; i < nfr; ++i) {
    Matrix2Df img(16, 16);
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x)
        img(y, x) = 40.0f + (i == 2 ? 500.0f : 0.0f)  // frame 2 forces clipping
                    + 3.0f * x + 2.0f * y +
                    11.0f * std::sin(0.6f * (x + y)) + 0.5f * i;
    imgs.push_back(std::move(img));
  }
  SourceImageProvider provider = [&](size_t i) -> const Matrix2Df & {
    return imgs[i];
  };
  Matrix2Df m = Matrix2Df::Constant(16, 16, 0.7f);
  FrameQualityProvider quality_of = [&](size_t) -> FrameQualityMaps {
    return {&m, &m, &m, &m};
  };
  MultibandStoreContract mbc;
  mbc.enabled = true;
  mbc.levels = 2;
  reconstruction::ForwardDrizzleCudaOptions attempt;
  attempt.attempt = true;

  SECTION("a local-warp frame -> hybrid path, store byte-identical to CPU") {
    if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
      SUCCEED("no CUDA device -- skipped");
      return;
    }
    auto p = plan;
    p.frames[1].has_smooth_local_model = true;
    p.frames[1].smooth_local_model.valid = true;
    p.frames[1].smooth_local_model.image_rows = 16;
    p.frames[1].smooth_local_model.image_cols = 16;
    p.frames[1].model_coordinate_scale = 1.0f;

    Fixture cpu_fx, gpu_fx;
    const auto cpu = persist_forward_drizzle_multiband(
        cpu_fx.root, p, provider, cfg, clip, mbc, quality_of);
    const auto gpu = persist_forward_drizzle_multiband(
        gpu_fx.root, p, provider, cfg, clip, mbc, quality_of, {}, {}, {},
        attempt);
    REQUIRE(gpu.cuda_timing.used);
    REQUIRE(gpu.cuda_timing.hybrid_local_frames == 1);
    REQUIRE(gpu.cuda_timing.hybrid_leaf_cells > 0);          // path ran
    REQUIRE(gpu.cuda_timing.hybrid_cpu_seconds >= 0.0);
    REQUIRE(gpu.cuda_timing.hybrid_gpu_raster_seconds >= 0.0);
    // The hybrid path runs strictly inside the accumulate_pair_by_frame_cuda
    // calls that stripe_seconds brackets -> its split must not exceed it.
    REQUIRE(gpu.cuda_timing.hybrid_cpu_seconds +
                gpu.cuda_timing.hybrid_gpu_raster_seconds <=
            gpu.cuda_timing.stripe_seconds + 1e-6);
    REQUIRE(cpu.clipping.candidate_contributions_clipped > 0);  // non-vacuous
    REQUIRE(gpu.clipping.candidate_contributions_clipped ==
            cpu.clipping.candidate_contributions_clipped);
    auto plane_digest = [](const fs::path &gen) {
      std::map<std::string, std::string> h;
      for (const auto &e : fs::directory_iterator(gen)) {
        const std::string n = e.path().filename().string();
        if (e.is_regular_file() && n.size() > 5 &&
            n.substr(n.size() - 5) == ".fits")
          h[n] = core::sha256_file(e.path());
      }
      REQUIRE(h.size() >= 12);
      return h;
    };
    REQUIRE(plane_digest(gpu.generation_dir) ==
            plane_digest(cpu.generation_dir));
  }

  SECTION("mode 2/1 -> device internal-2x bands folded 2x2 -> 1x on the host, "
          "store byte-identical to the CPU mode-2/1 build (§30.57)") {
    if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
      SUCCEED("no CUDA device -- skipped");
      return;
    }
    auto c = cfg;
    c.internal_scale = 2;
    c.output_scale = 1;  // native 16 -> internal 32 -> output 16

    auto plane_digest = [](const fs::path &gen) {
      std::map<std::string, std::string> h;
      for (const auto &e : fs::directory_iterator(gen)) {
        const std::string n = e.path().filename().string();
        if (e.is_regular_file() && n.size() > 5 &&
            n.substr(n.size() - 5) == ".fits")
          h[n] = core::sha256_file(e.path());
      }
      REQUIRE(h.size() >= 12);
      return h;
    };

    Fixture cpu_fx;
    const auto cpu = persist_forward_drizzle_multiband(
        cpu_fx.root, plan, provider, c, clip, mbc, quality_of);
    const auto ref = plane_digest(cpu.generation_dir);

    // Band boundary on an ODD internal row (chunk_rows=3 -> rows 0,3,6,...): the
    // exact case where a mis-aligned 2x2 fold would shift. And whole canvas.
    for (int cr : {3, 64}) {
      auto cc = c;
      cc.chunk_rows = cr;
      Fixture gpu_fx;
      const auto gpu = persist_forward_drizzle_multiband(
          gpu_fx.root, plan, provider, cc, clip, mbc, quality_of, {}, {}, {},
          attempt);
      REQUIRE(gpu.cuda_timing.used);
      REQUIRE(gpu.cuda_timing.hybrid_local_frames == 0);  // affine frames
      REQUIRE(plane_digest(gpu.generation_dir) == ref);
      REQUIRE(verify_drizzle_profile_store(gpu_fx.root, gpu.identity).usable);
    }
  }

  SECTION("mode 2/1 WITH a local-warp frame: hybrid §19.6.2 AND the 2x2 fold "
          "in one build, still byte-identical to CPU (§30.57)") {
    if (reconstruction::forward_drizzle_cuda_device_memory().free_bytes == 0) {
      SUCCEED("no CUDA device -- skipped");
      return;
    }
    auto c = cfg;
    c.internal_scale = 2;
    c.output_scale = 1;

    // A valid local model routes frame 2 through build_frame_records_hybrid_local
    // regardless of whether it subdivides (dispatch is on has_smooth_local_model).
    // The leaf_order > 0 sub-case is covered by the dedicated contrib-list test;
    // here the point is hybrid producer AND 2x2 fold in ONE build.
    auto p = plan;
    p.frames[2].has_smooth_local_model = true;
    p.frames[2].smooth_local_model.valid = true;
    p.frames[2].smooth_local_model.image_rows = 16;
    p.frames[2].smooth_local_model.image_cols = 16;
    p.frames[2].model_coordinate_scale = 1.0f;

    auto plane_digest = [](const fs::path &gen) {
      std::map<std::string, std::string> h;
      for (const auto &e : fs::directory_iterator(gen)) {
        const std::string n = e.path().filename().string();
        if (e.is_regular_file() && n.size() > 5 &&
            n.substr(n.size() - 5) == ".fits")
          h[n] = core::sha256_file(e.path());
      }
      REQUIRE(h.size() >= 12);
      return h;
    };

    Fixture cpu_fx, gpu_fx;
    const auto cpu = persist_forward_drizzle_multiband(
        cpu_fx.root, p, provider, c, clip, mbc, quality_of);
    const auto gpu = persist_forward_drizzle_multiband(
        gpu_fx.root, p, provider, c, clip, mbc, quality_of, {}, {}, {}, attempt);
    REQUIRE(gpu.cuda_timing.used);
    REQUIRE(gpu.cuda_timing.hybrid_local_frames == 1);   // the local-warp frame
    REQUIRE(gpu.cuda_timing.hybrid_leaf_cells > 0);
    REQUIRE(plane_digest(gpu.generation_dir) == plane_digest(cpu.generation_dir));
  }
}

TEST_CASE("plan-19.4 CUDA auto-chunking: reserve, fit, and the halving ladder",
          "[drizzle-store]") {
  using reconstruction::plan_cuda_chunking;

  SECTION("chunk fills usable memory after the reserve is withheld") {
    // 4 GiB free, 20% reserve -> ~3.2 GiB usable. 1 MiB/row -> ~3276 rows fit.
    const auto p = plan_cuda_chunking(4ull << 30, 1ull << 20, /*image_rows=*/8000);
    REQUIRE(p.feasible);
    REQUIRE(p.reserve_bytes == static_cast<std::size_t>((4ull << 30) * 0.20));
    REQUIRE(p.usable_bytes == (4ull << 30) - p.reserve_bytes);
    REQUIRE(p.chunk_rows == static_cast<int>(p.usable_bytes / (1ull << 20)));
    REQUIRE(p.chunk_rows < 8000);  // memory-bound, not image-bound
  }

  SECTION("image height caps the chunk when memory is ample") {
    const auto p = plan_cuda_chunking(16ull << 30, 1ull << 20, /*image_rows=*/512);
    REQUIRE(p.feasible);
    REQUIRE(p.chunk_rows == 512);
  }

  SECTION("a config ceiling caps the initial chunk") {
    const auto p = plan_cuda_chunking(16ull << 30, 1ull << 20, 4000,
                                      /*requested_chunk_rows=*/64);
    REQUIRE(p.chunk_rows == 64);
  }

  SECTION("reserve floor dominates the fraction for a small free figure") {
    // 300 MiB free, 20% = 60 MiB < 256 MiB floor -> floor wins.
    const auto p = plan_cuda_chunking(300ull << 20, 1ull << 20, 1000);
    REQUIRE(p.reserve_bytes == (static_cast<std::size_t>(256) << 20));
    REQUIRE(p.usable_bytes == (300ull << 20) - (256ull << 20));  // 44 MiB
    REQUIRE(p.feasible);
    REQUIRE(p.chunk_rows == 44);
  }

  SECTION("reserve floor exceeds free memory -> underwater, infeasible") {
    const auto p = plan_cuda_chunking(200ull << 20, 1ull << 20, 1000);
    REQUIRE(p.usable_bytes == 0);
    REQUIRE_FALSE(p.feasible);
  }

  SECTION("not enough for even one row -> infeasible, no negative sizes") {
    const auto p = plan_cuda_chunking(1ull << 30, 4ull << 30, 100);
    REQUIRE_FALSE(p.feasible);
    REQUIRE(p.chunk_rows == 0);
  }

  SECTION("retry ladder halves down to >= 1") {
    // Force chunk_rows == 100 via image height, ample memory.
    const auto p = plan_cuda_chunking(64ull << 30, 1ull << 20, 100);
    REQUIRE(p.chunk_rows == 100);
    // 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 : 6 halvings.
    REQUIRE(p.max_retries == 6);
    REQUIRE(p.min_chunk_rows == 1);
  }

  SECTION("degenerate inputs are rejected, not divided-by-zero") {
    REQUIRE_FALSE(plan_cuda_chunking(1ull << 30, 0, 100).feasible);
    REQUIRE_FALSE(plan_cuda_chunking(1ull << 30, 1 << 20, 0).feasible);
  }

  SECTION("§30.80: at real 3840x2160x2 geometry the host ClipCandidate term "
          "(cand_row) dominates bytes_per_row -> band collapse with frame "
          "count, and a host/device budget split alone does not fix it") {
    // bytes_per_row is assembled in persist_forward_drizzle_multiband
    // (drizzle_profile_store.cpp) as cand_row + rec_row + acc_row.
    //   cand_row = channels * dims.width * frames * sizeof(ClipCandidate)
    //   rec_row  = source_width * 16 * sizeof(CudaDrizzleContribRecord)
    //   acc_row  = channels * dims.width * 8 * sizeof(double)
    // These sizes are pinned so the test fails loudly if the structs change.
    constexpr std::size_t kClipCandidate = 64;         // 8 + 6*double + bool -> pad
    constexpr std::size_t kContribRecord = 40;         // 5*u32 -> pad + 2*double
    const int channels = 3;
    const int dims_width = 7680;   // (3840) * internal_scale 2
    const int dims_height = 4320;  // (2160) * 2
    const int source_width = 3840;
    const std::size_t rec_row =
        static_cast<std::size_t>(source_width) * 16 * kContribRecord;
    const std::size_t acc_row =
        static_cast<std::size_t>(channels) * dims_width * 8 * sizeof(double);
    const std::size_t free_bytes = 8ull << 30;  // a mid-range 8 GiB card

    struct Row { int frames; int band_rows; int bands; double cand_frac; };
    std::vector<Row> table;
    for (int frames : {40, 100, 300, 600}) {
      const std::size_t cand_row = static_cast<std::size_t>(channels) *
                                   dims_width * frames * kClipCandidate;
      const std::size_t bytes_per_row = cand_row + rec_row + acc_row;
      const auto p =
          plan_cuda_chunking(free_bytes, bytes_per_row, dims_height);
      REQUIRE(p.feasible);
      const int nbands = (dims_height + p.chunk_rows - 1) / p.chunk_rows;
      table.push_back({frames, p.chunk_rows, nbands,
                       static_cast<double>(cand_row) /
                           static_cast<double>(bytes_per_row)});
      std::printf("  frames %3d : band_rows %4d  bands %4d  "
                  "cand_row %6.1f MiB (%.1f%% of bytes_per_row)\n",
                  frames, p.chunk_rows, nbands,
                  static_cast<double>(cand_row) / (1024.0 * 1024.0),
                  100.0 * table.back().cand_frac);
    }
    // The finding: cand_row is the overwhelming majority at every realistic
    // frame count (93 % at 40 frames, rising to > 99 % at 600), and the band
    // height collapses toward single digits as N grows.
    for (const auto &r : table) REQUIRE(r.cand_frac > 0.90);
    REQUIRE(table.front().frames == 40);
    REQUIRE(table.front().cand_frac > 0.93);
    REQUIRE(table.back().frames == 600);
    REQUIRE(table.back().cand_frac > 0.99);
    REQUIRE(table.back().band_rows <= 16);   // 600 frames -> single-digit-ish
    REQUIRE(table.back().bands >= 256);      // ... i.e. hundreds of bands

    // A host/device budget split (R1.1): the device would see only
    // rec_row + acc_row, but the HOST ClipCandidate buffer cand_row*band_rows
    // must still fit an absolute host ceiling. Even a generous 16 GiB ceiling
    // only lifts 600-frame bands to ~18 rows -> still hundreds of bands.
    const std::size_t host_ceiling = 16ull << 30;
    const std::size_t cand_row_600 = static_cast<std::size_t>(channels) *
                                     dims_width * 600 * kClipCandidate;
    const int host_limited_rows =
        static_cast<int>(host_ceiling / cand_row_600);
    std::printf("  R1.1 split, 16 GiB host ceiling: 600-frame band_rows -> "
                "%d (%d bands)\n",
                host_limited_rows,
                (dims_height + host_limited_rows - 1) /
                    std::max(host_limited_rows, 1));
    REQUIRE(host_limited_rows < 32);  // still a collapse; R1.2/R2 needed
  }

  SECTION("§30.81 step 4 + 3a: band height and column tile width are derived "
          "JOINTLY from one host ceiling (the live regions sum to under it), "
          "AND --- with the per-band record memo removed in 3a --- the band "
          "height is frame-count-INDEPENDENT (no §30.80 collapse-with-N)") {
    constexpr std::size_t kClipCandidate = 64;
    constexpr std::size_t kContribRecord = 40;
    constexpr std::size_t kDrizzleContrib = 48;
    constexpr std::size_t kMemoCellsEst = 6;
    constexpr std::size_t kPlanePx = 3 * sizeof(float) + sizeof(std::uint8_t);
    const int channels = 3;
    const int dims_width = 7680;
    const int dims_height = 4320;
    const int source_width = 3840;
    const int source_height = 2160;
    // levels=3 + alpha -> 4 result planes (uniform+raw+fine+medium) + alpha.
    const std::size_t plane_px_bytes =
        static_cast<std::size_t>(4) * channels * kPlanePx + kPlanePx;
    const std::size_t stripe_row =
        plane_px_bytes * static_cast<std::size_t>(dims_width);
    const std::size_t rec_row =
        static_cast<std::size_t>(source_width) * 16 * kContribRecord;
    const std::size_t acc_row =
        static_cast<std::size_t>(channels) * dims_width * 8 * sizeof(double);
    const std::size_t device_bytes_per_row = rec_row + acc_row;
    const std::size_t free_bytes = 8ull << 30;
    const std::size_t host_budget = 2ull << 30;  // the default absolute ceiling
    const int kMinTileW = std::min(64, dims_width);
    const std::size_t src_q_const =
        static_cast<std::size_t>(2 + 4) * source_width * source_height *
        sizeof(float);

    // 3a: one frame's record scratch, N-INDEPENDENT (no frame_count factor).
    const std::size_t frame_rec_row =
        static_cast<std::size_t>(source_width) * kMemoCellsEst * kDrizzleContrib;

    std::printf("  §30.81 step-4+3a joint host budget (8 GiB card, 2 GiB host "
                "ceiling):\n");
    int first_band_rows = 0, last_band_rows = 0;
    for (int frames : {40, 100, 300, 600}) {
      const std::size_t cand_per_row_col =
          static_cast<std::size_t>(channels) * frames * kClipCandidate +
          static_cast<std::size_t>(channels) * sizeof(std::size_t) +
          plane_px_bytes;
      const std::size_t host_fixed_per_row = frame_rec_row + stripe_row;
      const std::size_t host_avail =
          host_budget > src_q_const ? host_budget - src_q_const : 0;
      const std::size_t host_row_cost =
          host_fixed_per_row +
          cand_per_row_col * static_cast<std::size_t>(kMinTileW);
      const int host_rows = static_cast<int>(std::clamp<std::size_t>(
          host_avail / host_row_cost, 1,
          static_cast<std::size_t>(dims_height)));

      const auto p = plan_cuda_chunking(free_bytes, device_bytes_per_row,
                                        dims_height, host_rows);
      REQUIRE(p.feasible);
      const int band_rows = p.chunk_rows;
      const int bands = (dims_height + band_rows - 1) / band_rows;

      const std::size_t rows_z = static_cast<std::size_t>(band_rows);
      const std::size_t fixed = host_fixed_per_row * rows_z + src_q_const;
      REQUIRE(host_budget > fixed);  // the band always leaves room for a tile
      const std::size_t for_cand = host_budget - fixed;
      const int tile_w = std::clamp(
          static_cast<int>(std::min<std::size_t>(
              for_cand / (cand_per_row_col * rows_z),
              static_cast<std::size_t>(dims_width))),
          1, dims_width);
      const int tiles = (dims_width + tile_w - 1) / tile_w;

      // The peak is the SUM of the simultaneously-live regions, each at the
      // resolved band height / tile width --- this is what must fit, and what
      // the pre-step-4 model failed to bound (memo and candidates each took the
      // whole ceiling).
      const std::size_t host_peak = host_fixed_per_row * rows_z +
                                    cand_per_row_col *
                                        static_cast<std::size_t>(tile_w) *
                                        rows_z +
                                    src_q_const;
      std::printf("  frames %3d : band_rows %4d (%d bands)  tile_w %5d  "
                  "tiles/band %3d  host peak %7.1f MiB\n",
                  frames, band_rows, bands, tile_w, tiles,
                  host_peak / (1024.0 * 1024.0));
      // Step 4: the JOINT peak is bounded and tiles never degrade below the
      // floor. Step 3a: with the memo gone, `host_fixed_per_row` carries no
      // frame_count factor. The band-height calc still reserves one kMinTileW-
      // wide candidate slice (so tiles never go below the floor), which leaves
      // a WEAK residual N-dependence in the band height --- but the §30.80 / B
      // collapse (42 -> 2 rows, 103 -> 2160 bands over 40..600) is gone: the
      // band stays in the hundreds of rows and the band count stays < 32.
      REQUIRE(host_peak <= host_budget);
      REQUIRE(tile_w >= std::min(kMinTileW, dims_width));
      if (first_band_rows == 0) first_band_rows = band_rows;
      REQUIRE(band_rows <= first_band_rows);   // weakly decreasing in N
      REQUIRE(band_rows >= 128);               // NOT a collapse (B: 2)
      REQUIRE(bands <= 32);                    // NOT hundreds (B: 2160)
      last_band_rows = band_rows;
    }
    // 600-frame band height still >= 1/4 of the 40-frame one (B: ~1/20).
    REQUIRE(last_band_rows * 4 >= first_band_rows);
  }

  SECTION("device-memory probe drives the plan") {
    const auto mem = reconstruction::forward_drizzle_cuda_device_memory();
    if (mem.free_bytes == 0) {
      // CUDA-free build, or no usable device: the plan must be infeasible so
      // the caller stays on the CPU reference path.
      REQUIRE(mem.total_bytes == 0);
      REQUIRE_FALSE(plan_cuda_chunking(mem.free_bytes, 1 << 20, 1000).feasible);
    } else {
      // Real device: free <= total, and a modest per-row cost yields a plan.
      REQUIRE(mem.free_bytes <= mem.total_bytes);
      const auto p = plan_cuda_chunking(mem.free_bytes, 1 << 20, 4096);
      REQUIRE(p.feasible);
      REQUIRE(p.chunk_rows >= 1);
    }
  }

  SECTION("the affine forward-drizzle CUDA path is enabled exactly when a "
          "usable device is present (§30.54/§30.55: wired + store-level "
          "byte-identical on real M31 data)") {
    const auto mem = reconstruction::forward_drizzle_cuda_device_memory();
    REQUIRE(reconstruction::forward_drizzle_cuda_runtime_available() ==
            (mem.free_bytes > 0));
  }
}

TEST_CASE("plan-19.4 CUDA chunk driver: retry ladder + hard-failure restart",
          "[drizzle-store]") {
  using reconstruction::run_cuda_chunked;
  using reconstruction::CudaAllocFailure;
  using reconstruction::CudaChunkPlan;

  auto plan = [](int chunk, int min_rows) {
    CudaChunkPlan p;
    p.feasible = true;
    p.chunk_rows = chunk;
    p.min_chunk_rows = min_rows;
    return p;
  };

  SECTION("happy path: contiguous bands that exactly cover the image") {
    std::vector<std::pair<int, int>> bands;
    const int n = run_cuda_chunked(plan(64, 1), 200,
                                   [&](int y0, int rows) { bands.push_back({y0, rows}); });
    REQUIRE(n == static_cast<int>(bands.size()));
    REQUIRE(bands.front().first == 0);
    int covered = 0;
    for (size_t i = 0; i < bands.size(); ++i) {
      if (i) REQUIRE(bands[i].first == bands[i - 1].first + bands[i - 1].second);
      covered += bands[i].second;
    }
    REQUIRE(covered == 200);
  }

  SECTION("OOM twice then succeed: the SAME band retries at halved height") {
    int calls = 0;
    std::vector<int> heights;
    reconstruction::CudaChunkRunStats stats;
    run_cuda_chunked(plan(100, 1), 100, [&](int, int rows) {
      heights.push_back(rows);
      if (++calls <= 2) throw CudaAllocFailure("oom");
    }, &stats);
    // 100 -> 50 -> 25, third attempt succeeds; then 25,25,25 for the rest.
    REQUIRE(heights[0] == 100);
    REQUIRE(heights[1] == 50);
    REQUIRE(heights[2] == 25);
    int covered = 0;
    for (size_t i = 2; i < heights.size(); ++i) covered += heights[i];
    REQUIRE(covered == 100);
    // §30.80 telemetry: two halvings on band 0 (collapsed to 25); band 1
    // starts fresh at the planned height and takes the remaining 75 rows.
    REQUIRE(stats.halvings == 2);
    REQUIRE(stats.bands == 2);
    REQUIRE(stats.min_band_rows == 25);
    REQUIRE(stats.max_band_rows == 75);
  }

  SECTION("§30.80: run stats on a clean drive report the true band heights") {
    reconstruction::CudaChunkRunStats stats;
    // 200 rows / 64-row bands -> 64, 64, 64, 8.
    const int n = run_cuda_chunked(plan(64, 1), 200, [](int, int) {}, &stats);
    REQUIRE(n == 4);
    REQUIRE(stats.bands == 4);
    REQUIRE(stats.halvings == 0);
    REQUIRE(stats.max_band_rows == 64);
    REQUIRE(stats.min_band_rows == 8);  // the ragged last band
  }

  SECTION("OOM forever at the floor -> ForwardDrizzleCudaError (CPU restart)") {
    REQUIRE_THROWS_AS(
        run_cuda_chunked(plan(16, 4), 64,
                         [](int, int) { throw CudaAllocFailure("oom"); }),
        reconstruction::ForwardDrizzleCudaError);
  }

  SECTION("a non-OOM throw is a hard failure -> propagates unchanged") {
    REQUIRE_THROWS_AS(
        run_cuda_chunked(plan(16, 1), 64,
                         [](int, int) { throw std::runtime_error("kernel launch failed"); }),
        std::runtime_error);
    // ... and specifically NOT swallowed into a ForwardDrizzleCudaError retry.
    bool caught_generic = false;
    try {
      run_cuda_chunked(plan(16, 1), 64,
                       [](int, int) { throw std::logic_error("bug"); });
    } catch (const reconstruction::ForwardDrizzleCudaError &) {
      FAIL("hard error was misclassified as a retryable CUDA fault");
    } catch (const std::logic_error &) {
      caught_generic = true;
    }
    REQUIRE(caught_generic);
  }

  SECTION("infeasible plan is rejected up front") {
    CudaChunkPlan bad;  // feasible == false
    REQUIRE_THROWS_AS(run_cuda_chunked(bad, 64, [](int, int) {}),
                      reconstruction::ForwardDrizzleCudaError);
  }
}
