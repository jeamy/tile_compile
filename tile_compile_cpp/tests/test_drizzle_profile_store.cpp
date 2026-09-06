#include "tile_compile/reconstruction/drizzle_profile_store.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/reconstruction/profile_store_manifest.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>

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
  reconstruction::fuse_multiband_store_to_image(fixture.root, identity,
      fixture.root / "x_cand.fits", fcfg, /*chunk_rows=*/5, 64, &cand);
  REQUIRE(cand.width == identity.width);
  REQUIRE(cand.height == identity.height);
  REQUIRE(cand.alpha_final_by_band.size() == 3u);
  REQUIRE(cand.alpha_final_by_band[2].empty());          // D3 <- Raw
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

  SECTION("attempt with no fault armed -> immediate throw, nothing created") {
    CudaFaultGuard guard(-1);
    Fixture fx;
    REQUIRE_THROWS_AS(
        persist_forward_drizzle_multiband(fx.root, plan, provider, cfg, clip, mbc,
                                          quality_of, {}, {}, {}, attempt),
        reconstruction::ForwardDrizzleCudaError);
    REQUIRE_FALSE(fs::exists(fx.root / "current.json"));
    REQUIRE(count_dirs(fx.root) == 0);  // not even a discarded generation
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
