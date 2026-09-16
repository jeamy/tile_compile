#include "tile_compile/reconstruction/drizzle_profile_store.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/reconstruction/profile_store_manifest.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
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
          p["bytes"] = fs::file_size(store.generation_dir / "uniform_L_value.fits");
    }
    if (kind == 2) fs::remove(store.generation_dir / "uniform_L_support.fits");
    if (kind == 3) {
      std::ofstream(store.generation_dir / "uniform_L_value.fits",std::ios::binary) << "broken FITS";
      for (auto &p : manifest["planes"])
        if (p["name"] == "uniform_L_value")
          p["bytes"] = fs::file_size(store.generation_dir / "uniform_L_value.fits");
    }
    // Rehash the generic manifest so completeness and FITS checks must act.
    ProfileStoreManifest parsed;
    parsed.profile = manifest["profile"];
    parsed.internal_width = 16; parsed.internal_height = 16;
    for (const auto &p : manifest["planes"])
      parsed.planes.push_back({p["name"], {}, p["bytes"], p["width"], p["height"]});
    manifest["manifest_hash"] = compute_profile_store_manifest_hash(parsed);
    replace_commit(fixture,commit);
    REQUIRE_FALSE(verify_drizzle_profile_store(fixture.root,make_drizzle_store_identity(plan,cfg)).usable);
  }
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
