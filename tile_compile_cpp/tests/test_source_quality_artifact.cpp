#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
using namespace tile_compile;
using namespace tile_compile::reconstruction;
using json=nlohmann::json;
namespace {
struct Fixture {
  core::AtomicOutput staging{fs::temp_directory_path()/"source-quality-test"};
  fs::path root=staging.path();
  registration::RegistrationSamplingPlan plan;
  Fixture() {
    fs::create_directory(root);
    plan.source_width=plan.source_height=32;
    plan.canvas_width_native=plan.canvas_height_native=32;
    plan.color_mode=ColorMode::MONO;
    plan.source_identity_hash="synthetic-source-calibration-v1";
    for (size_t i=0;i<2;++i) {
      registration::FrameSamplingTransform f;
      f.source_index=i;
      f.frame_id="source:"+std::to_string(i);
      f.valid=f.source_to_canvas_affine_valid=true;
      f.model_prediction_factor=i ? 0.75f : 1.0f;
      plan.frames.push_back(f);
      Matrix2Df pixels=Matrix2Df::Constant(32,32,10.0f+2*i);
      write(i,pixels);
    }
    plan.plan_hash=registration::compute_plan_hash(plan);
  }
  void write(size_t i,const Matrix2Df &pixels) {
    std::ofstream f(root/(std::to_string(i)+".raw"),std::ios::binary);
    f.write(reinterpret_cast<const char *>(pixels.data()),pixels.size()*sizeof(float));
  }
  ~Fixture() { std::error_code ec; fs::remove_all(root,ec); }
};
}

TEST_CASE("source cache: content bound frame loading rejects replacements and truncation", "[source-predecessors]") {
  Fixture f;
  const auto hash=publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache cache(f.root,f.plan,32);
  REQUIRE(cache.manifest_hash()==hash);
  REQUIRE(cache.load(0).minCoeff()==10.0f);
  REQUIRE(cache.load(1).minCoeff()==12.0f);
  REQUIRE_THROWS(cache.load(2));
  f.write(0,Matrix2Df::Constant(32,32,99.0f));
  REQUIRE_THROWS(cache.load(0));
  { std::ofstream file(f.root/"1.raw",std::ios::binary); file<<"short"; }
  REQUIRE_THROWS(cache.load(1));
  REQUIRE_THROWS(VerifiedNormalizedSourceCache(f.root,f.plan,32));
}

TEST_CASE("source cache: provenance mismatch and incomplete publication fail closed", "[source-predecessors]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);
  const auto before=core::read_text(f.root/"normalized_source_manifest.json");
  auto changed=f.plan;
  changed.source_identity_hash="other-calibration";
  REQUIRE_THROWS(VerifiedNormalizedSourceCache(f.root,changed));
  changed=f.plan; changed.cfa_origin_x=1;
  REQUIRE_THROWS(VerifiedNormalizedSourceCache(f.root,changed));
  fs::remove(f.root/"1.raw");
  REQUIRE_THROWS(publish_normalized_source_manifest(f.root,f.plan));
  REQUIRE(core::read_text(f.root/"normalized_source_manifest.json")==before);
}

TEST_CASE("quality artifact: frame identity binding is independent of artifact order", "[source-predecessors]") {
  Fixture f;
  GlobalQualityConfig cfg;
  VectorXf q(2); q<<0.25f,0.75f;
  auto quality=build_quality_frame_weight_plan(f.plan,q,compute_source_quality_config_hash(cfg));
  std::swap(quality.frames[0],quality.frames[1]);
  quality.plan_hash=compute_quality_frame_weight_plan_hash(quality);
  const auto weights=resolve_quality_frame_weights(quality,f.plan,cfg);
  REQUIRE(weights[0]==0.25f);
  REQUIRE(weights[1]==0.75f*0.75f);
  auto bad=quality;
  bad.frames[0].registration_residual_factor=0.5f;
  bad.frames[0].g_eff=bad.frames[0].g_quality*bad.frames[0].model_prediction_factor*0.5f;
  bad.plan_hash=compute_quality_frame_weight_plan_hash(bad);
  REQUIRE_THROWS(resolve_quality_frame_weights(bad,f.plan,cfg));
  auto other_cfg=cfg; other_cfg.w_noise=0.9f;
  REQUIRE_THROWS(resolve_quality_frame_weights(quality,f.plan,other_cfg));
}

TEST_CASE("quality artifact: persist load and raw reconstruction require matching predecessors", "[source-predecessors]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache cache(f.root,f.plan,32);
  GlobalQualityConfig quality_cfg;
  const auto artifact=f.root/"quality.json";
  const auto quality=persist_source_quality_artifact(artifact,f.plan,cache,quality_cfg,32);
  REQUIRE(load_source_quality_artifact(artifact,f.plan,cache,quality_cfg,32).plan_hash==quality.plan_hash);
  config::ReconstructionDrizzleConfig drizzle;
  drizzle.internal_scale=1; drizzle.pixfrac=1; drizzle.chunk_rows=3; drizzle.memory_budget_mb=32;
  config::ReconstructionClippingConfig clipping;
  clipping.min_n_eff=1;
  const auto store=persist_forward_drizzle_from_predecessors(f.root/"store",artifact,
      f.plan,cache,quality_cfg,drizzle,clipping);
  const auto weights=resolve_quality_frame_weights(quality,f.plan,quality_cfg);
  const auto expected=make_drizzle_store_identity(f.plan,drizzle,{},&clipping,weights,
      {cache.manifest_hash(),quality.plan_hash});
  REQUIRE(verify_drizzle_profile_store(f.root/"store",expected).usable);
  auto unbound=make_drizzle_store_identity(f.plan,drizzle,{},&clipping,weights);
  REQUIRE_FALSE(verify_drizzle_profile_store(f.root/"store",unbound).usable);
  const auto raw=read_drizzle_profile_region(f.root/"store",expected,"raw","L",0,0,2,2,16);
  const float mean=(10*weights[0]+12*weights[1])/(weights[0]+weights[1]);
  REQUIRE(std::abs(raw.value[0]-mean)<1e-5f);
  f.write(0,Matrix2Df::Constant(32,32,100.0f));
  REQUIRE_THROWS(persist_forward_drizzle_from_predecessors(f.root/"store",artifact,
      f.plan,cache,quality_cfg,drizzle,clipping));
  REQUIRE(verify_drizzle_profile_store(f.root/"store",expected).usable);
  publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache changed(f.root,f.plan,32);
  REQUIRE_THROWS(load_source_quality_artifact(artifact,f.plan,changed,quality_cfg,32));
}

TEST_CASE("quality artifact: memory preflight precedes cache reads and preserves old artifact", "[source-predecessors]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache cache(f.root,f.plan,32);
  const auto artifact=f.root/"quality.json";
  core::write_text_atomic(artifact,"previous");
  fs::remove(f.root/"0.raw");
  GlobalQualityConfig cfg;
  REQUIRE_THROWS_WITH(persist_source_quality_artifact(artifact,f.plan,cache,cfg,1),
      "SOURCE_QUALITY_MEMORY_BUDGET");
  REQUIRE(core::read_text(artifact)=="previous");
}

TEST_CASE("quality artifact: extreme source indices cannot cause unbounded weight allocation", "[source-predecessors]") {
  Fixture f;
  f.plan.frames[1].source_index=1000000000;
  f.plan.plan_hash=registration::compute_plan_hash(f.plan);
  GlobalQualityConfig cfg;
  VectorXf q=VectorXf::Constant(2,0.5f);
  const auto quality=build_quality_frame_weight_plan(f.plan,q,compute_source_quality_config_hash(cfg));
  REQUIRE_THROWS_WITH(resolve_quality_frame_weights(quality,f.plan,cfg,8),
      "SOURCE_QUALITY_WEIGHT_VECTOR_MEMORY_BUDGET");
}
