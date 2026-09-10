#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
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

// Plan §30.72 O2: the LRU serves a re-load of a resident, on-disk-unchanged
// frame without a read or a SHA-256, but still fails closed when the file is
// rewritten or truncated, and a per-worker clone is independent.
TEST_CASE("source cache: LRU hit skips hashing, tamper still fails closed",
          "[source-predecessors][cache-lru]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);

  SECTION("re-load of an unchanged frame does not re-hash") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64); // >= both frames fit
    const auto *p0=cache.load(0).data();
    REQUIRE(cache.hash_computation_count()==1);
    for (int i=0;i<5;++i) {
      REQUIRE(cache.load(0).data()==p0);           // same buffer, no reload
      REQUIRE(cache.load(0).minCoeff()==10.0f);
    }
    REQUIRE(cache.hash_computation_count()==1);    // still just the first touch
    cache.load(1);
    REQUIRE(cache.hash_computation_count()==2);
    REQUIRE(cache.resident_frame_count()==2);
  }

  SECTION("eviction under a tight budget, and re-touch re-hashes") {
    // 512x512 float = 1 MiB/frame; budget 2 MiB -> usable 1 MiB -> capacity 1.
    core::AtomicOutput big_staging{fs::temp_directory_path()/"src-cache-evict"};
    const fs::path br=big_staging.path();
    fs::create_directory(br);
    registration::RegistrationSamplingPlan bp;
    bp.source_width=bp.source_height=512;
    bp.canvas_width_native=bp.canvas_height_native=512;
    bp.color_mode=ColorMode::MONO;
    bp.source_identity_hash="evict-src";
    for (size_t i=0;i<2;++i) {
      registration::FrameSamplingTransform fr;
      fr.source_index=i; fr.frame_id="e:"+std::to_string(i);
      fr.valid=fr.source_to_canvas_affine_valid=true;
      bp.frames.push_back(fr);
      Matrix2Df px=Matrix2Df::Constant(512,512,3.0f+i);
      std::ofstream of(br/(std::to_string(i)+".raw"),std::ios::binary);
      of.write(reinterpret_cast<const char*>(px.data()),px.size()*sizeof(float));
    }
    bp.plan_hash=registration::compute_plan_hash(bp);
    publish_normalized_source_manifest(br,bp);
    VerifiedNormalizedSourceCache cache(br,bp,2);
    REQUIRE(cache.capacity_frames()==1);
    cache.load(0);
    cache.load(1);                                        // evicts 0
    REQUIRE(cache.resident_frame_count()==1);
    REQUIRE(cache.hash_computation_count()==2);
    cache.load(0);                                        // re-read + re-hash
    REQUIRE(cache.hash_computation_count()==3);
    fs::remove_all(br);
  }

  SECTION("rewrite with identical size is still caught on the hit path") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64);
    REQUIRE(cache.load(0).minCoeff()==10.0f);
    f.write(0,Matrix2Df::Constant(32,32,77.0f));          // same byte count
    REQUIRE_THROWS(cache.load(0));                        // mtime moved -> reverify
  }

  SECTION("per-worker clone shares the manifest but has an independent LRU") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64);
    cache.load(0);
    VerifiedNormalizedSourceCache worker(cache,8);
    REQUIRE(worker.manifest_hash()==cache.manifest_hash());
    REQUIRE(worker.resident_frame_count()==0);
    REQUIRE(worker.load(0).minCoeff()==10.0f);
    REQUIRE(worker.hash_computation_count()==1);          // its own first touch
    REQUIRE(cache.hash_computation_count()==1);           // unaffected
  }
}

// §30.81 step 3a-3: the run-internal source block-check index. Built from the
// same bytes as the whole-file SHA on first touch; a later read_rect reads and
// SHA-256-checks only the blocks its Y window covers. Verifies: region parity
// with load(), block-read amplification (X width is free, Y width is not), and
// the detection-semantics contract boundary (only touched blocks are checked).
TEST_CASE("source cache: block-check index --- parity, amplification, detection semantics",
          "[source-predecessors][cache-blocks]") {
  core::AtomicOutput st{fs::temp_directory_path()/"src-cache-blocks"};
  const fs::path br=st.path();
  fs::create_directory(br);
  const int W=512,H=512;
  registration::RegistrationSamplingPlan bp;
  bp.source_width=W; bp.source_height=H;
  bp.canvas_width_native=bp.canvas_height_native=W;
  bp.color_mode=ColorMode::MONO;
  bp.source_identity_hash="blk-src";
  auto write_frame=[&](size_t i,float bias){
    Matrix2Df px(H,W);
    for (int y=0;y<H;++y) for (int x=0;x<W;++x) px(y,x)=bias+y*1000.0f+x;
    std::ofstream of(br/(std::to_string(i)+".raw"),std::ios::binary);
    of.write(reinterpret_cast<const char*>(px.data()),px.size()*sizeof(float));
  };
  for (size_t i=0;i<2;++i) {
    registration::FrameSamplingTransform fr;
    fr.source_index=i; fr.frame_id="b:"+std::to_string(i);
    fr.valid=fr.source_to_canvas_affine_valid=true;
    bp.frames.push_back(fr);
    write_frame(i,i*0.5f);
  }
  bp.plan_hash=registration::compute_plan_hash(bp);
  publish_normalized_source_manifest(br,bp);
  VerifiedNormalizedSourceCache cache(br,bp,64);
  const int brows=static_cast<int>(cache.block_row_span());
  REQUIRE(brows>=1);
  REQUIRE(brows<H);   // frame spans several blocks

  SECTION("read_rect == load() slice, incl. edges and 1px-wide X") {
    const Matrix2Df full=cache.load(0);
    const std::array<std::array<int,4>,6> rects={{
      {{0,H,0,W}},{{10,42,0,W}},{{H-3,H,0,W}},{{100,140,7,9}},{{0,1,0,1}},
      {{200,201,W-1,W}}}};
    for (const auto &r:rects) {
      const Matrix2Df sub=cache.read_rect(0,r[0],r[1],r[2],r[3]);
      REQUIRE(sub.rows()==r[1]-r[0]);
      REQUIRE(sub.cols()==r[3]-r[2]);
      for (int y=r[0];y<r[1];++y) for (int x=r[2];x<r[3];++x)
        REQUIRE(sub(y-r[0],x-r[2])==full(y,x));
    }
    REQUIRE(cache.read_rect(0,5,5,0,W).rows()==0);   // empty range -> 0x0
  }

  SECTION("block-read amplification: narrowing X is free, narrowing Y is not") {
    cache.reset_io_counters();
    cache.read_rect(0,0,H,0,W);                       // first touch: builds index
    REQUIRE(cache.block_index_builds()==1);
    const auto full_blocks=cache.blocks_verified();
    REQUIRE(full_blocks>=2);

    cache.reset_io_counters();
    cache.read_rect(0,brows,2*brows,0,W);             // one block, full width
    const auto y_win=cache.blocks_verified();
    const auto y_win_bytes=cache.bytes_read();

    cache.reset_io_counters();
    cache.read_rect(0,brows,2*brows,3,4);             // SAME Y, 1px wide
    REQUIRE(cache.blocks_verified()==y_win);          // X width changed nothing
    REQUIRE(cache.bytes_read()==y_win_bytes);

    cache.reset_io_counters();
    cache.read_rect(0,0,H,3,4);                       // full Y, 1px wide
    REQUIRE(cache.blocks_verified()==full_blocks);    // all blocks again

    std::printf("[cache-blocks] block_rows=%d blocks/frame=%llu | 1-block "
                "read=%llu blk / %llu B | full=%llu blk\n",
                brows,(unsigned long long)full_blocks,
                (unsigned long long)y_win,(unsigned long long)y_win_bytes,
                (unsigned long long)full_blocks);
  }

  SECTION("detection semantics: only the blocks a read touches are verified") {
    cache.read_rect(0,0,1,0,W);                       // build the index
    const auto saved=fs::last_write_time(br/"0.raw");
    // Corrupt one float inside block 2, same file length, then roll the mtime
    // back so the staleness guard still passes: a region read does NOT
    // re-verify the whole file, by contract.
    const int trow=2*brows+1;
    {
      std::fstream fio(br/"0.raw",std::ios::in|std::ios::out|std::ios::binary);
      fio.seekp(std::streamoff(static_cast<size_t>(trow)*W*sizeof(float)));
      const float bad=-42.0f;
      fio.write(reinterpret_cast<const char*>(&bad),sizeof(float));
    }
    fs::last_write_time(br/"0.raw",saved);

    // A read that does NOT cover block 2 still succeeds...
    REQUIRE_NOTHROW(cache.read_rect(0,0,brows,0,W));
    REQUIRE_NOTHROW(cache.read_rect(0,3*brows,4*brows,0,W));
    // ...a read that DOES cover block 2 is caught only now, on access.
    REQUIRE_THROWS(cache.read_rect(0,trow,trow+1,0,W));
  }

  SECTION("whole-file replacement invalidates the index and re-verifies") {
    cache.read_rect(1,0,1,0,W);
    REQUIRE(cache.block_index_builds()>=1);
    write_frame(1,777.0f);                            // same length, mtime forward
    REQUIRE_THROWS(cache.read_rect(1,0,1,0,W));       // rebuild -> SHA mismatch
  }
  fs::remove_all(br);
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
