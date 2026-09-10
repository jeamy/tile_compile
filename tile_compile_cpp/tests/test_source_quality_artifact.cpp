#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <vector>
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

// A normalized-source cache directory with `n` frames of W x H whose pixel
// (i, y, x) is `fill(i, y, x)`. Publishes the manifest so a
// VerifiedNormalizedSourceCache can be constructed against `plan`.
struct BlkCacheDir {
  core::AtomicOutput st;
  fs::path root;
  registration::RegistrationSamplingPlan plan;
  int W, H;
  using Fill = std::function<float(size_t,int,int)>;
  BlkCacheDir(const char *id,int w,int h,size_t n,const Fill &fill)
      : st{fs::temp_directory_path()/id}, root(st.path()), W(w), H(h) {
    fs::create_directory(root);
    plan.source_width=W; plan.source_height=H;
    plan.canvas_width_native=plan.canvas_height_native=(W>H?W:H);
    plan.color_mode=ColorMode::MONO;
    plan.source_identity_hash=std::string("blk-")+id;
    for (size_t i=0;i<n;++i) {
      registration::FrameSamplingTransform fr;
      fr.source_index=i; fr.frame_id=std::string(id)+":"+std::to_string(i);
      fr.valid=fr.source_to_canvas_affine_valid=true;
      plan.frames.push_back(fr);
      write(i,fill);
    }
    plan.plan_hash=registration::compute_plan_hash(plan);
    publish_normalized_source_manifest(root,plan);
  }
  void write(size_t i,const Fill &fill) {
    std::vector<float> px(static_cast<size_t>(W)*H);
    for (int y=0;y<H;++y) for (int x=0;x<W;++x)
      px[static_cast<size_t>(y)*W+x]=fill(i,y,x);
    std::ofstream of(root/(std::to_string(i)+".raw"),std::ios::binary);
    of.write(reinterpret_cast<const char*>(px.data()),
             static_cast<std::streamsize>(px.size()*sizeof(float)));
  }
  ~BlkCacheDir() { std::error_code ec; fs::remove_all(root,ec); }
};
float f_from_u32(std::uint32_t b) { float f; std::memcpy(&f,&b,4); return f; }
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

// §30.81 step 3a-3 Part 1: the run-internal source block-check index. Built
// once --- from the buffer load() already verified, or by streaming the file
// one block at a time on a region-first touch --- and reused across image-LRU
// eviction. A read_rect reads and SHA-256-checks only the blocks its Y window
// covers. Covers: byte-exact region parity with load() (incl. NaN payloads and
// -0), block-read amplification (X width is free, Y width is not), production
// width 3840 -> 17 rows/block, incomplete last block + cross-block rects, index
// survival across a real eviction, fresh instance / per-worker clone, and the
// detection-semantics contract boundary (only touched blocks are verified).
namespace { using Catch::Matchers::ContainsSubstring;
bool same_bits(float a,float b){ return std::memcmp(&a,&b,sizeof(float))==0; } }

TEST_CASE("source cache: block-check index --- parity, amplification, detection semantics",
          "[source-predecessors][cache-blocks]") {
  // 500 rows over a 512-wide frame: block_rows=128 -> 4 blocks, the last one
  // only 116 rows (incomplete). Distinct value per (y,x) so a wrong row shows.
  const int W=512,H=500;
  BlkCacheDir dir("src-cache-blocks",W,H,3,
      [](size_t i,int y,int x){ return i*0.5f + y*1000.0f + x; });
  VerifiedNormalizedSourceCache cache(dir.root,dir.plan,64);
  const int brows=static_cast<int>(cache.block_row_span());
  REQUIRE(brows==128);
  REQUIRE(brows<H);                       // frame spans several blocks
  REQUIRE(H % brows != 0);                // ... and the last one is short

  SECTION("read_rect == load() slice, byte-exact, incl. edges, 1px-X, cross-block") {
    const Matrix2Df full=cache.load(0);
    const std::array<std::array<int,4>,7> rects={{
      {{0,H,0,W}},{{10,42,0,W}},{{H-3,H,0,W}},{{100,140,7,9}},{{0,1,0,1}},
      {{200,201,W-1,W}},{{brows-5,2*brows+5,0,W}}}};   // straddles 3 blocks
    for (const auto &r:rects) {
      const Matrix2Df sub=cache.read_rect(0,r[0],r[1],r[2],r[3]);
      REQUIRE(sub.rows()==r[1]-r[0]);
      REQUIRE(sub.cols()==r[3]-r[2]);
      for (int y=r[0];y<r[1];++y) for (int x=r[2];x<r[3];++x)
        REQUIRE(same_bits(sub(y-r[0],x-r[2]),full(y,x)));
    }
    REQUIRE(cache.read_rect(0,5,5,0,W).rows()==0);     // empty range -> 0x0
    REQUIRE(cache.read_rect(0,0,H,9,9).cols()==0);
  }

  SECTION("byte-exact through NaN payloads and -0.0") {
    BlkCacheDir nd("src-cache-blk-nan",64,40,1,[](size_t,int y,int x){
      switch ((y*64+x) % 4) {
        case 0: return f_from_u32(0x80000000u);        // -0.0
        case 1: return f_from_u32(0x7fc0deadu);        // quiet NaN, payload
        case 2: return f_from_u32(0x7f800001u);        // signalling NaN, payload
        default: return float(x)-32.0f;
      }
    });
    VerifiedNormalizedSourceCache nc(nd.root,nd.plan,8);
    const Matrix2Df full=nc.load(0);
    const Matrix2Df sub=nc.read_rect(0,0,40,0,64);
    for (int y=0;y<40;++y) for (int x=0;x<64;++x)
      REQUIRE(same_bits(sub(y,x),full(y,x)));           // == would mishandle both
  }

  SECTION("production width 3840 -> 17 rows per block, short last block") {
    BlkCacheDir pd("src-cache-blk-3840",3840,40,1,
        [](size_t,int y,int x){ return float(y*4096+x); });
    VerifiedNormalizedSourceCache pc(pd.root,pd.plan,32);
    REQUIRE(pc.block_row_span()==17);                   // 256 KiB / (3840*4) rounded down
    pc.reset_io_counters();
    pc.read_rect(0,0,40,0,3840);                        // 3 blocks: 17 + 17 + 6
    REQUIRE(pc.blocks_verified()==3);
    const Matrix2Df full=pc.load(0), sub=pc.read_rect(0,18,40,0,3840);
    for (int y=18;y<40;++y) for (int x=0;x<3840;++x)
      REQUIRE(same_bits(sub(y-18,x),full(y,x)));
  }

  SECTION("block-read amplification: narrowing X is free, narrowing Y is not") {
    cache.reset_io_counters();
    cache.read_rect(0,0,H,0,W);                         // region-first touch: streams
    REQUIRE(cache.block_index_builds()==1);
    const auto full_blocks=cache.blocks_verified();
    REQUIRE(full_blocks==4);

    cache.reset_io_counters();
    cache.read_rect(0,brows,2*brows,0,W);               // one block, full width
    const auto y_win=cache.blocks_verified();
    const auto y_win_bytes=cache.bytes_read();

    cache.reset_io_counters();
    cache.read_rect(0,brows,2*brows,3,4);               // SAME Y, 1px wide
    REQUIRE(cache.blocks_verified()==y_win);            // X width: no extra blocks
    REQUIRE(cache.bytes_read()==y_win_bytes);           // ... and no I/O saved

    cache.reset_io_counters();
    cache.read_rect(0,0,H,3,4);                         // full Y, 1px wide
    REQUIRE(cache.blocks_verified()==full_blocks);      // all blocks again

    std::printf("[cache-blocks] block_rows=%d blocks/frame=%llu | 1-block "
                "read=%llu blk / %llu B | full=%llu blk\n",
                brows,(unsigned long long)full_blocks,
                (unsigned long long)y_win,(unsigned long long)y_win_bytes,
                (unsigned long long)full_blocks);
  }

  SECTION("load() and the block index share one verification --- no second read") {
    cache.reset_io_counters();
    cache.load(0);                                      // builds the index too
    REQUIRE(cache.block_index_builds()==1);
    const auto h0=cache.hash_computation_count();
    REQUIRE(cache.bytes_read()==static_cast<std::uint64_t>(H)*W*4);   // one whole frame
    cache.read_rect(0,0,brows,0,W);                     // block 0 only
    REQUIRE(cache.block_index_builds()==1);             // NOT rebuilt
    REQUIRE(cache.hash_computation_count()==h0);        // no whole-file re-hash
    REQUIRE(cache.bytes_read()==
            static_cast<std::uint64_t>(H)*W*4 + static_cast<std::uint64_t>(brows)*W*4);
  }

  SECTION("index survives a real image-LRU eviction") {
    // 512x500 float ~ 0.98 MiB/frame; budget 2 MiB -> usable 1 MiB -> capacity 1.
    VerifiedNormalizedSourceCache tight(dir.root,dir.plan,2);
    REQUIRE(tight.capacity_frames()==1);
    tight.load(0);                                      // resident + index for 0
    tight.load(1);                                      // evicts image 0
    REQUIRE(tight.resident_frame_count()==1);
    const auto builds=tight.block_index_builds();
    const auto hashes=tight.hash_computation_count();
    const Matrix2Df sub=tight.read_rect(0,brows,2*brows,0,W);   // image 0 gone
    REQUIRE(tight.block_index_builds()==builds);        // index outlived the image
    REQUIRE(tight.hash_computation_count()==hashes);    // no re-verify
    REQUIRE(sub.rows()==brows);
  }

  SECTION("fresh instance and per-worker clone each build their own index") {
    cache.read_rect(0,0,1,0,W);
    VerifiedNormalizedSourceCache fresh(dir.root,dir.plan,64);
    fresh.reset_io_counters();
    fresh.read_rect(0,0,1,0,W);
    REQUIRE(fresh.block_index_builds()==1);             // nothing carried over
    VerifiedNormalizedSourceCache worker(cache,8);
    worker.reset_io_counters();
    worker.read_rect(0,0,1,0,W);
    REQUIRE(worker.block_index_builds()==1);
  }

  SECTION("detection semantics: only the blocks a read touches are verified") {
    cache.read_rect(0,0,1,0,W);                         // build the index
    const auto saved=fs::last_write_time(dir.root/"0.raw");
    // Corrupt one float inside block 2, same file length, then roll the mtime
    // back so the staleness guard still passes: a region read does NOT
    // re-verify the whole file, by contract.
    const int trow=2*brows+1;
    {
      std::fstream fio(dir.root/"0.raw",std::ios::in|std::ios::out|std::ios::binary);
      fio.seekp(std::streamoff(static_cast<size_t>(trow)*W*sizeof(float)));
      const float bad=-42.0f;
      fio.write(reinterpret_cast<const char*>(&bad),sizeof(float));
    }
    fs::last_write_time(dir.root/"0.raw",saved);

    // A read that does NOT cover block 2 still succeeds...
    REQUIRE_NOTHROW(cache.read_rect(0,0,brows,0,W));
    REQUIRE_NOTHROW(cache.read_rect(0,3*brows,H,0,W));
    // ...a read that DOES cover block 2 is caught only now, on access, and with
    // the block error --- not an unintended whole-file rebuild.
    REQUIRE_THROWS_WITH(cache.read_rect(0,trow,trow+1,0,W),
                        ContainsSubstring("NORMALIZED_CACHE_BLOCK_MISMATCH"));
  }

  SECTION("in-place whole-file rewrite: mtime moves forward -> rebuild -> SHA mismatch") {
    cache.read_rect(1,0,1,0,W);
    REQUIRE(cache.block_index_builds()>=1);
    dir.write(1,[](size_t,int y,int x){ return 777.0f + y - x; });
    REQUIRE_THROWS_WITH(cache.read_rect(1,0,1,0,W),
                        ContainsSubstring("NORMALIZED_CACHE_CONTENT_MISMATCH"));
  }

  SECTION("rename-swap replacement is caught on next access") {
    cache.read_rect(2,0,1,0,W);
    // Write a valid alternate frame beside the target, then rename it over the
    // original. The renamed-in file carries a fresh (forward) mtime, so the
    // staleness triple fails -> rebuild -> whole-file SHA mismatch.
    const fs::path tgt=dir.root/"2.raw", tmp=dir.root/"2.raw.swap";
    {
      std::vector<float> px(static_cast<size_t>(W)*H,3.5f);
      std::ofstream of(tmp,std::ios::binary);
      of.write(reinterpret_cast<const char*>(px.data()),
               static_cast<std::streamsize>(px.size()*sizeof(float)));
    }
    fs::rename(tmp,tgt);
    REQUIRE_THROWS_WITH(cache.read_rect(2,0,1,0,W),
                        ContainsSubstring("NORMALIZED_CACHE_CONTENT_MISMATCH"));
  }
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
