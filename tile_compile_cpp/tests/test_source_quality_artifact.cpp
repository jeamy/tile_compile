#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_production.hpp"
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
#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>
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

TEST_CASE("source cache: size-bound frame loading rejects truncation (trusted run)", "[source-predecessors]") {
  Fixture f;
  const auto hash=publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache cache(f.root,f.plan,32);
  REQUIRE(cache.manifest_hash()==hash);
  REQUIRE(cache.load(0).minCoeff()==10.0f);
  REQUIRE(cache.load(1).minCoeff()==12.0f);
  REQUIRE_THROWS(cache.load(2));
  // T1 trusted run: a same-size rewrite is NOT detected (no content hash).
  // The cache re-reads and serves the new content.
  f.write(0,Matrix2Df::Constant(32,32,99.0f));
  REQUIRE(cache.load(0).minCoeff()==99.0f);
  // Truncation IS detected via the size check.
  { std::ofstream file(f.root/"1.raw",std::ios::binary); file<<"short"; }
  REQUIRE_THROWS(cache.load(1));
  REQUIRE_THROWS(VerifiedNormalizedSourceCache(f.root,f.plan,32));
}

// Plan §30.72 O2 / T1: the LRU serves a re-load of a resident, on-disk-unchanged
// frame without a read (trusted run: no SHA-256), but still fails closed when
// the file is truncated, and a per-worker clone is independent.
TEST_CASE("source cache: LRU hit skips reading, truncation still fails closed",
          "[source-predecessors][cache-lru]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);

  SECTION("re-load of an unchanged frame does not re-read") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64); // >= both frames fit
    const auto *p0=cache.load(0).data();
    REQUIRE(cache.load_call_count()==1);
    for (int i=0;i<5;++i) {
      REQUIRE(cache.load(0).data()==p0);           // same buffer, no reload
      REQUIRE(cache.load(0).minCoeff()==10.0f);
    }
    REQUIRE(cache.lru_hit_count()==10);           // 5 iterations x 2 calls
    REQUIRE(cache.load_call_count()==11);          // 1 initial + 5x2 loop
    cache.load(1);
    REQUIRE(cache.resident_frame_count()==2);
  }

  SECTION("eviction under a tight budget, and re-touch re-reads") {
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
    REQUIRE(cache.eviction_count()==1);
    cache.load(0);                                        // re-read after eviction
    REQUIRE(cache.bytes_read()==3*512*512*4);           // 3 full-frame reads
    fs::remove_all(br);
  }

  SECTION("rewrite with identical size serves new content (trusted run)") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64);
    REQUIRE(cache.load(0).minCoeff()==10.0f);
    f.write(0,Matrix2Df::Constant(32,32,77.0f));          // same byte count
    REQUIRE(cache.load(0).minCoeff()==77.0f);             // re-read, new content
  }

  SECTION("truncation is caught on the hit path") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64);
    REQUIRE(cache.load(0).minCoeff()==10.0f);
    { std::ofstream file(f.root/"0.raw",std::ios::binary); file<<"short"; }
    REQUIRE_THROWS(cache.load(0));                        // size mismatch
  }

  SECTION("per-worker clone shares the manifest but has an independent LRU") {
    VerifiedNormalizedSourceCache cache(f.root,f.plan,64);
    cache.load(0);
    VerifiedNormalizedSourceCache worker(cache,8);
    REQUIRE(worker.manifest_hash()==cache.manifest_hash());
    REQUIRE(worker.resident_frame_count()==0);
    REQUIRE(worker.load(0).minCoeff()==10.0f);
    REQUIRE(worker.load_call_count()==1);                 // its own first touch
    REQUIRE(cache.resident_frame_count()==1);            // unaffected
  }
}

// T1: read_rect region reads (trusted run, no block SHA). Verifies byte-exact
// parity with load() slices, including NaN payloads and edge cases.
namespace { using Catch::Matchers::ContainsSubstring;
bool same_bits(float a,float b){ return std::memcmp(&a,&b,sizeof(float))==0; } }

TEST_CASE("source cache: read_rect region reads --- parity with load()",
          "[source-predecessors][cache-regions]") {
  const int W=512,H=500;
  BlkCacheDir dir("src-cache-blocks",W,H,3,
      [](size_t i,int y,int x){ return i*0.5f + y*1000.0f + x; });
  VerifiedNormalizedSourceCache cache(dir.root,dir.plan,64);

  SECTION("read_rect == load() slice, byte-exact, incl. edges, 1px-X") {
    const Matrix2Df full=cache.load(0);
    const std::array<std::array<int,4>,7> rects={{
      {{0,H,0,W}},{{10,42,0,W}},{{H-3,H,0,W}},{{100,140,7,9}},{{0,1,0,1}},
      {{200,201,W-1,W}},{{0,H,0,W}}}};
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

  SECTION("read_rect on a fresh index reads only the requested rows") {
    VerifiedNormalizedSourceCache fresh(dir.root,dir.plan,64);
    fresh.reset_io_counters();
    fresh.read_rect(0,100,140,0,W);                    // 40 rows
    REQUIRE(fresh.bytes_read()==static_cast<std::uint64_t>(40)*W*4);
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

TEST_CASE("quality artifact: memory preflight precedes cache reads and preserves old artifact", "[source-predecessors]") {
  Fixture f;
  publish_normalized_source_manifest(f.root,f.plan);
  VerifiedNormalizedSourceCache cache(f.root,f.plan,32);
  const auto artifact=f.root/"quality.json";
  core::write_text_atomic(artifact,"previous");
  fs::remove(f.root/"0.raw");
  GlobalQualityConfig cfg;
  // An impossible scratch demand (~275 GiB) exceeds any autogrow headroom:
  // the preflight still fails before touching the cache or the artifact.
  cfg.star_max_corners=std::numeric_limits<int>::max();
  REQUIRE_THROWS_WITH(persist_source_quality_artifact(artifact,f.plan,cache,cfg,1),
      Catch::Matchers::ContainsSubstring("DRIZZLE_MEMORY_BUDGET"));
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

TEST_CASE("source cache: read_rect_into fills reusable buffers with exact "
          "window reads",
          "[source-predecessors][cache-regions]") {
  const int W = 37, H = 29;  // odd dims; rects non-aligned to anything
  BlkCacheDir dir("src-cache-rectinto", W, H, 2,
                  [](size_t i, int y, int x) {
                    return i * 0.25f + y * 100.0f + x;
                  });
  VerifiedNormalizedSourceCache cache(dir.root, dir.plan, 64);
  const Matrix2Df full = cache.load(0);

  const std::array<std::array<int, 4>, 8> rects{{
      {{0, H, 0, W}},       // whole
      {{3, 11, 5, 9}},      // interior, odd
      {{0, 1, 0, 1}},       // 1x1 origin
      {{H - 2, H, W - 3, W}},
      {{7, 7, 0, W}},       // empty (y1<=y0)
      {{0, H, 20, 20}},     // empty (x1<=x0)
      {{-5, 8, -3, 6}},     // clamped at the origin
      {{H - 4, H + 9, W - 6, W + 9}},  // clamped at the far edge
  }};
  std::vector<float> buf;
  buf.reserve(static_cast<size_t>(W) * H);
  const float *const buf_ptr = buf.data();
  const std::size_t buf_cap = buf.capacity();
  cache.reset_io_counters();
  std::uint64_t expect_bytes = 0, expect_calls = 0;
  for (const auto &r : rects) {
    const int cy0 = std::max(0, r[0]), cy1 = std::min(H, r[1]);
    const int cx0 = std::max(0, r[2]), cx1 = std::min(W, r[3]);
    const Matrix2Df ref = cache.read_rect(0, r[0], r[1], r[2], r[3]);
    const auto bytes0 = cache.bytes_read();
    cache.read_rect_into(0, r[0], r[1], r[2], r[3], buf);
    const int rows = std::max(0, cy1 - cy0), cols = std::max(0, cx1 - cx0);
    REQUIRE(buf.size() == static_cast<size_t>(rows) * cols);
    // Pre-reserved output must never reallocate across varying sizes.
    REQUIRE(buf.data() == buf_ptr);
    REQUIRE(buf.capacity() == buf_cap);
    for (int y = 0; y < ref.rows(); ++y)
      for (int x = 0; x < ref.cols(); ++x)
        REQUIRE(same_bits(buf[static_cast<size_t>(y) * cols + x],
                          ref(y, x)));
    // Only the [x0,x1) span of each requested row is read (no full rows).
    const std::uint64_t span =
        static_cast<std::uint64_t>(rows) * cols * sizeof(float);
    if (rows > 0 && cols > 0) ++expect_calls;
    if (rows > 0 && cols > 0) expect_bytes += 2 * span;
    REQUIRE(cache.bytes_read() - bytes0 ==
            (rows > 0 && cols > 0 ? span : 0));
  }
  // Each non-empty rect counts once for the reference read_rect (which
  // delegates to read_rect_into) and once for the direct call.
  REQUIRE(cache.rect_read_calls() == 2 * expect_calls);
  REQUIRE(cache.bytes_read() == expect_bytes);
  // An unknown index fails closed exactly like read_rect.
  REQUIRE_THROWS(cache.read_rect_into(9, 0, 1, 0, 1, buf));
}

TEST_CASE("source cache: read_row_intervals_into packs ragged rows exactly "
          "(tranche 8)", "[source-predecessors]") {
  BlkCacheDir dir("tc8-intervals", 24, 16, 1,
                  [](size_t, int y, int x) {
                    return static_cast<float>(y * 100 + x);
                  });
  VerifiedNormalizedSourceCache cache(dir.root, dir.plan, 8);
  std::vector<DrizzleAffineSourceSpan> spans = {
      {2, 3, 9}, {5, 0, 24}, {11, 7, 8}};
  std::vector<float> out;
  out.reserve(64);
  const float *p0 = out.data();
  cache.read_row_intervals_into(0, spans, out);
  REQUIRE(out.size() == 6 + 24 + 1);
  for (int i = 0; i < 6; ++i)
    REQUIRE(out[i] == static_cast<float>(200 + 3 + i));
  for (int i = 0; i < 24; ++i)
    REQUIRE(out[6 + i] == static_cast<float>(500 + i));
  REQUIRE(out[30] == static_cast<float>(1100 + 7));
  // Exact byte accounting: sum of interval lengths x 4, one call.
  REQUIRE(cache.bytes_read() == 31u * sizeof(float));
  REQUIRE(cache.rect_read_calls() == 1);
  // Capacity preserved across a second read; invalid inputs fail closed.
  cache.read_row_intervals_into(0, spans, out);
  REQUIRE(out.data() == p0);
  REQUIRE(cache.rect_read_calls() == 2);
  std::vector<DrizzleAffineSourceSpan> bad = {{20, 0, 4}};
  REQUIRE_THROWS(cache.read_row_intervals_into(0, bad, out));
  bad = {{2, 9, 3}};
  REQUIRE_THROWS(cache.read_row_intervals_into(0, bad, out));
  bad = {{2, 0, 25}};
  REQUIRE_THROWS(cache.read_row_intervals_into(0, bad, out));
  bad = {{5, 0, 4}, {3, 0, 4}};  // not ascending
  REQUIRE_THROWS(cache.read_row_intervals_into(0, bad, out));
}

TEST_CASE("tranche 8 ragged samples: storage spans + sample builder equal "
          "the whole-plane oracle bitwise",
          "[source-predecessors]") {
  const int sw = 20, sh = 14;
  BlkCacheDir dir("tc8-samples", sw, sh, 1, [](size_t, int y, int x) {
    return static_cast<float>(std::sin(0.37 * x + 0.11 * y) * 40 + y + x);
  });
  VerifiedNormalizedSourceCache cache(dir.root, dir.plan, 8);
  Matrix2Df full(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      full(y, x) = static_cast<float>(std::sin(0.37 * x + 0.11 * y) * 40 +
                                      y + x);
  const double sig_n = 0.05, sig_reg = 0.3, half = 0.4;
  const auto oracle =
      forward_drizzle_v2_sigma2_plane(full, sig_n, sig_reg, half);
  // A rotated transform so the spans are genuinely ragged.
  registration::FrameSamplingTransform f = dir.plan.frames[0];
  const double th = 10.0 * M_PI / 180.0;
  f.source_to_canvas(0, 0) = static_cast<float>(std::cos(th));
  f.source_to_canvas(0, 1) = static_cast<float>(-std::sin(th));
  f.source_to_canvas(0, 2) = 3.0f;
  f.source_to_canvas(1, 0) = static_cast<float>(std::sin(th));
  f.source_to_canvas(1, 1) = static_cast<float>(std::cos(th));
  f.source_to_canvas(1, 2) = 2.0f;
  std::vector<DrizzleAffineSourceSpan> active, storage;
  std::vector<float> storage_vals;
  std::vector<int> ra, rs;
  std::vector<std::size_t> soff;
  std::vector<ForwardDrizzleV2SourceSample> samples;
  std::uint64_t total_storage_elems = 0;
  for (int by = 0; by < dir.plan.canvas_height_native; by += 4) {
    const int rows = std::min(4, dir.plan.canvas_height_native - by);
    drizzle_affine_source_spans_into(dir.plan, f, 0.8f, by, rows, active);
    if (active.empty()) continue;
    forward_drizzle_v2_affine_storage_spans(active, sw, sh, true, ra, storage);
    cache.read_row_intervals_into(0, storage, storage_vals);
    for (const auto &s : storage)
      total_storage_elems +=
          static_cast<std::uint64_t>(s.x_end - s.x_begin);
    forward_drizzle_v2_build_affine_samples(
        active, storage, storage_vals, true, sig_n, sig_reg, half, sw, sh,
        rs, soff, samples);
    std::size_t prev = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto &smp = samples[i];
      if (i > 0)
        REQUIRE(smp.source_y * sw + smp.source_x > prev);
      prev = smp.source_y * sw + smp.source_x;
      const int sx = static_cast<int>(smp.source_x);
      const int sy = static_cast<int>(smp.source_y);
      REQUIRE(smp.value == full(sy, sx));
      REQUIRE(smp.sigma2 ==
              oracle[static_cast<std::size_t>(sy) * sw + sx]);
    }
  }
  INFO("storage elems " << total_storage_elems << " vs " << sw * sh);
  // sigma2 absent => sigma2 field is 0 and never read as present.
  forward_drizzle_v2_build_affine_samples(
      active, storage, storage_vals, false, sig_n, sig_reg, half, sw, sh,
      rs, soff, samples);
  for (const auto &smp : samples) REQUIRE(smp.sigma2 == 0.0f);
}
