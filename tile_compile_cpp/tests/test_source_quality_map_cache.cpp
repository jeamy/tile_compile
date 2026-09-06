// M5 tests for the multi-stream source-space quality-map cache
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 13.3-13.5): uint16 quantization with a reserved zero-veto
// sentinel, the two new identity/config hashes (independent of registration
// and config.sha256), atomic multi-stream layout with a fail-closed reader,
// and source-region reads that preserve an exact zero-veto through storage.

#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/normalized_source_cache.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using Catch::Approx;
using registration::FrameSamplingTransform;
using registration::RegistrationSamplingPlan;
namespace fs = std::filesystem;

namespace {

struct TempDir {
  fs::path path;
  TempDir() {
    path = fs::temp_directory_path() /
           ("sqm-cache-test-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

RegistrationSamplingPlan make_plan(int w, int h) {
  RegistrationSamplingPlan plan;
  plan.source_width = w;
  plan.source_height = h;
  plan.canvas_width_native = 999;   // must not affect the identity hash
  plan.canvas_height_native = 999;
  plan.internal_scale = 2;          // must not affect the identity hash
  plan.output_scale = 1;
  plan.color_mode = ColorMode::MONO;
  plan.bayer_pattern = BayerPattern::UNKNOWN;
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;
  plan.source_identity_hash = "config-bound-hash-DO-NOT-REUSE";
  plan.plan_hash = "sampling-plan-hash";
  FrameSamplingTransform f0;
  f0.frame_id = "frame-0";
  f0.source_index = 0;
  f0.valid = true;
  FrameSamplingTransform f1;
  f1.frame_id = "frame-1";
  f1.source_index = 1;
  f1.valid = true;
  plan.frames = {f0, f1};
  return plan;
}

// Piecewise-constant on 2x2 blocks so a divisor-2 downsample is exact.
Matrix2Df block_map(int w, int h, float base) {
  Matrix2Df m(h, w);
  for (int by = 0; by < h; by += 2)
    for (int bx = 0; bx < w; bx += 2) {
      const float v =
          base + 0.01f * static_cast<float>(by / 2) +
          0.03f * static_cast<float>(bx / 2);
      for (int dy = 0; dy < 2 && by + dy < h; ++dy)
        for (int dx = 0; dx < 2 && bx + dx < w; ++dx)
          m(by + dy, bx + dx) = v;
    }
  return m;
}

void write_raw_frame(const fs::path &p, int w, int h, float seed) {
  std::vector<float> buf(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      buf[static_cast<size_t>(y) * w + x] =
          100.0f + seed + 8.0f * std::sin(0.3f * x + seed) +
          6.0f * std::cos(0.21f * y) +
          20.0f * std::exp(-((x - 0.4f * w) * (x - 0.4f * w) +
                             (y - 0.5f * h) * (y - 0.5f * h)) /
                           40.0f);
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char *>(buf.data()),
          static_cast<std::streamsize>(buf.size() * sizeof(float)));
}

}  // namespace

TEST_CASE("quantize_quality reserves 0 for the zero-veto and never maps a "
          "nonzero code to 0") {
  REQUIRE(quantize_quality(std::numeric_limits<float>::quiet_NaN()) == 0u);
  REQUIRE(quantize_quality(0.0f) == 0u);
  REQUIRE(quantize_quality(-0.3f) == 0u);
  REQUIRE(std::isnan(dequantize_quality(0u)));

  for (uint16_t q : {uint16_t{1}, uint16_t{2}, uint16_t{100}, uint16_t{32768},
                     uint16_t{65535}}) {
    REQUIRE(dequantize_quality(q) > 0.0f);
  }
  // A tiny positive quality must not be flushed to the veto code.
  REQUIRE(quantize_quality(1e-7f) >= 1u);
  REQUIRE(quantize_quality(1.0f) == 65535u);

  for (float v : {1e-4f, 0.05f, 0.5f, 0.9f, 0.999f, 1.0f}) {
    const float rt = dequantize_quality(quantize_quality(v));
    REQUIRE(rt == Approx(v).margin(1.0f / 65535.0f + 1e-6f));
  }
}

TEST_CASE("source-quality identity hash ignores registration / canvas / scale "
          "but tracks source content") {
  const auto base = make_plan(16, 12);
  const std::string nch = "normalized-cache-hash-A";
  const std::string h0 = compute_source_quality_identity_hash(base, nch);

  SECTION("invariant to registration-only changes") {
    auto p = base;
    p.internal_scale = 4;
    p.output_scale = 2;
    p.canvas_width_native = 12345;
    p.canvas_offset_x_native = 77;
    p.plan_hash = "totally-different";
    p.source_identity_hash = "totally-different-too";
    REQUIRE(compute_source_quality_identity_hash(p, nch) == h0);
  }
  SECTION("changes with source dimensions") {
    auto p = base;
    p.source_width = 17;
    REQUIRE(compute_source_quality_identity_hash(p, nch) != h0);
  }
  SECTION("changes with CFA origin / colour mode / Bayer pattern") {
    auto a = base; a.cfa_origin_x = 1;
    auto b = base; b.color_mode = ColorMode::RGB;
    auto c = base; c.bayer_pattern = BayerPattern::RGGB;
    REQUIRE(compute_source_quality_identity_hash(a, nch) != h0);
    REQUIRE(compute_source_quality_identity_hash(b, nch) != h0);
    REQUIRE(compute_source_quality_identity_hash(c, nch) != h0);
  }
  SECTION("changes with frame identity and normalized-cache hash") {
    auto p = base;
    p.frames[1].frame_id = "frame-1-rev2";
    REQUIRE(compute_source_quality_identity_hash(p, nch) != h0);
    REQUIRE(compute_source_quality_identity_hash(base, "normalized-cache-B") !=
            h0);
  }
}

TEST_CASE("scale-quality config hash tracks pyramid params, storage divisor "
          "and dtype") {
  config::AqmhPyramidConfig p;
  SourceQualityMapCacheConfig c;
  const std::string h0 = compute_scale_quality_config_hash(p, c);
  REQUIRE(compute_scale_quality_config_hash(p, c) == h0);  // stable

  auto p2 = p; p2.w_sharp += 0.1f;
  REQUIRE(compute_scale_quality_config_hash(p2, c) != h0);
  auto c2 = c; c2.storage_divisor = 4;
  REQUIRE(compute_scale_quality_config_hash(p, c2) != h0);
  auto c3 = c; c3.proxy_version = 2;
  REQUIRE(compute_scale_quality_config_hash(p, c3) != h0);
}

TEST_CASE("cache writer + fail-closed reader round-trip with a zero-veto "
          "block preserved through storage") {
  TempDir tmp;
  const fs::path root = tmp.path / "source_quality_maps";
  const int w = 8, h = 8;
  const auto plan = make_plan(w, h);
  const std::string nch = "ncache-hash-1";
  config::AqmhPyramidConfig pyr;
  SourceQualityMapCacheConfig ccfg;  // divisor 2, uint16

  Matrix2Df composite = block_map(w, h, 0.4f);
  // Veto the whole 2x2 block at source rows/cols [2,4).
  for (int y = 2; y < 4; ++y)
    for (int x = 2; x < 4; ++x)
      composite(y, x) = std::numeric_limits<float>::quiet_NaN();
  Matrix2Df scale0 = block_map(w, h, 0.6f);

  std::string identity, cfghash, cachehash;
  {
    SourceQualityMapCacheWriter wr(root, plan, nch, pyr, ccfg);
    identity = wr.identity_hash();
    cfghash = wr.config_hash();
    wr.put("composite", 0, composite);
    wr.put("scale_0", 0, scale0);
    const auto meta = wr.commit();
    cachehash = meta.source_quality_cache_hash;
    REQUIRE(meta.streams.size() == 2);
    REQUIRE(meta.source_width == w);
    REQUIRE(meta.storage_divisor == 2);
  }
  REQUIRE(identity ==
          compute_source_quality_identity_hash(plan, nch));
  REQUIRE(cfghash == compute_scale_quality_config_hash(pyr, ccfg));

  SECTION("usable reader returns blocky maps and preserves the veto exactly") {
    SourceQualityMapCacheReader rd(root, identity, cfghash);
    REQUIRE(rd.usable());
    REQUIRE(rd.has("composite", 0));
    REQUIRE(rd.has("scale_0", 0));
    REQUIRE_FALSE(rd.has("scale_1", 0));

    const Matrix2Df full = rd.read_full("composite", 0);
    REQUIRE(full.rows() == h);
    REQUIRE(full.cols() == w);

    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float v = full(y, x);
        if (y >= 2 && y < 4 && x >= 2 && x < 4) {
          REQUIRE(std::isnan(v));  // vetoed storage cell -> veto everywhere
        } else {
          REQUIRE(std::isfinite(v));
          REQUIRE(v > 0.0f);
          // Blocky reconstruction of the 2x2-constant source block.
          REQUIRE(v == Approx(composite(y, x)).margin(1.5f / 65535.0f));
        }
      }

    // Region read equals the matching rows of the full map.
    const Matrix2Df region = rd.read_region("composite", 0, 2, 6);
    REQUIRE(region.rows() == 4);
    for (int y = 2; y < 6; ++y)
      for (int x = 0; x < w; ++x) {
        const float a = region(y - 2, x);
        const float b = full(y, x);
        REQUIRE((std::isnan(a) == std::isnan(b)));
        if (std::isfinite(a)) REQUIRE(a == b);
      }
  }

  SECTION("wrong expected identity hash -> not usable") {
    SourceQualityMapCacheReader rd(root, "some-other-identity", cfghash);
    REQUIRE_FALSE(rd.usable());
    REQUIRE(rd.error() == "SQM_CACHE_IDENTITY_MISMATCH");
  }

  SECTION("tampered .bin payload -> fail closed on checksum") {
    const fs::path bin = root / "composite" / "source_quality_composite_000000.bin";
    REQUIRE(fs::exists(bin));
    {
      std::fstream f(bin, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(-2, std::ios::end);
      const char poke[2] = {0x7f, 0x33};
      f.write(poke, 2);
    }
    SourceQualityMapCacheReader rd(root, identity, cfghash);
    REQUIRE_FALSE(rd.usable());
    REQUIRE(rd.error().rfind("SQM_CACHE_FILE_CORRUPT", 0) == 0);
  }

  SECTION("tampered metadata.json -> manifest hash mismatch") {
    const fs::path meta = root / "metadata.json";
    std::string text;
    {
      std::ifstream f(meta);
      text.assign((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    }
    const auto pos = text.find("\"storage_divisor\": 2");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, std::string("\"storage_divisor\": 2").size(),
                 "\"storage_divisor\": 3");
    { std::ofstream(meta, std::ios::trunc) << text; }
    SourceQualityMapCacheReader rd(root, identity, cfghash);
    REQUIRE_FALSE(rd.usable());
  }
}

TEST_CASE("build_source_quality_map_cache orchestrates proxy -> streamed "
          "scale maps + composite + artifact for every frame (MONO)") {
  TempDir tmp;
  const int w = 96, h = 72;  // min_dim >= 64 so >= 2 pyramid scales compute
  auto plan = make_plan(w, h);
  plan.color_mode = ColorMode::MONO;
  plan.source_identity_hash = "mono-src-identity";  // required by the ncache

  const fs::path ncache_root = tmp.path / "normalized_frames";
  fs::create_directories(ncache_root);
  write_raw_frame(ncache_root / "0.raw", w, h, 0.0f);
  write_raw_frame(ncache_root / "1.raw", w, h, 5.0f);
  reconstruction::publish_normalized_source_manifest(ncache_root, plan);
  reconstruction::VerifiedNormalizedSourceCache ncache(ncache_root, plan);

  const fs::path sqm_root = tmp.path / "source_quality_maps";
  config::AqmhPyramidConfig pyr;
  const auto built = reconstruction::build_source_quality_map_cache(
      sqm_root, plan, ncache, pyr);

  REQUIRE(built.frames == 2);
  REQUIRE(built.computed_scales >= 2);
  // composite + artifact + one scale_N per computed scale.
  REQUIRE(std::find(built.streams.begin(), built.streams.end(), "composite") !=
          built.streams.end());
  REQUIRE(std::find(built.streams.begin(), built.streams.end(), "artifact") !=
          built.streams.end());
  REQUIRE(std::find(built.streams.begin(), built.streams.end(), "scale_0") !=
          built.streams.end());

  SourceQualityMapCacheReader rd(sqm_root, built.source_identity_hash,
                                 built.source_quality_config_hash);
  REQUIRE(rd.usable());
  REQUIRE(rd.metadata().source_quality_cache_hash ==
          built.source_quality_cache_hash);
  for (std::size_t fi : {std::size_t{0}, std::size_t{1}}) {
    REQUIRE(rd.has("composite", fi));
    REQUIRE(rd.has("artifact", fi));
    REQUIRE(rd.has("scale_0", fi));
    const Matrix2Df comp = rd.read_full("composite", fi);
    REQUIRE(comp.rows() == h);
    REQUIRE(comp.cols() == w);
    std::vector<float> vals;
    for (int i = 0; i < comp.size(); ++i)
      if (std::isfinite(comp.data()[i])) {
        REQUIRE(comp.data()[i] > 0.0f);
        REQUIRE(comp.data()[i] <= 1.0f);
        vals.push_back(comp.data()[i]);
      }
    // The value stream keeps positive-mean data next to unsupported borders
    // instead of decimating every straddling storage cell: the cached
    // composite must retain the large majority of the source-valid pixels,
    // not a handful of partition-aligned survivors. This synthetic proxy has
    // interior support almost everywhere, so expect > 90%.
    REQUIRE(static_cast<int>(vals.size()) > (comp.size() * 9) / 10);

    // The Q_composite / Fine / Medium weighting downstream is only meaningful
    // if the composite carries real per-pixel spread --- a near-constant map
    // would make Q_composite a multiplier that cancels in sum_wx_r/sum_w_r
    // and would make pow(Q_scale0, 4) noise. On a proxy with genuine
    // sharpness/SNR structure the geometric-mean composite must span a
    // useful range.
    std::sort(vals.begin(), vals.end());
    const float p05 = vals[vals.size() / 20];
    const float p50 = vals[vals.size() / 2];
    const float p95 = vals[(vals.size() * 19) / 20];
    INFO("composite Q p05/p50/p95 = " << p05 << " / " << p50 << " / " << p95
                                      << "  spread p95/p05 = " << (p95 / p05));
    REQUIRE(p95 / p05 > 1.5f);   // not degenerate
    REQUIRE(p95 - p05 > 0.10f);  // meaningful absolute range
  }
}

TEST_CASE("cache preserves an exact Q=0 hard veto through storage even when "
          "the storage cell also covers positive samples") {
  TempDir tmp;
  const fs::path root = tmp.path / "source_quality_maps";
  const int w = 8, h = 8;
  const auto plan = make_plan(w, h);
  config::AqmhPyramidConfig pyr;
  SourceQualityMapCacheConfig ccfg;  // divisor 2

  Matrix2Df comp = block_map(w, h, 0.5f);
  // A single hard veto (exact 0) inside an otherwise-positive 2x2 cell.
  comp(1, 1) = 0.0f;
  // A NaN (no support, NOT a hard veto) inside another otherwise-positive cell.
  comp(1, 5) = std::numeric_limits<float>::quiet_NaN();

  std::string id, cfgh;
  {
    SourceQualityMapCacheWriter wr(root, plan, "nch", pyr, ccfg);
    id = wr.identity_hash();
    cfgh = wr.config_hash();
    wr.put("composite", 0, comp);
    wr.commit();
  }
  SourceQualityMapCacheReader rd(root, id, cfgh);
  REQUIRE(rd.usable());
  const Matrix2Df full = rd.read_full("composite", 0);

  // Hard-veto cell (source rows/cols [0,2)) -> NaN everywhere, never positive.
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x) REQUIRE(std::isnan(full(y, x)));

  // No-support cell (source rows [0,2), cols [4,6)) keeps the positive mean of
  // its 3 supported samples -- good data is not decimated by a lone NaN.
  bool any_positive = false;
  for (int y = 0; y < 2; ++y)
    for (int x = 4; x < 6; ++x)
      if (std::isfinite(full(y, x)) && full(y, x) > 0.0f) any_positive = true;
  REQUIRE(any_positive);
}
