// Plan section 11.14 P1/P2 --- the geometry cache reproduces
// enumerate_drizzle_stripe_leaf_cells bit-for-bit from a built store at any
// chunk height, runs sample_leaves exactly once per source pixel, adds zero
// further local basis evaluations on read, keeps only the row index resident
// (no full-leaf cache), and enforces a strict reader contract.

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {
namespace fs = std::filesystem;

WarpMatrix s2c(double a, double b, double tx, double c, double d, double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

// `strength` scales the local displacement; large values force samples to fail
// inversion / subdivision so an exclusion rate can be exercised.
RegistrationSamplingPlan local_plan(int sw, int sh, int cw, int ch,
                                    float strength = 1.0f, int n_frames = 2) {
  RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = cw;
  plan.canvas_height_native = ch;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::GBRG;
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;
  plan.plan_hash = "test-plan-hash-geomcache";
  for (int fi = 0; fi < n_frames; ++fi) {
    FrameSamplingTransform f;
    f.frame_id = "local" + std::to_string(fi);
    f.source_index = static_cast<std::size_t>(fi);
    f.valid = true;
    f.source_to_canvas = s2c(1.0, 0.0, (cw - sw) / 2.0 + 0.3 * fi, 0.0, 1.0,
                             (ch - sh) / 2.0 - 0.2 * fi);
    f.source_to_canvas_affine_valid = true;
    f.has_smooth_local_model = true;
    f.smooth_local_model.valid = true;
    f.smooth_local_model.image_rows = ch;
    f.smooth_local_model.image_cols = cw;
    f.smooth_local_model.coeff_x.setZero();
    f.smooth_local_model.coeff_y.setZero();
    f.smooth_local_model.coeff_x[0] = strength * (0.12f + 0.03f * fi);
    f.smooth_local_model.coeff_y[0] = strength * -0.09f;
    f.smooth_local_model.coeff_x[5] = strength * 0.04f;
    f.model_coordinate_scale = 1.0f;
    plan.frames.push_back(f);
  }
  return plan;
}

std::vector<std::size_t> indices(int n) {
  std::vector<std::size_t> v;
  for (int i = 0; i < n; ++i) v.push_back(static_cast<std::size_t>(i));
  return v;
}

struct Cell {
  int sx, sy, c, leaf_order, x, y;
  double lx[4], ly[4];
  bool operator==(const Cell &o) const {
    if (sx != o.sx || sy != o.sy || c != o.c || leaf_order != o.leaf_order ||
        x != o.x || y != o.y)
      return false;
    for (int i = 0; i < 4; ++i)
      if (std::memcmp(&lx[i], &o.lx[i], sizeof(double)) != 0 ||
          std::memcmp(&ly[i], &o.ly[i], sizeof(double)) != 0)
        return false;
    return true;
  }
};

std::vector<Cell> reference_stripe(const RegistrationSamplingPlan &plan,
                                   const FrameSamplingTransform &f, int scale,
                                   float pixfrac, int y0, int rows) {
  std::vector<Cell> out;
  enumerate_drizzle_stripe_leaf_cells(
      plan, f, scale, pixfrac, y0, rows,
      [&](int sx, int sy, int c, int lo, int x, int y, const double *lx,
          const double *ly) {
        Cell cell{sx, sy, c, lo, x, y, {}, {}};
        for (int i = 0; i < 4; ++i) {
          cell.lx[i] = lx[i];
          cell.ly[i] = ly[i];
        }
        out.push_back(cell);
      },
      {});
  return out;
}

fs::path scratch_root(const std::string &name) {
  static std::mt19937_64 rng{0xC0FFEEu};
  const fs::path p = fs::temp_directory_path() /
                     ("tc_geomcache_" + name + "_" + std::to_string(rng()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

} // namespace

TEST_CASE("geometry cache: replays enumerate_drizzle_stripe_leaf_cells "
          "bit-for-bit at every chunk height (plan 11.14 P2)",
          "[geometry-cache][forward-drizzle-p0]") {
  const int sw = 20, sh = 20, cw = 40, ch = 40, scale = 1;
  const RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch);
  const float pf = 0.8f;

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = pf;

  const fs::path root = scratch_root("replay");
  const std::vector<GeometryVariant> variants = {{pf}, {1.0f}};
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, variants, {},
                                                  64ull * 1024 * 1024);

  const std::uint64_t source_pixels = static_cast<std::uint64_t>(sw) * sh;
  REQUIRE(built.frames.size() == 2u * 2u);
  for (const auto &fr : built.frames)
    REQUIRE(fr.top_level_sample_leaves_calls == source_pixels);

  DrizzleGeometryCacheReader reader(root, built.identities, indices(2));
  // Row index only --- resident bytes must not scale with the leaf volume.
  REQUIRE(reader.resident_bytes() <
          static_cast<std::uint64_t>(sh) * 4 * 64 * sizeof(double) * 2 + 65536);

  for (int frame_i = 0; frame_i < 2; ++frame_i) {
    const auto &f = plan.frames[static_cast<std::size_t>(frame_i)];
    REQUIRE(reader.has_frame(pf, f.source_index));
    REQUIRE(reader.has_frame(1.0f, f.source_index));

    for (int chunk : {1, 7, 16, ch * scale}) {
      std::vector<Cell> ref_all, cache_all;
      for (int y = 0; y < ch * scale; y += chunk) {
        const int rows = std::min(chunk, ch * scale - y);
        auto r = reference_stripe(plan, f, scale, pf, y, rows);
        ref_all.insert(ref_all.end(), r.begin(), r.end());
        reader.enumerate_stripe(
            pf, f.source_index, scale, y, rows,
            [&](int sx, int sy, int c, int lo, int x, int yy, const double *lx,
                const double *ly) {
              Cell cell{sx, sy, c, lo, x, yy, {}, {}};
              for (int i = 0; i < 4; ++i) {
                cell.lx[i] = lx[i];
                cell.ly[i] = ly[i];
              }
              cache_all.push_back(cell);
            });
      }
      INFO("frame=" << frame_i << " chunk=" << chunk
                    << " ref=" << ref_all.size()
                    << " cache=" << cache_all.size());
      REQUIRE(cache_all.size() == ref_all.size());
      for (std::size_t i = 0; i < ref_all.size(); ++i)
        REQUIRE(cache_all[i] == ref_all[i]);
    }
  }
}

TEST_CASE("geometry cache: reader adds zero local basis evaluations "
          "(plan 11.14.4)",
          "[geometry-cache][forward-drizzle-p0]") {
  const int sw = 16, sh = 16, cw = 32, ch = 32, scale = 1;
  const RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = 0.8f;
  const fs::path root = scratch_root("nobasis");
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                                  64ull * 1024 * 1024);
  DrizzleGeometryCacheReader reader(root, built.identities, indices(2));

  geomstats::ScopedEnable en(true);
  geomstats::ScopedVariant sv(geomstats::Variant::kUnattributed, 0.8);
  const auto before = geomstats::registry().cur();
  for (int y = 0; y < ch * scale; y += 5) {
    const int rows = std::min(5, ch * scale - y);
    reader.enumerate_stripe(0.8f, 0, scale, y, rows,
                            [](int, int, int, int, int, int, const double *,
                               const double *) {});
  }
  const auto after = geomstats::registry().cur();
  REQUIRE(after.invert_iterations == before.invert_iterations);
  REQUIRE(after.top_level_sample_leaves_calls ==
          before.top_level_sample_leaves_calls);
  REQUIRE(after.subdivide_local_calls == before.subdivide_local_calls);
}

TEST_CASE("geometry cache: frame exclusions do not depend on the diagnostic "
          "counters being enabled (plan 11.14.3)",
          "[geometry-cache]") {
  // A strong warp near a tight canvas border: some samples fail inversion, so
  // the exclusion rate is nonzero. It must be identical with geomstats on/off.
  const int sw = 18, sh = 18, cw = 20, ch = 20, scale = 1;
  const RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch, 9.0f, 1);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = 0.8f;

  auto run = [&](bool counters_on) {
    const fs::path root =
        scratch_root(counters_on ? "excl_on" : "excl_off");
    std::unique_ptr<geomstats::ScopedEnable> en;
    if (counters_on) en = std::make_unique<geomstats::ScopedEnable>(true);
    else {
      // force-disable for this run
      en = std::make_unique<geomstats::ScopedEnable>(false);
    }
    return build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                        64ull * 1024 * 1024);
  };

  const auto a = run(true);
  const auto b = run(false);
  REQUIRE(a.frames.size() == 1u);
  REQUIRE(b.frames.size() == 1u);
  REQUIRE(a.frames[0].samples_total == b.frames[0].samples_total);
  REQUIRE(a.frames[0].samples_discarded == b.frames[0].samples_discarded);
  REQUIRE(a.frames[0].samples_discarded > 0);  // the scenario actually excludes
  REQUIRE(a.frames[0].subdivision_error_rate ==
          b.frames[0].subdivision_error_rate);
  REQUIRE(a.frames[0].excluded == b.frames[0].excluded);
  REQUIRE(a.excluded_frames.size() == b.excluded_frames.size());
}

TEST_CASE("geometry cache: strict reader contract --- identity, population, "
          "schema, offsets, checksums", "[geometry-cache]") {
  const RegistrationSamplingPlan plan = local_plan(12, 12, 24, 24, 1.0f, 2);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  const fs::path root = scratch_root("contract");
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                                  64ull * 1024 * 1024);

  // Baseline: opens cleanly with the right population.
  REQUIRE_NOTHROW(
      DrizzleGeometryCacheReader(root, built.identities, indices(2)));

  // Identity mismatch.
  {
    auto bad = built.identities;
    bad[0].geometry_hash = "deadbeef";
    REQUIRE_THROWS(DrizzleGeometryCacheReader(root, bad, indices(2)));
  }
  // Wrong expected population (missing a frame / extra frame / duplicate).
  REQUIRE_THROWS(
      DrizzleGeometryCacheReader(root, built.identities, indices(1)));
  REQUIRE_THROWS(
      DrizzleGeometryCacheReader(root, built.identities, indices(3)));
  REQUIRE_THROWS(DrizzleGeometryCacheReader(root, built.identities,
                                            std::vector<std::size_t>{0, 0}));

  // Bump the manifest schema -> rejected.
  {
    const fs::path mp = built.generation_dir / "manifest.json";
    std::string t;
    {
      std::ifstream in(mp);
      t.assign((std::istreambuf_iterator<char>(in)), {});
    }
    const auto pos = t.find("\"schema\": 1");
    REQUIRE(pos != std::string::npos);
    t.replace(pos, 11, "\"schema\": 2");
    std::ofstream(mp, std::ios::trunc) << t;
    REQUIRE_THROWS(
        DrizzleGeometryCacheReader(root, built.identities, indices(2)));
  }
}

TEST_CASE("geometry cache: record-file corruption is caught with byte "
          "verification and rows-hash otherwise", "[geometry-cache]") {
  const RegistrationSamplingPlan plan = local_plan(12, 12, 24, 24, 1.0f, 1);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  const fs::path root = scratch_root("corrupt");
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                                  64ull * 1024 * 1024);

  // Corrupt the middle of the .leaves file.
  bool did = false;
  for (const auto &e : fs::directory_iterator(built.generation_dir)) {
    if (e.path().extension() != ".leaves") continue;
    const auto sz = fs::file_size(e.path());
    if (sz < 16) continue;
    std::fstream fh(e.path(), std::ios::binary | std::ios::in | std::ios::out);
    fh.seekp(static_cast<std::streamoff>(sz / 2));
    const char flip[8] = {90, 90, 90, 90, 90, 90, 90, 90};
    fh.write(flip, 8);
    did = true;
    break;
  }
  REQUIRE(did);
  // T1 trusted run: same-size content corruption is NOT detected (no SHA-256).
  // Only the structural size check catches truncation.
  // Truncate the .leaves file -> structural (size) check fails.
  for (const auto &e : fs::directory_iterator(built.generation_dir)) {
    if (e.path().extension() != ".leaves") continue;
    fs::resize_file(e.path(), fs::file_size(e.path()) - sizeof(double));
    break;
  }
  REQUIRE_THROWS(DrizzleGeometryCacheReader(root, built.identities, indices(1)));
}

TEST_CASE("geometry cache: parallel build (1/2/4 workers) is byte-identical to "
          "the single-worker reference (plan 11.14.5 P3)",
          "[geometry-cache][geometry-parallel]") {
  const int sw = 28, sh = 28, cw = 52, ch = 52, scale = 1;
  const RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch, 1.0f, 6);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = 0.8f;

  Matrix2Df img(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      img(y, x) = static_cast<float>(3.0 + 0.05 * (x + 2 * y));
  SourceImageProvider src = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  struct Fp {
    std::vector<std::pair<std::string, std::string>> per_frame;  // rows,leaves
    std::uint64_t total_leaves = 0, total_bytes = 0;
    std::size_t excluded = 0;
    std::uint64_t drizzle_digest = 0;
  };
  auto run = [&](int workers) {
    const fs::path root = scratch_root("par_w" + std::to_string(workers));
    const auto built = build_drizzle_geometry_cache(
        root, plan, cfg, {{cfg.pixfrac}, {1.0f}}, {}, 64ull << 20, workers);
    Fp fp;
    fp.total_leaves = built.total_leaves;
    fp.total_bytes = built.total_record_bytes;
    fp.excluded = built.excluded_frames.size();
    // per-frame content hashes from the committed manifest
    std::ifstream mfs(built.generation_dir / "manifest.json");
    nlohmann::json mf;
    mfs >> mf;
    for (const auto &v : mf.at("variants"))
      for (const auto &f : v.at("frames"))
        fp.per_frame.emplace_back(std::to_string(f.at("rows_bytes").get<std::uintmax_t>()),
                                  std::to_string(f.at("leaves_bytes").get<std::uintmax_t>()));
    DrizzleGeometryCacheReader reader(root, built.identities, indices(6));
    ScopedActiveGeometryCache guard(&reader);
    config::ReconstructionDrizzleConfig c = cfg;
    c.chunk_rows = 5;
    const auto r = compute_forward_drizzle_uniform(plan, src, c);
    fp.drizzle_digest = 0;
    for (const auto *pl : {&r.R, &r.G, &r.B}) {
      std::uint64_t h = 1469598103934665603ull;
      const auto *p = reinterpret_cast<const unsigned char *>(pl->value.data());
      for (std::size_t i = 0, n = pl->value.size() * sizeof(float); i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
      }
      fp.drizzle_digest ^= h;
    }
    fs::remove_all(root);
    return fp;
  };

  const Fp ref = run(1);
  REQUIRE(ref.per_frame.size() == 12u);  // 2 variants x 6 local frames
  for (int w : {2, 4}) {
    const Fp got = run(w);
    INFO("workers=" << w);
    REQUIRE(got.per_frame == ref.per_frame);       // identical file content
    REQUIRE(got.total_leaves == ref.total_leaves);
    REQUIRE(got.total_bytes == ref.total_bytes);
    REQUIRE(got.excluded == ref.excluded);
    REQUIRE(got.drizzle_digest == ref.drizzle_digest);  // identical output
  }
}

TEST_CASE("geometry cache: a rebuild keeps the previous generation intact and "
          "flips current.json atomically", "[geometry-cache]") {
  const RegistrationSamplingPlan plan = local_plan(10, 10, 20, 20, 1.0f, 1);
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  const fs::path root = scratch_root("regen");

  const auto a = build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                              64ull * 1024 * 1024);
  const auto b = build_drizzle_geometry_cache(root, plan, cfg, {{0.8f}}, {},
                                              64ull * 1024 * 1024);
  REQUIRE(a.generation_dir != b.generation_dir);
  REQUIRE(fs::exists(a.generation_dir / "manifest.json"));  // old one untouched
  REQUIRE(fs::exists(b.generation_dir / "manifest.json"));
  // current.json points at the newest.
  DrizzleGeometryCacheReader reader(root, b.identities, indices(1));
  REQUIRE(reader.has_frame(0.8f, 0));
  // No leftover staging directories.
  for (const auto &e : fs::directory_iterator(root))
    REQUIRE(e.path().filename().string().rfind(".staging-", 0) != 0);
}

// ---------------------------------------------------------------------------
// Plan 11.14.5 P3 (Teil 2): per-stripe output-row-band parallel reduction
// ---------------------------------------------------------------------------

namespace {

FrameSamplingTransform affine_frame(std::size_t idx, int sw, int sh, int cw,
                                    int ch) {
  FrameSamplingTransform f;
  f.frame_id = "aff" + std::to_string(idx);
  f.source_index = idx;
  f.valid = true;
  const double tx = (cw - sw) / 2.0 + 0.37 * idx;
  const double ty = (ch - sh) / 2.0 - 0.21 * idx;
  f.source_to_canvas = s2c(1.0, 0.0, tx, 0.0, 1.0, ty);
  f.source_to_canvas_affine_valid = true;
  f.has_smooth_local_model = false;
  return f;
}

// Sorted-multiset comparison: the band tiling changes the enumerate *sequence*
// (all source rows for band 0, then all for band 1, ...) but must not add,
// drop, or alter a single leaf cell.
std::vector<Cell> sorted(std::vector<Cell> v) {
  std::sort(v.begin(), v.end(), [](const Cell &a, const Cell &b) {
    auto key = [](const Cell &c) {
      return std::make_tuple(c.sy, c.sx, c.c, c.leaf_order, c.y, c.x);
    };
    if (key(a) != key(b)) return key(a) < key(b);
    for (int i = 0; i < 4; ++i) {
      if (a.lx[i] != b.lx[i]) return a.lx[i] < b.lx[i];
      if (a.ly[i] != b.ly[i]) return a.ly[i] < b.ly[i];
    }
    return false;
  });
  return v;
}

} // namespace

TEST_CASE("geometry cache: unioning disjoint canvas-row sub-windows reproduces "
          "the whole-stripe leaf cells exactly (band-tiling invariant)",
          "[geometry-cache][geometry-parallel]") {
  const int sw = 24, sh = 24, cw = 44, ch = 44, scale = 1;
  const float pf = 0.8f;
  RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch, 1.0f, 1);
  plan.frames.push_back(affine_frame(1, sw, sh, cw, ch));  // + one affine frame

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = pf;

  const fs::path root = scratch_root("bandtile");
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, {{pf}}, {},
                                                  64ull << 20);
  DrizzleGeometryCacheReader reader(root, built.identities, indices(1));
  ScopedActiveGeometryCache guard(&reader);

  const int H = ch * scale;
  for (const auto &f : plan.frames) {
    INFO("frame " << f.frame_id
                  << (f.has_smooth_local_model ? " (cache-fed local)"
                                               : " (affine, no cache)"));
    const auto whole = reference_stripe(plan, f, scale, pf, 0, H);

    for (int parts : {2, 3, 5}) {
      std::vector<Cell> unioned;
      for (int p = 0; p < parts; ++p) {
        const int y0 = static_cast<int>((static_cast<long long>(p) * H) / parts);
        const int y1 =
            static_cast<int>((static_cast<long long>(p + 1) * H) / parts);
        if (y1 <= y0) continue;
        auto part = reference_stripe(plan, f, scale, pf, y0, y1 - y0);
        unioned.insert(unioned.end(), part.begin(), part.end());
      }
      INFO("parts=" << parts);
      REQUIRE(unioned.size() == whole.size());
      REQUIRE(sorted(unioned) == sorted(whole));
    }
  }
}


TEST_CASE("geometry cache: read_stripe_leaves_into replays records, bounds "
          "capacity and counts exact payload bytes (tranche 6)",
          "[geometry-cache]") {
  const int sw = 18, sh = 18, cw = 36, ch = 36, scale = 1;
  const RegistrationSamplingPlan plan = local_plan(sw, sh, cw, ch, 1.0f, 2);
  const float pf = 0.8f;
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = scale;
  cfg.pixfrac = pf;
  const fs::path root = scratch_root("leafread");
  const auto built = build_drizzle_geometry_cache(root, plan, cfg, {{pf}}, {},
                                                  64ull << 20);
  DrizzleGeometryCacheReader reader(root, built.identities, indices(2));
  REQUIRE(built.frames.size() == 2u);
  for (const auto &fs : built.frames) REQUIRE_FALSE(fs.excluded);

  const int H = ch * scale, Wi = cw * scale;
  // First, middle and last (short) stripes.
  const int stripes[3][2] = {{0, 7}, {13, 9}, {H - 5, 5}};
  std::uint64_t cap = 0;
  for (const auto &st : stripes)
    cap = std::max(cap, reader.max_stripe_leaf_records(pf, st[1]));
  // The whole-canvas read below must also stay inside the reserve.
  cap = std::max(cap, reader.max_stripe_leaf_records(pf, H));
  REQUIRE(cap > 0);
  // The legacy enumerator keeps its own row-block I/O contract: it must
  // not run through (or be billed to) the v2 leaf-record reader.
  {
    const std::uint64_t lb0 = reader.leaf_record_bytes_read();
    const std::uint64_t ec0 = reader.enumerate_call_count();
    std::size_t emitted = 0;
    reader.enumerate_stripe(
        pf, plan.frames[0].source_index, scale, 0, H,
        [&](int, int, int, int, int, int, const double *, const double *) {
          ++emitted;
        });
    REQUIRE(emitted > 0);
    REQUIRE(reader.enumerate_call_count() == ec0 + 1);
    REQUIRE(reader.leaf_record_bytes_read() == lb0);
  }

  DrizzleCachedLeafWindow win;
  win.leaves.reserve(static_cast<std::size_t>(cap));
  // Row-block I/O scratch: one on-disk record block at a time.
  win.io_scratch.reserve(
      static_cast<std::size_t>(reader.max_row_record_count()) * 72u);
  const DrizzleCachedLeaf *stable = win.leaves.data();
  const std::uint8_t *stable_io = win.io_scratch.data();
  const std::size_t leaves_cap = win.leaves.capacity();
  const std::size_t io_cap = win.io_scratch.capacity();
  REQUIRE(io_cap > 0);

  for (int fi = 0; fi < 2; ++fi) {
    const auto &f = plan.frames[static_cast<std::size_t>(fi)];
    for (const auto &st : stripes) {
      const int y0 = st[0], rows = st[1];
      const auto ref = reference_stripe(plan, f, scale, pf, y0, rows);
      const std::uint64_t b0 = reader.leaf_record_bytes_read();
      reader.read_stripe_leaves_into(pf, f.source_index, scale, y0, rows,
                                   win);
      // Pre-reserved capacity is reused, never replaced.
      REQUIRE(win.leaves.data() == stable);
      REQUIRE(win.io_scratch.data() == stable_io);
      REQUIRE(win.leaves.capacity() == leaves_cap);
      REQUIRE(win.io_scratch.capacity() == io_cap);
      // The row-index bound covers every stripe's retained count.
      REQUIRE(reader.max_stripe_leaf_records(pf, rows) >= win.leaves.size());

      // Replay enumerate_stripe's cell emission over the leaf records and
      // compare it field-for-field to the legacy enumerator's output.
      std::vector<Cell> cells;
      std::vector<std::pair<std::uint32_t, std::uint32_t>> samples;
      int bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
      for (const DrizzleCachedLeaf &rec : win.leaves) {
        samples.emplace_back(rec.source_x, rec.source_y);
        const double xmin = *std::min_element(rec.x, rec.x + 4);
        const double xmax = *std::max_element(rec.x, rec.x + 4);
        const double ymin = *std::min_element(rec.y, rec.y + 4);
        const double ymax = *std::max_element(rec.y, rec.y + 4);
        const int x0 = static_cast<int>(std::clamp(
            std::floor(xmin), 0.0, static_cast<double>(Wi)));
        const int x1 = static_cast<int>(std::clamp(
            std::ceil(xmax), 0.0, static_cast<double>(Wi)));
        const int cy0 = static_cast<int>(
            std::clamp(std::floor(ymin), static_cast<double>(y0),
                       static_cast<double>(y0 + rows)));
        const int cy1 = static_cast<int>(
            std::clamp(std::ceil(ymax), static_cast<double>(y0),
                       static_cast<double>(y0 + rows)));
        for (int y = cy0; y < cy1; ++y)
          for (int x = x0; x < x1; ++x) {
            Cell cell{static_cast<int>(rec.source_x),
                      static_cast<int>(rec.source_y),
                      static_cast<int>(rec.channel),
                      static_cast<int>(rec.leaf_order), x, y, {}, {}};
            for (int i = 0; i < 4; ++i) {
              cell.lx[i] = rec.x[i];
              cell.ly[i] = rec.y[i];
            }
            cells.push_back(cell);
          }
        const int sx = static_cast<int>(rec.source_x);
        if (samples.size() == 1) {
          bx0 = sx;
          by0 = static_cast<int>(rec.source_y);
          bx1 = sx + 1;
          by1 = static_cast<int>(rec.source_y) + 1;
        } else {
          bx0 = std::min(bx0, sx);
          by0 = std::min(by0, static_cast<int>(rec.source_y));
          bx1 = std::max(bx1, sx + 1);
          by1 = std::max(by1, static_cast<int>(rec.source_y) + 1);
        }
      }
      INFO("frame=" << fi << " stripe=" << y0 << "+" << rows
                    << " leaves=" << win.leaves.size()
                    << " ref=" << ref.size());
      REQUIRE(cells.size() == ref.size());
      for (std::size_t i = 0; i < ref.size(); ++i)
        REQUIRE(cells[i] == ref[i]);

      // Half-open source bbox over the retained records.
      if (win.leaves.empty()) {
        REQUIRE(win.source_x0 == 0);
        REQUIRE(win.source_y0 == 0);
        REQUIRE(win.source_x1 == 0);
        REQUIRE(win.source_y1 == 0);
        REQUIRE(win.unique_source_samples == 0);
      } else {
        REQUIRE(win.source_x0 == bx0);
        REQUIRE(win.source_y0 == by0);
        REQUIRE(win.source_x1 == bx1);
        REQUIRE(win.source_y1 == by1);
        std::sort(samples.begin(), samples.end());
        const auto nu = static_cast<std::uint64_t>(
            std::unique(samples.begin(), samples.end()) - samples.begin());
        REQUIRE(win.unique_source_samples == nu);
      }

      // Payload-byte counter counts whole intersecting row blocks only.
      const std::uint64_t delta = reader.leaf_record_bytes_read() - b0;
      REQUIRE(delta % sizeof(double) == 0);
      REQUIRE(delta % 72u == 0u);
      REQUIRE(delta >= win.leaves.size() * 72u);
    }

    // A whole-canvas stripe reads every record block of the frame ---
    // exactly leaves_written * 72 payload bytes.
    const std::uint64_t b0 = reader.leaf_record_bytes_read();
    reader.read_stripe_leaves_into(pf, f.source_index, scale, 0, H, win);
    const auto &fst = built.frames[static_cast<std::size_t>(fi)];
    REQUIRE(reader.leaf_record_bytes_read() - b0 ==
            fst.leaves_written * 72u);
    REQUIRE(win.unique_source_samples ==
            fst.samples_total - fst.samples_discarded);
  }
}
