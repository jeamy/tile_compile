// Plan section 11.14 P0 --- forward-drizzle geometry scaling characterisation.
//
// Drives the CPU reference (`compute_forward_drizzle_uniform` ->
// `stream_forward_drizzle_uniform`) on a small synthetic LOCAL-WARP scene at a
// range of chunk heights {1, 16, 64, 256, full} and reads the P0 geometry
// counters (tile_compile/reconstruction/drizzle_geometry_stats.hpp) to measure
// the K-factor (stripe count) that plan 11.14.1 names as the O(frames *
// source_pixels * stripes * consumers) blow-up.
//
// It asserts two properties P1/P2 must preserve and P0 must be able to prove:
//   1. COMPUTE INVARIANCE: the profile digest is bit-identical across every
//      admitted chunk height (plan: "Chunkvariation aendert keine Kandidaten-/
//      Masken-/Auswahlsemantik").
//   2. K-SCALING (today's defect): for a local-warp frame the per-stripe
//      enumeration rescans the FULL source, so top_level_sample_leaves_calls
//      grows linearly with the stripe count K, plus one extra full sweep from
//      prepare_drizzle_frames. After P2 this must collapse to
//      eligible_local_frames * source_pixels, invariant in K --- that future
//      target is checked as a soft expectation and logged, not asserted.
//
// Budget-incompatible chunk heights (plan 11.14.2: "nur budgetkompatible
// Kombinationen ausfuehren; Budgetablehnungen ausdruecklich protokollieren")
// are caught and recorded, not treated as failures.

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include "tile_compile/registration/sampling_geometry.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {

WarpMatrix source_to_canvas(double a, double b, double tx, double c, double d,
                            double ty) {
  WarpMatrix m;
  m(0, 0) = static_cast<float>(a);
  m(0, 1) = static_cast<float>(b);
  m(0, 2) = static_cast<float>(tx);
  m(1, 0) = static_cast<float>(c);
  m(1, 1) = static_cast<float>(d);
  m(1, 2) = static_cast<float>(ty);
  return m;
}

// One local-warp frame with a small, smoothly varying displacement so the
// fixed-point inversion actually iterates (and the RBF basis is evaluated) at
// every source sample, without ever exceeding the subdivision error budget.
RegistrationSamplingPlan make_local_warp_plan(int src_w, int src_h,
                                              int canvas_w, int canvas_h) {
  RegistrationSamplingPlan plan;
  plan.source_width = src_w;
  plan.source_height = src_h;
  plan.canvas_width_native = canvas_w;
  plan.canvas_height_native = canvas_h;
  plan.color_mode = ColorMode::MONO;

  FrameSamplingTransform f;
  f.frame_id = "local0";
  f.source_index = 0;
  f.valid = true;
  f.source_to_canvas =
      source_to_canvas(1.0, 0.0, (canvas_w - src_w) / 2.0, 0.0, 1.0,
                       (canvas_h - src_h) / 2.0);
  f.source_to_canvas_affine_valid = true;
  f.has_smooth_local_model = true;
  f.smooth_local_model.valid = true;
  f.smooth_local_model.image_rows = canvas_h;
  f.smooth_local_model.image_cols = canvas_w;
  f.smooth_local_model.coeff_x.setZero();
  f.smooth_local_model.coeff_y.setZero();
  // Sub-pixel displacement: small enough to stay inside the position/area
  // epsilons at depth 0, large enough that the Newton loop runs > 1 step.
  f.smooth_local_model.coeff_x[0] = 0.15f;
  f.smooth_local_model.coeff_y[0] = -0.10f;
  f.smooth_local_model.coeff_x[5] = 0.05f;
  f.model_coordinate_scale = 1.0f;
  plan.frames.push_back(f);
  return plan;
}

std::uint64_t digest_plane(const std::vector<float> &v) {
  // FNV-1a over the raw float bytes.
  std::uint64_t h = 1469598103934665603ull;
  const auto *p = reinterpret_cast<const unsigned char *>(v.data());
  const std::size_t n = v.size() * sizeof(float);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

struct Run {
  int requested_chunk_rows = 0;
  bool admitted = false;
  std::string rejection;
  int stripe_count = 0;
  std::uint64_t digest = 0;
  geomstats::VariantCounters uniform;
  geomstats::VariantCounters prepare;
};

} // namespace

TEST_CASE("forward drizzle P0: local-warp geometry cost scales with the stripe "
          "count K and is compute-invariant across chunk heights (plan 11.14)",
          "[forward-drizzle-p0][geometry-scaling]") {
  const int src_w = 24, src_h = 24;
  const int canvas_w = 48, canvas_h = 48; // internal_scale=1 -> 48 internal rows
  const RegistrationSamplingPlan plan =
      make_local_warp_plan(src_w, src_h, canvas_w, canvas_h);
  const std::uint64_t source_pixels =
      static_cast<std::uint64_t>(src_w) * src_h;

  Matrix2Df img(src_h, src_w);
  for (int y = 0; y < src_h; ++y)
    for (int x = 0; x < src_w; ++x)
      img(y, x) = static_cast<float>(1.0 + 0.01 * (x + y));
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  const std::vector<int> chunk_heights = {1, 16, 64, 256, 0}; // 0 == full height
  std::vector<Run> runs;

  for (int ch : chunk_heights) {
    Run r;
    r.requested_chunk_rows = ch;
    config::ReconstructionDrizzleConfig cfg;
    cfg.internal_scale = 1;
    cfg.pixfrac = 0.8f;
    cfg.chunk_rows = ch;
    cfg.memory_budget_mb = 256; // generous; this scene is tiny

    ForwardDrizzleUniformResult result;
    try {
      geomstats::ScopedEnable _e(true);
      result = compute_forward_drizzle_uniform(plan, source_of, cfg);
      const auto &reg = geomstats::registry();
      r.admitted = true;
      r.stripe_count = reg.stripe_count;
      r.uniform =
          reg.v[static_cast<std::size_t>(geomstats::Variant::kUniformDiagnostic)];
      r.prepare = reg.v[static_cast<std::size_t>(
          geomstats::Variant::kPrepareExclusionScan)];
    } catch (const std::exception &e) {
      r.rejection = e.what();
      runs.push_back(r);
      std::printf("[P0] chunk_rows=%d REJECTED: %s\n", ch, e.what());
      continue;
    }

    r.digest = digest_plane(result.L.value) ^ digest_plane(result.L.weight_sum);
    runs.push_back(r);

    std::printf(
        "[P0] chunk_rows=%-3d K=%-2d | uniform: enum=%llu rows_scanned=%llu "
        "top_sample_leaves=%llu invert_iters=%llu leaves=%llu cells=%llu "
        "wall=%.4fs | prepare: top_sample_leaves=%llu invert_iters=%llu\n",
        ch, r.stripe_count,
        (unsigned long long)r.uniform.enumerate_calls,
        (unsigned long long)r.uniform.source_rows_scanned,
        (unsigned long long)r.uniform.top_level_sample_leaves_calls,
        (unsigned long long)r.uniform.invert_iterations,
        (unsigned long long)r.uniform.leaves_generated,
        (unsigned long long)r.uniform.leaf_cells_emitted,
        r.uniform.geometry_wall_s,
        (unsigned long long)r.prepare.top_level_sample_leaves_calls,
        (unsigned long long)r.prepare.invert_iterations);
  }

  std::vector<const Run *> ok;
  for (const auto &r : runs)
    if (r.admitted)
      ok.push_back(&r);
  REQUIRE(ok.size() >= 2); // need at least two admitted heights to compare

  // --- Property 1: compute invariance across every admitted chunk height. ---
  for (const auto *r : ok)
    REQUIRE(r->digest == ok.front()->digest);

  // --- The local-warp frame is on the subdivision path at every height. ---
  for (const auto *r : ok) {
    REQUIRE(r->uniform.top_level_sample_leaves_calls > 0);
    REQUIRE(r->uniform.invert_iterations > 0);
    REQUIRE(r->uniform.subdivide_local_calls > 0);
  }

  // --- prepare_drizzle_frames does exactly ONE full-source sweep, K-independent.
  for (const auto *r : ok) {
    REQUIRE(r->prepare.enumerate_calls == 1);
    REQUIRE(r->prepare.top_level_sample_leaves_calls == source_pixels);
  }

  // --- Property 2: today's defect --- per-stripe enumeration rescans the FULL
  // source, so the uniform variant's work is K * source_pixels, i.e. it grows
  // strictly with the stripe count. ---
  const Run *k1 = nullptr;   // smallest K (== full height, 1 stripe)
  const Run *kmax = nullptr; // largest K
  for (const auto *r : ok) {
    if (!k1 || r->stripe_count < k1->stripe_count)
      k1 = r;
    if (!kmax || r->stripe_count > kmax->stripe_count)
      kmax = r;
  }
  REQUIRE(k1->stripe_count == 1);
  REQUIRE(kmax->stripe_count > k1->stripe_count);

  for (const auto *r : ok) {
    // one enumerate per stripe (single frame), full-source rescan each time
    REQUIRE(r->uniform.enumerate_calls ==
            static_cast<std::uint64_t>(r->stripe_count));
    REQUIRE(r->uniform.source_rows_scanned ==
            static_cast<std::uint64_t>(r->stripe_count) * src_h);
    REQUIRE(r->uniform.top_level_sample_leaves_calls ==
            static_cast<std::uint64_t>(r->stripe_count) * source_pixels);
  }

  // The blow-up factor is exactly K: chunk_rows=1 does `full_height` times the
  // per-sample geometry work of the single-stripe run.
  REQUIRE(kmax->uniform.top_level_sample_leaves_calls ==
          k1->uniform.top_level_sample_leaves_calls * kmax->stripe_count);
  REQUIRE(kmax->uniform.invert_iterations >
          k1->uniform.invert_iterations * 4); // grossly super-linear vs K=1

  // --- Soft expectation (P2 acceptance, plan 11.14.4): after P1/P2 the uniform
  // variant must reach eligible_local_frames * source_pixels top-level
  // sample_leaves calls, invariant in K. Today it does not for K>1 --- record
  // the gap so the P2 change has a concrete before/after number. ---
  const std::uint64_t p2_target = source_pixels; // 1 eligible local frame
  for (const auto *r : ok) {
    const double ratio =
        static_cast<double>(r->uniform.top_level_sample_leaves_calls) /
        static_cast<double>(p2_target);
    std::printf("[P0] chunk_rows=%-3d uniform top_level_sample_leaves / "
                "P2_target = %.1fx (target 1.0x, invariant in K)\n",
                r->requested_chunk_rows, ratio);
  }
  // Guard the direction of the future fix without over-constraining it.
  REQUIRE(k1->uniform.top_level_sample_leaves_calls == p2_target);
  REQUIRE(kmax->uniform.top_level_sample_leaves_calls > p2_target);
}

// The SAMPLING_GEOMETRY phase (compute_geometric_coverage) is the phase that
// cost ~36 min single-threaded on the real 100-frame M66 run (§30.61). It runs
// TWO enumerations per frame per stripe: the CFA droplet at cfg.pixfrac and the
// dense footprint at pixfrac=1.0. This case proves (a) both are separate
// geometry variants with distinct pixfrac, and (b) each carries the same K
// blow-up as the uniform path.
TEST_CASE("forward drizzle P0: coverage geometry runs two pixfrac-distinct "
          "variants, both scaling with K (plan 11.14.1 / 11.14.3)",
          "[forward-drizzle-p0][geometry-scaling]") {
  namespace gs = geomstats;
  const int src_w = 24, src_h = 24;
  const int canvas_w = 48, canvas_h = 48;
  const RegistrationSamplingPlan plan =
      make_local_warp_plan(src_w, src_h, canvas_w, canvas_h);
  const std::uint64_t source_pixels =
      static_cast<std::uint64_t>(src_w) * src_h;

  const config::ReconstructionCoverageGateConfig gate_cfg{};
  const std::vector<int> chunk_heights = {1, 16, 64, 0};

  struct CovRun {
    int chunk_rows = 0;
    bool admitted = false;
    int stripe_count = 0;
    gs::VariantCounters cfa;
    gs::VariantCounters foot;
  };
  std::vector<CovRun> runs;

  for (int ch : chunk_heights) {
    CovRun r;
    r.chunk_rows = ch;
    config::ReconstructionDrizzleConfig res;
    res.internal_scale = 1;
    res.pixfrac = 0.8f;
    res.chunk_rows = ch;
    res.memory_budget_mb = 256;
    try {
      gs::ScopedEnable _e(true);
      auto coverage = registration::compute_geometric_coverage(
          plan, /*internal_scale=*/1, /*pixfrac=*/0.8f, gate_cfg,
          /*common_overlap_required_fraction=*/0.5f, /*num_workers=*/1, res,
          /*retain_channel_counts=*/false);
      (void)coverage;
      const auto &reg = gs::registry();
      r.admitted = true;
      r.stripe_count = reg.stripe_count;
      r.cfa = reg.v[static_cast<std::size_t>(gs::Variant::kCoverageCfa)];
      r.foot = reg.v[static_cast<std::size_t>(gs::Variant::kCoverageFootprint)];
    } catch (const std::exception &e) {
      std::printf("[P0] coverage chunk_rows=%d REJECTED: %s\n", ch, e.what());
      runs.push_back(r);
      continue;
    }
    runs.push_back(r);
    std::printf(
        "[P0] coverage chunk_rows=%-3d K=%-2d | cfa(pf=%.2f): enum=%llu "
        "top_sample_leaves=%llu invert_iters=%llu | footprint(pf=%.2f): "
        "enum=%llu top_sample_leaves=%llu invert_iters=%llu\n",
        ch, r.stripe_count, r.cfa.pixfrac,
        (unsigned long long)r.cfa.enumerate_calls,
        (unsigned long long)r.cfa.top_level_sample_leaves_calls,
        (unsigned long long)r.cfa.invert_iterations, r.foot.pixfrac,
        (unsigned long long)r.foot.enumerate_calls,
        (unsigned long long)r.foot.top_level_sample_leaves_calls,
        (unsigned long long)r.foot.invert_iterations);
  }

  std::vector<const CovRun *> ok;
  for (const auto &r : runs)
    if (r.admitted)
      ok.push_back(&r);
  REQUIRE(ok.size() >= 2);

  for (const auto *r : ok) {
    // Two DISTINCT variants with distinct pixfrac (plan 11.14.3): the CFA
    // droplet at cfg.pixfrac (0.8f), the dense footprint at exactly 1.0.
    REQUIRE(r->cfa.pixfrac == static_cast<double>(0.8f));
    REQUIRE(r->foot.pixfrac == 1.0);
    REQUIRE(r->cfa.pixfrac != r->foot.pixfrac);
    // One enumeration per stripe, per variant, per (single) frame.
    REQUIRE(r->cfa.enumerate_calls ==
            static_cast<std::uint64_t>(r->stripe_count));
    REQUIRE(r->foot.enumerate_calls ==
            static_cast<std::uint64_t>(r->stripe_count));
    // Full-source rescan every stripe on the local-warp path -> K blow-up on
    // BOTH coverage variants, independently.
    REQUIRE(r->cfa.top_level_sample_leaves_calls ==
            static_cast<std::uint64_t>(r->stripe_count) * source_pixels);
    REQUIRE(r->foot.top_level_sample_leaves_calls ==
            static_cast<std::uint64_t>(r->stripe_count) * source_pixels);
    REQUIRE(r->cfa.invert_iterations > 0);
    REQUIRE(r->foot.invert_iterations > 0);
  }

  const CovRun *k1 = nullptr;
  const CovRun *kmax = nullptr;
  for (const auto *r : ok) {
    if (!k1 || r->stripe_count < k1->stripe_count)
      k1 = r;
    if (!kmax || r->stripe_count > kmax->stripe_count)
      kmax = r;
  }
  REQUIRE(k1->stripe_count == 1);
  REQUIRE(kmax->stripe_count > 1);
  // Combined SAMPLING_GEOMETRY per-frame geometry cost is (cfa + footprint),
  // i.e. 2 * K * source_pixels top-level sample_leaves for a local-warp frame.
  const std::uint64_t k1_total =
      k1->cfa.top_level_sample_leaves_calls +
      k1->foot.top_level_sample_leaves_calls;
  const std::uint64_t kmax_total =
      kmax->cfa.top_level_sample_leaves_calls +
      kmax->foot.top_level_sample_leaves_calls;
  REQUIRE(k1_total == 2 * source_pixels);
  REQUIRE(kmax_total == k1_total * kmax->stripe_count);
}

namespace {
std::filesystem::path p2_scratch() {
  static std::mt19937_64 rng{0xC0DE1234ull};
  const auto p = std::filesystem::temp_directory_path() /
                 ("tc_geomcache_scaling_" + std::to_string(rng()));
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  return p;
}
} // namespace

// P2 acceptance (plan 11.14.4): with a published geometry cache, EVERY geometry
// consumer's top-level sample_leaves count collapses to
// eligible_local_frames * source_pixels, invariant across chunk heights; other
// consumers add ZERO local basis evaluations; the prepare_drizzle_frames sweep
// disappears; and the computed profile is bit-identical to the no-cache run.
TEST_CASE("forward drizzle P2: a published geometry cache removes the K factor "
          "from every consumer and stays bit-exact (plan 11.14.4)",
          "[forward-drizzle-p0][geometry-scaling][geometry-cache]") {
  namespace gs = geomstats;
  const int src_w = 24, src_h = 24, canvas_w = 48, canvas_h = 48;
  const RegistrationSamplingPlan plan =
      make_local_warp_plan(src_w, src_h, canvas_w, canvas_h);
  const std::uint64_t source_pixels =
      static_cast<std::uint64_t>(src_w) * src_h;
  const std::uint64_t eligible_local_frames = 1; // make_local_warp_plan: 1 frame

  Matrix2Df img(src_h, src_w);
  for (int y = 0; y < src_h; ++y)
    for (int x = 0; x < src_w; ++x)
      img(y, x) = static_cast<float>(1.0 + 0.01 * (x + y));
  SourceImageProvider source_of = [&](std::size_t) -> const Matrix2Df & {
    return img;
  };

  config::ReconstructionDrizzleConfig base_cfg;
  base_cfg.internal_scale = 1;
  base_cfg.pixfrac = 0.8f;
  base_cfg.memory_budget_mb = 256;

  auto digest_uniform = [&](int chunk_rows, bool with_cache,
                            const DrizzleGeometryCacheReader *reader) {
    config::ReconstructionDrizzleConfig cfg = base_cfg;
    cfg.chunk_rows = chunk_rows;
    std::optional<ScopedActiveGeometryCache> guard;
    if (with_cache) guard.emplace(reader);
    const auto r = compute_forward_drizzle_uniform(plan, source_of, cfg);
    return digest_plane(r.L.value) ^ digest_plane(r.L.weight_sum);
  };

  // --- no-cache baseline digest ---
  const std::uint64_t baseline = digest_uniform(0, false, nullptr);
  REQUIRE(digest_uniform(7, false, nullptr) == baseline);

  // --- build the cache (both variants) ---
  const auto root = p2_scratch();
  const std::vector<GeometryVariant> variants = {{0.8f}, {1.0f}};
  const auto built = build_drizzle_geometry_cache(root, plan, base_cfg, variants,
                                                  {}, 64ull * 1024 * 1024);
  std::vector<GeometryCacheIdentity> ids = built.identities;
  DrizzleGeometryCacheReader reader(root, ids, std::vector<std::size_t>{0});

  for (int chunk : {1, 16, 64, 0}) {
    // uniform path under the cache
    {
      gs::ScopedEnable en(true);
      const std::uint64_t d = digest_uniform(chunk, true, &reader);
      REQUIRE(d == baseline); // bit-exact vs no-cache
      const auto &u = gs::registry().v[static_cast<std::size_t>(
          gs::Variant::kUniformDiagnostic)];
      const auto &prep = gs::registry().v[static_cast<std::size_t>(
          gs::Variant::kPrepareExclusionScan)];
      // K factor gone: the reader replays pre-built cells and never enters
      // sample_leaves at all, at any chunk height.
      REQUIRE(u.top_level_sample_leaves_calls == 0);
      REQUIRE(u.invert_iterations == 0);
      REQUIRE(u.subdivide_local_calls == 0);
      // prepare's separate full sweep is folded into the build:
      REQUIRE(prep.enumerate_calls == 0);
      REQUIRE(prep.source_samples_visited == 0);
      REQUIRE(prep.top_level_sample_leaves_calls == 0);
    }
    // coverage path under the cache
    {
      gs::ScopedEnable en(true);
      config::ReconstructionDrizzleConfig res = base_cfg;
      res.chunk_rows = chunk;
      const config::ReconstructionCoverageGateConfig gate_cfg{};
      {
        ScopedActiveGeometryCache guard(&reader);
        auto cov = registration::compute_geometric_coverage(
            plan, 1, 0.8f, gate_cfg, 0.5f, 1, res, false);
        (void)cov;
      }
      const auto &cfa = gs::registry().v[static_cast<std::size_t>(
          gs::Variant::kCoverageCfa)];
      const auto &foot = gs::registry().v[static_cast<std::size_t>(
          gs::Variant::kCoverageFootprint)];
      REQUIRE(cfa.invert_iterations == 0);
      REQUIRE(foot.invert_iterations == 0);
      REQUIRE(cfa.subdivide_local_calls == 0);
      REQUIRE(foot.subdivide_local_calls == 0);
    }
  }

  // The one-time build did exactly V * eligible_local_frames * source_pixels
  // top-level sample_leaves calls total (V = 2 variants), and nothing else does.
  std::uint64_t build_calls = 0;
  for (const auto &fr : built.frames)
    build_calls += fr.top_level_sample_leaves_calls;
  REQUIRE(build_calls == 2 * eligible_local_frames * source_pixels);
}
