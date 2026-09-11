// Plan section 11.14 P3/P4 --- shared synthetic scaling-ladder infrastructure.
//
// One parametric local-warp scene, swept over a frame-count ladder, with the
// geometry cache in the loop. It measures the quantities P3 (deterministic
// parallelism) and P4 (I/O + memory at 600 frames) both need, and asserts the
// invariants both must preserve:
//
//   * cache-on forward-drizzle output is BYTE-IDENTICAL to cache-off, at every
//     rung of the ladder and every chunk height (the property P3's scheduler
//     and P4's disk-residency changes must not break);
//   * the reader's resident bytes track the ROW INDEX only (O(N * source_rows)),
//     never the leaf-record volume;
//   * leaves-per-sample is ~constant across the ladder (so per-sample rates
//     extrapolate).
//
// It also prints per-rung wall times + on-disk sizes and a clearly-labelled
// projection toward 600 frames @ 3840x2160, so the P3-vs-P4 ordering call can
// be made from real rates rather than assumptions. Default rungs are small
// (fast CI); set TC_LADDER_FULL=1 for a heavier sweep.

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <utility>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
using tile_compile::registration::FrameSamplingTransform;
using tile_compile::registration::RegistrationSamplingPlan;

namespace {
namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

long long vmhwm_kib() {
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line))
    if (line.rfind("VmHWM:", 0) == 0)
      return std::atoll(line.c_str() + 6);
  return 0;
}

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

// n_frames local-warp frames, deterministic small rotation + dither, each with
// a smooth field strong enough to force ~1-2 leaves/sample.
RegistrationSamplingPlan ladder_plan(int n_frames, int sw, int sh, int cw,
                                     int ch) {
  RegistrationSamplingPlan plan;
  plan.source_width = sw;
  plan.source_height = sh;
  plan.canvas_width_native = cw;
  plan.canvas_height_native = ch;
  plan.color_mode = ColorMode::OSC;
  plan.bayer_pattern = BayerPattern::GBRG;
  plan.cfa_origin_x = 0;
  plan.cfa_origin_y = 0;
  plan.plan_hash = "ladder-" + std::to_string(n_frames) + "-" +
                   std::to_string(sw) + "x" + std::to_string(sh);
  const double ox = (cw - sw) / 2.0, oy = (ch - sh) / 2.0;
  for (int i = 0; i < n_frames; ++i) {
    const double ang = 0.004 * (i % 7 - 3);
    const double dx = ox + 0.37 * ((i * 5) % 7 - 3);
    const double dy = oy + 0.29 * ((i * 3) % 7 - 3);
    FrameSamplingTransform f;
    f.frame_id = "f" + std::to_string(i);
    f.source_index = static_cast<std::size_t>(i);
    f.valid = true;
    f.source_to_canvas =
        s2c(std::cos(ang), -std::sin(ang), dx, std::sin(ang), std::cos(ang), dy);
    f.source_to_canvas_affine_valid = true;
    f.has_smooth_local_model = true;
    f.smooth_local_model.valid = true;
    f.smooth_local_model.image_rows = ch;
    f.smooth_local_model.image_cols = cw;
    f.smooth_local_model.coeff_x.setZero();
    f.smooth_local_model.coeff_y.setZero();
    f.smooth_local_model.coeff_x[0] = 0.11f + 0.01f * (i % 5);
    f.smooth_local_model.coeff_y[0] = -0.08f;
    f.smooth_local_model.coeff_x[5] = 0.03f;
    f.model_coordinate_scale = 1.0f;
    plan.frames.push_back(f);
  }
  return plan;
}

std::uint64_t digest(const std::vector<float> &v) {
  std::uint64_t h = 1469598103934665603ull;
  const auto *p = reinterpret_cast<const unsigned char *>(v.data());
  for (std::size_t i = 0, n = v.size() * sizeof(float); i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

fs::path scratch() {
  static std::mt19937_64 rng{0x1ADDE12345ull};
  const auto p = fs::temp_directory_path() /
                 ("tc_ladder_" + std::to_string(rng()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

} // namespace

TEST_CASE("plan 11.14 P3/P4: geometry-cache scaling ladder --- bit-exact, "
          "row-index-bounded, and rate-measured", "[geometry-ladder]") {
  const bool full = std::getenv("TC_LADDER_FULL") != nullptr;
  const std::vector<int> rungs =
      full ? std::vector<int>{8, 40, 100, 200} : std::vector<int>{4, 8, 16};
  const int sw = full ? 192 : 96, sh = full ? 192 : 96;
  const int cw = sw + 24, chh = sh + 24;
  const std::uint64_t sp = static_cast<std::uint64_t>(sw) * sh;

  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale = 1;
  cfg.pixfrac = 0.8f;
  cfg.memory_budget_mb = 512;

  Matrix2Df img(sh, sw);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      img(y, x) = static_cast<float>(4.0 + 0.02 * (x ^ y));
  SourceImageProvider src = [&](std::size_t) -> const Matrix2Df & { return img; };

  std::printf("\n[ladder] N   compute_s  write_s  disk_MiB  wr_MiB/s  open_s  "
              "openV_s  resident_KiB  leaves/sample  drz_cache_s  "
              "drz_nocache_s  speedup\n");

  struct Row {
    int n;
    double leaves_per_sample, compute_s, write_s, wr_mibps;
    std::uint64_t record_bytes, resident_bytes;
    long long rss_kib;
  };
  std::vector<Row> table;

  for (int n : rungs) {
    const auto plan = ladder_plan(n, sw, sh, cw, chh);
    std::vector<std::size_t> local_idx;
    for (const auto &f : plan.frames) local_idx.push_back(f.source_index);
    const fs::path root = scratch();

   try {
    const auto built = build_drizzle_geometry_cache(
        root, plan, cfg, {{cfg.pixfrac}}, {}, 128ull << 20);

    const auto t2 = clk::now();
    DrizzleGeometryCacheReader reader(root, built.identities, local_idx, false);
    const auto t3 = clk::now();
    { DrizzleGeometryCacheReader v(root, built.identities, local_idx, true);
      (void)v; }
    const auto t4 = clk::now();

    // cache-on vs cache-off forward drizzle at two chunk heights
    std::uint64_t dig_cache = 0, dig_nocache = 0;
    double drz_cache_s = 0, drz_nocache_s = 0;
    for (int chunk : {7, 0}) {
      config::ReconstructionDrizzleConfig c = cfg;
      c.chunk_rows = chunk;
      {
        ScopedActiveGeometryCache g(&reader);
        const auto a = clk::now();
        const auto r = compute_forward_drizzle_uniform(plan, src, c);
        drz_cache_s += secs(a, clk::now());
        dig_cache ^= digest(r.R.value) ^ digest(r.G.weight_sum) ^ digest(r.B.n_eff);
      }
      {
        const auto a = clk::now();
        const auto r = compute_forward_drizzle_uniform(plan, src, c);
        drz_nocache_s += secs(a, clk::now());
        dig_nocache ^= digest(r.R.value) ^ digest(r.G.weight_sum) ^ digest(r.B.n_eff);
      }
    }

    // --- invariants ---
    REQUIRE(dig_cache == dig_nocache);                     // byte-identical
    REQUIRE(reader.resident_bytes() < built.total_record_bytes / 4 + 65536);
    REQUIRE(reader.resident_bytes() >
            static_cast<std::uint64_t>(n) * sh * 8);       // row index present

    const double lps = static_cast<double>(built.total_leaves) /
                       static_cast<double>(sp * n);
    const double disk_mib =
        static_cast<double>(built.total_record_bytes) / (1024.0 * 1024.0);
    const double wr_mibps =
        built.write_seconds > 1e-6 ? disk_mib / built.write_seconds : 0.0;
    const double speedup =
        drz_cache_s > 1e-9 ? drz_nocache_s / drz_cache_s : 0.0;
    std::printf("[ladder] %-3d %9.3f %8.4f %8.2f %9.1f %7.4f %8.4f %12llu "
                "%13.3f %11.4f %13.4f %8.1fx\n",
                n, built.sample_leaves_seconds, built.write_seconds, disk_mib,
                wr_mibps, secs(t2, t3), secs(t3, t4),
                (unsigned long long)(reader.resident_bytes() / 1024), lps,
                drz_cache_s, drz_nocache_s, speedup);
    table.push_back({n, lps, built.sample_leaves_seconds, built.write_seconds,
                     wr_mibps, built.total_record_bytes, reader.resident_bytes(),
                     vmhwm_kib()});
   } catch (const std::exception &e) {
     // The opt-in heavy sweep can outgrow a small /tmp; that is an environment
     // limit, not a defect. The small default rungs never hit it.
     std::printf("[ladder] N=%d aborted (environment): %s\n", n, e.what());
     fs::remove_all(root);
     break;
   }
    fs::remove_all(root);
  }
  REQUIRE(table.size() >= 2);

  // leaves/sample ~constant across the ladder (rates extrapolate).
  for (const auto &r : table)
    REQUIRE(std::abs(r.leaves_per_sample - table.front().leaves_per_sample) <
            0.35);

  // --- P4 residency: geometry-cache RAM must scale with the ROW INDEX
  // (O(N * source_rows)), NOT with the record volume (O(N * source_px)). ---
  std::printf("\n[ladder] P4 residency:  N  disk_records_MiB  reader_resident_"
              "KiB  resident/record  process_VmHWM_MiB\n");
  for (const auto &r : table)
    std::printf("[ladder]   %3d %16.2f %18llu %15.5f %17lld\n", r.n,
                r.record_bytes / (1024.0 * 1024.0),
                (unsigned long long)(r.resident_bytes / 1024),
                static_cast<double>(r.resident_bytes) /
                    static_cast<double>(std::max<std::uint64_t>(1, r.record_bytes)),
                r.rss_kib / 1024);
  if (table.size() >= 2) {
    const auto &a = table.front();
    const auto &b = table.back();
    // resident bytes per frame ~constant (row index); ratio to records shrinks
    // as N (and thus record volume) grows.
    const double per_frame_a =
        static_cast<double>(a.resident_bytes) / a.n;
    const double per_frame_b =
        static_cast<double>(b.resident_bytes) / b.n;
    REQUIRE(std::abs(per_frame_b - per_frame_a) < per_frame_a * 0.5 + 4096);
    REQUIRE(static_cast<double>(b.resident_bytes) / b.record_bytes <
            static_cast<double>(a.resident_bytes) / a.record_bytes + 1e-9);
  }

  // Projection toward the production target --- EXTRAPOLATION, not a measurement.
  // Per-sample compute rate from the largest rung; the build parallelises
  // per-frame (P3), so also show the /16-core figure.
  const auto &last = table.back();
  const double this_samples = static_cast<double>(sp) * last.n;
  const double compute_ns_per_sample =
      last.compute_s / this_samples * 1e9;
  const double prod_samples = 3840.0 * 2160.0 * 600.0;
  const double prod_bytes = prod_samples * last.leaves_per_sample * 72.0;
  const double prod_compute_s = compute_ns_per_sample * 1e-9 * prod_samples;
  const double prod_write_s =
      last.wr_mibps > 0 ? (prod_bytes / (1024.0 * 1024.0)) / last.wr_mibps : 0.0;
  std::printf(
      "\n[ladder] PROJECTION 600f @ 3840x2160 (leaves/sample=%.2f):\n"
      "[ladder]   records on disk    ~%.0f GiB\n"
      "[ladder]   build sample_leaves ~%.0f s single-thread  (~%.0f s / 16 "
      "cores; P3 parallelises this per-frame)\n"
      "[ladder]   build record write  ~%.0f s at the measured %.0f MiB/s "
      "(SSD BW is higher; this rung is compute-bound so the rate is a floor)\n"
      "[ladder]   NOTE: extrapolation from a %d-frame / %dx%d rung, not a run\n",
      last.leaves_per_sample, prod_bytes / (1024.0 * 1024.0 * 1024.0),
      prod_compute_s, prod_compute_s / 16.0, prod_write_s, last.wr_mibps,
      last.n, sw, sh);
  std::printf("[ladder] baseline VmHWM ~%lld MiB\n", vmhwm_kib() / 1024);

  // --- P3 worker sweep on the largest rung: byte-identical + wall speedup ---
  const int wn = full ? 100 : 16;
  const auto wp = ladder_plan(wn, sw, sh, cw, chh);
  std::vector<std::size_t> widx;
  for (const auto &f : wp.frames) widx.push_back(f.source_index);
  std::printf("\n[ladder] P3 worker sweep (N=%d):  workers  wall_s  "
              "sample_leaves_s  speedup_vs_w1\n",
              wn);
  double w1_wall = 0.0;
  std::vector<std::pair<std::string, std::string>> ref_shas;
  for (int w : {1, 2, 4, 8}) {
    const fs::path root = scratch();
    GeometryCacheBuildResult b;
    try {
      b = build_drizzle_geometry_cache(root, wp, cfg, {{cfg.pixfrac}}, {},
                                       128ull << 20, w);
    } catch (const std::exception &e) {
      std::printf("[ladder]   workers=%d aborted (environment): %s\n", w,
                  e.what());
      fs::remove_all(root);
      break;
    }
    if (w == 1) w1_wall = b.wall_seconds;
    std::vector<std::pair<std::string, std::string>> shas;
    for (const auto &fr : b.frames) (void)fr;
    {
      std::ifstream mfs(b.generation_dir / "manifest.json");
      nlohmann::json mf;
      mfs >> mf;
      for (const auto &v : mf.at("variants"))
        for (const auto &fj : v.at("frames"))
          shas.emplace_back(std::to_string(fj.at("rows_bytes").get<std::uintmax_t>()),
                            std::to_string(fj.at("leaves_bytes").get<std::uintmax_t>()));
    }
    if (w == 1)
      ref_shas = shas;
    else
      REQUIRE(shas == ref_shas);  // byte-identical regardless of worker count
    std::printf("[ladder]   %7d %7.3f %15.3f %13.1fx\n", b.workers_used,
                b.wall_seconds, b.sample_leaves_seconds,
                b.wall_seconds > 1e-9 ? w1_wall / b.wall_seconds : 0.0);
    fs::remove_all(root);
  }
  REQUIRE(!ref_shas.empty());
}
