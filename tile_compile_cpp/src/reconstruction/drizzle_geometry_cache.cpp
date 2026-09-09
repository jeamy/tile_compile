// Plan section 11.14 P1/P2 --- see the header.
//
// On-disk layout (committed generations are immutable and never deleted; the
// live one is named by <root>/current.json):
//   <root>/current.json                       {schema, generation, variants[]}
//   <root>/generation-<hash>-<uid>/manifest.json   identities + per-frame stats
//                                                  + per-file sha256 (written LAST)
//   <root>/generation-.../v<vi>_f<idx>.rows    source_height x RowEntry (32 B)
//   <root>/generation-.../v<vi>_f<idx>.leaves  leaves x LeafRecord (72 B)
//
// Build streams: each completed SOURCE ROW's leaf records are appended to the
// .leaves file and freed before the next row starts (bounded staging buffer,
// capped by memory_budget_bytes). Discards are counted directly from
// sample_leaves' return value --- never from the optional geomstats registry.
//
// Reader loads only the .rows indices; enumerate_stripe() seek-reads the row
// blocks intersecting the requested stripe. Leaf records are never fully
// resident. Every read replays the SAME (source_y, source_x, leaf_order, y, x)
// canonical order and the SAME bbox clamp the stripe enumerator uses.
//
// Commit is crash-safe: unique generation + staging directory names, the old
// generation is left untouched, all payload is fsync'd before the manifest,
// the manifest is fsync'd before the directory rename, and current.json is
// swapped last via a temp-file rename. An abort at any point leaves the
// previous committed generation fully usable.

#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"

#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/registration/sampling_geometry.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <fcntl.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <exception>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace tile_compile::reconstruction {
namespace {
using json = nlohmann::json;
using registration::RegistrationSamplingPlan;

constexpr int kSchemaVersion = 1;
constexpr int kAlgorithmVersion = 1;

#pragma pack(push, 1)
struct RowEntry {
  double canvas_ymin;           // min leaf y over this source row (raw)
  double canvas_ymax;           // max leaf y over this source row
  std::uint64_t record_offset;  // byte offset into the .leaves file
  std::uint64_t record_count;   // LeafRecords in this row's contiguous block
};
struct LeafRecord {
  std::uint32_t source_x;
  std::uint16_t channel;     // 0..2 (OSC) / 0 (MONO)
  std::uint16_t leaf_order;  // li within the sample's leaves vector
  double x[4];
  double y[4];
};
#pragma pack(pop)
static_assert(sizeof(RowEntry) == 32, "RowEntry layout");
static_assert(sizeof(LeafRecord) == 72, "LeafRecord layout");

// ---- streaming SHA-256 ----------------------------------------------------
class Sha256Stream {
public:
  Sha256Stream() : ctx_(EVP_MD_CTX_new()) {
    if (!ctx_ || EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SHA_INIT");
  }
  ~Sha256Stream() {
    if (ctx_) EVP_MD_CTX_free(ctx_);
  }
  Sha256Stream(const Sha256Stream &) = delete;
  Sha256Stream &operator=(const Sha256Stream &) = delete;
  void update(const void *p, std::size_t n) {
    if (n && EVP_DigestUpdate(ctx_, p, n) != 1)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SHA_UPDATE");
  }
  std::string hex() {
    unsigned char h[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    if (EVP_DigestFinal_ex(ctx_, h, &n) != 1)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SHA_FINAL");
    static const char *k = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (unsigned int i = 0; i < n; ++i) {
      s.push_back(k[h[i] >> 4]);
      s.push_back(k[h[i] & 0xF]);
    }
    return s;
  }

private:
  EVP_MD_CTX *ctx_;
};

// ---- POSIX durable IO ---------------------------------------------------
// Append-writes a byte span to an open fd, retrying short writes.
void write_all(int fd, const void *data, std::size_t n) {
  const auto *p = static_cast<const char *>(data);
  while (n) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("DRIZZLE_GEOMETRY_CACHE_WRITE: ") +
                               std::strerror(errno));
    }
    p += w;
    n -= static_cast<std::size_t>(w);
  }
}
void fsync_path(const fs::path &p, bool dir) {
  const int flags = dir ? (O_RDONLY | O_DIRECTORY) : O_RDONLY;
  const int fd = ::open(p.c_str(), flags);
  if (fd < 0) {
    if (dir) return;  // some filesystems reject O_DIRECTORY fsync; best effort
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC_OPEN: " + p.string());
  }
  const int r = ::fsync(fd);
  ::close(fd);
  if (r != 0 && !dir)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC: " + p.string());
}

std::string variant_tag(std::size_t vi) { return "v" + std::to_string(vi); }
std::string rows_name(std::size_t vi, std::size_t idx) {
  return variant_tag(vi) + "_f" + std::to_string(idx) + ".rows";
}
std::string leaves_name(std::size_t vi, std::size_t idx) {
  return variant_tag(vi) + "_f" + std::to_string(idx) + ".leaves";
}

std::string fresh_uid() {
  std::random_device rd;
  std::uniform_int_distribution<std::uint64_t> d;
  std::uint64_t a = (static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^
                    static_cast<std::uint64_t>(::getpid());
  std::uint64_t b = (static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^ d(rd);
  char buf[33];
  std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b));
  return std::string(buf);
}

// One independent (variant, frame) build task. No shared mutable state: its own
// sample_leaves sweep, its own .rows/.leaves files, its own SHA contexts. Safe
// to run concurrently for distinct (vi, source_index). geomstats must be
// disabled by the caller across the (possibly parallel) region.
struct FrameBuildOut {
  bool present = false;  // false => (vi, fidx) was not a local frame
  json fj;
  GeometryCacheFrameStats stats;
  std::uint64_t record_bytes = 0;
  std::uint64_t index_bytes = 0;
  bool excluded = false;
  std::string excluded_frame_id;
  double excluded_rate = 0.0;
  double t_compute = 0.0, t_write = 0.0;
};

FrameBuildOut build_one_frame(
    const fs::path &staging, std::size_t vi, float pixfrac,
    const RegistrationSamplingPlan &plan,
    const registration::FrameSamplingTransform &f,
    const ForwardDrizzleSubdivisionParams &subdivision, int scale,
    int internal_w, int internal_h, std::uint64_t source_pixels) {
  using bclk = std::chrono::steady_clock;
  FrameBuildOut out;
  if (!f.valid || !f.has_smooth_local_model) return out;
  if (!f.source_to_canvas_affine_valid || !f.source_to_canvas.allFinite())
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_BAD_TRANSFORM");
  out.present = true;

  const fs::path rp = staging / rows_name(vi, f.source_index);
  const fs::path lp = staging / leaves_name(vi, f.source_index);
  const int lfd = ::open(lp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (lfd < 0)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_OPEN: " + lp.string());
  Sha256Stream leaf_sha;

  std::vector<Leaf> leaves;
  leaves.reserve(16);
  std::vector<LeafRecord> row_buf;
  std::vector<RowEntry> rows(static_cast<std::size_t>(plan.source_height));
  std::uint64_t running = 0, leaves_total = 0, discarded = 0;

  for (int sy = 0; sy < plan.source_height; ++sy) {
    row_buf.clear();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    const std::uint64_t row_start = running;
    const auto row_c0 = bclk::now();
    for (int sx = 0; sx < plan.source_width; ++sx) {
      if (!sample_leaves(plan, f, sx, sy, scale, pixfrac, subdivision, leaves)) {
        ++discarded;
        continue;
      }
      std::uint16_t channel = 0;
      if (plan.color_mode == ColorMode::OSC) {
        const auto ch = cfa_channel_for_source_pixel(
            sx, sy, plan.bayer_pattern, plan.cfa_origin_x, plan.cfa_origin_y);
        channel = ch == CfaChannel::R ? 0 : ch == CfaChannel::G ? 1 : 2;
      }
      for (std::size_t li = 0; li < leaves.size(); ++li) {
        LeafRecord rec{};
        rec.source_x = static_cast<std::uint32_t>(sx);
        rec.channel = channel;
        rec.leaf_order = static_cast<std::uint16_t>(li);
        for (int i = 0; i < 4; ++i) {
          rec.x[i] = leaves[li].x[i];
          rec.y[i] = leaves[li].y[i];
          ymin = std::min(ymin, leaves[li].y[i]);
          ymax = std::max(ymax, leaves[li].y[i]);
        }
        row_buf.push_back(rec);
      }
    }
    const auto row_w0 = bclk::now();
    out.t_compute += std::chrono::duration<double>(row_w0 - row_c0).count();
    if (!row_buf.empty()) {
      const std::size_t nb = row_buf.size() * sizeof(LeafRecord);
      write_all(lfd, row_buf.data(), nb);
      leaf_sha.update(row_buf.data(), nb);
      running += nb;
      leaves_total += row_buf.size();
    }
    out.t_write += std::chrono::duration<double>(bclk::now() - row_w0).count();
    RowEntry &re = rows[static_cast<std::size_t>(sy)];
    re.record_offset = row_start;
    re.record_count = (running - row_start) / sizeof(LeafRecord);
    re.canvas_ymin = re.record_count ? ymin : 0.0;
    re.canvas_ymax = re.record_count ? ymax : 0.0;
  }

  const auto tail_w0 = bclk::now();
  if (::fsync(lfd) != 0) {
    ::close(lfd);
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC: " + lp.string());
  }
  ::close(lfd);
  const std::string leaves_hash = leaf_sha.hex();
  {
    const int rfd = ::open(rp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (rfd < 0)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_OPEN: " + rp.string());
    write_all(rfd, rows.data(), rows.size() * sizeof(RowEntry));
    if (::fsync(rfd) != 0) {
      ::close(rfd);
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC: " + rp.string());
    }
    ::close(rfd);
  }
  const std::string rows_hash = core::sha256_file(rp);
  out.t_write += std::chrono::duration<double>(bclk::now() - tail_w0).count();

  const double rate = source_pixels
                          ? static_cast<double>(discarded) /
                                static_cast<double>(source_pixels)
                          : 0.0;
  const bool excluded = rate > subdivision.per_frame_inversion_error_rate_max;

  out.fj = json{{"source_index", f.source_index},
                {"frame_id", f.frame_id},
                {"source_width", plan.source_width},
                {"source_height", plan.source_height},
                {"internal_width", internal_w},
                {"internal_height", internal_h},
                {"samples_total", source_pixels},
                {"samples_discarded", discarded},
                {"leaves", leaves_total},
                {"leaves_bytes", running},
                {"subdivision_error_rate", rate},
                {"excluded", excluded},
                {"rows_file", rows_name(vi, f.source_index)},
                {"leaves_file", leaves_name(vi, f.source_index)},
                {"rows_sha256", rows_hash},
                {"leaves_sha256", leaves_hash}};
  out.stats.source_index = f.source_index;
  out.stats.samples_total = source_pixels;
  out.stats.top_level_sample_leaves_calls = source_pixels;
  out.stats.leaves_written = leaves_total;
  out.stats.samples_discarded = discarded;
  out.stats.subdivision_error_rate = rate;
  out.stats.excluded = excluded;
  out.record_bytes = running;
  out.index_bytes = rows.size() * sizeof(RowEntry);
  out.excluded = excluded;
  out.excluded_frame_id = f.frame_id;
  out.excluded_rate = rate;
  return out;
}

} // namespace

// --------------------------------------------------------------------------
// Identity
// --------------------------------------------------------------------------

GeometryCacheIdentity make_geometry_cache_identity(
    const RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const GeometryVariant &variant,
    const ForwardDrizzleSubdivisionParams &subdivision) {
  config::ReconstructionDrizzleConfig vcfg = cfg;
  vcfg.pixfrac = variant.pixfrac;
  const std::string base = registration::compute_coverage_geometry_hash(
      plan, vcfg, /*common_fraction=*/0.0f);
  std::string s = "drizzle-geometry-cache:schema" +
                  std::to_string(kSchemaVersion) + ":algo" +
                  std::to_string(kAlgorithmVersion) +
                  ":edge-centers:exact-polygons:" + base;
  std::vector<std::uint8_t> bytes(s.begin(), s.end());
  auto u32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
      bytes.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  };
  auto f32 = [&](float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    u32(u);
  };
  const registration::LocalInversionParams inv{};
  u32(static_cast<std::uint32_t>(inv.max_iter));
  f32(inv.tol_px);
  f32(inv.safety_margin_px);
  u32(static_cast<std::uint32_t>(subdivision.max_subdivision_depth));
  f32(subdivision.position_epsilon_internal_px);
  f32(subdivision.area_relative_epsilon);
  f32(subdivision.per_frame_inversion_error_rate_max);
  u32(static_cast<std::uint32_t>(plan.canvas_width_native));
  u32(static_cast<std::uint32_t>(plan.canvas_height_native));
  u32(static_cast<std::uint32_t>(cfg.internal_scale));
  u32(static_cast<std::uint32_t>(plan.bayer_pattern));
  u32(static_cast<std::uint32_t>(plan.cfa_origin_x));
  u32(static_cast<std::uint32_t>(plan.cfa_origin_y));
  u32(static_cast<std::uint32_t>(plan.color_mode));

  GeometryCacheIdentity id;
  id.geometry_hash = core::sha256_bytes(bytes);
  id.sampling_plan_hash = plan.plan_hash;
  id.canvas_width_native = plan.canvas_width_native;
  id.canvas_height_native = plan.canvas_height_native;
  id.internal_scale = cfg.internal_scale;
  id.pixfrac = variant.pixfrac;
  id.color_mode = plan.color_mode;
  return id;
}

// --------------------------------------------------------------------------
// Builder --- streams one completed source row at a time
// --------------------------------------------------------------------------

GeometryCacheBuildResult build_drizzle_geometry_cache(
    const fs::path &root, const RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const std::vector<GeometryVariant> &variants_in,
    const ForwardDrizzleSubdivisionParams &subdivision,
    std::uint64_t memory_budget_bytes, int max_workers) {
  if (plan.source_width <= 0 || plan.source_height <= 0)
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_BAD_PLAN");
  if (static_cast<long long>(plan.source_width) > 0xFFFFFFFF)
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_SOURCE_TOO_WIDE");
  if (cfg.internal_scale < 1 || cfg.internal_scale > 2)
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_BAD_SCALE");

  std::vector<GeometryVariant> variants;
  for (const auto &v : variants_in) {
    bool dup = false;
    for (const auto &e : variants) dup = dup || (e.pixfrac == v.pixfrac);
    if (!dup) variants.push_back(v);
  }
  if (variants.empty())
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_NO_VARIANT");

  GeometryCacheBuildResult result;
  for (const auto &v : variants)
    result.identities.push_back(
        make_geometry_cache_identity(plan, cfg, v, subdivision));

  const int scale = cfg.internal_scale;
  const int internal_w = plan.canvas_width_native * scale;
  const int internal_h = plan.canvas_height_native * scale;
  const std::uint64_t source_pixels =
      static_cast<std::uint64_t>(plan.source_width) * plan.source_height;

  // The staging buffer is ONE completed source row of leaf records --- a
  // bounded batch (<= source_width * max_leaves_per_sample * 72 B, independent
  // of frame count and canvas height, satisfying plan 11.14.3). It is written
  // and freed before the next row. memory_budget_bytes is a floor sanity check
  // only: a single row must never exceed it.
  const std::uint64_t max_leaves_per_sample = 1ull
      << (2 * static_cast<unsigned>(std::max(0, subdivision.max_subdivision_depth)));
  const std::uint64_t row_bytes_bound =
      static_cast<std::uint64_t>(plan.source_width) * max_leaves_per_sample *
      sizeof(LeafRecord);
  if (memory_budget_bytes && row_bytes_bound > memory_budget_bytes)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROW_EXCEEDS_BUDGET");

  const std::string uid = fresh_uid();
  const std::string gen_id =
      "generation-" + result.identities.front().geometry_hash.substr(0, 16) +
      "-" + uid.substr(0, 12);
  const fs::path staging = root / (".staging-" + uid);
  std::error_code ec;
  fs::create_directories(staging, ec);
  if (ec) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_STAGING: " + ec.message());

  json manifest;
  manifest["schema"] = kSchemaVersion;
  manifest["algorithm_version"] = kAlgorithmVersion;
  manifest["source_width"] = plan.source_width;
  manifest["source_height"] = plan.source_height;
  manifest["internal_width"] = internal_w;
  manifest["internal_height"] = internal_h;
  manifest["variants"] = json::array();

  // Plan 11.14.5 P3: one independent task per (variant, frame). Assemble the
  // manifest in deterministic (variant, source_index) order afterwards, so the
  // committed store is byte-identical regardless of worker count / scheduling.
  struct Task { std::size_t vi, fidx; };
  std::vector<Task> tasks;
  for (std::size_t vi = 0; vi < variants.size(); ++vi)
    for (std::size_t fidx = 0; fidx < plan.frames.size(); ++fidx)
      tasks.push_back({vi, fidx});
  std::vector<FrameBuildOut> outs(tasks.size());

  const int workers = std::max(1, max_workers);
  const auto wall0 = std::chrono::steady_clock::now();

  // The sample_leaves path touches the process-global geomstats counters when
  // enabled --- a data race under parallelism, and the build is not a consumer
  // anyway. Disable across the region, restore after (single-threaded here).
  auto &greg = geomstats::registry();
  const bool prev_geom_enabled = greg.enabled;
  greg.enabled = false;

  std::exception_ptr eptr;
#ifdef _OPENMP
  if (workers > 1) omp_set_num_threads(workers);
#endif
#pragma omp parallel for schedule(dynamic, 1) if (workers > 1)
  for (long long k = 0; k < static_cast<long long>(tasks.size()); ++k) {
    if (eptr) continue;
    try {
      outs[static_cast<std::size_t>(k)] = build_one_frame(
          staging, tasks[static_cast<std::size_t>(k)].vi,
          variants[tasks[static_cast<std::size_t>(k)].vi].pixfrac, plan,
          plan.frames[tasks[static_cast<std::size_t>(k)].fidx], subdivision,
          scale, internal_w, internal_h, source_pixels);
    } catch (...) {
#pragma omp critical
      { if (!eptr) eptr = std::current_exception(); }
    }
  }
  greg.enabled = prev_geom_enabled;
  if (eptr) std::rethrow_exception(eptr);

  double t_compute = 0.0, t_write = 0.0;
  for (std::size_t vi = 0; vi < variants.size(); ++vi) {
    json vjson;
    vjson["pixfrac"] = variants[vi].pixfrac;
    vjson["geometry_hash"] = result.identities[vi].geometry_hash;
    vjson["sampling_plan_hash"] = plan.plan_hash;
    vjson["canvas_width_native"] = plan.canvas_width_native;
    vjson["canvas_height_native"] = plan.canvas_height_native;
    vjson["internal_scale"] = scale;
    vjson["color_mode"] = plan.color_mode == ColorMode::MONO ? "MONO" : "OSC";
    vjson["frames"] = json::array();
    // tasks were pushed in (vi, fidx) order == source_index order.
    for (std::size_t k = 0; k < tasks.size(); ++k) {
      if (tasks[k].vi != vi) continue;
      const auto &o = outs[k];
      if (!o.present) continue;
      vjson["frames"].push_back(o.fj);
      result.frames.push_back(o.stats);
      result.total_leaves += o.stats.leaves_written;
      result.total_record_bytes += o.record_bytes;
      result.index_bytes += o.index_bytes;
      if (o.excluded)
        result.excluded_frames.emplace_back(o.excluded_frame_id, o.excluded_rate);
      t_compute += o.t_compute;
      t_write += o.t_write;
    }
    manifest["variants"].push_back(vjson);
  }
  result.workers_used = workers;
  result.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0)
          .count();

  // ---- crash-safe commit ----
  const std::string mtext = manifest.dump(2);
  {
    const fs::path mp = staging / "manifest.json";
    const int mfd = ::open(mp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (mfd < 0)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_OPEN: " + mp.string());
    write_all(mfd, mtext.data(), mtext.size());
    if (::fsync(mfd) != 0) {
      ::close(mfd);
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC: manifest");
    }
    ::close(mfd);
  }
  fsync_path(staging, /*dir=*/true);

  const fs::path gen_dir = root / gen_id;
  fs::rename(staging, gen_dir, ec);
  if (ec) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_COMMIT: " + ec.message());
  fsync_path(root, /*dir=*/true);

  json current;
  current["schema"] = kSchemaVersion;
  current["generation"] = gen_id;
  current["variants"] = json::array();
  for (std::size_t vi = 0; vi < variants.size(); ++vi)
    current["variants"].push_back(
        {{"pixfrac", variants[vi].pixfrac},
         {"geometry_hash", result.identities[vi].geometry_hash}});
  const std::string ctext = current.dump(2);
  const fs::path cur = root / "current.json";
  const fs::path cur_tmp = root / (".current-" + uid + ".json");
  {
    const int cfd = ::open(cur_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (cfd < 0)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_OPEN: current tmp");
    write_all(cfd, ctext.data(), ctext.size());
    if (::fsync(cfd) != 0) {
      ::close(cfd);
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FSYNC: current tmp");
    }
    ::close(cfd);
  }
  fs::rename(cur_tmp, cur, ec);
  if (ec) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_COMMIT: current");
  fsync_path(root, /*dir=*/true);

  result.generation_dir = gen_dir;
  result.sample_leaves_seconds = t_compute;
  result.write_seconds = t_write;
  return result;
}

// --------------------------------------------------------------------------
// Reader
// --------------------------------------------------------------------------

struct DrizzleGeometryCacheReader::Impl {
  struct FrameData {
    fs::path leaves_path;
    std::vector<RowEntry> rows;
    int source_height = 0;
    bool excluded = false;
    std::uint64_t samples_total = 0;
    std::uint64_t samples_discarded = 0;
    double subdivision_error_rate = 0.0;
    std::uint64_t leaves_bytes = 0;
  };
  struct VariantData {
    float pixfrac = 0.0f;
    int canvas_width_native = 0;
    int internal_scale = 0;
    std::map<std::size_t, FrameData> frames;
  };
  std::vector<VariantData> variants;
  std::vector<std::pair<std::string, double>> excluded;
  std::uint64_t resident = 0;
  // Incremented once per enumerate_stripe() call. Atomic so a band-parallel
  // reduction (plan 11.14.5 P3 Teil 2) can assert the cache is still being
  // consulted at workers > 1 without racing.
  std::atomic<std::uint64_t> enumerate_calls{0};

  const VariantData *find_variant(float pixfrac) const {
    for (const auto &v : variants)
      if (v.pixfrac == pixfrac) return &v;
    return nullptr;
  }
};

DrizzleGeometryCacheReader::DrizzleGeometryCacheReader(
    const fs::path &root, const std::vector<GeometryCacheIdentity> &expected,
    const std::vector<std::size_t> &expected_local_source_indices,
    bool verify_record_bytes)
    : impl_(std::make_unique<Impl>()) {
  const fs::path cur = root / "current.json";
  if (!fs::is_regular_file(cur))
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_NO_CURRENT");
  json current;
  { std::ifstream in(cur); in >> current; }
  if (current.value("schema", -1) != kSchemaVersion)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_CURRENT_SCHEMA");

  const fs::path gen = root / current.at("generation").get<std::string>();
  const fs::path mpath = gen / "manifest.json";
  if (!fs::is_regular_file(mpath))
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_NO_MANIFEST");
  json manifest;
  { std::ifstream in(mpath); in >> manifest; }
  if (manifest.value("schema", -1) != kSchemaVersion ||
      manifest.value("algorithm_version", -1) != kAlgorithmVersion)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SCHEMA_MISMATCH");

  const int m_sw = manifest.at("source_width").get<int>();
  const int m_sh = manifest.at("source_height").get<int>();
  const int m_iw = manifest.at("internal_width").get<int>();
  const int m_ih = manifest.at("internal_height").get<int>();
  if (m_sw <= 0 || m_sh <= 0 || m_iw <= 0 || m_ih <= 0)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_BAD_DIMS");

  const auto &vseq = manifest.at("variants");
  if (vseq.size() != expected.size())
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_VARIANT_COUNT");

  std::vector<std::size_t> want(expected_local_source_indices);
  std::sort(want.begin(), want.end());
  if (std::adjacent_find(want.begin(), want.end()) != want.end())
    throw std::invalid_argument("DRIZZLE_GEOMETRY_CACHE_DUP_EXPECTED");

  for (std::size_t vi = 0; vi < vseq.size(); ++vi) {
    const auto &vj = vseq[vi];
    const auto &exp = expected[vi];
    if (vj.at("geometry_hash").get<std::string>() != exp.geometry_hash)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_IDENTITY_MISMATCH");
    if (vj.at("canvas_width_native").get<int>() != exp.canvas_width_native ||
        vj.at("canvas_height_native").get<int>() != exp.canvas_height_native ||
        vj.at("internal_scale").get<int>() != exp.internal_scale)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_GEOMETRY_MISMATCH");
    const std::string cm = vj.at("color_mode").get<std::string>();
    if ((cm == "MONO") != (exp.color_mode == ColorMode::MONO))
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_COLOR_MISMATCH");
    const float vpf = vj.at("pixfrac").get<float>();
    if (vpf != exp.pixfrac)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_PIXFRAC_MISMATCH");
    if (vj.at("canvas_width_native").get<int>() * exp.internal_scale != m_iw)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_DIM_CONSISTENCY");

    Impl::VariantData vd;
    vd.pixfrac = vpf;
    vd.canvas_width_native = vj.at("canvas_width_native").get<int>();
    vd.internal_scale = exp.internal_scale;

    std::vector<std::size_t> got;
    for (const auto &fj : vj.at("frames")) {
      Impl::FrameData fd;
      const std::size_t sidx = fj.at("source_index").get<std::size_t>();
      fd.source_height = fj.at("source_height").get<int>();
      if (fd.source_height != m_sh ||
          fj.at("source_width").get<int>() != m_sw ||
          fj.at("internal_width").get<int>() != m_iw ||
          fj.at("internal_height").get<int>() != m_ih)
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_FRAME_DIMS");
      fd.excluded = fj.at("excluded").get<bool>();
      fd.samples_total = fj.at("samples_total").get<std::uint64_t>();
      fd.samples_discarded = fj.at("samples_discarded").get<std::uint64_t>();
      fd.subdivision_error_rate = fj.at("subdivision_error_rate").get<double>();
      fd.leaves_bytes = fj.at("leaves_bytes").get<std::uint64_t>();
      if (fd.samples_total !=
          static_cast<std::uint64_t>(m_sw) * static_cast<std::uint64_t>(m_sh))
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SAMPLES_TOTAL");
      if (fd.samples_discarded > fd.samples_total)
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_DISCARD_RANGE");
      {
        const double want_rate =
            fd.samples_total ? static_cast<double>(fd.samples_discarded) /
                                   static_cast<double>(fd.samples_total)
                             : 0.0;
        if (std::abs(want_rate - fd.subdivision_error_rate) > 1e-12)
          throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_RATE_CONSISTENCY");
      }

      const fs::path rp = gen / fj.at("rows_file").get<std::string>();
      const fs::path lp = gen / fj.at("leaves_file").get<std::string>();
      if (!fs::is_regular_file(rp) || !fs::is_regular_file(lp))
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_MISSING_FILE");
      if (core::sha256_file(rp) != fj.at("rows_sha256").get<std::string>())
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROWS_CHECKSUM");
      if (verify_record_bytes &&
          core::sha256_file(lp) != fj.at("leaves_sha256").get<std::string>())
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_LEAVES_CHECKSUM");

      const auto rsz = fs::file_size(rp);
      const auto lsz = fs::file_size(lp);
      if (rsz % sizeof(RowEntry) ||
          rsz / sizeof(RowEntry) != static_cast<std::uint64_t>(fd.source_height))
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROW_COUNT");
      if (lsz != fd.leaves_bytes || lsz % sizeof(LeafRecord))
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_LEAVES_SIZE");

      fd.rows.resize(static_cast<std::size_t>(fd.source_height));
      {
        std::ifstream in(rp, std::ios::binary);
        in.read(reinterpret_cast<char *>(fd.rows.data()),
                static_cast<std::streamsize>(rsz));
        if (!in) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROWS_READ");
      }
      // Offsets must tile [0, lsz) exactly, in order, aligned.
      std::uint64_t expect_off = 0;
      std::uint64_t leaf_sum = 0;
      for (const auto &re : fd.rows) {
        if (re.record_offset != expect_off ||
            re.record_offset % sizeof(LeafRecord) != 0)
          throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROW_OFFSET");
        const std::uint64_t block = re.record_count * sizeof(LeafRecord);
        if (re.record_offset + block > lsz)
          throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROW_BOUNDS");
        expect_off += block;
        leaf_sum += re.record_count;
        if (re.record_count &&
            !(re.canvas_ymin <= re.canvas_ymax))
          throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_ROW_YRANGE");
      }
      if (expect_off != lsz ||
          leaf_sum != fj.at("leaves").get<std::uint64_t>())
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_LEAF_SUM");

      fd.leaves_path = lp;
      if (fd.excluded)
        impl_->excluded.emplace_back(fj.at("frame_id").get<std::string>(),
                                     fd.subdivision_error_rate);
      impl_->resident += fd.rows.size() * sizeof(RowEntry) + sizeof(fd);
      got.push_back(sidx);
      if (!vd.frames.emplace(sidx, std::move(fd)).second)
        throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_DUP_FRAME");
    }
    std::sort(got.begin(), got.end());
    if (got != want)
      throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_POPULATION");
    impl_->variants.push_back(std::move(vd));
  }
}

DrizzleGeometryCacheReader::~DrizzleGeometryCacheReader() = default;

bool DrizzleGeometryCacheReader::has_frame(float pixfrac,
                                           std::size_t source_index) const {
  const auto *v = impl_->find_variant(pixfrac);
  if (!v) return false;
  auto it = v->frames.find(source_index);
  return it != v->frames.end() && !it->second.excluded;
}

DrizzleGeometryCacheReader::FrameStatsView
DrizzleGeometryCacheReader::frame_stats(float pixfrac,
                                        std::size_t source_index) const {
  FrameStatsView out;
  const auto *v = impl_->find_variant(pixfrac);
  if (!v) return out;
  auto it = v->frames.find(source_index);
  if (it == v->frames.end()) return out;
  out.present = true;
  out.excluded = it->second.excluded;
  out.samples_total = it->second.samples_total;
  out.samples_discarded = it->second.samples_discarded;
  out.subdivision_error_rate = it->second.subdivision_error_rate;
  return out;
}

void DrizzleGeometryCacheReader::enumerate_stripe(
    float pixfrac, std::size_t source_index, int scale, int y_begin, int rows,
    const DrizzleLeafCellSink &sink) const {
  impl_->enumerate_calls.fetch_add(1, std::memory_order_relaxed);
  const auto *v = impl_->find_variant(pixfrac);
  if (!v) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_UNKNOWN_VARIANT");
  auto it = v->frames.find(source_index);
  if (it == v->frames.end())
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_UNKNOWN_FRAME");
  const auto &fd = it->second;
  if (scale != v->internal_scale)
    throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_SCALE_MISMATCH");
  const int Wi = v->canvas_width_native * scale;
  const double y_lo = static_cast<double>(y_begin);
  const double y_hi = static_cast<double>(y_begin + rows);

  std::ifstream in(fd.leaves_path, std::ios::binary);
  if (!in) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_LEAVES_OPEN");
  std::vector<LeafRecord> block;  // reused per row

  for (int sy = 0; sy < fd.source_height; ++sy) {
    const RowEntry &re = fd.rows[static_cast<std::size_t>(sy)];
    if (re.record_count == 0) continue;
    if (!(re.canvas_ymin < y_hi && re.canvas_ymax > y_lo)) continue;

    block.resize(static_cast<std::size_t>(re.record_count));
    in.seekg(static_cast<std::streamoff>(re.record_offset));
    in.read(reinterpret_cast<char *>(block.data()),
            static_cast<std::streamsize>(re.record_count * sizeof(LeafRecord)));
    if (!in) throw std::runtime_error("DRIZZLE_GEOMETRY_CACHE_STRIPE_READ");

    for (const LeafRecord &rec : block) {
      const double xmin = *std::min_element(rec.x, rec.x + 4);
      const double xmax = *std::max_element(rec.x, rec.x + 4);
      const double ymin = *std::min_element(rec.y, rec.y + 4);
      const double ymax = *std::max_element(rec.y, rec.y + 4);
      const int x0 = static_cast<int>(
          std::clamp(std::floor(xmin), 0.0, static_cast<double>(Wi)));
      const int x1 = static_cast<int>(
          std::clamp(std::ceil(xmax), 0.0, static_cast<double>(Wi)));
      const int cy0 = static_cast<int>(std::clamp(
          std::floor(ymin), static_cast<double>(y_begin),
          static_cast<double>(y_begin + rows)));
      const int cy1 = static_cast<int>(std::clamp(
          std::ceil(ymax), static_cast<double>(y_begin),
          static_cast<double>(y_begin + rows)));
      for (int y = cy0; y < cy1; ++y)
        for (int x = x0; x < x1; ++x)
          sink(static_cast<int>(rec.source_x), sy, rec.channel, rec.leaf_order,
               x, y, rec.x, rec.y);
    }
  }
}

const std::vector<std::pair<std::string, double>> &
DrizzleGeometryCacheReader::excluded_frames() const {
  return impl_->excluded;
}

std::uint64_t DrizzleGeometryCacheReader::resident_bytes() const {
  return impl_->resident;
}

std::uint64_t DrizzleGeometryCacheReader::enumerate_call_count() const {
  return impl_->enumerate_calls.load(std::memory_order_relaxed);
}

// --------------------------------------------------------------------------
// Thread-local active reader (P3: one guard per worker)
// --------------------------------------------------------------------------

namespace {
thread_local const DrizzleGeometryCacheReader *g_active_geometry_cache = nullptr;
}

ScopedActiveGeometryCache::ScopedActiveGeometryCache(
    const DrizzleGeometryCacheReader *reader)
    : prev_(g_active_geometry_cache) {
  g_active_geometry_cache = reader;
}
ScopedActiveGeometryCache::~ScopedActiveGeometryCache() {
  g_active_geometry_cache = prev_;
}
const DrizzleGeometryCacheReader *active_geometry_cache() {
  return g_active_geometry_cache;
}

} // namespace tile_compile::reconstruction
