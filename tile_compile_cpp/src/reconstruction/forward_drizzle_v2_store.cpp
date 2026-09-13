#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {

using json = nlohmann::json;

constexpr std::uint32_t kBandMagic = 0x32564446U;  // "FDV2" little-endian
constexpr std::size_t kBandHeaderBytes = 64;
constexpr std::size_t kMaxManifestBytes = 4 * 1024 * 1024;
const char *kGenerationPrefix = "forward_drizzle_v2_generation-";

// Self-describing band artifact header, host byte order like the record
// payload it precedes (trusted-run derived artifact, same-binary reader).
struct BandHeader {
  std::uint32_t magic = kBandMagic;
  std::uint32_t schema_version = 1;
  std::int32_t band_index = 0;
  std::int32_t y_begin = 0;
  std::int32_t rows = 0;
  std::int32_t native_cols = 0;
  std::int32_t channels = 0;
  std::uint32_t record_bytes = sizeof(ForwardDrizzleV2PixelResult);
  std::uint64_t record_count = 0;
  std::uint64_t dense_overlap_count = 0;
  std::uint64_t reserved0 = 0;
  std::uint64_t reserved1 = 0;
};
static_assert(sizeof(BandHeader) == kBandHeaderBytes);
static_assert(sizeof(ForwardDrizzleV2PixelResult) == 64);

std::string digest(const json &j) {
  const auto text = j.dump();  // object keys sort -> canonical bytes
  return core::sha256_bytes(std::vector<std::uint8_t>(text.begin(), text.end()));
}

std::string band_artifact_name(int index) {
  char name[32];
  std::snprintf(name, sizeof(name), "band-%04d.bin", index);
  return name;
}

std::string band_profiles_artifact_name(int index) {
  char name[40];
  std::snprintf(name, sizeof(name), "band-%04d.profiles.bin", index);
  return name;
}

json plan_to_json(const ForwardDrizzleV2RunPlan &p, bool with_hash) {
  json j = {
      {"schema_version", p.schema_version},
      {"pipeline_contract", p.pipeline_contract},
      {"contract_version", p.contract_version},
      {"source_identity_hash", p.source_identity_hash},
      {"normalized_cache_hash", p.normalized_cache_hash},
      {"quality_plan_hash", p.quality_plan_hash},
      {"sampling_plan_hash", p.sampling_plan_hash},
      {"config_snapshot_hash", p.config_snapshot_hash},
      {"enumeration", p.enumeration},
      {"estimator", p.estimator},
      {"reservoir_size", p.reservoir_size},
      {"reservoir_seed", p.reservoir_seed},
      {"min_clip_contributors", p.min_clip_contributors},
      {"min_candidates", p.min_candidates},
      {"robust_passes", p.robust_passes},
      {"sigma_low", p.sigma_low},
      {"sigma_high", p.sigma_high},
      {"support_fold_contract", p.support_fold_contract},
      {"numerics", p.numerics},
      {"sigma2_enabled", p.sigma2_enabled},
      {"local_warp_representation", p.local_warp_representation},
      {"trusted_run", p.trusted_run},
      {"emit_profiles", p.emit_profiles},
      {"multiband_levels", p.multiband_levels},
      {"fine_quality_exponent", p.fine_quality_exponent},
      {"medium_quality_exponent", p.medium_quality_exponent},
      {"native_width", p.native_width},
      {"native_height", p.native_height},
      {"channels", p.channels},
      {"internal_scale", p.internal_scale},
      {"color_mode", p.color_mode},
      {"pixfrac", p.pixfrac},
      {"bayer_pattern", p.bayer_pattern},
      {"cfa_origin_x", p.cfa_origin_x},
      {"cfa_origin_y", p.cfa_origin_y},
      {"frame_count", p.frame_count},
      {"band_rows", p.band_rows},
      {"band_count", p.band_count},
      {"tile_cols", p.tile_cols},
      {"halo_rows", p.halo_rows},
      {"x_tiled", p.x_tiled},
  };
  // Absent when empty so pre-field stores keep their stored plan_hash
  // reproducible (the hash input then matches the old serialization).
  if (!p.source_quality_cache_hash.empty())
    j["source_quality_cache_hash"] = p.source_quality_cache_hash;
  if (with_hash) j["plan_hash"] = p.plan_hash;
  return j;
}

ForwardDrizzleV2RunPlan plan_from_json(const json &j) {
  ForwardDrizzleV2RunPlan p;
  p.schema_version = j.at("schema_version").get<int>();
  p.pipeline_contract = j.at("pipeline_contract").get<std::string>();
  p.contract_version = j.at("contract_version").get<int>();
  p.source_identity_hash = j.at("source_identity_hash").get<std::string>();
  p.normalized_cache_hash =
      j.value("normalized_cache_hash", std::string{});
  p.quality_plan_hash = j.value("quality_plan_hash", std::string{});
  p.sampling_plan_hash = j.at("sampling_plan_hash").get<std::string>();
  p.config_snapshot_hash = j.at("config_snapshot_hash").get<std::string>();
  p.source_quality_cache_hash =
      j.value("source_quality_cache_hash", std::string{});
  p.enumeration = j.at("enumeration").get<std::string>();
  p.estimator = j.at("estimator").get<std::string>();
  p.reservoir_size = j.at("reservoir_size").get<int>();
  p.reservoir_seed = j.at("reservoir_seed").get<std::uint64_t>();
  p.min_clip_contributors = j.at("min_clip_contributors").get<int>();
  p.min_candidates = j.at("min_candidates").get<int>();
  p.robust_passes = j.at("robust_passes").get<int>();
  p.sigma_low = j.at("sigma_low").get<double>();
  p.sigma_high = j.at("sigma_high").get<double>();
  p.support_fold_contract = j.at("support_fold_contract").get<std::string>();
  p.numerics = j.at("numerics").get<std::string>();
  p.sigma2_enabled = j.at("sigma2_enabled").get<bool>();
  p.local_warp_representation =
      j.at("local_warp_representation").get<std::string>();
  p.trusted_run = j.at("trusted_run").get<bool>();
  p.emit_profiles = j.value("emit_profiles", false);
  p.multiband_levels = j.value("multiband_levels", 0);
  p.fine_quality_exponent = j.value("fine_quality_exponent", 4.0f);
  p.medium_quality_exponent = j.value("medium_quality_exponent", 2.0f);
  p.native_width = j.at("native_width").get<int>();
  p.native_height = j.at("native_height").get<int>();
  p.channels = j.at("channels").get<int>();
  p.internal_scale = j.at("internal_scale").get<int>();
  p.color_mode = j.at("color_mode").get<std::string>();
  p.pixfrac = j.at("pixfrac").get<double>();
  p.bayer_pattern = j.at("bayer_pattern").get<int>();
  p.cfa_origin_x = j.at("cfa_origin_x").get<int>();
  p.cfa_origin_y = j.at("cfa_origin_y").get<int>();
  p.frame_count = j.at("frame_count").get<std::uint64_t>();
  p.band_rows = j.at("band_rows").get<int>();
  p.band_count = j.at("band_count").get<int>();
  p.tile_cols = j.at("tile_cols").get<int>();
  p.halo_rows = j.at("halo_rows").get<int>();
  p.x_tiled = j.at("x_tiled").get<bool>();
  p.plan_hash = j.value("plan_hash", std::string{});
  return p;
}

void validate_plan_fields(const ForwardDrizzleV2RunPlan &p) {
  const bool bad =
      p.schema_version != ForwardDrizzleV2RunPlan::kSchemaVersion ||
      p.pipeline_contract.empty() || p.contract_version != 1 ||
      p.source_identity_hash.empty() || p.sampling_plan_hash.empty() ||
      p.config_snapshot_hash.empty() || p.enumeration.empty() ||
      p.estimator.empty() || p.reservoir_size <= 0 ||
      p.min_clip_contributors <= 0 || p.min_candidates <= 0 ||
      p.robust_passes <= 0 || !(p.sigma_low > 0.0) || !(p.sigma_high > 0.0) ||
      p.support_fold_contract.empty() || p.numerics.empty() ||
      p.local_warp_representation.empty() || p.native_width <= 0 ||
      p.native_height <= 0 || (p.channels != 1 && p.channels != 3) ||
      (p.internal_scale != 1 && p.internal_scale != 2) ||
      (p.channels == 1) != (p.color_mode == "MONO") ||
      (p.channels == 3) != (p.color_mode == "OSC") || !(p.pixfrac > 0.0) ||
      p.pixfrac > 1.0 || p.bayer_pattern < 0 || p.cfa_origin_x < 0 ||
      p.multiband_levels < 0 || p.multiband_levels > 4 ||
      (!p.emit_profiles && p.multiband_levels != 0) ||
      !(p.fine_quality_exponent > 0.0f) ||
      !(p.medium_quality_exponent > 0.0f) ||
      p.cfa_origin_y < 0 || p.frame_count == 0 || p.band_rows <= 0 ||
      p.band_count <= 0 || p.halo_rows < 0 ||
      // Bands tile [0, native_height) contiguously; the last band is
      // allowed to be short.
      static_cast<std::int64_t>(p.band_rows) * (p.band_count - 1) >=
          p.native_height ||
      static_cast<std::int64_t>(p.band_rows) * p.band_count <
          p.native_height ||
      (p.x_tiled ? (p.tile_cols <= 0 || p.tile_cols >= p.native_width)
                 : (p.tile_cols != 0 && p.tile_cols != p.native_width));
  if (bad) throw std::invalid_argument("FDV2_STORE_INVALID_PLAN");
}

json checkpoint_to_json(const ForwardDrizzleV2Checkpoint &c, bool with_hash) {
  json bands = json::array();
  for (const auto &b : c.bands) {
    bands.push_back({{"band_index", b.band_index},
                     {"y_begin", b.y_begin},
                     {"rows", b.rows},
                     {"native_cols", b.native_cols},
                     {"channels", b.channels},
                     {"dense_overlap_count", b.dense_overlap_count},
                     {"artifact", b.artifact},
                     {"bytes", b.bytes},
                     {"sha256", b.sha256},
                     {"profiles_artifact", b.profiles_artifact},
                     {"profiles_bytes", b.profiles_bytes},
                     {"profiles_sha256", b.profiles_sha256}});
  }
  json j = {{"schema_version", c.schema_version},
            {"plan_hash", c.plan_hash},
            {"band_count", c.band_count},
            {"bands", bands}};
  if (with_hash) j["checkpoint_hash"] = c.checkpoint_hash;
  return j;
}

ForwardDrizzleV2Checkpoint checkpoint_from_json(const json &j) {
  ForwardDrizzleV2Checkpoint c;
  c.schema_version = j.at("schema_version").get<int>();
  c.plan_hash = j.at("plan_hash").get<std::string>();
  c.band_count = j.at("band_count").get<int>();
  for (const auto &e : j.at("bands")) {
    ForwardDrizzleV2BandCommit b;
    b.band_index = e.at("band_index").get<int>();
    b.y_begin = e.at("y_begin").get<int>();
    b.rows = e.at("rows").get<int>();
    b.native_cols = e.at("native_cols").get<int>();
    b.channels = e.at("channels").get<int>();
    b.dense_overlap_count = e.at("dense_overlap_count").get<std::uint64_t>();
    b.artifact = e.at("artifact").get<std::string>();
    b.bytes = e.at("bytes").get<std::uintmax_t>();
    b.sha256 = e.at("sha256").get<std::string>();
    b.profiles_artifact =
        e.value("profiles_artifact", std::string{});
    b.profiles_bytes = e.value("profiles_bytes", std::uintmax_t{0});
    b.profiles_sha256 =
        e.value("profiles_sha256", std::string{});
    c.bands.push_back(std::move(b));
  }
  c.checkpoint_hash = j.value("checkpoint_hash", std::string{});
  return c;
}

void write_checkpoint(const fs::path &generation,
                      ForwardDrizzleV2Checkpoint &checkpoint) {
  checkpoint.checkpoint_hash = digest(checkpoint_to_json(checkpoint, false));
  core::write_text_atomic(generation / "checkpoint.json",
                          checkpoint_to_json(checkpoint, true).dump(2));
}

json read_small_json(const fs::path &path) {
  std::error_code ec;
  if (!fs::is_regular_file(fs::symlink_status(path, ec)))
    throw std::runtime_error("FDV2_STORE_NOT_REGULAR_FILE: " +
                             path.filename().string());
  if (fs::file_size(path, ec) > kMaxManifestBytes)
    throw std::runtime_error("FDV2_STORE_OVERSIZED_MANIFEST");
  std::ifstream file(path);
  json j = json::parse(file, nullptr, true, true);
  if (file.bad())
    throw std::runtime_error("FDV2_STORE_MANIFEST_READ_FAILED");
  return j;
}

// Verifies one committed band artifact on disk: existence, size and
// content hash against its checkpoint entry. Returns false with `error`
// set on any failure (fail closed, never throws for store state).
bool verify_band_artifact(const fs::path &generation,
                          const ForwardDrizzleV2BandCommit &b,
                          std::string &error) {
  std::error_code ec;
  const fs::path path = generation / b.artifact;
  if (!fs::is_regular_file(fs::symlink_status(path, ec))) {
    error = "missing band artifact " + b.artifact;
    return false;
  }
  const auto size = fs::file_size(path, ec);
  if (ec || size != b.bytes) {
    error = "size mismatch on " + b.artifact;
    return false;
  }
  if (core::sha256_file(path) != b.sha256) {
    error = "sha256 mismatch on " + b.artifact;
    return false;
  }
  if (b.profiles_artifact.empty()) return true;
  const fs::path ppath = generation / b.profiles_artifact;
  if (!fs::is_regular_file(fs::symlink_status(ppath, ec))) {
    error = "missing band artifact " + b.profiles_artifact;
    return false;
  }
  const auto psize = fs::file_size(ppath, ec);
  if (ec || psize != b.profiles_bytes) {
    error = "size mismatch on " + b.profiles_artifact;
    return false;
  }
  if (core::sha256_file(ppath) != b.profiles_sha256) {
    error = "sha256 mismatch on " + b.profiles_artifact;
    return false;
  }
  return true;
}

// Verifies a checkpoint's band list forms a contiguous prefix whose
// artifacts are all intact. `expected_cols` is the band width the plan
// implies. Returns the verified prefix length via `bands` (already
// parsed).
bool verify_checkpoint_bands(const fs::path &generation,
                             const ForwardDrizzleV2Checkpoint &c,
                             int expected_cols, int native_height,
                             bool emit_profiles, std::string &error) {
  int y = 0;
  for (std::size_t i = 0; i < c.bands.size(); ++i) {
    const auto &b = c.bands[i];
    const bool profiles_expected =
        emit_profiles && !b.profiles_artifact.empty() &&
        b.profiles_bytes > 0 && !b.profiles_sha256.empty() &&
        b.profiles_artifact ==
            band_profiles_artifact_name(b.band_index);
    const bool profiles_absent = b.profiles_artifact.empty() &&
                                 b.profiles_bytes == 0 &&
                                 b.profiles_sha256.empty();
    if (b.band_index != static_cast<int>(i) || b.y_begin != y ||
        b.rows <= 0 || b.native_cols != expected_cols ||
        b.channels <= 0 || b.artifact != band_artifact_name(b.band_index) ||
        b.sha256.empty() || y + b.rows > native_height ||
        (emit_profiles ? !profiles_expected : !profiles_absent)) {
      error = "non-contiguous or malformed checkpoint band " +
              std::to_string(i);
      return false;
    }
    if (!verify_band_artifact(generation, b, error)) return false;
    y += b.rows;
  }
  return true;
}

}  // namespace

void finalize_forward_drizzle_v2_run_plan(ForwardDrizzleV2RunPlan &plan) {
  validate_plan_fields(plan);
  plan.plan_hash = digest(plan_to_json(plan, false));
}

std::string serialize_forward_drizzle_v2_plan(
    const ForwardDrizzleV2RunPlan &plan) {
  return plan_to_json(plan, true).dump(2);
}

bool parse_forward_drizzle_v2_plan(const std::string &text,
                                   ForwardDrizzleV2RunPlan &out,
                                   std::string &error) {
  try {
    out = plan_from_json(json::parse(text));
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

std::string serialize_forward_drizzle_v2_checkpoint(
    const ForwardDrizzleV2Checkpoint &checkpoint) {
  return checkpoint_to_json(checkpoint, true).dump(2);
}

bool parse_forward_drizzle_v2_checkpoint(const std::string &text,
                                         ForwardDrizzleV2Checkpoint &out,
                                         std::string &error) {
  try {
    out = checkpoint_from_json(json::parse(text));
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

ForwardDrizzleV2StoreWriter::ForwardDrizzleV2StoreWriter(
    fs::path root, ForwardDrizzleV2RunPlan plan)
    : root_(std::move(root)), plan_(std::move(plan)) {
  if (plan_.plan_hash.empty())
    throw std::invalid_argument("FDV2_STORE_PLAN_NOT_FINALIZED");
}

ForwardDrizzleV2StoreWriter::~ForwardDrizzleV2StoreWriter() {
  if (!publish_attempted_ && !generation_.empty()) {
    std::error_code ec;
    fs::remove_all(generation_, ec);  // only this writer's own generation
  }
}

void ForwardDrizzleV2StoreWriter::begin() {
  if (begun_) throw std::runtime_error("FDV2_STORE_BEGIN_TWICE");
  fs::create_directories(root_);
  static std::atomic<std::uint64_t> sequence{0};
  for (int attempt = 0; attempt < 64; ++attempt) {
    auto dir =
        root_ / (kGenerationPrefix +
                 std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count()) +
                 "-" + std::to_string(sequence++));
    if (fs::create_directory(dir)) {
      generation_ = dir;
      break;
    }
  }
  if (generation_.empty())
    throw std::runtime_error("FDV2_STORE_GENERATION_FAILED");
  core::write_text_atomic(generation_ / "plan.json",
                          serialize_forward_drizzle_v2_plan(plan_));
  ForwardDrizzleV2Checkpoint checkpoint;
  checkpoint.plan_hash = plan_.plan_hash;
  checkpoint.band_count = plan_.band_count;
  write_checkpoint(generation_, checkpoint);
  begun_ = true;
}

void ForwardDrizzleV2StoreWriter::adopt(
    const fs::path &generation, int resume_from_band,
    std::vector<ForwardDrizzleV2BandCommit> committed) {
  if (begun_) throw std::runtime_error("FDV2_STORE_BEGIN_TWICE");
  if (resume_from_band != static_cast<int>(committed.size()) ||
      resume_from_band > plan_.band_count)
    throw std::invalid_argument("FDV2_STORE_ADOPT_MISMATCH");
  std::error_code ec;
  if (!fs::is_directory(fs::symlink_status(generation, ec)))
    throw std::runtime_error("FDV2_STORE_ADOPT_MISSING_GENERATION");
  // The adopted generation must provably belong to this plan even when the
  // caller fabricated the inspection state.
  ForwardDrizzleV2RunPlan stored;
  std::string parse_error;
  if (!parse_forward_drizzle_v2_plan(
          read_small_json(generation / "plan.json").dump(), stored,
          parse_error) ||
      stored.plan_hash != plan_.plan_hash ||
      stored.plan_hash != digest(plan_to_json(stored, false)))
    throw std::runtime_error("FDV2_STORE_ADOPT_MISMATCH");
  int y = 0;
  for (std::size_t i = 0; i < committed.size(); ++i) {
    const auto &b = committed[i];
    if (b.band_index != static_cast<int>(i) || b.y_begin != y || b.rows <= 0)
      throw std::invalid_argument("FDV2_STORE_ADOPT_MISMATCH");
    if (plan_.emit_profiles !=
        (!b.profiles_artifact.empty() && b.profiles_bytes > 0 &&
         !b.profiles_sha256.empty()))
      throw std::invalid_argument("FDV2_STORE_ADOPT_MISMATCH");
    y += b.rows;
  }
  if (y > plan_.native_height)
    throw std::invalid_argument("FDV2_STORE_ADOPT_MISMATCH");
  generation_ = generation;
  bands_ = std::move(committed);
  next_y_ = y;
  begun_ = true;
}

void ForwardDrizzleV2StoreWriter::commit_band(
    int band_index, int y_begin, int rows,
    std::span<const ForwardDrizzleV2PixelResult> results,
    std::span<const ForwardDrizzleV2ProfileResult> profiles,
    std::uint64_t dense_overlap_count) {
  if (!begun_ || publish_attempted_)
    throw std::runtime_error("FDV2_STORE_BAND_ORDER");
  const int expected_cols = plan_.x_tiled ? plan_.tile_cols : plan_.native_width;
  if (band_index != static_cast<int>(bands_.size()) ||
      band_index >= plan_.band_count || y_begin != next_y_ || rows <= 0 ||
      y_begin + rows > plan_.native_height)
    throw std::runtime_error("FDV2_STORE_BAND_ORDER");
  const std::uint64_t record_count =
      static_cast<std::uint64_t>(rows) * expected_cols * plan_.channels;
  if (results.size() != record_count ||
      record_count > (std::numeric_limits<std::uint64_t>::max() -
                      kBandHeaderBytes) /
                         sizeof(ForwardDrizzleV2PixelResult))
    throw std::runtime_error("FDV2_STORE_INVALID_BAND");
  if (plan_.emit_profiles != (profiles.size() == record_count))
    throw std::runtime_error("FDV2_STORE_INVALID_BAND");

  const auto name = band_artifact_name(band_index);
  core::AtomicOutput output(generation_ / name);
  {
    BandHeader header;
    header.band_index = band_index;
    header.y_begin = y_begin;
    header.rows = rows;
    header.native_cols = expected_cols;
    header.channels = plan_.channels;
    header.record_count = record_count;
    header.dense_overlap_count = dense_overlap_count;
    std::ofstream file(output.path(), std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(results.data()),
               static_cast<std::streamsize>(record_count *
                                            sizeof(results[0])));
    file.flush();
    if (!file) throw std::runtime_error("FDV2_STORE_BAND_WRITE_FAILED");
  }
  output.commit();  // fsync + rename + dir fsync: artifact durable first

  ForwardDrizzleV2BandCommit commit;
  commit.band_index = band_index;
  commit.y_begin = y_begin;
  commit.rows = rows;
  commit.native_cols = expected_cols;
  commit.channels = plan_.channels;
  commit.dense_overlap_count = dense_overlap_count;
  commit.artifact = name;
  commit.bytes = kBandHeaderBytes +
                 record_count * sizeof(ForwardDrizzleV2PixelResult);
  commit.sha256 = core::sha256_file(generation_ / name);
  if (plan_.emit_profiles) {
    const auto pname = band_profiles_artifact_name(band_index);
    core::AtomicOutput poutput(generation_ / pname);
    {
      BandHeader header;
      header.band_index = band_index;
      header.y_begin = y_begin;
      header.rows = rows;
      header.native_cols = expected_cols;
      header.channels = plan_.channels;
      header.record_bytes = sizeof(ForwardDrizzleV2ProfileResult);
      header.record_count = record_count;
      header.dense_overlap_count = dense_overlap_count;
      std::ofstream file(poutput.path(), std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char *>(&header), sizeof(header));
      file.write(reinterpret_cast<const char *>(profiles.data()),
                 static_cast<std::streamsize>(
                     record_count * sizeof(profiles[0])));
      file.flush();
      if (!file) throw std::runtime_error("FDV2_STORE_BAND_WRITE_FAILED");
    }
    poutput.commit();
    commit.profiles_artifact = pname;
    commit.profiles_bytes =
        kBandHeaderBytes +
        record_count * sizeof(ForwardDrizzleV2ProfileResult);
    commit.profiles_sha256 =
        core::sha256_file(generation_ / pname);
  }
  bands_.push_back(std::move(commit));
  next_y_ += rows;

  // The commit mark is the last durable step of the band transaction.
  ForwardDrizzleV2Checkpoint checkpoint;
  checkpoint.plan_hash = plan_.plan_hash;
  checkpoint.band_count = plan_.band_count;
  checkpoint.bands = bands_;
  write_checkpoint(generation_, checkpoint);
}

fs::path ForwardDrizzleV2StoreWriter::finish(
    const ForwardDrizzleV2CommitGate &gate) {
  if (!begun_ || publish_attempted_)
    throw std::runtime_error("FDV2_STORE_FINISH_ORDER");
  if (static_cast<int>(bands_.size()) != plan_.band_count ||
      next_y_ != plan_.native_height)
    throw std::runtime_error("FDV2_STORE_INCOMPLETE_BANDS");
  if (gate.nonfinite_pixels_inside_source_support != 0 ||
      gate.bands_processed != static_cast<std::uint64_t>(plan_.band_count))
    throw std::runtime_error("FDV2_STORE_COMMIT_GATE_FAILED");
  std::string error;
  for (const auto &b : bands_)
    if (!verify_band_artifact(generation_, b, error))
      throw std::runtime_error("FDV2_STORE_CORRUPT_PLANES: " + error);

  ForwardDrizzleV2Checkpoint checkpoint;
  checkpoint.plan_hash = plan_.plan_hash;
  checkpoint.band_count = plan_.band_count;
  checkpoint.bands = bands_;
  write_checkpoint(generation_, checkpoint);

  json telemetry = json::object();
  if (!gate.telemetry_json.empty())
    telemetry = json::parse(gate.telemetry_json);
  json commit = {
      {"schema_version", 1},
      {"generation", generation_.filename().string()},
      {"plan_hash", plan_.plan_hash},
      {"checkpoint_hash", checkpoint.checkpoint_hash},
      {"band_count", plan_.band_count},
      {"commit_gate",
       {{"nonfinite_pixels_inside_source_support",
         gate.nonfinite_pixels_inside_source_support},
        {"bands_processed", gate.bands_processed}}},
      {"telemetry", telemetry}};
  commit["commit_hash"] = digest(commit);
  core::write_text_atomic(generation_ / "commit.json", commit.dump(2));

  // If the rename succeeded but the directory sync failed, current.json
  // may already name this generation: never delete it afterwards.
  publish_attempted_ = true;
  core::write_text_atomic(root_ / "current.json", commit.dump(2));
  return generation_;
}

bool load_forward_drizzle_v2_published_plan(
    const fs::path &root, ForwardDrizzleV2RunPlan &out, std::string &error) {
  error.clear();
  std::error_code ec;
  const fs::path current_path = root / "current.json";
  if (!fs::is_regular_file(fs::symlink_status(current_path, ec)))
    return false;
  try {
    const json current = read_small_json(current_path);
    const auto generation_name = current.value("generation", std::string{});
    if (generation_name.empty())
      throw std::runtime_error("malformed current.json");
    const fs::path generation = root / generation_name;
    std::string parse_error;
    if (!parse_forward_drizzle_v2_plan(
            read_small_json(generation / "plan.json").dump(), out,
            parse_error))
      throw std::runtime_error("unparseable plan.json: " + parse_error);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("FDV2_STORE_PLAN_LOAD: ") + e.what());
  }
  // The parse round-trip is structural only; the identity is verified by a
  // following inspect() call (plan_hash recompute + commit chain).
  ForwardDrizzleV2RunPlan reparsed;
  std::string re_error;
  if (!parse_forward_drizzle_v2_plan(
          serialize_forward_drizzle_v2_plan(out), reparsed, re_error) ||
      reparsed.plan_hash != out.plan_hash)
    throw std::runtime_error("FDV2_STORE_PLAN_LOAD_ROUNDTRIP");
  return true;
}

std::vector<ForwardDrizzleV2PixelResult> read_forward_drizzle_v2_band(
    const fs::path &generation, const ForwardDrizzleV2BandCommit &commit) {
  const fs::path path = generation / commit.artifact;
  std::error_code ec;
  if (!fs::is_regular_file(fs::symlink_status(path, ec)) ||
      fs::file_size(path, ec) != commit.bytes)
    throw std::runtime_error("FDV2_STORE_BAND_MISSING_OR_SIZED");
  if (core::sha256_file(path) != commit.sha256)
    throw std::runtime_error("FDV2_STORE_BAND_HASH_MISMATCH");
  std::ifstream file(path, std::ios::binary);
  BandHeader header;
  file.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!file || header.magic != kBandMagic ||
      header.schema_version != 1 || header.band_index != commit.band_index ||
      header.y_begin != commit.y_begin || header.rows != commit.rows ||
      header.native_cols != commit.native_cols ||
      header.channels != commit.channels ||
      header.record_bytes != sizeof(ForwardDrizzleV2PixelResult) ||
      header.record_count !=
          static_cast<std::uint64_t>(commit.rows) * commit.native_cols *
              commit.channels ||
      header.dense_overlap_count != commit.dense_overlap_count)
    throw std::runtime_error("FDV2_STORE_BAND_HEADER_MISMATCH");
  std::vector<ForwardDrizzleV2PixelResult> records(header.record_count);
  file.read(reinterpret_cast<char *>(records.data()),
            static_cast<std::streamsize>(header.record_count *
                                         sizeof(records[0])));
  if (!file) throw std::runtime_error("FDV2_STORE_BAND_READ_FAILED");
  return records;
}

std::vector<ForwardDrizzleV2ProfileResult>
read_forward_drizzle_v2_band_profiles(
    const fs::path &generation, const ForwardDrizzleV2BandCommit &commit) {
  if (commit.profiles_artifact.empty())
    throw std::runtime_error("FDV2_STORE_BAND_NO_PROFILES");
  const fs::path path = generation / commit.profiles_artifact;
  std::error_code ec;
  if (!fs::is_regular_file(fs::symlink_status(path, ec)) ||
      fs::file_size(path, ec) != commit.profiles_bytes)
    throw std::runtime_error("FDV2_STORE_BAND_MISSING_OR_SIZED");
  if (core::sha256_file(path) != commit.profiles_sha256)
    throw std::runtime_error("FDV2_STORE_BAND_HASH_MISMATCH");
  std::ifstream file(path, std::ios::binary);
  BandHeader header;
  file.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!file || header.magic != kBandMagic ||
      header.schema_version != 1 || header.band_index != commit.band_index ||
      header.y_begin != commit.y_begin || header.rows != commit.rows ||
      header.native_cols != commit.native_cols ||
      header.channels != commit.channels ||
      header.record_bytes != sizeof(ForwardDrizzleV2ProfileResult) ||
      header.record_count !=
          static_cast<std::uint64_t>(commit.rows) * commit.native_cols *
              commit.channels)
    throw std::runtime_error("FDV2_STORE_BAND_HEADER_MISMATCH");
  std::vector<ForwardDrizzleV2ProfileResult> records(header.record_count);
  file.read(reinterpret_cast<char *>(records.data()),
            static_cast<std::streamsize>(header.record_count *
                                         sizeof(records[0])));
  if (!file) throw std::runtime_error("FDV2_STORE_BAND_READ_FAILED");
  return records;
}

ForwardDrizzleV2StoreInspection inspect_forward_drizzle_v2_store(
    const fs::path &root, const ForwardDrizzleV2RunPlan &expected_plan) {
  if (expected_plan.plan_hash.empty())
    throw std::invalid_argument("FDV2_STORE_PLAN_NOT_FINALIZED");
  ForwardDrizzleV2StoreInspection out;
  const auto fail = [&out](std::string error) {
    out.status = ForwardDrizzleV2StoreStatus::corrupt;
    out.error = std::move(error);
    return out;
  };
  std::error_code ec;
  if (!fs::is_directory(fs::symlink_status(root, ec))) return out;  // fresh

  const int expected_cols = expected_plan.x_tiled ? expected_plan.tile_cols
                                                  : expected_plan.native_width;
  const fs::path current_path = root / "current.json";
  if (fs::is_regular_file(fs::symlink_status(current_path, ec))) {
    json current;
    try {
      current = read_small_json(current_path);
    } catch (const std::exception &e) {
      return fail(std::string("unreadable current.json: ") + e.what());
    }
    const auto generation_name =
        current.value("generation", std::string{});
    const auto plan_hash = current.value("plan_hash", std::string{});
    const auto commit_hash = current.value("commit_hash", std::string{});
    if (current.value("schema_version", 0) != 1 || generation_name.empty() ||
        commit_hash.empty())
      return fail("malformed current.json");
    if (plan_hash != expected_plan.plan_hash)
      return fail("current.json plan_hash mismatch");
    const fs::path generation = root / generation_name;
    if (!fs::is_directory(fs::symlink_status(generation, ec)))
      return fail("current.json points at missing generation");
    json committed;
    try {
      committed = read_small_json(generation / "commit.json");
    } catch (const std::exception &e) {
      return fail(std::string("unreadable commit.json: ") + e.what());
    }
    committed.erase("commit_hash");
    if (digest(committed) != commit_hash ||
        committed.value("plan_hash", std::string{}) != plan_hash)
      return fail("commit.json hash mismatch");
    // The final checkpoint.json is hash-chained into commit.json via
    // checkpoint_hash: parse and bind it so a complete-store consumer gets
    // the same verified band list a resumable inspection exposes. Every
    // band read still re-checks its sha256 against the commit entry.
    ForwardDrizzleV2Checkpoint final_checkpoint;
    std::string fc_parse_error;
    try {
      if (!parse_forward_drizzle_v2_checkpoint(
              read_small_json(generation / "checkpoint.json").dump(),
              final_checkpoint, fc_parse_error))
        return fail("unparseable final checkpoint.json: " + fc_parse_error);
    } catch (const std::exception &e) {
      return fail(std::string("unreadable final checkpoint.json: ") +
                  e.what());
    }
    if (final_checkpoint.schema_version !=
            ForwardDrizzleV2Checkpoint::kSchemaVersion ||
        digest(checkpoint_to_json(final_checkpoint, false)) !=
            final_checkpoint.checkpoint_hash ||
        final_checkpoint.plan_hash != expected_plan.plan_hash ||
        final_checkpoint.band_count != expected_plan.band_count ||
        final_checkpoint.checkpoint_hash !=
            committed.value("checkpoint_hash", std::string{}))
      return fail("final checkpoint.json binding mismatch");
    out.status = ForwardDrizzleV2StoreStatus::complete;
    out.generation = generation;
    out.commit_hash = commit_hash;
    out.committed = std::move(final_checkpoint.bands);
    return out;
  }

  std::vector<fs::path> generations;
  for (const auto &entry : fs::directory_iterator(root, ec)) {
    if (ec) break;
    if (entry.is_directory(ec) &&
        entry.path().filename().string().rfind(kGenerationPrefix, 0) == 0)
      generations.push_back(entry.path());
  }
  if (generations.empty()) return out;  // fresh
  if (generations.size() > 1)
    return fail("ambiguous: multiple unpublished generations");
  const fs::path generation = generations.front();

  ForwardDrizzleV2RunPlan stored_plan;
  std::string parse_error;
  try {
    if (!parse_forward_drizzle_v2_plan(
            read_small_json(generation / "plan.json").dump(), stored_plan,
            parse_error))
      return fail("unparseable plan.json: " + parse_error);
  } catch (const std::exception &e) {
    return fail(std::string("unreadable plan.json: ") + e.what());
  }
  if (stored_plan.schema_version != ForwardDrizzleV2RunPlan::kSchemaVersion)
    return fail("plan schema_version mismatch");
  const std::string recomputed = digest(plan_to_json(stored_plan, false));
  if (stored_plan.plan_hash != recomputed)
    return fail("plan.json hash mismatch (tampered)");
  if (stored_plan.plan_hash != expected_plan.plan_hash)
    return fail("plan context mismatch");

  ForwardDrizzleV2Checkpoint checkpoint;
  try {
    if (!parse_forward_drizzle_v2_checkpoint(
            read_small_json(generation / "checkpoint.json").dump(), checkpoint,
            parse_error))
      return fail("unparseable checkpoint.json: " + parse_error);
  } catch (const std::exception &e) {
    return fail(std::string("unreadable checkpoint.json: ") + e.what());
  }
  if (checkpoint.schema_version != ForwardDrizzleV2Checkpoint::kSchemaVersion)
    return fail("checkpoint schema_version mismatch");
  if (digest(checkpoint_to_json(checkpoint, false)) !=
      checkpoint.checkpoint_hash)
    return fail("checkpoint hash mismatch (tampered)");
  if (checkpoint.plan_hash != expected_plan.plan_hash ||
      checkpoint.band_count != expected_plan.band_count)
    return fail("checkpoint context mismatch");

  std::string verify_error;
  if (!verify_checkpoint_bands(generation, checkpoint, expected_cols,
                               expected_plan.native_height,
                               expected_plan.emit_profiles, verify_error))
    return fail(verify_error);

  out.status = ForwardDrizzleV2StoreStatus::resumable;
  out.next_band = static_cast<int>(checkpoint.bands.size());
  out.generation = generation;
  out.committed = std::move(checkpoint.bands);
  return out;
}

}  // namespace tile_compile::reconstruction
