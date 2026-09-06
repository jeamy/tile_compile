#include "tile_compile/reconstruction/drizzle_profile_store.hpp"

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/atrous_decomposition.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/reconstruction/profile_store_manifest.hpp"

#include <fitsio.h>
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <map>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {
using json = nlohmann::json;
constexpr size_t io_reserve = 8 * 1024 * 1024;

void check(int status) {
  if (status) throw std::runtime_error("DRIZZLE_STORE_FITS: " + std::to_string(status));
}
std::string digest(const json &j) {
  const auto text = j.dump();
  return core::sha256_bytes(std::vector<uint8_t>(text.begin(), text.end()));
}
json identity_json(const DrizzleStoreIdentity &i) {
  json j = {{"source_identity_hash", i.source_identity_hash},
            {"sampling_plan_hash", i.sampling_plan_hash},
            {"reconstruction_hash", i.reconstruction_hash},
            {"normalized_cache_hash", i.normalized_cache_hash},
            {"quality_plan_hash", i.quality_plan_hash}, {"mode", i.mode},
            {"width", i.width}, {"height", i.height},
            {"color_mode", i.color_mode == ColorMode::MONO ? "MONO" : "OSC"}};
  // Additive: a non-multiband store keeps its exact pre-M6 identity JSON.
  if (i.multiband_levels > 0) j["multiband_levels"] = i.multiband_levels;
  return j;
}
// The four channel-min alpha-confidence maps live as single-field pseudo-
// planes ("<name>_X_value") so read_drizzle_profile_region's base+field path
// works unchanged.
const std::array<const char *, 4> kAlphaMapNames = {
    "alpha_separation", "alpha_artifact", "alpha_registration", "alpha_support"};
bool is_multiband_mode(const std::string &mode) {
  return mode == "uniform_raw_multiband_clipped";
}
std::vector<std::string> plane_names(const DrizzleStoreIdentity &i) {
  const bool mb = is_multiband_mode(i.mode);
  if (i.width <= 0 || i.height <= 0 || i.source_identity_hash.empty() ||
      i.sampling_plan_hash.empty() || i.reconstruction_hash.empty() ||
      (i.mode != "uniform_unclipped" && i.mode != "uniform_raw_clipped" && !mb) ||
      (i.color_mode != ColorMode::MONO && i.color_mode != ColorMode::OSC) ||
      (mb ? (i.multiband_levels < 1 || i.multiband_levels > 4)
          : i.multiband_levels != 0))
    throw std::invalid_argument("DRIZZLE_STORE_INVALID_IDENTITY");
  std::vector<std::string> names;
  std::vector<std::string> profiles = {"uniform"};
  if (i.mode != "uniform_unclipped") profiles.push_back("raw");
  if (mb) {
    profiles.push_back("fine");
    if (i.multiband_levels >= 2) profiles.push_back("medium");
  }
  for (const auto &profile : profiles) {
    for (const auto &channel : {"L", "R", "G", "B"}) {
      if ((std::string(channel) == "L") != (i.color_mode == ColorMode::MONO)) continue;
      for (const auto &field : {"value", "weight_sum", "n_eff", "support"})
        names.push_back(profile + "_" + channel + "_" + field);
    }
  }
  if (mb)
    for (const auto *m : kAlphaMapNames)
      names.push_back(std::string(m) + "_X_value");
  std::sort(names.begin(), names.end());
  return names;
}
void check_file(const fs::path &path) {
  if (!fs::is_regular_file(fs::symlink_status(path)))
    throw std::runtime_error("DRIZZLE_STORE_NOT_REGULAR_FILE");
}
json read_small_json(const fs::path &path) {
  check_file(path);
  if (fs::file_size(path) > 1024 * 1024)
    throw std::runtime_error("DRIZZLE_STORE_OVERSIZED_MANIFEST");
  std::ifstream file(path);
  json j = json::parse(file);
  if (file.bad()) throw std::runtime_error("DRIZZLE_STORE_MANIFEST_READ_FAILED");
  return j;
}
void check_fits(const fs::path &path, int width, int height) {
  fitsfile *file = nullptr;
  int status = 0;
  fits_open_diskfile(&file, path.string().c_str(), READONLY, &status);
  check(status);
  int axes = 0, type = 0;
  long size[2] = {0, 0};
  fits_get_img_param(file, 2, &type, &axes, size, &status);
  char roworder[FLEN_VALUE] = {};
  fits_read_key(file, TSTRING, "ROWORDER", roworder, nullptr, &status);
  int close_status = 0;
  fits_close_file(file, &close_status);
  check(status);
  check(close_status);
  if (axes != 2 || type != FLOAT_IMG || size[0] != width || size[1] != height ||
      std::string(roworder) != "TOP-DOWN")
    throw std::runtime_error("DRIZZLE_STORE_FITS_SHAPE_OR_TYPE_MISMATCH");
}
void validate_generation(const fs::path &dir, const json &commit,
                         const DrizzleStoreIdentity &expected) {
  if (commit.at("schema_version") != 2 || commit.at("identity") != identity_json(expected))
    throw std::runtime_error("DRIZZLE_STORE_CONTEXT_MISMATCH");
  const auto manifests = commit.at("planes");
  ProfileStoreManifest manifest;
  std::string error;
  if (!parse_profile_store_manifest(manifests.dump(), manifest, error))
    throw std::runtime_error(error);
  const auto names = plane_names(expected);
  std::vector<std::string> actual;
  for (const auto &p : manifest.planes) actual.push_back(p.name);
  if (actual != names || manifest.internal_width != expected.width ||
      manifest.internal_height != expected.height || manifest.profile != expected.mode)
    throw std::runtime_error("DRIZZLE_STORE_INCOMPLETE_PLANE_SET");
  const auto verified = verify_profile_store(dir, manifest);
  if (!verified.usable) throw std::runtime_error("DRIZZLE_STORE_CORRUPT_PLANES");
  for (const auto &name : names) check_fits(dir / (name + ".fits"), expected.width, expected.height);
}

struct PlaneFile {
  core::AtomicOutput output;
  fitsfile *file = nullptr;
  PlaneFile(const fs::path &path, int width, int height) : output(path) {
    int status = 0;
    fits_create_diskfile(&file, output.path().string().c_str(), &status);
    if (!status) {
      long dims[2] = {width, height};
      fits_create_img(file, FLOAT_IMG, 2, dims, &status);
      char roworder[] = "TOP-DOWN";
      fits_update_key(file, TSTRING, "ROWORDER", roworder, nullptr, &status);
    }
    if (status && file) { int ignored = 0; fits_close_file(file, &ignored); file = nullptr; }
    check(status);
  }
  ~PlaneFile() { if (file) { int s = 0; fits_close_file(file, &s); } }
  void write(int y, int width, std::span<const float> values) {
    int status = 0;
    fits_write_img(file, TFLOAT, static_cast<LONGLONG>(y) * width + 1,
                   static_cast<LONGLONG>(values.size()),
                   const_cast<float *>(values.data()), &status);
    check(status);
  }
  void finish() {
    int status = 0;
    fits_close_file(file, &status);
    file = nullptr;
    check(status);
    output.commit();
  }
};

class StoreWriter {
  fs::path root_, generation_;
  DrizzleStoreIdentity identity_;
  std::map<std::string, std::unique_ptr<PlaneFile>> files_;
  int next_y_ = 0;
  bool publish_attempted_ = false;
public:
  StoreWriter(fs::path root, DrizzleStoreIdentity identity)
      : root_(std::move(root)), identity_(std::move(identity)) {}
  ~StoreWriter() {
    files_.clear();
    if (!publish_attempted_ && !generation_.empty()) {
      std::error_code ec;
      fs::remove_all(generation_, ec); // only this writer's privately reserved generation
    }
  }
  void start() {
    fs::create_directories(root_);
    const auto names = plane_names(identity_);
    const uint64_t pixels = static_cast<uint64_t>(identity_.width) * identity_.height;
    const uint64_t per_pixel = names.size() * sizeof(float);
    if (pixels > (std::numeric_limits<uint64_t>::max() - io_reserve) / per_pixel ||
        fs::space(root_).available < pixels * per_pixel + io_reserve)
      throw std::runtime_error("DRIZZLE_STORE_DISK_BUDGET");
    static std::atomic<uint64_t> sequence{0};
    for (int attempt = 0; attempt < 64; ++attempt) {
      auto dir = root_ / ("generation-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
          std::to_string(sequence++));
      if (fs::create_directory(dir)) { generation_ = dir; break; }
    }
    if (generation_.empty()) throw std::runtime_error("DRIZZLE_STORE_GENERATION_FAILED");
    for (const auto &name : names)
      files_.emplace(name, std::make_unique<PlaneFile>(generation_ / (name + ".fits"),
                                                     identity_.width, identity_.height));
  }
  void write_profile(const std::string &profile, int y, const ForwardDrizzleUniformResult &stripe) {
    const int rows = stripe.internal_height;
    if (stripe.internal_width != identity_.width || stripe.color_mode != identity_.color_mode ||
        rows <= 0 || rows > identity_.height - y)
      throw std::runtime_error("DRIZZLE_STORE_INVALID_STRIPE");
    for (const auto &[channel, p] : std::array<std::pair<const char *, const ProfilePlane *>, 4>{
             {{"R", &stripe.R}, {"G", &stripe.G}, {"B", &stripe.B}, {"L", &stripe.L}}}) {
      if ((std::string(channel) == "L") != (identity_.color_mode == ColorMode::MONO)) continue;
      const size_t count = static_cast<size_t>(rows) * identity_.width;
      if (p->width != identity_.width || p->height != rows || p->value.size() != count ||
          p->weight_sum.size() != count || p->n_eff.size() != count || p->support.size() != count)
        throw std::runtime_error("DRIZZLE_STORE_INVALID_PLANE");
      const std::string base = profile + "_" + channel + "_";
      files_.at(base + "value")->write(y, identity_.width, p->value);
      files_.at(base + "weight_sum")->write(y, identity_.width, p->weight_sum);
      files_.at(base + "n_eff")->write(y, identity_.width, p->n_eff);
      std::vector<float> row(identity_.width);
      for (int dy = 0; dy < rows; ++dy) {
        for (int x = 0; x < identity_.width; ++x)
          row[x] = p->support[static_cast<size_t>(dy) * identity_.width + x] ? 1.0f : 0.0f;
        files_.at(base + "support")->write(y + dy, identity_.width, row);
      }
    }
  }
  void write_alpha_map(const std::string &name, int y, int rows,
                       std::span<const float> values) {
    if (values.size() != static_cast<size_t>(rows) * identity_.width)
      throw std::runtime_error("DRIZZLE_STORE_INVALID_ALPHA_MAP");
    files_.at(name + "_X_value")->write(y, identity_.width, values);
  }
  void stripe(int y, const ForwardDrizzleUniformResult &uniform,
              const ForwardDrizzleUniformResult *raw = nullptr) {
    const bool raw_expected = identity_.mode == "uniform_raw_clipped" ||
                              is_multiband_mode(identity_.mode);
    if (y != next_y_ || (raw != nullptr) != raw_expected)
      throw std::runtime_error("DRIZZLE_STORE_STRIPE_ORDER");
    if (generation_.empty()) start();
    write_profile("uniform", y, uniform);
    if (raw) {
      if (raw->internal_height != uniform.internal_height)
        throw std::runtime_error("DRIZZLE_STORE_STRIPE_HEIGHT_MISMATCH");
      write_profile("raw", y, *raw);
    }
    next_y_ += uniform.internal_height;
  }
  // Multiband stripe: uniform + raw + fine + (medium) + the four channel-min
  // alpha-confidence maps, all covering the same `rows`.
  void multiband_stripe(int y, const ForwardDrizzleUniformAndRawResult &s) {
    if (!is_multiband_mode(identity_.mode))
      throw std::runtime_error("DRIZZLE_STORE_STRIPE_ORDER");
    if (y != next_y_) throw std::runtime_error("DRIZZLE_STORE_STRIPE_ORDER");
    if (generation_.empty()) start();
    const int rows = s.uniform.internal_height;
    if (rows <= 0 || s.raw.internal_height != rows ||
        s.fine.internal_height != rows ||
        (identity_.multiband_levels >= 2 && s.medium.internal_height != rows))
      throw std::runtime_error("DRIZZLE_STORE_STRIPE_HEIGHT_MISMATCH");
    write_profile("uniform", y, s.uniform);
    write_profile("raw", y, s.raw);
    write_profile("fine", y, s.fine);
    if (identity_.multiband_levels >= 2) write_profile("medium", y, s.medium);
    std::vector<float> sup(static_cast<size_t>(rows) * identity_.width);
    for (size_t i = 0; i < sup.size(); ++i)
      sup[i] = (i < s.alpha_confidence_support.size() &&
                s.alpha_confidence_support[i])
                   ? 1.0f
                   : 0.0f;
    write_alpha_map("alpha_separation", y, rows, s.a_separation);
    write_alpha_map("alpha_artifact", y, rows, s.a_artifact);
    write_alpha_map("alpha_registration", y, rows, s.a_registration);
    write_alpha_map("alpha_support", y, rows, sup);
    next_y_ += rows;
  }
  fs::path finish() {
    if (next_y_ != identity_.height) throw std::runtime_error("DRIZZLE_STORE_INCOMPLETE_STRIPES");
    for (auto &[name, file] : files_) file->finish();
    files_.clear();
    auto manifest = build_profile_store_manifest(identity_.mode, identity_.width,
        identity_.height, generation_, plane_names(identity_));
    json commit = {{"schema_version", 2}, {"generation", generation_.filename().string()},
                   {"identity", identity_json(identity_)},
                   {"planes", json::parse(serialize_profile_store_manifest(manifest))}};
    validate_generation(generation_, commit, identity_);
    commit["commit_hash"] = digest(commit);
    core::write_text_atomic(generation_ / "commit.json", commit.dump(2));
    // If rename succeeds but directory fsync fails, current.json may already
    // name this generation. Never delete it after a publication attempt.
    publish_attempted_ = true;
    core::write_text_atomic(root_ / "current.json", commit.dump(2));
    return generation_;
  }
};
size_t writer_reserve(const DrizzleStoreIdentity &identity) {
  return io_reserve + static_cast<size_t>(identity.width) * sizeof(float);
}
} // namespace

DrizzleStoreIdentity make_drizzle_store_identity(
    const registration::RegistrationSamplingPlan &plan,
    const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const config::ReconstructionClippingConfig *clipping, const std::vector<float> &g_eff,
    const DrizzleStorePredecessors &predecessors,
    const MultibandStoreContract &multiband) {
  const auto dims = plan_drizzle_memory(plan, cfg, 1);
  const OutputScaleMode osm{cfg.internal_scale, cfg.output_scale};
  if (!osm.valid())
    throw std::invalid_argument("DRIZZLE_STORE_INVALID_OUTPUT_SCALE");
  const bool mb = multiband.enabled;
  if (mb) {
    if (!clipping)
      throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_REQUIRES_CLIPPING");
    if (multiband.levels < 1 || multiband.levels > 4)
      throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_LEVELS_RANGE");
  }
  DrizzleStoreIdentity i;
  i.source_identity_hash = plan.source_identity_hash;
  i.normalized_cache_hash = predecessors.normalized_cache_hash;
  i.quality_plan_hash = predecessors.quality_plan_hash;
  i.sampling_plan_hash = registration::compute_plan_hash(plan);
  if (!plan.plan_hash.empty() && plan.plan_hash != i.sampling_plan_hash)
    throw std::invalid_argument("DRIZZLE_STORE_STALE_SAMPLING_HASH");
  i.mode = !clipping ? "uniform_unclipped"
           : mb      ? "uniform_raw_multiband_clipped"
                     : "uniform_raw_clipped";
  i.multiband_levels = mb ? multiband.levels : 0;
  // Stored geometry: internal, or halved once for the 2/1 area-average
  // (plan 12.1). 1/1 and 2/2 store at their internal resolution.
  i.width = osm.needs_2x2_downsample() ? dims.width / 2 : dims.width;
  i.height = osm.needs_2x2_downsample() ? dims.height / 2 : dims.height;
  i.color_mode = plan.color_mode;
  json algorithm = {{"version", 1}, {"mode", i.mode}, {"kernel", cfg.kernel},
      {"scale", cfg.internal_scale}, {"output_scale", cfg.output_scale},
      {"pixfrac", cfg.pixfrac},
      {"position_epsilon", subdivision.position_epsilon_internal_px},
      {"area_epsilon", subdivision.area_relative_epsilon},
      {"max_depth", subdivision.max_subdivision_depth},
      {"frame_error_limit", subdivision.per_frame_inversion_error_rate_max}};
  if (clipping) {
    algorithm["clipping"] = {{"min_contributors", cfg.min_clip_contributors},
      {"passes", cfg.robust_passes}, {"sigma_low", clipping->clip_sigma_low},
      {"sigma_high", clipping->clip_sigma_high}, {"min_fraction", clipping->min_fraction},
      {"min_n_eff", clipping->min_n_eff}, {"g_eff", g_eff}};
    // M5: only present when Raw actually consumed Q_composite, so a store
    // built without Q-maps keeps its previous reconstruction hash.
    if (!predecessors.source_quality_cache_hash.empty())
      algorithm["clipping"]["source_quality_cache_hash"] =
          predecessors.source_quality_cache_hash;
  } else if (!g_eff.empty()) throw std::invalid_argument("DRIZZLE_STORE_UNIFORM_HAS_QUALITY_WEIGHTS");
  // M6: multiband_config_hash content (plan 16.4) --- additive, only present
  // for a multiband store so uniform/uniform_raw stores keep their hash.
  if (mb) {
    const auto &a = multiband.alpha;
    const auto &g = multiband.guard;
    const auto &c = multiband.confidence;
    algorithm["multiband"] = {
        {"levels", multiband.levels},
        {"fine_quality_exponent", multiband.fine_quality_exponent},
        {"medium_quality_exponent", multiband.medium_quality_exponent},
        {"atrous_den_min", kAtrousDenMinFraction},
        {"atrous_version", kAtrousDecompositionVersion},
        {"alpha", {{"alpha_cap", a.alpha_cap},
                   {"min_effective_samples", a.min_effective_samples},
                   {"full_effective_samples", a.full_effective_samples}}},
        {"guard", {{"energy_limit", g.energy_limit},
                   {"bisection_iters", g.bisection_iters},
                   {"min_window_pixels", g.min_window_pixels}}},
        {"confidence",
         {{"min_quality_separation", c.min_quality_separation},
          {"full_quality_separation", c.full_quality_separation},
          {"min_artifact_contributors", c.min_artifact_contributors},
          {"direct_fraction_lo", c.direct_fraction_lo},
          {"direct_fraction_hi", c.direct_fraction_hi},
          {"residual_p20_lo", c.residual_p20_lo},
          {"residual_p20_hi", c.residual_p20_hi},
          {"artifact_lo", c.artifact_lo},
          {"artifact_hi", c.artifact_hi}}}};
  }
  i.reconstruction_hash = digest(algorithm);
  plane_names(i);
  return i;
}

DrizzleStoreResult persist_forward_drizzle_uniform(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of, const config::ReconstructionDrizzleConfig &cfg,
    const ForwardDrizzleSubdivisionParams &subdivision) {
  if (cfg.internal_scale == 2 && cfg.output_scale == 1)
    throw std::invalid_argument(
        "DRIZZLE_STORE_UNIFORM_ONLY_2_1_UNSUPPORTED: use "
        "persist_forward_drizzle_uniform_and_raw for the 2/1 output scale");
  const auto identity = make_drizzle_store_identity(plan, cfg, subdivision);
  StoreWriter writer(root, identity);
  DrizzleStoreResult result;
  result.diagnostics = stream_forward_drizzle_uniform(plan, source_of, cfg,
      [&](int y, const ForwardDrizzleUniformResult &stripe) { writer.stripe(y, stripe); },
      subdivision, writer_reserve(identity));
  result.generation_dir = writer.finish();
  result.identity = identity;
  return result;
}
DrizzleStoreResult persist_forward_drizzle_uniform_and_raw(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of, const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clipping,
    const ForwardDrizzleSubdivisionParams &subdivision, const std::vector<float> &g_eff,
    const DrizzleStorePredecessors &predecessors,
    const FrameQualityProvider &quality_of) {
  const auto identity = make_drizzle_store_identity(plan, cfg, subdivision, &clipping, g_eff, predecessors);
  StoreWriter writer(root, identity);
  DrizzleStoreResult result;
  const auto sink = [&](int y, const ForwardDrizzleUniformAndRawResult &stripe) {
    writer.stripe(y, stripe.uniform, &stripe.raw);
  };
  ForwardDrizzlePairDiagnostics summary;
  if (cfg.internal_scale == 2 && cfg.output_scale == 1) {
    // Plan 12.1 mode 2/1: the store receives stripes already area-averaged
    // to output (1x) resolution --- never a full internal-resolution image.
    summary = stream_forward_drizzle_uniform_and_raw_2x2(plan, source_of, cfg, clipping, sink,
                                                         subdivision, g_eff, writer_reserve(identity),
                                                         quality_of);
  } else {
    summary = stream_forward_drizzle_uniform_and_raw(plan, source_of, cfg, clipping, sink,
                                                     subdivision, g_eff, writer_reserve(identity),
                                                     quality_of);
  }
  result.diagnostics = summary.diagnostics;
  result.clipping = summary.clipping;
  result.generation_dir = writer.finish();
  result.identity = identity;
  return result;
}
DrizzleStoreResult persist_forward_drizzle_multiband(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &cfg,
    const config::ReconstructionClippingConfig &clipping,
    const MultibandStoreContract &multiband,
    const FrameQualityProvider &quality_of,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff,
    const DrizzleStorePredecessors &predecessors,
    const ForwardDrizzleCudaOptions &cuda) {
  if (!multiband.enabled)
    throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_NOT_ENABLED");
  if (!quality_of)
    throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_REQUIRES_QUALITY");
  const auto identity = make_drizzle_store_identity(plan, cfg, subdivision,
                                                   &clipping, g_eff,
                                                   predecessors, multiband);
  // Plan 19: the CUDA droplet/clipping/profile kernels are a later slice. When
  // the CUDA path is requested we either fail immediately, or --- for the
  // fault-injection restart test --- after `fault_after` committed stripes.
  const int fault_after =
      cuda.attempt ? forward_drizzle_cuda_fault_after_chunks() : -1;
  if (cuda.attempt && fault_after < 0)
    throw ForwardDrizzleCudaError(
        "forward_drizzle CUDA path not implemented (M7 slice 1: "
        "transactional-restart contract only)");
  StoreWriter writer(root, identity);

  MultibandProfileParams mb;
  mb.emit_fine = true;                        // D1 <- Fine, always
  mb.emit_medium = multiband.levels >= 2;     // D2 <- Medium
  mb.emit_alpha_confidence = true;
  mb.fine_quality_exponent = multiband.fine_quality_exponent;
  mb.medium_quality_exponent = multiband.medium_quality_exponent;
  mb.alpha_confidence = multiband.confidence;

  DrizzleStoreResult result;
  int committed_stripes = 0;
  const auto sink = [&](int y, const ForwardDrizzleUniformAndRawResult &stripe) {
    if (fault_after >= 0 && committed_stripes >= fault_after)
      throw ForwardDrizzleCudaError(
          "forward_drizzle CUDA: injected fault after " +
          std::to_string(fault_after) + " stripe(s)");
    writer.multiband_stripe(y, stripe);
    ++committed_stripes;
  };
  ForwardDrizzlePairDiagnostics summary;
  if (cfg.internal_scale == 2 && cfg.output_scale == 1) {
    // Plan 12.1 mode 2/1: stripes arrive already area-averaged to output (1x)
    // resolution --- fine/medium via the same 2x2 mean, the channel-min
    // confidence maps via 2x2 min + AND support (plan 14.4).
    summary = stream_forward_drizzle_uniform_and_raw_2x2(
        plan, source_of, cfg, clipping, sink, subdivision, g_eff,
        writer_reserve(identity), quality_of, mb);
  } else {
    summary = stream_forward_drizzle_uniform_and_raw(
        plan, source_of, cfg, clipping, sink, subdivision, g_eff,
        writer_reserve(identity), quality_of, mb);
  }
  result.diagnostics = summary.diagnostics;
  result.clipping = summary.clipping;
  result.generation_dir = writer.finish();
  result.identity = identity;
  return result;
}
DrizzleStoreValidation verify_drizzle_profile_store(
    const fs::path &root, const DrizzleStoreIdentity &expected) {
  DrizzleStoreValidation result;
  try {
    auto commit = read_small_json(root / "current.json");
    const auto hash = commit.at("commit_hash").get<std::string>();
    commit.erase("commit_hash");
    if (hash != digest(commit)) throw std::runtime_error("DRIZZLE_STORE_COMMIT_HASH_MISMATCH");
    const auto generation = commit.at("generation").get<std::string>();
    if (!generation.starts_with("generation-") || generation.size() > 128 ||
        generation.find_first_not_of("generation-0123456789") != std::string::npos)
      throw std::runtime_error("DRIZZLE_STORE_INVALID_GENERATION");
    const fs::path dir = root / generation;
    if (!fs::is_directory(fs::symlink_status(dir)))
      throw std::runtime_error("DRIZZLE_STORE_MISSING_GENERATION");
    auto saved_commit = read_small_json(dir / "commit.json");
    if (saved_commit.at("commit_hash") != hash)
      throw std::runtime_error("DRIZZLE_STORE_GENERATION_HASH_MISMATCH");
    saved_commit.erase("commit_hash");
    if (saved_commit != commit) throw std::runtime_error("DRIZZLE_STORE_GENERATION_COMMIT_MISMATCH");
    validate_generation(dir, commit, expected);
    result.generation_dir = dir;
    result.usable = true;
  } catch (const std::exception &e) { result.error = e.what(); }
  return result;
}
ProfilePlane read_drizzle_profile_region_preverified(
    const fs::path &generation_dir, const DrizzleStoreIdentity &expected,
    const std::string &profile, const std::string &channel,
    int x, int y, int width, int height, size_t memory_budget_mb) {
  if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
      x > expected.width || y > expected.height ||
      width > expected.width - x || height > expected.height - y)
    throw std::invalid_argument("DRIZZLE_STORE_INVALID_REGION");
  const uint64_t pixels = static_cast<uint64_t>(width) * height;
  const uint64_t overhead = io_reserve + static_cast<uint64_t>(width) * sizeof(float);
  if (memory_budget_mb > std::numeric_limits<size_t>::max() / (1024 * 1024) ||
      pixels > (std::numeric_limits<uint64_t>::max() - overhead) / 13 ||
      pixels * 13 + overhead > static_cast<uint64_t>(memory_budget_mb) * 1024 * 1024)
    throw std::runtime_error("DRIZZLE_STORE_REGION_MEMORY_BUDGET");
  const auto names = plane_names(expected);
  const std::string base = profile + "_" + channel + "_";
  if (!std::binary_search(names.begin(), names.end(), base + "value"))
    throw std::invalid_argument("DRIZZLE_STORE_UNKNOWN_PROFILE_CHANNEL");
  ProfilePlane result;
  result.allocate(width, height);
  auto read = [&](const std::string &field, std::vector<float> *values) {
    fitsfile *file = nullptr;
    int status = 0;
    fits_open_diskfile(&file, (generation_dir / (base + field + ".fits")).string().c_str(),
                       READONLY, &status);
    check(status);
    try {
      std::vector<float> row;
      if (!values) row.resize(width);
      for (int dy = 0; dy < height; ++dy) {
        long first[2] = {static_cast<long>(x) + 1, static_cast<long>(y) + dy + 1};
        float *dst = values ? values->data() + static_cast<size_t>(dy) * width : row.data();
        float null_value = std::numeric_limits<float>::quiet_NaN();
        int any_null = 0;
        fits_read_pix(file, TFLOAT, first, width, &null_value, dst, &any_null, &status);
        check(status);
        if (!values) for (int dx = 0; dx < width; ++dx) {
          if (dst[dx] != 0.0f && dst[dx] != 1.0f)
            throw std::runtime_error("DRIZZLE_STORE_INVALID_SUPPORT");
          result.support[static_cast<size_t>(dy) * width + dx] = dst[dx] == 1.0f;
        }
      }
      int close_status = 0;
      fits_close_file(file, &close_status);
      file = nullptr;
      check(close_status);
    } catch (...) {
      if (file) { int ignored = 0; fits_close_file(file, &ignored); }
      throw;
    }
  };
  read("value", &result.value);
  // Alpha-confidence pseudo-planes ("alpha_*_X") carry only a value field.
  const bool alpha_map =
      channel == "X" &&
      std::find(kAlphaMapNames.begin(), kAlphaMapNames.end(), profile) !=
          kAlphaMapNames.end();
  if (!alpha_map) {
    read("weight_sum", &result.weight_sum);
    read("n_eff", &result.n_eff);
    read("support", nullptr);
  } else {
    for (int k = 0; k < width * height; ++k)
      result.support[k] = std::isfinite(result.value[k]) ? 1u : 0u;
  }
  return result;
}

ProfilePlane read_drizzle_profile_region(
    const fs::path &root, const DrizzleStoreIdentity &expected,
    const std::string &profile, const std::string &channel,
    int x, int y, int width, int height, size_t memory_budget_mb) {
  const auto verified = verify_drizzle_profile_store(root, expected);
  if (!verified.usable) throw std::runtime_error(verified.error);
  return read_drizzle_profile_region_preverified(
      verified.generation_dir, expected, profile, channel, x, y, width, height,
      memory_budget_mb);
}

} // namespace tile_compile::reconstruction
