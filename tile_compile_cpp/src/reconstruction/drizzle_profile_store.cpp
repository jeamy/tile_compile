#include "tile_compile/reconstruction/drizzle_profile_store.hpp"

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/atrous_decomposition.hpp"
#include "tile_compile/reconstruction/forward_drizzle_contrib_list.hpp"
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
#include <optional>
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
    const FrameQualityProvider &quality_of, int workers) {
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
                                                         quality_of, {}, workers);
  } else {
    summary = stream_forward_drizzle_uniform_and_raw(plan, source_of, cfg, clipping, sink,
                                                     subdivision, g_eff, writer_reserve(identity),
                                                     quality_of, {}, workers);
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
    const FrameQualityRectProvider &quality_of,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::vector<float> &g_eff,
    const DrizzleStorePredecessors &predecessors,
    const ForwardDrizzleCudaOptions &cuda, int workers) {
  if (!multiband.enabled)
    throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_NOT_ENABLED");
  if (!quality_of)
    throw std::invalid_argument("DRIZZLE_STORE_MULTIBAND_REQUIRES_QUALITY");
  // §30.81 step 3a-2: the CPU streaming sub-paths take a full-source provider;
  // adapt the rect provider back (full extent) for them. Only the CUDA stripe
  // path uses the real per-tile rectangle.
  const FrameQualityProvider quality_full =
      [&quality_of](std::size_t si) { return quality_of(si, -1, -1, -1, -1); };
  const auto identity = make_drizzle_store_identity(plan, cfg, subdivision,
                                                   &clipping, g_eff,
                                                   predecessors, multiband);
  // Plan 19.4 / 19.6.1: the CUDA path drives the internal canvas in
  // device-sized bands, each one a deterministic per-frame contribution-list
  // pass (accumulate_pair_by_frame_cuda). It handles AFFINE frame sets only,
  // outside mode 2/1, with a usable device. A local-warp frame (transcendental
  // Gauss basis, §19.6.1), mode 2/1, or no device makes the whole store fall
  // back to the CPU reference path; the reason lands in `cuda_fallback_reason`.
  // The `fault_after >= 0` branch is the slice-1 restart-contract test: it
  // still runs the CPU streaming path with a sink that throws after N stripes.
  const int fault_after =
      cuda.attempt ? forward_drizzle_cuda_fault_after_chunks() : -1;
  const bool mode_2_1 = cfg.internal_scale == 2 && cfg.output_scale == 1;
  const bool cuda_stripe_path = cuda.attempt && fault_after < 0;
  CudaDeviceMemory devmem;
  if (cuda_stripe_path) {
    // Mode 2/1 is NO LONGER declined: the device produces internal-2x stripes
    // exactly as it does for 2/2 (real M31 2/2 run, §30.55); a
    // Downsample2x2StripeAdapter folds each band 2x2 -> 1x on the host with the
    // plan-12.1 operator before the StoreWriter --- byte-identical to the CPU
    // mode-2/1 build. Local-warp frames take the plan-19.6.2 hybrid path inside
    // accumulate_pair_by_frame_cuda. Only "no device" still declines here.
    devmem = forward_drizzle_cuda_device_memory();
    if (devmem.free_bytes == 0)
      throw ForwardDrizzleCudaError("forward_drizzle CUDA: no usable device");
  }
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
  // §4.4: prepare_drizzle_frames() is band-invariant — build it ONCE before
  // the band loop and pass it through to every accumulate_pair_by_frame_cuda
  // call, instead of rebuilding it per band inside accumulate_pair_impl.
  const auto prepared_frames = cuda_stripe_path
      ? std::optional<PreparedDrizzleFrames>(
            prepare_drizzle_frames(plan, cfg, subdivision))
      : std::nullopt;
  if (cuda_stripe_path) {
    // NB: `workers` (P3 Teil 2 CPU-reduction band parallelism) does NOT apply
    // here --- this path has its own device-sized band chunking. A
    // ForwardDrizzleCudaError restarts the whole build on the CPU branch below,
    // which does honour `workers`.
    // Device-sized bands over the internal canvas; each band is one
    // accumulate_pair_by_frame_cuda pass feeding the SAME multiband stripe
    // sink. Bit-identical to the CPU streaming build (verified store-level by
    // the [drizzle-store][cuda-parity] test) --- §19.6's chunk-height
    // invariance covers the differing band boundaries.
    const auto dims = plan_drizzle_memory(plan, cfg, 1);
    const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
    const std::size_t frame_count = plan.frames.size();
    // Working-set estimate per internal output row, used ONLY to pick the
    // initial band height. It is deliberately the SUM of the host and device
    // terms (not the max): the host flat ClipCandidate buffer
    // (channels * W * frames), the device per-frame contribution vector
    // (~ source_width * a few cells), the 8 double stripe accumulators per
    // channel. At real geometry the host term dominates, so the resolved band
    // is smaller than raw VRAM would allow --- conservative on purpose.
    const std::size_t cand_row = static_cast<std::size_t>(channels) *
                                 dims.width * frame_count * sizeof(ClipCandidate);
    const std::size_t rec_row = static_cast<std::size_t>(plan.source_width) *
                                16 * sizeof(CudaDrizzleContribRecord);
    const std::size_t acc_row = static_cast<std::size_t>(channels) * dims.width *
                                8 * sizeof(double);
    const std::size_t bytes_per_row = cand_row + rec_row + acc_row;
    // The per-band host buffers are bounded by an ABSOLUTE ceiling (the explicit
    // drizzle budget, else 2 GiB). A band too tall for it throws
    // DRIZZLE_CONTRIB_LIST_BUDGET and run_cuda_chunked's halving ladder makes
    // real progress (plan 19.4).
    const std::size_t host_budget =
        cfg.memory_budget_mb
            ? static_cast<std::size_t>(cfg.memory_budget_mb) << 20
            : (static_cast<std::size_t>(2) << 30);
    // §30.81 step 4 + step 3a: the band height and the per-band column tile
    // width are derived JOINTLY from the one `host_budget` ceiling so the
    // regions that are live SIMULTANEOUSLY during the tiled reduction sum to
    // under it. Before step 4 the memo cap and the tile-width cap each divided
    // the FULL `host_budget` and the full-width reassembly `stripe` was counted
    // nowhere. Step 3a then removed the per-band record memo entirely --- its
    // `frame_count * source_width * cells` per-row term was what dragged the
    // band into the §30.80 collapse-with-N.
    //
    // Per internal output row, the host regions now are:
    //   frame_rec_row  one frame's contribution records for the band (produced
    //                  per column tile, freed after the fold) --- N-INDEPENDENT
    //                  (no frame_count factor). kMemoCellsEst is a generous
    //                  records-per-source-pixel estimate (typical 1--4, kernel
    //                  capacity 32); counted per INTERNAL row while the records
    //                  are per source row, so at internal_scale 2 it is ~2x
    //                  conservative. If the real count exceeds it,
    //                  accumulate_pair_impl throws DRIZZLE_CONTRIB_LIST_BUDGET
    //                  and run_cuda_chunked's halving ladder shortens the band.
    //   stripe_row     the full-width band reassembly `stripe` (uniform + raw +
    //                  optional fine/medium planes + optional alpha), 13 B/px.
    //   cand/col       one column tile's flat ClipCandidate buffer + counts +
    //                  the tile result `r` (width tile_w, <= tile_w/W of stripe).
    // Plus ~constant: source image + quality maps (single frame slot).
    constexpr std::size_t kMemoCellsEst = 6;
    constexpr std::size_t kPlanePxBytes = 3 * sizeof(float) + sizeof(std::uint8_t);
    const int n_result_planes =
        2 + (mb.emit_fine ? 1 : 0) + (mb.emit_medium ? 1 : 0);
    const std::size_t plane_px_bytes =
        static_cast<std::size_t>(n_result_planes) *
            static_cast<std::size_t>(channels) * kPlanePxBytes +
        (mb.emit_alpha_confidence ? kPlanePxBytes : 0);
    const std::size_t frame_rec_row =
        static_cast<std::size_t>(plan.source_width) * kMemoCellsEst *
        sizeof(DrizzleContrib);
    const std::size_t stripe_row =
        plane_px_bytes * static_cast<std::size_t>(dims.width);
    const int q_map_count =
        quality_of ? 1 + (mb.emit_fine ? 1 : 0) + (mb.emit_medium ? 1 : 0) +
                         (mb.emit_alpha_confidence ? 1 : 0)
                   : 0;
    // §30.81 step 3a-2: the Q maps are now per-tile source RECTANGLES, not
    // full-source. Their bytes scale with tile_w -> fold into cand_per_row_col
    // (<= q_map_count floats per INTERNAL cell touched; source rect is smaller
    // at internal_scale > 1, so conservative). src_const keeps only the two
    // resident source frames (full, until the block-index cut).
    const std::size_t cand_per_row_col =
        static_cast<std::size_t>(channels) * frame_count * sizeof(ClipCandidate) +
        static_cast<std::size_t>(channels) * sizeof(std::size_t) + plane_px_bytes +
        static_cast<std::size_t>(q_map_count) * sizeof(float);
    const std::size_t src_const = static_cast<std::size_t>(2) *
                                    static_cast<std::size_t>(plan.source_width) *
                                    static_cast<std::size_t>(plan.source_height) *
                                    sizeof(float);
    // Narrower-than-min column tiles are rejected (edge overlap + per-call cost,
    // §30.81 step 4): the band is shrunk until a >= kMinTileW tile fits. Capped
    // at the canvas width --- for a canvas narrower than the floor there is no
    // tiling to be had and only the band height can move.
    const int kMinTileW = std::min(64, dims.width);
    const std::size_t host_fixed_per_row = frame_rec_row + stripe_row;
    const std::size_t host_avail =
        host_budget > src_const ? host_budget - src_const : 0;
    const std::size_t host_row_cost =
        host_fixed_per_row +
        cand_per_row_col * static_cast<std::size_t>(kMinTileW);
    const int host_rows =
        (host_avail && host_row_cost)
            ? static_cast<int>(std::clamp<std::size_t>(
                  host_avail / host_row_cost, 1,
                  static_cast<std::size_t>(dims.height)))
            : dims.height;
    const int requested_rows =
        cfg.chunk_rows > 0 ? std::min(cfg.chunk_rows, host_rows) : host_rows;
    const std::size_t device_bytes_per_row = rec_row + acc_row;
    const auto chunk_plan = plan_cuda_chunking(
        devmem.free_bytes, device_bytes_per_row, dims.height, requested_rows);
    if (!chunk_plan.feasible)
      throw ForwardDrizzleCudaError(
          "forward_drizzle CUDA: device memory below one internal row");
    int hybrid_local_frames = 0;
    for (const auto &f : plan.frames)
      if (f.has_smooth_local_model) ++hybrid_local_frames;
    result.cuda_timing.used = true;
    result.cuda_timing.hybrid_local_frames = hybrid_local_frames;
    result.cuda_timing.resolved_chunk_rows = chunk_plan.chunk_rows;
    result.cuda_timing.min_chunk_rows = chunk_plan.min_chunk_rows;
    result.cuda_timing.bytes_per_row = bytes_per_row;
    result.cuda_timing.device_free_bytes = devmem.free_bytes;
    // §30.80: the three-way split + host ceiling, for the priority-3 budget
    // analysis (cand_row dominates bytes_per_row at real geometry).
    result.cuda_timing.cand_row_bytes = cand_row;
    result.cuda_timing.rec_row_bytes = rec_row;
    result.cuda_timing.acc_row_bytes = acc_row;
    result.cuda_timing.host_budget_bytes = host_budget;
    using store_clock = std::chrono::steady_clock;
    const auto t_all0 = store_clock::now();
    ForwardDrizzleClippingDiagnostics clip_total;
    ForwardDrizzleDiagnostics last_diag;
    HybridPathStats hybrid_stats;
    // T4c: aggregate source/Q band cache stats across all bands.
    long long tot_q_hits = 0, tot_q_misses = 0, tot_q_bytes = 0;
    long long tot_src_hits = 0, tot_src_misses = 0, tot_src_bytes = 0;
    // Mode 2/1: the device bands are internal-2x; fold them 2x2 -> 1x on the
    // host (plan 12.1) before `sink`, exactly as the CPU mode-2/1 path does.
    // Exactly one adapter per build --- a ForwardDrizzleCudaError discards this
    // generation and persist_multiband_store_from_predecessors restarts on a
    // fresh CPU path with its own adapter, never nested.
    std::unique_ptr<Downsample2x2StripeAdapter> down;
    if (mode_2_1)
      down = std::make_unique<Downsample2x2StripeAdapter>(
          [&](int y, const ForwardDrizzleUniformAndRawResult &o) { sink(y, o); },
          dims.width, plan.color_mode == ColorMode::MONO);
    int first_tile_w = 0, min_tile_w = dims.width, max_tiles_per_band = 0;
    CudaChunkRunStats chunk_stats;
    const int bands = run_cuda_chunked(
        chunk_plan, dims.height, [&](int y0, int rows) {
          // §30.81 step 4 + 3a: widest column tile such that, at THIS band
          // height,
          //   (frame_rec_row + stripe_row) * rows    [1 frame records + reassembly]
          // + cand_per_row_col * tile_w * rows        [one tile + its result]
          // + src_const                              [two resident source frames]
          // stays under `host_budget`. Derived per call so a halving that
          // shrinks `rows` re-enters here and recomputes it --- never stale.
          const std::size_t rows_z = static_cast<std::size_t>(std::max(rows, 1));
          const std::size_t fixed = host_fixed_per_row * rows_z + src_const;
          const std::size_t for_cand =
              host_budget > fixed ? host_budget - fixed : 0;
          const std::size_t denom = cand_per_row_col * rows_z;
          int tile_w = (for_cand && denom)
                           ? static_cast<int>(std::min<std::size_t>(
                                 for_cand / denom,
                                 static_cast<std::size_t>(dims.width)))
                           : (for_cand ? dims.width : 1);
          tile_w = std::clamp(tile_w, 1, dims.width);
          const int tiles =
              (dims.width + tile_w - 1) / std::max(tile_w, 1);
          if (first_tile_w == 0) first_tile_w = tile_w;
          min_tile_w = std::min(min_tile_w, tile_w);
          max_tiles_per_band = std::max(max_tiles_per_band, tiles);

          ForwardDrizzleUniformAndRawResult stripe;
          auto alloc_plane = [&](ForwardDrizzleUniformResult &p, bool on) {
            p.color_mode = plan.color_mode;
            p.internal_width = dims.width;
            p.internal_height = on ? rows : 0;
            if (!on) return;
            if (plan.color_mode == ColorMode::MONO) {
              p.L.allocate(dims.width, rows);
            } else {
              p.R.allocate(dims.width, rows);
              p.G.allocate(dims.width, rows);
              p.B.allocate(dims.width, rows);
            }
          };
          alloc_plane(stripe.uniform, true);
          alloc_plane(stripe.raw, true);
          alloc_plane(stripe.fine, mb.emit_fine);
          alloc_plane(stripe.medium, mb.emit_medium);
          const std::size_t full_n =
              static_cast<std::size_t>(dims.width) * static_cast<std::size_t>(rows);
          if (mb.emit_alpha_confidence) {
            stripe.a_separation.assign(full_n,
                                       std::numeric_limits<float>::quiet_NaN());
            stripe.a_artifact.assign(full_n,
                                     std::numeric_limits<float>::quiet_NaN());
            stripe.a_registration.assign(
                full_n, std::numeric_limits<float>::quiet_NaN());
            stripe.alpha_confidence_support.assign(full_n, 0u);
          }

          const auto t0 = store_clock::now();
          // §30.81 step-5 (B): a single call produces + sorts each frame ONCE
          // per band, then reduces `tile_w`-wide column tiles from that per-band
          // memo. Each reduced tile is blitted into the full-width band stripe
          // here. This removes the per-tile produce + sort repetition (the
          // baseline in §30.81 step 5 showed the sort scaled exactly with the
          // tile count).
          auto blit = [&](ForwardDrizzleUniformResult &dst,
                          const ForwardDrizzleUniformResult &s, int xb, int tw) {
            auto planes = [&](ForwardDrizzleUniformResult &r) {
              return plan.color_mode == ColorMode::MONO
                         ? std::array<ProfilePlane *, 3>{&r.L, nullptr, nullptr}
                         : std::array<ProfilePlane *, 3>{&r.R, &r.G, &r.B};
            };
            const auto dp = planes(dst);
            auto sp_planes = [&](const ForwardDrizzleUniformResult &r) {
              return plan.color_mode == ColorMode::MONO
                         ? std::array<const ProfilePlane *, 3>{&r.L, nullptr,
                                                               nullptr}
                         : std::array<const ProfilePlane *, 3>{&r.R, &r.G, &r.B};
            };
            const auto sp = sp_planes(s);
            for (int c = 0; c < channels; ++c) {
              if (!dp[c] || sp[c]->value.empty()) continue;
              for (int ry = 0; ry < rows; ++ry)
                for (int rx = 0; rx < tw; ++rx) {
                  const std::size_t si = static_cast<std::size_t>(ry) * tw + rx;
                  const std::size_t di =
                      static_cast<std::size_t>(ry) * dims.width + (xb + rx);
                  dp[c]->value[di] = sp[c]->value[si];
                  dp[c]->weight_sum[di] = sp[c]->weight_sum[si];
                  dp[c]->n_eff[di] = sp[c]->n_eff[si];
                  dp[c]->support[di] = sp[c]->support[si];
                }
            }
          };
          PairTileSink tile_sink = [&](int xb, int tw,
                                       const ForwardDrizzleUniformAndRawResult
                                           &part) {
            blit(stripe.uniform, part.uniform, xb, tw);
            blit(stripe.raw, part.raw, xb, tw);
            if (mb.emit_fine) blit(stripe.fine, part.fine, xb, tw);
            if (mb.emit_medium) blit(stripe.medium, part.medium, xb, tw);
            if (mb.emit_alpha_confidence && !part.a_separation.empty()) {
              for (int ry = 0; ry < rows; ++ry)
                for (int rx = 0; rx < tw; ++rx) {
                  const std::size_t si = static_cast<std::size_t>(ry) * tw + rx;
                  const std::size_t di =
                      static_cast<std::size_t>(ry) * dims.width + (xb + rx);
                  stripe.a_separation[di] = part.a_separation[si];
                  stripe.a_artifact[di] = part.a_artifact[si];
                  stripe.a_registration[di] = part.a_registration[si];
                  stripe.alpha_confidence_support[di] =
                      part.alpha_confidence_support[si];
                }
            }
          };
          // §30.81 step 4 + 3a: the budget passed INWARD is what remains after
          // the store-owned full-width `stripe`, one frame's record scratch and
          // the ~constant source/Q buffers --- so accumulate_pair_impl's
          // per-tile candidate guard checks against what is actually free, not
          // the full ceiling (before step 4 both the inner guard and this call
          // used the full `host_budget`, so candidates + stripe could exceed
          // it). Floored at 64 MiB: too tight for one producer batch -> throws
          // ForwardDrizzleCudaError -> full CPU restart, not the halving ladder.
          const std::size_t fixed_bytes =
              host_fixed_per_row * static_cast<std::size_t>(std::max(rows, 1)) +
              src_const;
          const std::size_t inner_budget = std::max<std::size_t>(
              host_budget > fixed_bytes ? host_budget - fixed_bytes : 0,
              static_cast<std::size_t>(64) << 20);
          ForwardDrizzleUniformAndRawResult agg;
          try {
            // The literals mirror the accumulate_pair_by_frame_cuda header
            // defaults; spelled out to reach the trailing &hybrid_stats and the
            // §30.81 tile-sink params. On the hybrid path a batch that cannot
            // fit even at the floor throws ForwardDrizzleCudaError -> full CPU
            // restart (plan 19.6.2), NOT CudaAllocFailure -> band halving.
            agg = accumulate_pair_by_frame_cuda(
                plan, source_of, cfg, clipping, y0, rows, subdivision, g_eff,
                quality_of, mb, inner_budget, /*max_cells_per_pixel=*/32,
                /*max_batch_items=*/static_cast<std::size_t>(1) << 20,
                &hybrid_stats, /*target_x_begin=*/0, /*target_cols=*/-1,
                &tile_sink, tile_w,
                prepared_frames ? &*prepared_frames : nullptr);
          } catch (const std::runtime_error &e) {
            // The per-band record memo or a tile's candidate buffer exceeded the
            // host ceiling -> ask run_cuda_chunked for a shorter band (plan 19.4
            // halving ladder); with `tile_w` and the memo-limited band height
            // derived above this is the safety net.
            const std::string what = e.what();
            if (what.find("DRIZZLE_CONTRIB_LIST_BUDGET") != std::string::npos)
              throw CudaAllocFailure(what);
            throw;
          }
          stripe.clipping = agg.clipping;  // aggregated over tiles by the driver
          last_diag = agg.diagnostics;
          // T4c: accumulate per-band cache stats.
          tot_q_hits += agg.diagnostics.q_band_cache_hits;
          tot_q_misses += agg.diagnostics.q_band_cache_misses;
          tot_q_bytes += agg.diagnostics.q_bytes_read;
          tot_src_hits += agg.diagnostics.source_band_cache_hits;
          tot_src_misses += agg.diagnostics.source_band_cache_misses;
          tot_src_bytes += agg.diagnostics.source_bytes_read;
          result.cuda_timing.stripe_seconds +=
              std::chrono::duration<double>(store_clock::now() - t0).count();
          if (down) {
            // The row-buffered fold desynchronises silently if a band's plane
            // set differs from band 0's --- assert full-height planes first.
            if (stripe.uniform.internal_height != rows ||
                (mb.emit_fine && stripe.fine.internal_height != rows) ||
                (mb.emit_medium && stripe.medium.internal_height != rows))
              throw std::runtime_error(
                  "DRIZZLE_STORE_CUDA_BAND_PLANE_HEIGHT_MISMATCH");
            down->feed(y0, stripe);
          } else {
            sink(y0, stripe);
          }
          clip_total.pixel_channel_evaluations +=
              stripe.clipping.pixel_channel_evaluations;
          clip_total.pixel_channel_rejected +=
              stripe.clipping.pixel_channel_rejected;
          clip_total.candidate_contributions_clipped +=
              stripe.clipping.candidate_contributions_clipped;
          // `last_diag` is set per column tile inside the loop above (all tiles
          // of a band carry the same frame-level diagnostics).
        }, &chunk_stats);
    if (down) down->finish();  // asserts the total internal height was even
    result.cuda_timing.bands = bands;
    result.cuda_timing.band_halvings = chunk_stats.halvings;
    result.cuda_timing.min_band_rows = chunk_stats.min_band_rows;
    result.cuda_timing.max_band_rows = chunk_stats.max_band_rows;
    result.cuda_timing.resolved_tile_w = first_tile_w;
    result.cuda_timing.min_tile_w = min_tile_w;
    result.cuda_timing.max_tiles_per_band = max_tiles_per_band;
    result.cuda_timing.total_seconds =
        std::chrono::duration<double>(store_clock::now() - t_all0).count();
    result.cuda_timing.hybrid_cpu_seconds = hybrid_stats.cpu_seconds;
    result.cuda_timing.hybrid_gpu_raster_seconds =
        hybrid_stats.gpu_raster_seconds;
    result.cuda_timing.hybrid_leaf_cells = hybrid_stats.leaf_cells;
    summary.diagnostics = last_diag;
    // T4c: replace last-band values with run-wide aggregates.
    summary.diagnostics.q_band_cache_hits = tot_q_hits;
    summary.diagnostics.q_band_cache_misses = tot_q_misses;
    summary.diagnostics.q_bytes_read = tot_q_bytes;
    summary.diagnostics.source_band_cache_hits = tot_src_hits;
    summary.diagnostics.source_band_cache_misses = tot_src_misses;
    summary.diagnostics.source_bytes_read = tot_src_bytes;
    // Keep the FORWARD_DRIZZLE diagnostics the runner emits populated on the
    // CUDA path (they otherwise come from the streaming planner, which did not
    // run here).
    summary.diagnostics.resolved_chunk_rows = chunk_plan.chunk_rows;
    // §30.81 step 4: real peak = device band buffers + the host regions that
    // are live together (memo + full-width reassembly + one column tile + its
    // result + source/Q), each modelled at the resolved band height / tile
    // width rather than assuming the whole `host_budget` is consumed.
    {
      const std::size_t band_z =
          static_cast<std::size_t>(std::max(chunk_plan.chunk_rows, 1));
      const std::size_t tile_z =
          static_cast<std::size_t>(std::max(first_tile_w, 1));
      summary.diagnostics.estimated_peak_bytes =
          device_bytes_per_row * band_z + host_fixed_per_row * band_z +
          cand_per_row_col * tile_z * band_z + src_const;
    }
    summary.clipping = clip_total;
  } else if (mode_2_1) {
    // Plan 12.1 mode 2/1: stripes arrive already area-averaged to output (1x)
    // resolution --- fine/medium via the same 2x2 mean, the channel-min
    // confidence maps via 2x2 min + AND support (plan 14.4).
    summary = stream_forward_drizzle_uniform_and_raw_2x2(
        plan, source_of, cfg, clipping, sink, subdivision, g_eff,
        writer_reserve(identity), quality_full, mb, workers);
  } else {
    summary = stream_forward_drizzle_uniform_and_raw(
        plan, source_of, cfg, clipping, sink, subdivision, g_eff,
        writer_reserve(identity), quality_full, mb, workers);
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
