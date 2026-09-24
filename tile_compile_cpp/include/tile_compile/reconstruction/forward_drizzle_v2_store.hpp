#pragma once

// Forward-Drizzle-v2 Gate-7: versioned artifact schema, checkpoint
// identities and band-boundary resume with atomic commit. Implements the
// frozen contract in
// docs/forward_drizzle_v2_gate7_transaction_resume_spec_2026-09-12.json.
//
// Layout under a caller-chosen store root:
//
//   <root>/forward_drizzle_v2_generation-<unique>/
//       plan.json          binding run plan (written first, immutable)
//       band-%04d.bin      self-describing band artifact (64 B header +
//                          record_count * 64 B ForwardDrizzleV2PixelResult)
//       band-%04d.profiles.bin  profile payload (Gate-9/10, only when
//                          plan.emit_profiles): same header shape,
//                          record_count * 80 B ForwardDrizzleV2ProfileResult
//       checkpoint.json    committed band prefix, rewritten atomically
//                          after every band commit
//       commit.json        final marker, after full re-verification
//   <root>/current.json    pointer; written atomically as the LAST step
//
// Resume granularity (frozen decision): atomically committed band
// boundaries only. A partially written band is discarded and recomputed;
// reservoir/accumulator state is transient and never persisted. A phase
// event alone never establishes resumability --- only checkpoint.json
// content does. Unpublished generations are owned and removed by their
// writer; after a publication attempt they are never deleted.
//
// Not wired into the production runner.

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace tile_compile::reconstruction {

// Binding run plan (spec plan_binding_fields). Every identity a band
// result depends on is recorded; plan_hash binds the canonical
// serialization of all other fields, so two contexts produce different
// hashes iff any bound field differs.
// Plan estimator contracts. The default keeps the historical reservoir value;
// the pilot/full-frame contract streams the hash-selected pilot frames first,
// freezes their clip bounds and accumulates every accepted frame.
inline constexpr const char *kFdV2EstimatorReservoir = "reservoir_sigma_clip";
inline constexpr const char *kFdV2EstimatorPilotFullFrame =
    "reservoir_pilot_full_frame";

struct ForwardDrizzleV2RunPlan {
  static constexpr int kSchemaVersion = 1;
  int schema_version = kSchemaVersion;
  std::string pipeline_contract = "cfa_forward_drizzle_v2";
  int contract_version = 1;

  // Predecessor identities (opaque hashes supplied by the caller).
  std::string source_identity_hash;
  std::string normalized_cache_hash;
  std::string quality_plan_hash;
  std::string sampling_plan_hash;
  std::string config_snapshot_hash;
  // Gate-10: the source-quality map cache the profile quality streams were
  // folded from. Empty for stores that predate the field (the key is then
  // absent from plan.json, keeping those stores' plan_hash reproducible).
  std::string source_quality_cache_hash;

  // Frozen gate decisions.
  std::string enumeration = "dense_scatter";        // gate 1
  std::string estimator = "reservoir_sigma_clip";   // gate 3
  int reservoir_size = 64;
  std::uint64_t reservoir_seed = 0x9e3779b97f4a7c15ULL;
  int min_clip_contributors = 5;
  int min_candidates = 5;
  int robust_passes = 3;
  double sigma_low = 3.0;
  double sigma_high = 3.0;
  // See config::ReconstructionClippingConfig::shared_frame_rejection.
  // Default false: identical to the pre-existing independent-per-channel
  // clip. Implemented on both the CPU and CUDA forward-drizzle kernels.
  bool shared_frame_rejection = false;
  double shared_frame_rejection_consensus = 0.5;
  // See config::ReconstructionClippingConfig::bimodal_veto. Implemented on
  // both the CPU and CUDA forward-drizzle kernels.
  bool bimodal_veto = false;
  double bimodal_veto_gap_sigma = 2.5;
  std::string support_fold_contract = "gate2_v1";   // gate 2
  std::string numerics = "fp64_accumulators";       // gate 4
  bool sigma2_enabled = true;
  // "affine_only" | "smooth_local_coefficients" (gate 8)
  std::string local_warp_representation = "affine_only";
  bool trusted_run = true;

  // Gate-9/10 profiled mode: every band transaction also commits a
  // band-%04d.profiles.bin payload of ForwardDrizzleV2ProfileResult
  // records. multiband_levels binds the fusion level count (0 = no
  // multiband contract; still valid for uniform/raw-only production).
  bool emit_profiles = false;
  int multiband_levels = 0;
  // Profile weight exponents (plan 11.9): bound by plan_hash because they
  // change the committed profile records.
  float fine_quality_exponent = 4.0f;
  float medium_quality_exponent = 2.0f;

  // Geometry and band plan.
  int native_width = 0;
  int native_height = 0;
  int channels = 0;                 // 1 (MONO) or 3 (OSC)
  int internal_scale = 2;
  std::string color_mode;           // "MONO" | "OSC"
  double pixfrac = 0.8;
  int bayer_pattern = 0;
  int cfa_origin_x = 0;
  int cfa_origin_y = 0;
  std::uint64_t frame_count = 0;
  int band_rows = 0;                // core rows per band
  int band_count = 0;
  int tile_cols = 0;                // native cols per X-tile (0 = full width)
  int halo_rows = 0;
  bool x_tiled = false;

  std::string plan_hash;  // sha256 over the canonical serialization sans
                          // this field; filled by finalize_run_plan()
};

// Validates the bound fields (geometry, contract labels, band plan
// consistency) and computes plan_hash. Throws invalid_argument on a
// malformed plan. Call before begin()/inspect.
void finalize_forward_drizzle_v2_run_plan(ForwardDrizzleV2RunPlan &plan);

std::string serialize_forward_drizzle_v2_plan(
    const ForwardDrizzleV2RunPlan &plan);
bool parse_forward_drizzle_v2_plan(const std::string &text,
                                   ForwardDrizzleV2RunPlan &out,
                                   std::string &error);

// One committed band entry of checkpoint.json.
struct ForwardDrizzleV2BandCommit {
  int band_index = 0;
  int y_begin = 0;
  int rows = 0;
  int native_cols = 0;
  int channels = 0;
  std::uint64_t dense_overlap_count = 0;
  std::string artifact;        // "band-%04d.bin"
  std::uintmax_t bytes = 0;    // file size
  std::string sha256;          // payload content hash
  // Profile payload (present iff the run plan has emit_profiles):
  std::string profiles_artifact;      // "band-%04d.profiles.bin"
  std::uintmax_t profiles_bytes = 0;
  std::string profiles_sha256;
};

struct ForwardDrizzleV2Checkpoint {
  static constexpr int kSchemaVersion = 1;
  int schema_version = kSchemaVersion;
  std::string plan_hash;
  int band_count = 0;
  std::vector<ForwardDrizzleV2BandCommit> bands;  // contiguous prefix
  std::string checkpoint_hash;
};

std::string serialize_forward_drizzle_v2_checkpoint(
    const ForwardDrizzleV2Checkpoint &checkpoint);
bool parse_forward_drizzle_v2_checkpoint(const std::string &text,
                                         ForwardDrizzleV2Checkpoint &out,
                                         std::string &error);

// Commit gate evaluated by finish() (spec section 19: a successful commit
// with nonfinite pixels inside source support is forbidden).
struct ForwardDrizzleV2CommitGate {
  std::uint64_t nonfinite_pixels_inside_source_support = 0;  // must be 0
  std::uint64_t bands_processed = 0;   // must equal plan.band_count
  std::string telemetry_json;          // recorded verbatim in commit.json
};

// Band writer. begin() creates the unpublished generation and writes
// plan.json plus an empty checkpoint. commit_band() appends exactly the
// next expected band index in a contiguous y order: payload staged,
// fsynced, renamed, hashed, THEN checkpoint.json is updated atomically
// (the commit mark is always the last durable step). finish() re-verifies
// every artifact, evaluates the gate, writes commit.json and publishes
// current.json. The destructor removes the generation iff publish was
// never attempted.
class ForwardDrizzleV2StoreWriter {
 public:
  ForwardDrizzleV2StoreWriter(fs::path root, ForwardDrizzleV2RunPlan plan);
  ~ForwardDrizzleV2StoreWriter();
  ForwardDrizzleV2StoreWriter(const ForwardDrizzleV2StoreWriter &) = delete;
  ForwardDrizzleV2StoreWriter &operator=(
      const ForwardDrizzleV2StoreWriter &) = delete;

  // Creates (fresh) or adopts (resumed) the unpublished generation and
  // persists plan.json + checkpoint.json. Must be called before
  // commit_band().
  void begin();
  // Adopts the generation named by a resumable inspect state instead of
  // creating a new one; `resume_from_band` must equal the verified
  // committed prefix length.
  void adopt(const fs::path &generation, int resume_from_band,
             std::vector<ForwardDrizzleV2BandCommit> committed);

  // Appends band `band_index` covering native rows [y_begin, y_begin+rows).
  // `results` must hold rows*native_cols*channels records (channel-major).
  // When plan_.emit_profiles is set, `profiles` must hold the same count of
  // ForwardDrizzleV2ProfileResult records; it is committed inside the same
  // band transaction (durable before the checkpoint mark). When the plan
  // has no emit_profiles, `profiles` must be empty.
  // Enforces the contiguous prefix and the plan's band geometry.
  void commit_band(int band_index, int y_begin, int rows,
                   std::span<const ForwardDrizzleV2PixelResult> results,
                   std::span<const ForwardDrizzleV2ProfileResult> profiles,
                   std::uint64_t dense_overlap_count);

  // Verifies completeness, evaluates `gate`, writes commit.json and
  // publishes current.json. Throws on any violation; a thrown finish()
  // leaves the generation unpublished (destructor cleans it up).
  fs::path finish(const ForwardDrizzleV2CommitGate &gate);

  const ForwardDrizzleV2RunPlan &plan() const { return plan_; }
  const fs::path &generation() const { return generation_; }
  int committed_bands() const { return static_cast<int>(bands_.size()); }
  bool published() const { return publish_attempted_; }

 private:
  fs::path root_;
  ForwardDrizzleV2RunPlan plan_;
  fs::path generation_;
  std::vector<ForwardDrizzleV2BandCommit> bands_;
  int next_y_ = 0;
  bool begun_ = false;
  bool publish_attempted_ = false;
};

enum class ForwardDrizzleV2StoreStatus {
  fresh,      // no generation: caller starts a new run
  resumable,  // verified committed prefix; resume at next_band
  complete,   // current.json points at a fully committed generation
  corrupt     // fail closed; unbound or inconsistent state
};

struct ForwardDrizzleV2StoreInspection {
  ForwardDrizzleV2StoreStatus status = ForwardDrizzleV2StoreStatus::fresh;
  int next_band = 0;                       // resume point (resumable only)
  fs::path generation;                     // adopted dir (resumable) or
                                           // published dir (complete)
  std::vector<ForwardDrizzleV2BandCommit> committed;  // verified prefix
  std::string commit_hash;                 // complete only
  std::string error;                       // corrupt only
};

// Fail-closed store inspection (spec resume_semantics). Purely read-only:
// never deletes, never creates. `expected_plan` must already carry a valid
// plan_hash (finalize_forward_drizzle_v2_run_plan).
ForwardDrizzleV2StoreInspection inspect_forward_drizzle_v2_store(
    const fs::path &root, const ForwardDrizzleV2RunPlan &expected_plan);

// Store autodetection for consumers that arrive without the plan (the
// MULTIBAND phase): resolves <root>/current.json -> <generation>/plan.json
// and parses it. Returns false when no published store exists (out stays
// default); throws runtime_error on a present-but-malformed store. The
// returned plan is the caller's expected_plan for inspect().
bool load_forward_drizzle_v2_published_plan(
    const fs::path &root, ForwardDrizzleV2RunPlan &out, std::string &error);

// Reads one committed band artifact back (header validation + sha256
// re-check against `commit`). Throws runtime_error on any mismatch.
std::vector<ForwardDrizzleV2PixelResult> read_forward_drizzle_v2_band(
    const fs::path &generation, const ForwardDrizzleV2BandCommit &commit);

// Same for the band's profile payload. Throws runtime_error when the
// commit carries no profiles artifact.
std::vector<ForwardDrizzleV2ProfileResult>
read_forward_drizzle_v2_band_profiles(
    const fs::path &generation, const ForwardDrizzleV2BandCommit &commit);

}  // namespace tile_compile::reconstruction
