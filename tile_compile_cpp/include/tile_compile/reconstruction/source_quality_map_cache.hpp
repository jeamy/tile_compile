#pragma once

// Multi-stream source-space quality-map cache --- milestone M5 of the
// CFA-forward-drizzle plan (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_
// implementierungsplan_de.md, sections 13.3-13.5).
//
// Layout (plan 13.4):
//   cache/source_quality_maps/
//     metadata.json                        <- sole commit point
//     composite/source_quality_composite_NNNNNN.bin
//     scale_0/ .. scale_3/ source_quality_sK_NNNNNN.bin
//     artifact/source_quality_artifact_NNNNNN.bin
//
// Each .bin is a small header + row-major uint16 on the storage grid
// (source geometry downsampled by storage_divisor). Value semantics: a
// reserved code 0 = veto (Q=0 or unsupported); positive quality in (0,1] maps
// to [1,65535]. This guarantees an exact zero-veto never decodes positive and
// no nonzero code decodes to 0 (plan 13.5).

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace tile_compile::reconstruction {

namespace fs = std::filesystem;

// --- Quantization (plan 13.5) ------------------------------------------------
// 0.0f and any non-finite value -> 0 (veto). v in (0,1] -> max(1, round(v*
// 65535)). Decode: 0 -> NaN, q -> q/65535 (always > 0 for q >= 1).
uint16_t quantize_quality(float v);
float dequantize_quality(uint16_t q);

struct SourceQualityMapCacheConfig {
  int storage_divisor = 2;  // spatial: storage grid = ceil(source / divisor)
  std::string dtype = "uint16";
  int proxy_version = 1;
};

// Identity of the SOURCE content only (plan 13.4): ordered frame IDs +
// source_index, source dimensions, colour mode, Bayer pattern + CFA origin,
// and the normalized-cache manifest hash. Excludes registration, canvas
// geometry, internal_scale and output_scale, so a pure re-registration does
// not invalidate unchanged source Q-maps. This is deliberately NOT
// RegistrationSamplingPlan::source_identity_hash (which folds in config.sha256).
std::string compute_source_quality_identity_hash(
    const registration::RegistrationSamplingPlan &plan,
    const std::string &normalized_cache_hash);

// Config identity (plan 13.4): proxy version, pyramid/Q parameters, storage
// divisor and dtype.
std::string compute_scale_quality_config_hash(
    const config::AqmhPyramidConfig &pyramid,
    const SourceQualityMapCacheConfig &cache_cfg);

struct SourceQualityCacheFileEntry {
  std::string stream;
  std::size_t source_index = 0;
  std::string name;    // path relative to the cache root
  std::string sha256;
};

struct SourceQualityCacheMetadata {
  int schema_version = 1;
  std::string coordinate_space = "source_cfa";
  int source_width = 0;
  int source_height = 0;
  int storage_divisor = 2;
  std::string dtype = "uint16";
  std::string source_identity_hash;
  std::string normalized_cache_hash;
  std::string source_quality_config_hash;
  std::string source_quality_cache_hash;  // excluded from its own computation
  int proxy_version = 1;
  int cfa_origin_x = 0;
  int cfa_origin_y = 0;
  std::vector<std::string> streams;
  std::vector<SourceQualityCacheFileEntry> files;
};

// Writer. Per-.bin files are written atomically; metadata.json is written
// atomically last and is the sole commit point --- a crash before it leaves no
// usable cache.
class SourceQualityMapCacheWriter {
 public:
  SourceQualityMapCacheWriter(
      fs::path root, const registration::RegistrationSamplingPlan &plan,
      std::string normalized_cache_hash,
      const config::AqmhPyramidConfig &pyramid,
      SourceQualityMapCacheConfig cache_cfg = {});

  // Store one source-geometry map (values in [0,1], NaN/<=0 = veto) for one
  // frame into one stream ("composite", "scale_0".."scale_3", "artifact").
  // Downsample to the storage grid is conservative: a storage cell that
  // covers ANY veto source pixel is stored as veto (plan 13.5).
  //
  // Thread-safe (plan §30.72 R4): the downsample/quantize/write/sha256 work is
  // lock-free (each (stream, source_index) writes its own uniquely named file
  // via AtomicOutput); only the parent-dir creation and the file-list update
  // take the internal mutex. commit() sorts the list, so the committed bytes
  // and source_quality_cache_hash do not depend on the call order.
  void put(const std::string &stream, std::size_t source_index,
           const Matrix2Df &source_geom_map);

  // Writes metadata.json (sole commit point) and returns it.
  SourceQualityCacheMetadata commit();

  const std::string &identity_hash() const { return identity_hash_; }
  const std::string &config_hash() const { return config_hash_; }

 private:
  fs::path root_;
  int source_width_ = 0, source_height_ = 0;
  int cfa_origin_x_ = 0, cfa_origin_y_ = 0;
  std::string normalized_cache_hash_, identity_hash_, config_hash_;
  SourceQualityMapCacheConfig cfg_;
  std::vector<SourceQualityCacheFileEntry> files_;
  std::mutex files_mu_;  // guards files_ + parent-dir creation in put()
};

// Reader. Fail-closed: usable() is true only when metadata.json parses, its
// schema/identity/config hashes match what is expected, its declared
// source_quality_cache_hash recomputes, and every listed .bin file checksum
// still matches.
class SourceQualityMapCacheReader {
 public:
  SourceQualityMapCacheReader(fs::path root,
                              std::string expected_identity_hash,
                              std::string expected_config_hash);

  bool usable() const { return usable_; }
  const std::string &error() const { return error_; }
  const SourceQualityCacheMetadata &metadata() const { return meta_; }
  bool has(const std::string &stream, std::size_t source_index) const;

  // Full source-geometry map for one (stream, frame). Nearest-neighbour
  // upsample from the storage grid over the same partition the writer used;
  // any veto storage cell yields veto (NaN) for every source pixel it covers.
  Matrix2Df read_full(const std::string &stream,
                      std::size_t source_index) const;

  // Region read (plan 13.5): decode only source rows [y0, y1). Returns a
  // (y1 - y0) x source_width map with identical values to the corresponding
  // rows of read_full().
  Matrix2Df read_region(const std::string &stream, std::size_t source_index,
                        int y0, int y1) const;

  // Rectangle read (§30.81 step 3a-2): decode only the source rectangle
  // [y0, y1) x [x0, x1). Returns a (y1 - y0) x (x1 - x0) map whose element
  // (y - y0, x - x0) equals read_full()(y, x) for every (y, x) in the rect ---
  // the storage-grid cell is picked from the ABSOLUTE (y, x), so the
  // edge-clamp against the last storage row/column is unchanged. Ranges are
  // clamped to the source geometry; an empty rect returns Matrix2Df(0, 0).
  Matrix2Df read_rect(const std::string &stream, std::size_t source_index,
                      int y0, int y1, int x0, int x1) const;

  // §30.81 step 3a-2: I/O accounting. `bin_loads` counts .bin files decoded
  // (a non-empty read_rect still loads and decodes the WHOLE compact grid ---
  // only the float expansion is bounded; the seek-read is 3a-2b).
  // `expanded_floats` counts map elements actually materialised.
  std::uint64_t bin_loads() const { return bin_loads_.load(); }
  std::uint64_t expanded_floats() const { return expanded_floats_.load(); }
  void reset_io_counters() const {
    bin_loads_.store(0);
    expanded_floats_.store(0);
  }

 private:
  fs::path file_path(const std::string &stream,
                     std::size_t source_index) const;

  fs::path root_;
  bool usable_ = false;
  std::string error_;
  SourceQualityCacheMetadata meta_;
  mutable std::atomic<std::uint64_t> bin_loads_{0};
  mutable std::atomic<std::uint64_t> expanded_floats_{0};
};

struct SourceQualityMapsBuildResult {
  std::string source_identity_hash;
  std::string source_quality_config_hash;
  std::string source_quality_cache_hash;
  std::vector<std::string> streams;
  int frames = 0;
  int computed_scales = 0;
};

// Orchestrator for the SOURCE_QUALITY_MAPS phase. For every valid frame in the
// plan: load the normalized source, build the CFA-aware analysis proxy
// (compute_source_quality_proxy_v1), run compute_source_quality_maps() with a
// sink that streams each scale straight into the cache, and also store the
// composite and artifact_confidence streams. No more than one full
// source-geometry scale map is resident at a time (plan 13.3). metadata.json
// is the sole commit point.
// workers > 1 computes the per-frame proxy + quality maps concurrently, one
// VerifiedNormalizedSourceCache clone per worker (plan §30.72 O3). writer.put
// is serialised; metadata.json sorts its file list, so the committed store and
// source_quality_cache_hash are byte-identical to the 1-worker run.
SourceQualityMapsBuildResult build_source_quality_map_cache(
    const fs::path &cache_root,
    const registration::RegistrationSamplingPlan &plan,
    VerifiedNormalizedSourceCache &cache,
    const config::AqmhPyramidConfig &pyramid,
    SourceQualityMapCacheConfig cache_cfg = {}, int workers = 1);

}  // namespace tile_compile::reconstruction
