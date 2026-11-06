#pragma once

#include "tile_compile/registration/registration_sampling_plan.hpp"
#include "tile_compile/astrometry/astap_paths.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/config/path_check.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/image/hypermetric_stretch.hpp"

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string>
#include <thread>
#include <atomic>
#include <vector>

namespace tile_compile::io {
struct FitsHeader;
struct RGBImage;
}

namespace tile_compile::runner {

/// Result of the native near-solve that matches detected image stars against
/// the installed Siril-format Gaia DR3 catalogue.  This is deliberately a
/// local fallback: it needs valid approximate pointing and optical metadata,
/// but never starts Siril or makes a network request.
struct LocalGaiaPlateSolveResult {
  bool success = false;
  astrometry::WCS wcs;
  int image_stars = 0;
  int catalog_stars = 0;
  int inlier_stars = 0;
  bool reflected_solution = false;
  std::string error_message;
};

/// Solve an RGB FITS image against the locally installed Gaia DR3 catalogue.
/// The header must provide RA, DEC, FOCALLEN and XPIXSZ/YPIXSZ.  The returned
/// WCS uses the regular FITS TAN/CD convention and is safe to persist in the
/// same way as an ASTAP WCS result.
LocalGaiaPlateSolveResult solve_with_local_gaia_catalog(
    const io::RGBImage &rgb);

/// Persist a WCS as a FITS-card compatible sidecar that parse_wcs_file can
/// read.  Used for native solutions because no external solver created a WCS
/// file to copy into the run artifacts.
bool write_wcs_sidecar(const astrometry::WCS &wcs,
                       const std::filesystem::path &path);

/// Format a byte count for human-readable logs and diagnostics.
std::string format_bytes(uint64_t bytes);

/// Sum file sizes for input-frame disk-space planning.
uint64_t estimate_total_file_bytes(const std::vector<std::filesystem::path> &paths);

/// Workload class used to tune runner worker counts for CPU, IO, or mixed phases.
enum class WorkerParallelProfile {
  MixedIo,
  IoHeavy,
};

/// Choose a conservative worker count for the current phase and machine limits.
///
/// Combines `runtime_limits.parallel_workers`, memory budget, workload type,
/// and input-frame sizes. CPU-heavy phases can safely use more workers; IO-heavy
/// phases are throttled to reduce disk pressure.
int compute_adaptive_worker_count(
    const config::Config &cfg, size_t task_count,
    const std::vector<std::filesystem::path> &frames,
    WorkerParallelProfile profile);

/// Temporarily limits OpenCV's global worker pool while application-level
/// workers run. Construction and destruction must happen outside all worker
/// threads because cv::setNumThreads() is process-global and not thread-safe.
class ScopedOpenCvThreadLimit {
public:
  explicit ScopedOpenCvThreadLimit(int outer_workers) noexcept;
  ~ScopedOpenCvThreadLimit() noexcept;

  ScopedOpenCvThreadLimit(const ScopedOpenCvThreadLimit &) = delete;
  ScopedOpenCvThreadLimit &operator=(const ScopedOpenCvThreadLimit &) = delete;

private:
  int previous_threads_ = 0;
  bool changed_ = false;
};

/// Determine default parallel workers for CPU bound tasks without memory capping.
int default_parallel_workers(size_t items, int requested_workers = 0);

/// Platform-aware shell quoting for external commands.
std::string shell_quote(const std::string &s);

/// Convert config::HyperMetricStretchConfig to image::HyperMetricStretchConfig
/// (an alias of the same struct; kept as the single conversion point).
image::HyperMetricStretchConfig to_image_hms_config(
    const config::HyperMetricStretchConfig &src);

/// Wrap a command for execution via std::system (cmd /c "..." on Windows).
std::string system_cmd(const std::string &cmd);

// ASTAP path resolution lives in the library so `tile_compile_cli
// validate-config` and the runner share one implementation.
using astrometry::AstapPaths;
using astrometry::resolve_astap_binary_path;
using astrometry::resolve_astap_paths;

/// Check the configured paths before any work starts. Every finding is
/// emitted as a `warning` event (GUI run-warning banner) and written to
/// `artifacts/path_check.json` (only when `write_artifact`; a resume must not
/// rewrite the original run's artifacts). When a path whose data the run needs is
/// missing and no standard location replaces it, emits `run_end` with
/// status "error" and returns false; the caller must then stop.
bool run_path_preflight(const std::string &run_id, const config::Config &cfg,
                        const std::filesystem::path &project_root,
                        const config::PathCheckScope &scope,
                        const std::filesystem::path &run_dir,
                        core::EventEmitter &emitter, std::ostream &log,
                        bool write_artifact = true);

/// Emit already computed path findings the same way (warnings, optional
/// artifact at `artifact_path`, `run_end` error on a blocking finding).
bool emit_path_preflight(const std::string &run_id,
                         const std::vector<config::PathIssue> &issues,
                         const std::filesystem::path &artifact_path,
                         core::EventEmitter &emitter, std::ostream &log);

/// Estimate the optical image-height FOV for ASTAP from original frame height
/// and FITS focal/pixel-size metadata.
std::optional<double> estimate_astap_sensor_fov_deg(const io::FitsHeader &header,
                                                    int image_height);

/// Simple parallel-for loop over indices [0, count).
template <typename Fn>
void parallel_for_indices(size_t count, int workers, Fn fn) {
  if (count == 0) return;
  workers = std::max(1, std::min<int>(workers, static_cast<int>(count)));
  if (workers == 1) {
    for (size_t i = 0; i < count; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&]() {
      while (true) {
        const size_t i = next.fetch_add(1);
        if (i >= count) break;
        fn(i);
      }
    });
  }
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }
}


/// Memory-budget plan for frame sub-batching.
///
/// This mirrors the Tile Compile OSC reconstruction rule:
/// `workers * sub_batch * pixels_per_worker * channels * sizeof(float)` should
/// fit into 80% of `runtime_limits.memory_budget`.
struct FrameSubBatchPlan {
  size_t frame_sub_batch_size = 0;
  int effective_workers = 1;
  uint64_t memory_budget_bytes = 0;
  uint64_t bytes_per_frame_per_worker = 0;
  bool budget_too_small_for_requested_workers = false;
  bool sub_batch_limited = false;
};

FrameSubBatchPlan compute_memory_capped_frame_sub_batch(
    size_t frame_count,
    size_t pixels_per_worker,
    int channels,
    int requested_workers,
    int memory_budget_mb);

/// Resolve the PCC aperture FWHM used by auto-radius photometry.
///
/// Estimates a representative star size from RGB channels and falls back to a
/// caller-provided FWHM when image-based estimation is unavailable. `source_out`
/// reports whether the value came from image measurement, fallback, or default.
double resolve_pcc_auto_fwhm_px(const Matrix2Df &R, const Matrix2Df &G,
                                const Matrix2Df &B,
                                bool have_fallback_fwhm = false,
                                double fallback_fwhm_px = 0.0,
                                std::string *source_out = nullptr);

/// Detect common disk-full messages from exception strings and library errors.
bool message_indicates_disk_full(const std::string &message);

/// Load a mono FITS canvas mask and validate it against expected dimensions.
bool load_canvas_mask_fits(const std::filesystem::path &mask_path, int rows,
                           int cols, std::vector<uint8_t> &out_mask,
                           std::string &error_out);

/// Load and validate the canvas mask that belongs to RGB output planes.
bool load_canvas_mask_for_rgb(const std::filesystem::path &mask_path,
                              const Matrix2Df &R, const Matrix2Df &G,
                              const Matrix2Df &B,
                              std::vector<uint8_t> &out_mask,
                              int &rows_out, int &cols_out,
                              std::string &error_out);

/// Integer bounds of a set of warped frame corners on the output canvas.
struct WarpBounds {
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;

  [[nodiscard]] int width() const { return max_x - min_x; }
  [[nodiscard]] int height() const { return max_y - min_y; }
};

/// Invert a 2x3 affine warp matrix, returning false for singular matrices.
bool invert_affine_warp(const WarpMatrix &w, WarpMatrix &inv);

/// Compute the minimal canvas bounds that contain all warped input frames.
WarpBounds compute_warps_bounds(int width, int height,
                                const std::vector<WarpMatrix> &warps);


/// Convert runner configuration into the image-module BGE runtime config.
image::BGEConfig to_image_bge_config(const config::BGEConfig &src);
void apply_autobge_exclusion_polygons(
    const config::BGEConfig &src, int rows, int cols,
    image::BGEConfig &dst);
/// Convert runner configuration into the astrometry-module PCC runtime config.
astrometry::PCCConfig to_astrometry_pcc_config(const config::PCCConfig &src);

/// Build the downsampled registration proxy used by global registration.
///
/// OSC inputs use a CFA-aware green-channel proxy; mono inputs use a simple
/// 2x2 mean downsample. The returned image is intentionally lower resolution
/// than the source frame, so translation components must be scaled before they
/// are applied to full-resolution frames.
Matrix2Df build_registration_proxy(const Matrix2Df &img, ColorMode detected_mode,
                                   const std::string &detected_bayer_str);

/// Serialize BGE diagnostics into the artifact/report JSON shape.
tile_compile::core::json bge_diag_to_json(const image::BGEDiagnostics &diag,
                                          bool requested,
                                          bool have_tile_data,
                                          bool metrics_tiles_match);

/// Result of selecting/querying the PCC star catalog backend.
struct PCCCatalogQueryResult {
  std::vector<astrometry::GaiaStar> stars;
  std::string used_source;
};

/// Query the configured PCC catalog source for stars covering the solved WCS.
PCCCatalogQueryResult query_pcc_catalog_stars(const astrometry::WCS &wcs,
                                              const config::PCCConfig &cfg,
                                              std::ostream &log_stream,
                                              const std::string &log_prefix);

/// Stream buffer that mirrors writes to two destination stream buffers.
class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *a, std::streambuf *b);

protected:
  int overflow(int c) override;
  int sync() override;

private:
  std::streambuf *a_;
  std::streambuf *b_;
};

/// Disk-backed fixed-size frame store.
///
/// Frames are written as raw float matrices in a run-local cache directory and
/// loaded/memory-mapped on demand. This keeps long runs under the configured
/// memory budget while still allowing random tile extraction in later phases.
class DiskCacheFrameStore {
public:
  DiskCacheFrameStore();
  DiskCacheFrameStore(const std::filesystem::path &cache_dir, size_t n_frames,
                      int rows, int cols, bool attach_existing = false);
  ~DiskCacheFrameStore();

  DiskCacheFrameStore(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore &operator=(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept;
  DiskCacheFrameStore &operator=(DiskCacheFrameStore &&o) noexcept;

  /// Persist one full frame at index `fi`.
  void store(size_t fi, const Matrix2Df &frame);
  /// Load a full frame into memory.
  Matrix2Df load(size_t fi) const;
  /// Return a mapped pointer to full-frame data; valid until cache cleanup.
  const float *frame_data(size_t fi) const;
  /// Extract a tile by value, applying optional canvas coordinate offsets.
  Matrix2Df extract_tile(size_t fi, const Tile &t, int offset_x = 0,
                         int offset_y = 0) const;
  /// Extract a tile into an existing matrix to avoid repeated allocations.
  bool extract_tile_into(size_t fi, const Tile &t, Matrix2Df &out,
                         int offset_x = 0, int offset_y = 0) const;
  /// Release all currently mapped frame views without deleting cached files.
  void clear_mappings() const;
  /// Release the mmap mapping for one frame; the cached file remains.
  void invalidate_mapping(size_t fi) const;

  /// Whether a frame has been stored for `fi`.
  bool has_data(size_t fi) const;
  /// Number of frame slots managed by the store.
  size_t size() const;
  /// Stored frame row count.
  int rows() const;
  /// Stored frame column count.
  int cols() const;

  /// Remove cached files and release all mappings.
  void cleanup();
  /// Keep or remove cached files when the store is destroyed.
  void set_preserve_files(bool preserve);

private:
  const float *mapped_frame_ptr(size_t fi) const;
  std::filesystem::path frame_path(size_t fi) const;

  std::filesystem::path cache_dir_;
  int rows_ = 0;
  int cols_ = 0;
  size_t frame_bytes_ = 0;
  std::vector<uint8_t> has_data_;
  mutable std::mutex mapped_mutex_;
  mutable std::vector<void *> mapped_views_;
  bool preserve_files_ = false;
};

/// Shared per-run cache for normalized frames and registration proxies.
///
/// Normalized full frames are disk-backed through `DiskCacheFrameStore`; compact
/// registration proxies stay in memory because they are repeatedly reused by
/// direct registration, anchor promotion, and rescue passes.
class RunnerFrameCache {
public:
  RunnerFrameCache();
  RunnerFrameCache(const std::filesystem::path &cache_dir, size_t n_frames,
                   int rows, int cols);

  RunnerFrameCache(const RunnerFrameCache &) = delete;
  RunnerFrameCache &operator=(const RunnerFrameCache &) = delete;
  RunnerFrameCache(RunnerFrameCache &&) = delete;
  RunnerFrameCache &operator=(RunnerFrameCache &&) = delete;

  /// Store a normalized full-resolution frame in the disk cache.
  void store_normalized(size_t fi, const Matrix2Df &frame);
  /// Load a normalized full-resolution frame from the disk cache.
  Matrix2Df load_normalized(size_t fi) const;
  /// Try to load a normalized frame, returning false if not available.
  bool try_load_normalized(size_t fi, Matrix2Df &out) const;
  /// Whether normalized frame data is available for `fi`.
  bool has_normalized(size_t fi) const;

  /// Store the downsampled registration proxy for `fi`.
  void store_registration_proxy(size_t fi, const Matrix2Df &proxy);
  /// Load a cached registration proxy if present.
  bool try_load_registration_proxy(size_t fi, Matrix2Df &out) const;

  /// Seal a complete normalized cache and retain it beyond this object lifetime.
  void seal_normalized_cache(const registration::RegistrationSamplingPlan &plan);
  void release_registration_proxies();

  /// Number of frame slots in the cache.
  size_t size() const;
  /// Normalized frame row count.
  int rows() const;
  /// Normalized frame column count.
  int cols() const;
  /// Remove disk-backed normalized frames and clear in-memory proxies.
  void cleanup();

private:
  std::filesystem::path normalized_cache_dir_;
  bool normalized_cache_sealed_ = false;
  DiskCacheFrameStore normalized_frames_;
  mutable std::mutex proxy_mutex_;
  std::vector<uint8_t> has_registration_proxy_;
  std::vector<Matrix2Df> registration_proxies_;
};

} // namespace tile_compile::runner
