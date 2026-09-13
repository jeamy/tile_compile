#pragma once

// CFA-forward-drizzle CUDA path --- milestone M7 (plan section 19).
//
// M7 splits into slices. THIS slice ships only the transactional restart
// contract of plan 19.4:
//
//   * chunks write exclusively to a phase-local, uncommitted profile-store
//     generation (StoreWriter already discards an unpublished generation in
//     its destructor);
//   * if the CUDA path fails in ANY chunk after its allowed retries, the
//     whole FORWARD_DRIZZLE phase restarts on the CPU reference path and only
//     a fully computed, validated, hashed CPU result is committed --- never a
//     mixed CPU/CUDA image or a half-accumulated pixel.
//
// The droplet / clipping / profile-accumulation kernels (plan 19.2 stages
// 3--7) and their parity matrix (plan 19.5) are a LATER slice. Until then
// `forward_drizzle_cuda_runtime_available()` is false and every "attempt CUDA"
// resolves to an immediate ForwardDrizzleCudaError, which the caller turns
// into a clean CPU run.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

namespace tile_compile::reconstruction {

// Thrown when the CUDA forward-drizzle path cannot complete a chunk (real
// device/allocation failure, or an injected fault). The caller MUST discard
// any uncommitted profile-store generation and restart the ENTIRE
// FORWARD_DRIZZLE phase on the CPU reference path (plan 19.4). It is never a
// partial-commit or a per-chunk fallback.
struct ForwardDrizzleCudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// True iff this binary has a usable custom forward-drizzle CUDA path. Slice 1
// has no kernels, so this is always false; slice 2 makes it probe
// TILE_COMPILE_WITH_CUDA && a present device.
bool forward_drizzle_cuda_runtime_available();

// Test-only fault injection for the plan-19.4 restart contract. When set to
// n >= 0, an attempted CUDA persist throws ForwardDrizzleCudaError after n
// committed stripes (n == 0 => before the first stripe, i.e. an immediate
// failure). -1 (the default) disables injection. Process-global; a test that
// sets it must reset it. Also honoured from the environment variable
// TILE_COMPILE_FORWARD_DRIZZLE_CUDA_FAULT_AFTER_CHUNKS at first read.
void set_forward_drizzle_cuda_fault_after_chunks(int n);
int forward_drizzle_cuda_fault_after_chunks();

// Options threaded into persist_forward_drizzle_multiband to request the CUDA
// path. `attempt == false` is the plain CPU reference path (default).
struct ForwardDrizzleCudaOptions {
  bool attempt = false;
};

// --- Plan 19.4: device memory + auto-chunking ------------------------------

// Free / total bytes of the active CUDA device. {0, 0} when no usable device
// (or the binary was built without CUDA). Slice-2 device probe.
struct CudaDeviceMemory {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};
CudaDeviceMemory forward_drizzle_cuda_device_memory();

// The plan-19.4 chunking decision: an initial chunk height plus a bounded
// retry ladder that halves the height on each CUDA allocation failure, down
// to `min_chunk_rows`. A safety margin (`reserve_bytes`) is held back for the
// driver / OpenCV before any chunk is sized.
struct CudaChunkPlan {
  int chunk_rows = 0;         // initial attempt
  int min_chunk_rows = 1;     // floor; below this the phase restarts on CPU
  int max_retries = 0;        // halvings available from chunk_rows to the floor
  std::size_t reserve_bytes = 0;  // margin withheld for driver / OpenCV
  std::size_t usable_bytes = 0;   // free_bytes - reserve_bytes (0 if underwater)
  std::size_t bytes_per_row = 0;  // the working-set estimate that was used
  bool feasible = false;     // false => not enough memory for even one row
};

// Pure arithmetic (no device calls): pick the chunk plan from a free-memory
// figure and a per-output-row working-set estimate. `requested_chunk_rows > 0`
// caps the initial height (a config ceiling); 0 means "as large as fits".
// `reserve_fraction` of free memory is held back, but never less than
// `reserve_floor_bytes`. Deterministic and unit-tested without a GPU.
CudaChunkPlan plan_cuda_chunking(std::size_t free_bytes,
                                 std::size_t bytes_per_row, int image_rows,
                                 int requested_chunk_rows = 0,
                                 double reserve_fraction = 0.20,
                                 std::size_t reserve_floor_bytes =
                                     static_cast<std::size_t>(256) << 20);

// A per-chunk device allocation failed: the driver of run_cuda_chunked() then
// halves the chunk height and retries. Any OTHER exception from a chunk
// processor is a hard failure --- run_cuda_chunked lets it propagate and the
// caller restarts FORWARD_DRIZZLE on the CPU reference path (plan 19.4).
struct CudaAllocFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Process one output-row band [y0, y0 + rows). Throw CudaAllocFailure to ask
// for a smaller band; return normally on success.
using CudaChunkProcessor = std::function<void(int y0, int rows)>;

// §30.80 (P6 priority-3 groundwork): observed band structure of a
// run_cuda_chunked drive. `halvings` counts CudaAllocFailure catches (band
// collapse indicator); `min_band_rows` / `max_band_rows` are the actual
// processed band heights (the planned `chunk_rows` does not reveal what the
// ladder collapsed to). Pure telemetry, no behaviour change.
struct CudaChunkRunStats {
  int bands = 0;
  int halvings = 0;
  int min_band_rows = 0;
  int max_band_rows = 0;
};

// --- Plan 19.2 stage 3/5 kernel building block ----------------------------

// The exact square-droplet vs. output-cell overlap area (Sutherland-Hodgman
// convex clip + shoelace), evaluated ON THE DEVICE for a batch of
// (convex quad, axis-aligned rectangle) pairs. Ported 1:1 from the CPU
// polygon_rectangle_intersection_area() so the parity matrix (plan 19.5) can
// pin CPU == CUDA on the numerically hardest piece before the full rasterizer
// is wired into persist_forward_drizzle_multiband.
//   quad_xy  : n*8 doubles  --- x0,y0, x1,y1, x2,y2, x3,y3  per quad
//   rect     : n*4 doubles  --- rx0,ry0, rx1,ry1            per cell
//   out_area : n doubles    --- caller-allocated
// Returns false (out_area untouched) on a CUDA-free build, no usable device,
// or any CUDA error; true on success.
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *quad_xy,
                                                  const double *rect, int n,
                                                  double *out_area);

// The exact affine square-droplet corner map, on the device, for a batch of
// native-source samples under ONE 2x3 affine (source->canvas, row-major
// a00,a01,a02, a10,a11,a12). Per sample: the square
// [sx-half, sx+half] x [sy-half, sy+half] is mapped through the affine and then
// scaled by `internal_scale` (native-canvas -> internal-canvas), giving the 4
// leaf corners in CCW order matching build_affine_leaf() + to_internal().
//   sample_xy    : n*2 doubles (sx, sy)
//   out_corners  : n*8 doubles (x0,y0, x1,y1, x2,y2, x3,y3), caller-allocated
// Returns false (out untouched) on a CUDA-free build, no device, or CUDA error.
bool forward_drizzle_cuda_affine_leaf_corners_batch(const double affine6[6],
                                                    int internal_scale,
                                                    double half,
                                                    const double *sample_xy,
                                                    int n, double *out_corners);

// --- Plan 19.2/19.6 stage 3: the affine droplet rasterizer on the device ---

// One positive-area forward contribution for a single stripe, as produced by
// the device rasterizer. `target_y` is stripe-local. Mirrors the CPU
// DrizzleContrib key (minus `frame_order`, which the caller supplies per batch,
// and `leaf_order`, which is always 0 on the affine path).
struct CudaDrizzleContribRecord {
  std::uint32_t channel = 0;
  std::uint32_t target_y = 0;
  std::uint32_t target_x = 0;
  std::uint32_t source_y = 0;
  std::uint32_t source_x = 0;
  double area = 0.0;   // exact polygon/rectangle overlap, > 0
  double value = 0.0;  // source(source_y, source_x)
};

// Rasterize ONE affine frame's stripe-band contributions on the device. The
// square [x-half, x+half]^2 (x = sx+0.5) is mapped through `affine6`
// (source->canvas, row-major), scaled by `internal_scale`, and overlapped with
// each internal-canvas cell in the stripe --- a 1:1 device port of
// build_affine_leaf + the rasterize_drizzle_stripe bbox/area loop, compiled
// --fmad=false so every area is bit-identical to the CPU reference.
//
//   half            : pixfrac / 2
//   y_begin, rows   : the internal-canvas stripe [y_begin, y_begin+rows)
//   canvas_w_internal : W = canvas_width_native * internal_scale
//   band_sy0/1      : the source-row range to scan (caller derives it from the
//                     inverse affine exactly like the CPU path; [0, source_h]
//                     is always safe, just slower). Clamped to [0, source_h].
//   band_sx0/1      : T5 X+Y windowing — the source-column range to scan,
//                     derived from the tile window's inverse affine. Clamped
//                     to [0, source_w]. [0, source_w] is always safe (full
//                     width), just slower.
//   source_values   : host pointer to the BAND-LOCAL source buffer ---
//                     (band_sy1 - band_sy0) * (band_sx1 - band_sx0) row-major
//                     floats, row 0 == source row band_sy0, col 0 == source
//                     col band_sx0. The whole image is never copied.
//   mono            : true => channel is always 0; false => CFA classification
//   max_cells_per_pixel : per-source-pixel record capacity (a leaf spanning
//                     more cells than this makes the call fail -> CPU fallback)
//
// `records_out` must hold band_rows * band_cols * max_cells_per_pixel entries
// (band_cols = band_sx1 - band_sx0).
// Unused slots are left with area == 0. `*out_written` gets the compacted count
// after the call packs the positive-area records to the front, preserving the
// (source_y, source_x, emit) order. Returns false (and the caller falls back to
// the CPU path for this frame) on: a CUDA-free build, no device, any CUDA
// error, or a leaf exceeding `max_cells_per_pixel`.
bool forward_drizzle_cuda_affine_frame_contributions(
    const double affine6[6], int internal_scale, double half, int y_begin,
    int rows, int canvas_w_internal, int band_sy0, int band_sy1,
    int band_sx0, int band_sx1, int source_w,
    int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, int max_cells_per_pixel,
    CudaDrizzleContribRecord *records_out, long long records_capacity,
    long long *out_written);

// Forward-Drizzle-v2 Gate-1 prototype G: record-free affine target gather for
// one frame and one target rectangle.  One CUDA thread owns one target cell,
// scans its conservative inverse-affine source neighbourhood in canonical
// (source_y, source_x) order and writes dense frame-local A/B planes.  Output
// layout is channel-major [channel][row][column], with 1 channel for MONO and
// 3 for OSC.  This prototype intentionally owns its temporary allocations;
// Gate 6 replaces them with a persistent workspace after Gate 1 chooses the
// enumeration.
bool forward_drizzle_cuda_affine_target_gather(
    const double affine6[6], const double inverse6[6], int internal_scale,
    double half, int target_x_begin, int target_y_begin, int target_cols,
    int target_rows, int source_w, int source_h, const float *source_values,
    int bayer_pattern, int cfa_origin_x, int cfa_origin_y, bool mono,
    double *out_a, double *out_b, unsigned long long *out_source_candidates,
    unsigned long long *out_positive_overlaps);

// Gate-1 prototype S: one thread per source sample scatters directly into
// dense frame-local A/B target planes with device atomics.  It creates no
// records and uses memory independent of frame_count.  Repeatability and
// numerical drift are measured against target gather before either prototype
// can be selected for Gate 6.
bool forward_drizzle_cuda_affine_dense_scatter(
    const double affine6[6], int internal_scale, double half,
    int target_x_begin, int target_y_begin, int target_cols, int target_rows,
    int source_w, int source_h, const float *source_values, int bayer_pattern,
    int cfa_origin_x, int cfa_origin_y, bool mono, double *out_a,
    double *out_b, unsigned long long *out_positive_overlaps);

// --- Forward-Drizzle-v2 Gate-8: local warp descriptor ---------------------
//
// The persisted SmoothLocalWarpModel plus the frozen inversion/subdivision
// contract (LocalInversionParams + ForwardDrizzleSubdivisionParams of the
// production CPU oracle). ~144 bytes per frame; passed by value into the
// device scatter kernel, which performs the bounded fixed-point inversion
// q_{n+1} = u - d(q_n) and the adaptive 3x3 subdivision on device. No
// deformation grid is ever materialized (spec 11.2 option 4).
struct ForwardDrizzleV2LocalWarp {
  float coeff_x[16] = {};
  float coeff_y[16] = {};
  int image_rows = 0;             // model image dims
  int image_cols = 0;
  int model_valid = 1;            // SmoothLocalWarpModel::valid
  float model_coordinate_scale = 1.0f;
  float model_offset_x = 0.0f;
  float model_offset_y = 0.0f;
  // LocalInversionParams of the CPU oracle.
  int max_iter = 6;
  float tol_px = 1.0e-3f;
  float safety_margin_px = 64.0f;
  // ForwardDrizzleSubdivisionParams of the CPU oracle.
  float position_epsilon_internal_px = 0.05f;
  int max_subdivision_depth = 2;
  float area_relative_epsilon = 0.005f;
};

// Gate-8 debug/parity scatter: one local-warp frame scattered into dense
// frame-local internal planes (channel-major [c][row][col]), mirroring
// forward_drizzle_cuda_affine_dense_scatter but running the on-device
// fixed-point inversion + adaptive subdivision per source sample. Returns
// A (area-weighted value), B_src (finite-value geometry) and B_geo (all
// geometry) planes plus positive overlap and discarded-sample counts.
// Returns false on a CUDA-free build, no device, invalid warp or any CUDA
// error; the caller falls back to the CPU leaf oracle.
bool forward_drizzle_cuda_local_dense_scatter(
    const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
    int internal_scale, double half, int target_cols, int target_rows,
    int source_w, int source_h, const float *source_values,
    int bayer_pattern, int cfa_origin_x, int cfa_origin_y, bool mono,
    int canvas_w_native, int canvas_h_native, double *out_a,
    double *out_b_src, double *out_b_geo,
    unsigned long long *out_positive_overlaps,
    unsigned long long *out_discarded);

struct ForwardDrizzleV2CudaWorkspaceStats {
  std::uint64_t allocations = 0;
  std::uint64_t calls = 0;
  std::uint64_t source_bytes_uploaded = 0;
  std::uint64_t result_bytes_downloaded = 0;
  std::uint64_t positive_overlaps = 0;
  std::uint64_t device_global_synchronizations = 0;
  std::uint64_t stream_synchronizations = 0;
  std::size_t reserved_device_bytes = 0;
  double upload_seconds = 0.0;
  double kernel_seconds = 0.0;
  double download_seconds = 0.0;
};

// Gate-1 selected dense-scatter persistent-buffer spike. It proves allocation
// reuse and the selected affine enumeration only; it does not satisfy Gate 6
// (no overlapped slot pipeline, Q data, robust reduction, coverage, fold or
// transaction yet).
// reserve() is called before entering the frame loop; run_dense_scatter()
// performs no cudaMalloc/cudaFree and exposes the transfer/kernel split
// unconditionally through stats().
class ForwardDrizzleV2CudaWorkspace {
 public:
  ForwardDrizzleV2CudaWorkspace();
  ~ForwardDrizzleV2CudaWorkspace();
  ForwardDrizzleV2CudaWorkspace(const ForwardDrizzleV2CudaWorkspace &) = delete;
  ForwardDrizzleV2CudaWorkspace &operator=(
      const ForwardDrizzleV2CudaWorkspace &) = delete;

  bool reserve(std::size_t source_elements, std::size_t target_plane_elements,
               int channels);
  bool run_dense_scatter(
      const double affine6[6], int internal_scale, double half,
      int target_x_begin, int target_y_begin, int target_cols, int target_rows,
      int source_w, int source_h, const float *source_values,
      int bayer_pattern, int cfa_origin_x, int cfa_origin_y, bool mono,
      double *out_a, double *out_b);
  const ForwardDrizzleV2CudaWorkspaceStats &stats() const { return stats_; }

 private:
  void *device_source_ = nullptr;
  void *device_a_ = nullptr;
  void *device_b_ = nullptr;
  void *device_overlaps_ = nullptr;
  void *stream_ = nullptr;
  std::size_t source_capacity_ = 0;
  std::size_t plane_capacity_ = 0;
  int channel_capacity_ = 0;
  ForwardDrizzleV2CudaWorkspaceStats stats_;
};

// --- Forward-Drizzle-v2 Gate-6: minimal affine prototype kernel ------------
//
// Persistent device workspace implementing the frozen gates 1-5 pipeline per
// output band: dense scatter into frame-local internal planes (A, B_src,
// B_geo, optional S2), a fused per-frame fold+accumulate into native-pixel
// accumulators (full-stream A/B/B2, coverage, support masks, footprint,
// hash reservoir with per-candidate sigma2), and a band-end finalize kernel
// running the bit-exact CPU-oracle clip on each pixel's reservoir.
// No host contribution records, no cudaMalloc/cudaFree after reserve(), no
// cudaDeviceSynchronize; the only stream sync is the band finalize.
// Unpublished: nothing in the production runner calls this.

struct ForwardDrizzleV2KernelConfig {
  int internal_scale = 2;  // internal subpixels per native axis
  int reservoir_size = 64;
  std::uint64_t reservoir_seed = 0x9e3779b97f4a7c15ULL;
  std::uint64_t stream_length = 0;  // N for the keep predicate (required > 0)
  int min_clip_contributors = 5;
  int min_candidates = 5;
  int robust_passes = 3;
  double sigma_low = 3.0;
  double sigma_high = 3.0;
  double half = 0.4;
  int bayer_pattern = 1;
  int cfa_origin_x = 0;
  int cfa_origin_y = 0;
  bool mono = false;
  // When false, no sigma2 frame plane is allocated and every candidate
  // contributes zero sigma (CPU "no candidate_sigma2" semantics).
  bool sigma2_plane = true;
  // Full native target-canvas dimensions for the gate-8 local-warp
  // inversion bounds check. The workspace band can be shorter than the
  // canvas (reserve()'s native_rows is the band height), while
  // invert_local_source_to_canvas must bound against the full canvas.
  // 0 = the band covers the whole canvas (use the band dims).
  int canvas_width_native = 0;
  int canvas_height_native = 0;
  // Gate-10 band window origin in native canvas pixels. The affine6
  // argument of accumulate_* is ALWAYS the persisted source->canvas
  // transform in canvas coordinates; emitted internal coordinates are
  // (q - band_origin) * internal_scale. The local-warp inversion still
  // runs in canvas coordinates (seed, model evaluation and bounds), so
  // per-sample discard decisions are identical to the whole-canvas oracle
  // for every band. 0 = the band starts at the canvas origin.
  int band_origin_x_native = 0;
  int band_origin_y_native = 0;
  // Gate-9: profile production. When true, reserve() additionally allocates
  // the five quality frame planes (Q_c, Q_0, Q_1, Q_a, Q_aflag), the float4
  // reservoir side array, the per-frame meta table and the profile result
  // buffer; finalize() then also emits ForwardDrizzleV2ProfileResult
  // records. Exponents follow the plan-11.9 profile contract.
  bool emit_profiles = false;
  float fine_quality_exponent = 4.0f;
  float medium_quality_exponent = 2.0f;
};

// Gate-9 optional per-frame quality source planes (source resolution,
// row-major float, same extent as `source`). A null pointer marks an absent
// stream: its folded candidate mean is 1.0 and the artifact stream reports
// qa_has_data=false. Per-sample NaN/<=0 contributes 0 to the area-weighted
// mean (explicit veto, matching the CPU contract).
struct ForwardDrizzleV2FrameQuality {
  const float *q_composite = nullptr;
  const float *q_scale0 = nullptr;
  const float *q_scale1 = nullptr;
  const float *q_artifact = nullptr;
};

// Per-native-pixel per-channel band result (AoS record, downloaded once).
struct ForwardDrizzleV2PixelResult {
  double value = 0.0;          // clip center (or uniform fallback mean)
  double b = 0.0;              // full-stream folded B
  double n_eff = 0.0;
  double confidence = 0.0;
  float geometry_fraction = 0.0f;
  float source_fraction = 0.0f;
  float estimator_fraction = 0.0f;
  float profile_fraction = 0.0f;
  std::uint32_t contributors = 0;
  std::uint8_t robust_state = 0;    // ForwardDrizzleV2RobustState
  std::uint8_t confidence_state = 0;  // ForwardDrizzleV2ConfidenceState
  std::uint64_t conf_degraded = 0;
};

struct ForwardDrizzleV2PrototypeStats {
  std::uint64_t allocations = 0;
  std::uint64_t device_global_synchronizations = 0;
  std::uint64_t stream_synchronizations = 0;
  std::uint64_t frames_processed = 0;
  std::uint64_t source_bytes_uploaded = 0;
  std::uint64_t result_bytes_downloaded = 0;
  std::uint64_t positive_overlaps = 0;
  std::uint64_t candidates_streamed = 0;
  std::uint64_t reservoir_kept_total = 0;
  std::uint64_t slot_transitions = 0;
  // Gate-8: source samples discarded all-or-nothing by local-warp
  // inversion/subdivision failure across all processed local frames. The
  // driver applies the per-frame exclusion policy to this count.
  std::uint64_t local_samples_discarded = 0;
  std::size_t reserved_device_bytes = 0;
  double upload_seconds = 0.0;   // event-timed
  double kernel_seconds = 0.0;
  double download_seconds = 0.0;
  // Worst per-frame upload+kernel time (event deltas); the gate-1 bound is
  // checked against this, not the mean.
  double max_frame_seconds = 0.0;
};

class ForwardDrizzleV2CudaPrototypeKernel {
 public:
  ForwardDrizzleV2CudaPrototypeKernel();
  ~ForwardDrizzleV2CudaPrototypeKernel();
  ForwardDrizzleV2CudaPrototypeKernel(
      const ForwardDrizzleV2CudaPrototypeKernel &) = delete;
  ForwardDrizzleV2CudaPrototypeKernel &operator=(
      const ForwardDrizzleV2CudaPrototypeKernel &) = delete;

  // Reserve every role for one band: native window cols x rows, internal
  // planes at internal_scale resolution, full source frame + optional sigma2
  // slot. Counted as the single allowed allocation batch.
  bool reserve(int native_cols, int native_rows, int source_w, int source_h,
               const ForwardDrizzleV2KernelConfig &cfg);
  // One frame: upload source (+sigma2, +quality planes when profiles are
  // enabled), scatter, fold+accumulate. All async on the workspace stream;
  // no allocations, no sync. `affine6` is the persisted source->canvas
  // transform in CANVAS coordinates; cfg.band_origin_*_native selects the
  // band window. When cfg.emit_profiles is set, `meta_or_null`
  // must supply the frame's g_eff/is_direct/residual_factor row (the call
  // fails otherwise); without profiles both extras may stay null.
  bool accumulate_frame(const double affine6[6], const float *source,
                        const float *sigma2_or_null, std::uint64_t frame_order,
                        const ForwardDrizzleV2FrameQuality *quality_or_null =
                            nullptr,
                        const ForwardDrizzleV2FrameMeta *meta_or_null =
                            nullptr);
  // Gate-8 local-warp variant: same contract as accumulate_frame but the
  // per-sample geometry runs the on-device fixed-point inversion +
  // adaptive subdivision instead of the single affine leaf. affine6 remains
  // the affine seed. CPU-oracle failure semantics are preserved: an
  // invalid model, non-finite coefficients/scale or a failed inversion
  // discards the affected samples (model_valid == 0 discards every
  // sample); only a subdivision depth > 2 (not executable by the implicit
  // 21-node tree) rejects the call. Discards are counted in
  // stats().local_samples_discarded.
  bool accumulate_frame_local(const double affine6[6],
                              const ForwardDrizzleV2LocalWarp &warp,
                              const float *source,
                              const float *sigma2_or_null,
                              std::uint64_t frame_order,
                              const ForwardDrizzleV2FrameQuality *quality_or_null =
                                  nullptr,
                              const ForwardDrizzleV2FrameMeta *meta_or_null =
                                  nullptr);
  // Band end: finalize kernel + single stream sync + result download.
  // results must hold native_cols*native_rows*channels entries
  // (channel-major); when cfg.emit_profiles is set, profiles_or_null must
  // hold the same number of ForwardDrizzleV2ProfileResult records (the call
  // fails otherwise). dense_overlap_count receives the number of native
  // pixels covered by every processed frame.
  bool finalize(ForwardDrizzleV2PixelResult *results,
                ForwardDrizzleV2ProfileResult *profiles_or_null,
                std::uint64_t *dense_overlap_count);
  const ForwardDrizzleV2PrototypeStats &stats() const { return stats_; }
  std::size_t device_bytes_per_native_pixel() const {
    return bytes_per_native_pixel_;
  }

 private:
  // Shared frame pipeline of accumulate_frame/accumulate_frame_local:
  // warp == nullptr selects the affine scatter kernel.
  bool accumulate_frame_impl(const double affine6[6],
                             const ForwardDrizzleV2LocalWarp *warp,
                             const float *source,
                             const float *sigma2_or_null,
                             std::uint64_t frame_order,
                             const ForwardDrizzleV2FrameQuality *quality_or_null,
                             const ForwardDrizzleV2FrameMeta *meta_or_null);
  struct Impl;
  Impl *impl_ = nullptr;
  ForwardDrizzleV2PrototypeStats stats_;
  std::size_t bytes_per_native_pixel_ = 0;
};

// §30.81 step-5 baseline instrumentation. Coarse wall-clock accumulators for
// the affine CUDA pair path, split so the per-tile repetition factor is
// attributable (producer / device phases vs the host sort + reduce). Enabled
// only when the env var TC_FD_CUDA_PROFILE is set; every write is guarded by
// forward_drizzle_cuda_profile_enabled(), so a normal run pays nothing and the
// numbers never touch compute. NOT thread-safe against concurrent
// accumulate_pair_impl calls --- the CUDA store path drives it serially.
struct ForwardDrizzleCudaProfile {
  std::atomic<double> dev_malloc_s{0.0};   // cudaMalloc/Memset in the .cu wrapper
  std::atomic<double> dev_upload_s{0.0};   // H2D source copy
  std::atomic<double> dev_kernel_s{0.0};   // kernel launch + cudaDeviceSynchronize
  std::atomic<double> dev_download_s{0.0}; // D2H records + count
  std::atomic<double> produce_s{0.0};      // whole PairFrameRecordProducer call
  std::atomic<double> sort_s{0.0};         // std::sort of the frame's records
  std::atomic<double> reduce_s{0.0};       // reduce_pixel_profiles loop
  std::atomic<std::uint64_t> calls{0};     // accumulate_pair_impl invocations
  std::atomic<std::uint64_t> records_sorted{0};  // sum of recs.size() over sorts
  void reset() {
    dev_malloc_s = dev_upload_s = dev_kernel_s = dev_download_s = 0.0;
    produce_s = sort_s = reduce_s = 0.0;
    calls = 0;
    records_sorted = 0;
  }
};
ForwardDrizzleCudaProfile &forward_drizzle_cuda_profile();
bool forward_drizzle_cuda_profile_enabled();
// Add `dt` seconds to `slot` (std::atomic<double>, C++20 fetch_add).
inline void forward_drizzle_cuda_profile_add(std::atomic<double> &slot,
                                             double dt) {
  slot.fetch_add(dt, std::memory_order_relaxed);
}

// Plan-19.4 chunk driver (host-only, no device calls of its own). Walks the
// image in bands of `plan.chunk_rows`; on CudaAllocFailure it halves the
// CURRENT band height and retries the SAME band, down to `plan.min_chunk_rows`.
// If a band still cannot be processed at the floor height, every temporary
// CUDA store is void and the whole phase must restart on CPU --- signalled by
// throwing ForwardDrizzleCudaError. Returns the number of bands committed.
// `plan.feasible` must be true. `stats` (optional): observed band structure
// (§30.80); filled on a normal return, untouched on a throw.
int run_cuda_chunked(const CudaChunkPlan &plan, int image_rows,
                     const CudaChunkProcessor &process,
                     CudaChunkRunStats *stats = nullptr);

}  // namespace tile_compile::reconstruction
