#pragma once

// Gate-10 host CPU port of the forward-drizzle v2 band kernel and the
// backend-neutral boundary the production band driver runs against.
//
// The host port mirrors ForwardDrizzleV2CudaPrototypeKernel's contract
// exactly: reserve() allocates the whole band workspace once, frames are
// streamed through accumulate_frame()/accumulate_frame_local() (affine6 is
// source->canvas in the caller's band-local target frame), and finalize()
// emits one ForwardDrizzleV2PixelResult (+ optional
// ForwardDrizzleV2ProfileResult) record per native pixel and channel. The
// scatter, fold, reservoir keep predicate, sigma-clip, confidence and
// profile/alpha semantics are the same mathematical contract the device
// kernels implement --- verified against the CPU oracles and, where a device
// exists, bit-tolerant against the CUDA output.

#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"

namespace tile_compile::reconstruction {

// Backend-neutral kernel boundary for the gate-10 band driver. Both
// implementations share the device kernel's call contract; the only
// observable difference is stats() (no CUDA counters on the host) and
// device_backend().
class ForwardDrizzleV2Kernel {
 public:
  virtual ~ForwardDrizzleV2Kernel() = default;
  virtual bool reserve(int native_cols, int native_rows, int source_w,
                       int source_h,
                       const ForwardDrizzleV2KernelConfig &cfg) = 0;
  // Rebind the reserved workspace to the next band (active rows <= the
  // reserved maximum) without allocating; clears all per-band state.
  // Allowed right after reserve() (before any frame) or after a successful
  // finalize(); only native_rows and band_origin_y_native may change.
  virtual bool begin_band(
      int native_rows, const ForwardDrizzleV2KernelConfig &cfg) = 0;
  virtual bool accumulate_frame(
      const double affine6[6], const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  virtual bool accumulate_frame_local(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  // Window variants: `source` is a packed row-major buffer of
  // window.width*window.height (absolute origin x_begin/y_begin, optional
  // halo); only the ACTIVE rect (window.active_*, all-zero = whole buffer)
  // is iterated. Float sigma2/Q planes are packed over the ACTIVE rect and
  // indexed by the active-local tid; packed Q descriptors decode by absolute
  // (sx, sy). sigma2_or_null and an ENABLED sigma2_model_or_null are
  // mutually exclusive. The window buffer must be positive and contained in
  // the reserved full source extent.
  virtual bool accumulate_frame_window(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  virtual bool accumulate_frame_local_window(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  // Geometry-cache variant (tranche 6): the frame's geometry is
  // `leaf_count` committed cache leaves (raw internal canvas coordinates,
  // canonical order); leaf source coordinates must lie inside the ACTIVE
  // window rect. affine6 is validated but not applied. Requires
  // leaf_count <= cfg.cached_leaf_capacity.
  virtual bool accumulate_frame_cached_leaves(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
      std::uint64_t unique_source_samples, std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  // Tranche 8 canonical affine path: one-shot full-target frame fed by the
  // ragged source-sample list (canonical ascending (y,x) order, all
  // coordinates in the reserved source extent, count <= source_w*source_h).
  // sigma2_present=false reproduces the absent-sigma semantics. Quality is
  // the aligned packed contract (presence bit => code array, null veto =>
  // no veto; code 0 or veto decodes to NaN).
  virtual bool accumulate_frame_affine_samples(
      const double affine6[6], const ForwardDrizzleV2SourceSample *samples,
      std::size_t sample_count, bool sigma2_present,
      const ForwardDrizzleV2AlignedQuality *quality_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  // Affine target-tile piece lifecycle (tranche 7): one affine frame may be
  // emitted as several target-x pieces; each piece scatters+folds only the
  // internal x columns [target_x_begin*scale, (begin+cols)*scale) so
  // overlapping source scan boxes never double-count. Pieces must be
  // ordered by target x and non-overlapping, and the frame's quality
  // stream presence must be identical on every piece. The local-warp and
  // cached-geometry paths are always single-call and must not interleave
  // with an open piece frame.
  virtual bool begin_affine_frame(
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  virtual bool accumulate_affine_piece(
      const double affine6[6], int target_x_begin_native,
      int target_cols_native,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr) = 0;
  virtual bool finish_affine_frame(std::uint64_t frame_order) = 0;
  // Frame whose source window is empty for this band: advances stream
  // bookkeeping (and the meta row when profiles are enabled) without
  // scatter/fold.
  virtual bool skip_frame(
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) = 0;
  virtual bool finalize(ForwardDrizzleV2PixelResult *results,
                        ForwardDrizzleV2ProfileResult *profiles_or_null,
                        std::uint64_t *dense_overlap_count) = 0;
  virtual const ForwardDrizzleV2PrototypeStats &stats() const = 0;
  virtual bool device_backend() const = 0;
  // Diagnosis hook: the CUDA adapter reports cudaGetErrorString() of the
  // failing runtime call after a false return; the CPU kernel always
  // returns "".
  virtual std::string last_device_error() const { return {}; }
};

// Host CPU port. Bounded memory: all state is allocated in reserve() sized
// by the band window, source frame and stream length. accumulate_* performs
// the same scatter + fold as the device kernels (polygon-clip droplet
// scatter, K-fold per native pixel, deterministic hash reservoir) and
// finalize runs the production sigma-clip plus the shared profile reduction
// (forward_drizzle_v2_profile_reduce) over the clip-accepted set.
class ForwardDrizzleV2CpuKernel final : public ForwardDrizzleV2Kernel {
 public:
  ForwardDrizzleV2CpuKernel();
  ~ForwardDrizzleV2CpuKernel() override;
  ForwardDrizzleV2CpuKernel(const ForwardDrizzleV2CpuKernel &) = delete;
  ForwardDrizzleV2CpuKernel &operator=(const ForwardDrizzleV2CpuKernel &) =
      delete;

  bool reserve(int native_cols, int native_rows, int source_w, int source_h,
               const ForwardDrizzleV2KernelConfig &cfg) override;
  bool begin_band(int native_rows,
                  const ForwardDrizzleV2KernelConfig &cfg) override;
  bool accumulate_frame(
      const double affine6[6], const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_frame_local(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_frame_window(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_frame_local_window(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_frame_cached_leaves(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
      std::uint64_t unique_source_samples, std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_frame_affine_samples(
      const double affine6[6], const ForwardDrizzleV2SourceSample *samples,
      std::size_t sample_count, bool sigma2_present,
      const ForwardDrizzleV2AlignedQuality *quality_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool begin_affine_frame(
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override;
  bool accumulate_affine_piece(
      const double affine6[6], int target_x_begin_native,
      int target_cols_native,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2FrameQuality *quality_or_null =
          nullptr) override;
  bool finish_affine_frame(std::uint64_t frame_order) override;
  bool skip_frame(std::uint64_t frame_order,
                  const ForwardDrizzleV2FrameMeta *meta_or_null =
                      nullptr) override;
  bool finalize(ForwardDrizzleV2PixelResult *results,
                ForwardDrizzleV2ProfileResult *profiles_or_null,
                std::uint64_t *dense_overlap_count) override;
  const ForwardDrizzleV2PrototypeStats &stats() const override {
    return stats_;
  }
  bool device_backend() const override { return false; }
  // Diagnostic for the memory plan: reserved host bytes per native pixel.
  std::size_t host_bytes_per_native_pixel() const {
    return bytes_per_native_pixel_;
  }

 private:
  bool accumulate_frame_impl(
      const double affine6[6], const ForwardDrizzleV2LocalWarp *warp,
      const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
      std::uint64_t unique_source_samples, std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null,
      const ForwardDrizzleV2FrameMeta *meta_or_null);
  // Fold of the frame planes into the band accumulators for native target
  // columns [nx0, nx1) only (all rows/channels) --- shared by the one-shot
  // calls (full width) and the affine piece path.
  void fold_native_range(int nx0, int nx1, bool q_frame,
                         unsigned int qmask, std::uint64_t frame_order,
                         bool keep_all, std::uint64_t threshold);
  struct Impl;
  Impl *impl_ = nullptr;
  ForwardDrizzleV2PrototypeStats stats_;
  std::size_t bytes_per_native_pixel_ = 0;
  std::size_t capacity_bytes_ = 0;
  // True until the first begin_band consumes the reservation reported by
  // stats() so the driver sums exactly one workspace reservation.
  bool pending_reservation_ = false;
  // Open affine-piece frame state (tranche 7).
  bool frame_open_ = false;
  std::uint64_t open_order_ = 0;
  int open_pieces_ = 0;
  int open_tx_end_ = 0;          // lowest target x the next piece may start
  unsigned int open_qmask_ = 0;  // stream presence fixed on piece 1
  bool open_qframe_ = false;
};

// Device adapter: forwards the prototype kernel through the same boundary so
// the driver can hold either backend without a variant. Never allocates; the
// wrapped kernel keeps its exact semantics.
class ForwardDrizzleV2CudaKernel final : public ForwardDrizzleV2Kernel {
 public:
  bool reserve(int native_cols, int native_rows, int source_w, int source_h,
               const ForwardDrizzleV2KernelConfig &cfg) override {
    return kernel_.reserve(native_cols, native_rows, source_w, source_h, cfg);
  }
  bool begin_band(int native_rows,
                  const ForwardDrizzleV2KernelConfig &cfg) override {
    return kernel_.begin_band(native_rows, cfg);
  }
  bool accumulate_frame(
      const double affine6[6], const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame(affine6, source, sigma2_or_null,
                                    frame_order, quality_or_null,
                                    meta_or_null);
  }
  bool accumulate_frame_local(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame_local(affine6, warp, source,
                                          sigma2_or_null, frame_order,
                                          quality_or_null, meta_or_null);
  }
  bool accumulate_frame_window(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame_window(affine6, window, source,
                                           sigma2_or_null,
                                           sigma2_model_or_null, frame_order,
                                           quality_or_null, meta_or_null);
  }
  bool accumulate_frame_local_window(
      const double affine6[6], const ForwardDrizzleV2LocalWarp &warp,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame_local_window(
        affine6, warp, window, source, sigma2_or_null, sigma2_model_or_null,
        frame_order, quality_or_null, meta_or_null);
  }
  bool accumulate_frame_cached_leaves(
      const double affine6[6], const ForwardDrizzleV2SourceWindow &window,
      const float *source, const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2CachedLeaf *leaves, std::size_t leaf_count,
      std::uint64_t unique_source_samples, std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null = nullptr,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame_cached_leaves(
        affine6, window, source, sigma2_or_null, sigma2_model_or_null,
        leaves, leaf_count, unique_source_samples, frame_order,
        quality_or_null, meta_or_null);
  }
  bool accumulate_frame_affine_samples(
      const double affine6[6], const ForwardDrizzleV2SourceSample *samples,
      std::size_t sample_count, bool sigma2_present,
      const ForwardDrizzleV2AlignedQuality *quality_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.accumulate_frame_affine_samples(
        affine6, samples, sample_count, sigma2_present, quality_or_null,
        frame_order, meta_or_null);
  }
  bool begin_affine_frame(
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameMeta *meta_or_null = nullptr) override {
    return kernel_.begin_affine_frame(frame_order, meta_or_null);
  }
  bool accumulate_affine_piece(
      const double affine6[6], int target_x_begin_native,
      int target_cols_native,
      const ForwardDrizzleV2SourceWindow &window, const float *source,
      const float *sigma2_or_null,
      const ForwardDrizzleV2Sigma2FrameModel *sigma2_model_or_null,
      const ForwardDrizzleV2FrameQuality *quality_or_null =
          nullptr) override {
    return kernel_.accumulate_affine_piece(
        affine6, target_x_begin_native, target_cols_native, window, source,
        sigma2_or_null, sigma2_model_or_null, quality_or_null);
  }
  bool finish_affine_frame(std::uint64_t frame_order) override {
    return kernel_.finish_affine_frame(frame_order);
  }
  bool skip_frame(std::uint64_t frame_order,
                  const ForwardDrizzleV2FrameMeta *meta_or_null =
                      nullptr) override {
    return kernel_.skip_frame(frame_order, meta_or_null);
  }
  bool finalize(ForwardDrizzleV2PixelResult *results,
                ForwardDrizzleV2ProfileResult *profiles_or_null,
                std::uint64_t *dense_overlap_count) override {
    return kernel_.finalize(results, profiles_or_null, dense_overlap_count);
  }
  const ForwardDrizzleV2PrototypeStats &stats() const override {
    return kernel_.stats();
  }
  bool device_backend() const override { return true; }
  std::string last_device_error() const override {
    return kernel_.last_device_error();
  }

 private:
  ForwardDrizzleV2CudaPrototypeKernel kernel_;
};

}  // namespace tile_compile::reconstruction
