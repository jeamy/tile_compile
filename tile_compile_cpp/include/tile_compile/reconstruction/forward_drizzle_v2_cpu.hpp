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
  virtual bool finalize(ForwardDrizzleV2PixelResult *results,
                        ForwardDrizzleV2ProfileResult *profiles_or_null,
                        std::uint64_t *dense_overlap_count) = 0;
  virtual const ForwardDrizzleV2PrototypeStats &stats() const = 0;
  virtual bool device_backend() const = 0;
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
      const float *source, const float *sigma2_or_null,
      std::uint64_t frame_order,
      const ForwardDrizzleV2FrameQuality *quality_or_null,
      const ForwardDrizzleV2FrameMeta *meta_or_null);
  struct Impl;
  Impl *impl_ = nullptr;
  ForwardDrizzleV2PrototypeStats stats_;
  std::size_t bytes_per_native_pixel_ = 0;
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
  bool finalize(ForwardDrizzleV2PixelResult *results,
                ForwardDrizzleV2ProfileResult *profiles_or_null,
                std::uint64_t *dense_overlap_count) override {
    return kernel_.finalize(results, profiles_or_null, dense_overlap_count);
  }
  const ForwardDrizzleV2PrototypeStats &stats() const override {
    return kernel_.stats();
  }
  bool device_backend() const override { return true; }

 private:
  ForwardDrizzleV2CudaPrototypeKernel kernel_;
};

}  // namespace tile_compile::reconstruction
