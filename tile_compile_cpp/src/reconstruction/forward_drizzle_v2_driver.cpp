// Gate-10 band driver: see the header for the contract. The driver is
// deliberately serial --- the kernel owns all parallelism; the bounded-
// memory contract comes from the kernel's single reserve() per band.

#include "tile_compile/reconstruction/forward_drizzle_v2_driver.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tile_compile::reconstruction {
namespace {

// Process-global test hook; armed once from the environment.
std::atomic<int> g_fd_v2_cuda_fault_after_bands{-1};
std::atomic<bool> g_fd_v2_fault_env_read{false};

void read_fault_env_once() {
  if (g_fd_v2_fault_env_read.exchange(true)) return;
  if (const char *e =
          std::getenv("TILE_COMPILE_FD_V2_CUDA_FAULT_AFTER_BANDS"))
    g_fd_v2_cuda_fault_after_bands.store(std::atoi(e));
}

ForwardDrizzleV2KernelConfig kernel_config_for_band(
    const ForwardDrizzleV2RunPlan &plan, int y_begin, int rows) {
  (void)rows;
  ForwardDrizzleV2KernelConfig k;
  k.internal_scale = plan.internal_scale;
  k.reservoir_size = plan.reservoir_size;
  k.reservoir_seed = plan.reservoir_seed;
  k.stream_length = plan.frame_count;
  k.min_clip_contributors = plan.min_clip_contributors;
  k.min_candidates = plan.min_candidates;
  k.robust_passes = plan.robust_passes;
  k.sigma_low = plan.sigma_low;
  k.sigma_high = plan.sigma_high;
  k.half = 0.5 * plan.pixfrac;
  k.bayer_pattern = plan.bayer_pattern;
  k.cfa_origin_x = plan.cfa_origin_x;
  k.cfa_origin_y = plan.cfa_origin_y;
  k.mono = plan.channels == 1;
  k.sigma2_plane = plan.sigma2_enabled;
  k.canvas_width_native = plan.native_width;
  k.canvas_height_native = plan.native_height;
  k.band_origin_y_native = y_begin;
  k.emit_profiles = plan.emit_profiles;
  k.fine_quality_exponent = plan.fine_quality_exponent;
  k.medium_quality_exponent = plan.medium_quality_exponent;
  return k;
}

std::unique_ptr<ForwardDrizzleV2Kernel> make_kernel(bool cuda) {
  if (cuda) return std::make_unique<ForwardDrizzleV2CudaKernel>();
  return std::make_unique<ForwardDrizzleV2CpuKernel>();
}

void add_stats(ForwardDrizzleV2PrototypeStats &t,
               const ForwardDrizzleV2PrototypeStats &s) {
  t.allocations += s.allocations;
  t.device_global_synchronizations += s.device_global_synchronizations;
  t.stream_synchronizations += s.stream_synchronizations;
  t.frames_processed += s.frames_processed;
  t.source_bytes_uploaded += s.source_bytes_uploaded;
  t.result_bytes_downloaded += s.result_bytes_downloaded;
  t.positive_overlaps += s.positive_overlaps;
  t.candidates_streamed += s.candidates_streamed;
  t.reservoir_kept_total += s.reservoir_kept_total;
  t.slot_transitions += s.slot_transitions;
  t.local_samples_discarded += s.local_samples_discarded;
}

// One backend attempt over the committed-prefix tail of the store.
// Returns false iff the device backend reported a kernel-level failure
// (the caller then restarts the whole phase on CPU). `device_failure`
// additionally captures why. Any other failure throws.
bool attempt_backend(const fs::path &store_root,
                     const ForwardDrizzleV2RunPlan &plan, int source_w,
                     int source_h, const ForwardDrizzleV2FrameProvider &provider,
                     bool cuda, ForwardDrizzleV2DriverResult &result,
                     const ForwardDrizzleV2DriverOptions &options,
                     std::string *device_failure) {
  // Re-inspect: a prior failed attempt may have deleted the generation.
  const auto insp = inspect_forward_drizzle_v2_store(store_root, plan);
  if (insp.status == ForwardDrizzleV2StoreStatus::corrupt)
    throw std::runtime_error("FDV2_DRIVER_STORE_CORRUPT: " + insp.error);
  if (insp.status == ForwardDrizzleV2StoreStatus::complete) {
    result.generation_dir = insp.generation;
    result.committed = true;
    result.bands_reused = plan.band_count;
    result.commit_hash = insp.commit_hash;
    return true;
  }

  // The writer owns the generation until publish; on a device failure it is
  // destroyed here (scope exit), which deletes the unpublished generation
  // --- including an adopted prefix --- per the spec's discard contract.
  // Heap-allocated so the simulated-kill path can leak it deliberately
  // (a real SIGKILL never runs destructors either).
  auto writer =
      std::make_unique<ForwardDrizzleV2StoreWriter>(store_root, plan);
  int first_band = 0;
  if (insp.status == ForwardDrizzleV2StoreStatus::resumable) {
    writer->adopt(insp.generation, insp.next_band, insp.committed);
    first_band = insp.next_band;
    result.bands_reused += insp.next_band;
  } else {
    writer->begin();
  }

  const int cols = plan.native_width;  // full-width bands only
  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  ForwardDrizzleV2FrameInput in;
  int committed_this_attempt = 0;
  std::uint64_t nonfinite_in_support = 0;

  for (int band = first_band; band < plan.band_count; ++band) {
    // Test hook: pretend the device dies before this band's kernel work.
    if (cuda && g_fd_v2_cuda_fault_after_bands.load() >= 0 &&
        committed_this_attempt >= g_fd_v2_cuda_fault_after_bands.load()) {
      *device_failure = "injected fault after " +
                        std::to_string(committed_this_attempt) + " bands";
      return false;
    }
    const int y_begin = band * plan.band_rows;
    const int rows = std::min(plan.band_rows, plan.native_height - y_begin);
    if (options.progress) options.progress(band, plan.band_count);

    auto kernel = make_kernel(cuda);
    if (!kernel->reserve(cols, rows, source_w, source_h,
                         kernel_config_for_band(plan, y_begin, rows))) {
      if (!cuda) throw std::runtime_error("FDV2_DRIVER_RESERVE_FAILED");
      *device_failure = "reserve() reported a device failure";
      return false;
    }
    for (std::uint64_t fr = 0; fr < plan.frame_count; ++fr) {
      if (!provider(fr, in) || in.source == nullptr)
        throw std::runtime_error("FDV2_DRIVER_FRAME_PROVIDER_FAILED");
      const bool ok =
          in.has_local_model
              ? kernel->accumulate_frame_local(in.affine6, in.warp, in.source,
                                               in.sigma2, fr, &in.quality,
                                               &in.meta)
              : kernel->accumulate_frame(in.affine6, in.source, in.sigma2,
                                         fr, &in.quality, &in.meta);
      if (!ok) {
        if (!cuda)
          throw std::runtime_error("FDV2_DRIVER_ACCUMULATE_FAILED band=" +
                                   std::to_string(band) +
                                   " frame=" + std::to_string(fr));
        *device_failure = "accumulate() reported a device failure";
        return false;
      }
    }
    const std::size_t nrec = static_cast<std::size_t>(rows) * cols *
                             static_cast<std::size_t>(plan.channels);
    records.assign(nrec, ForwardDrizzleV2PixelResult{});
    if (plan.emit_profiles)
      profiles.assign(nrec, ForwardDrizzleV2ProfileResult{});
    std::uint64_t dense = 0;
    if (!kernel->finalize(records.data(),
                          plan.emit_profiles ? profiles.data() : nullptr,
                          &dense)) {
      if (!cuda) throw std::runtime_error("FDV2_DRIVER_FINALIZE_FAILED");
      *device_failure = "finalize() reported a device failure";
      return false;
    }
    // Commit gate input (spec section 19): a nonfinite centre inside source
    // support is a hard defect.
    for (const auto &r : records)
      if (r.source_fraction > 0.0f && !std::isfinite(r.value))
        ++nonfinite_in_support;
    writer->commit_band(band, y_begin, rows, records,
                        plan.emit_profiles
                            ? std::span<const ForwardDrizzleV2ProfileResult>(
                                  profiles)
                            : std::span<const ForwardDrizzleV2ProfileResult>{},
                        dense);
    ++committed_this_attempt;
    ++result.bands_committed;
    add_stats(result.totals, kernel->stats());
    result.local_samples_discarded += kernel->stats().local_samples_discarded;

    // Test-only kill simulation: leak the writer so the unpublished
    // generation survives, then report the "kill".
    if (options.simulate_kill_after_bands >= 0 &&
        committed_this_attempt >= options.simulate_kill_after_bands) {
      writer.release();
      throw ForwardDrizzleV2SimulatedKill("FDV2_DRIVER_SIMULATED_KILL");
    }
  }

  ForwardDrizzleV2CommitGate gate;
  gate.nonfinite_pixels_inside_source_support = nonfinite_in_support;
  gate.bands_processed = static_cast<std::uint64_t>(plan.band_count);
  gate.telemetry_json =
      "{\"backend\":\"" + std::string(cuda ? "cuda_v2" : "cpu_v2") +
      "\",\"frames_processed\":" +
      std::to_string(result.totals.frames_processed) +
      ",\"candidates_streamed\":" +
      std::to_string(result.totals.candidates_streamed) +
      ",\"reservoir_kept_total\":" +
      std::to_string(result.totals.reservoir_kept_total) +
      ",\"local_samples_discarded\":" +
      std::to_string(result.local_samples_discarded) + "}";
  result.generation_dir = writer->finish(gate);
  result.committed = true;
  result.backend_used = cuda ? "cuda_v2" : "cpu_v2";
  return true;
}

}  // namespace

void set_forward_drizzle_v2_cuda_fault_after_bands(int n) {
  g_fd_v2_cuda_fault_after_bands.store(n);
}

int forward_drizzle_v2_cuda_fault_after_bands() {
  read_fault_env_once();
  return g_fd_v2_cuda_fault_after_bands.load();
}

ForwardDrizzleV2DriverResult run_forward_drizzle_v2(
    const fs::path &store_root, const ForwardDrizzleV2RunPlan &plan,
    int source_width, int source_height,
    const ForwardDrizzleV2FrameProvider &provider,
    const ForwardDrizzleV2DriverOptions &options) {
  if (plan.plan_hash.empty())
    throw std::invalid_argument("FDV2_DRIVER_PLAN_NOT_FINALIZED");
  if (plan.x_tiled)
    throw std::invalid_argument("FDV2_DRIVER_X_TILED_UNSUPPORTED");
  if (source_width <= 0 || source_height <= 0 || !provider)
    throw std::invalid_argument("FDV2_DRIVER_INVALID_ARGS");
  read_fault_env_once();

  ForwardDrizzleV2DriverResult result;
  result.bands_total = plan.band_count;

  const bool try_cuda =
      options.prefer_cuda && forward_drizzle_cuda_runtime_available();
  if (try_cuda) {
    std::string device_failure;
    try {
      if (attempt_backend(store_root, plan, source_width, source_height,
                          provider, true, result, options, &device_failure))
        return result;
    } catch (const ForwardDrizzleCudaError &e) {
      device_failure = e.what();
    }
    // Spec: discard the uncommitted generation, restart the WHOLE phase on
    // CPU. The failed attempt's writer already destructed (removing the
    // generation); reset the attempt-scoped accounting.
    result.cuda_fallback_reason = std::move(device_failure);
    result.bands_committed = 0;
    result.bands_reused = 0;
    result.totals = ForwardDrizzleV2PrototypeStats{};
    result.local_samples_discarded = 0;
  }

  if (!attempt_backend(store_root, plan, source_width, source_height,
                       provider, false, result, options, nullptr))
    throw std::runtime_error("FDV2_DRIVER_CPU_BACKEND_FAILED");
  return result;
}

}  // namespace tile_compile::reconstruction
