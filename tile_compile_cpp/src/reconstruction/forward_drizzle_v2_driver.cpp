// Gate-10 band driver: see the header for the contract. The driver is
// deliberately serial --- the kernel owns all parallelism; the bounded-
// memory contract comes from the kernel's single reserve() per band.

#include "tile_compile/reconstruction/forward_drizzle_v2_driver.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    const ForwardDrizzleV2RunPlan &plan, int y_begin, int rows,
    std::uint64_t cached_leaf_capacity) {
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
  k.cached_leaf_capacity = cached_leaf_capacity;
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
  t.quality_bytes_uploaded += s.quality_bytes_uploaded;
  t.source_samples_launched += s.source_samples_launched;
  t.quality_frames_processed += s.quality_frames_processed;
  t.frames_skipped_empty_window += s.frames_skipped_empty_window;
  t.quality_expanded_floats += s.quality_expanded_floats;
  t.workspace_reservations += s.workspace_reservations;
  t.band_resets += s.band_resets;
  t.cached_leaf_records_launched += s.cached_leaf_records_launched;
  t.cached_leaf_bytes_uploaded += s.cached_leaf_bytes_uploaded;
  t.affine_pieces_processed += s.affine_pieces_processed;
  t.affine_samples_processed += s.affine_samples_processed;
  t.affine_span_rows += s.affine_span_rows;
  t.reserved_device_bytes =
      std::max(t.reserved_device_bytes, s.reserved_device_bytes);
  t.upload_seconds += s.upload_seconds;
  t.kernel_seconds += s.kernel_seconds;
  t.download_seconds += s.download_seconds;
  t.max_frame_seconds = std::max(t.max_frame_seconds, s.max_frame_seconds);
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
  // One persistent kernel workspace per backend attempt: reserve for the
  // maximum band height once, rebind per band via begin_band. On a device
  // failure the kernel destructs (frees once) and the CPU attempt creates
  // its own single workspace.
  auto kernel = make_kernel(cuda);
  // Appends the backend's last_device_error() detail (CUDA error string)
  // when present so the fallback reason identifies the actual failure.
  auto fail_device = [&](const char *where) {
    *device_failure = std::string(where) + " reported a device failure";
    const std::string detail = kernel->last_device_error();
    if (!detail.empty()) *device_failure += ": " + detail;
  };
  if (!kernel->reserve(
          cols, plan.band_rows, source_w, source_h,
          kernel_config_for_band(plan, 0, plan.band_rows,
                                 options.cached_leaf_capacity))) {
    if (!cuda) throw std::runtime_error("FDV2_DRIVER_RESERVE_FAILED");
    fail_device("reserve()");
    return false;
  }
  std::vector<ForwardDrizzleV2PixelResult> records;
  std::vector<ForwardDrizzleV2ProfileResult> profiles;
  const std::size_t max_nrec =
      static_cast<std::size_t>(plan.band_rows) * cols *
      static_cast<std::size_t>(plan.channels);
  records.reserve(max_nrec);
  if (plan.emit_profiles) profiles.reserve(max_nrec);
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

    const std::size_t rec_cap0 =
        records.capacity() + profiles.capacity();
    if (!kernel->begin_band(
            rows, kernel_config_for_band(plan, y_begin, rows,
                                         options.cached_leaf_capacity))) {
      if (!cuda)
        throw std::runtime_error("FDV2_DRIVER_BEGIN_BAND_FAILED");
      fail_device("begin_band()");
      return false;
    }
    for (std::uint64_t fr = 0; fr < plan.frame_count; ++fr) {
      const auto frame_t0 = std::chrono::steady_clock::now();
      // Piece-stream contract (tranche 7): the provider invokes the sink
      // exactly once with skip=true, or with one-or-more non-skip pieces.
      // Affine pieces carry ordered, non-overlapping native target ranges
      // (individual empty tiles may simply be omitted); local-warp and
      // cached-geometry frames are always exactly one full-width piece.
      // Each piece is dispatched inside the sink call because the provider
      // may reuse its buffers once the call returns.
      bool any_piece = false;
      bool saw_skip = false;
      bool affine_open = false;
      int affine_tx_end = 0;
      bool emit_failed = false;
      bool kernel_ok = true;
      ForwardDrizzleV2FrameInput first_piece{};
      const ForwardDrizzleV2FramePieceSink sink =
          [&](const ForwardDrizzleV2FrameInput &piece) -> bool {
        if (emit_failed) return false;
        auto fail = [&]() {
          emit_failed = true;
          return false;
        };
        if (piece.skip) {
          if (any_piece) return fail();  // mixed skip/non-skip
          any_piece = true;
          saw_skip = true;
          kernel_ok = kernel->skip_frame(fr, &piece.meta);
          return kernel_ok;
        }
        if (saw_skip) return fail();  // non-skip after skip
        // A zero-sized window means "full source" (the FrameInput default);
        // anything else must be a positive buffer contained in the source
        // with a positive active rect contained in the buffer. The
        // tranche-8 sample path carries no window/source pointer at all.
        ForwardDrizzleV2SourceWindow w = piece.source_window;
        if (w.width == 0 && w.height == 0 && w.x_begin == 0 &&
            w.y_begin == 0)
          w = {0, 0, source_w, source_h, 0, 0, 0, 0};
        int aw = w.active_width, ah = w.active_height;
        if (aw == 0 && ah == 0 && w.active_x == 0 && w.active_y == 0) {
          aw = w.width;
          ah = w.height;
        }
        if (piece.has_affine_samples) {
          // One-shot full-target piece: canonical sample list, no source
          // buffer/window, no target-tile fields.
          if (piece.affine_samples == nullptr ||
              piece.affine_sample_count == 0 ||
              piece.affine_sample_count >
                  static_cast<std::size_t>(source_w) * source_h ||
              piece.target_x_begin_native != 0 ||
              piece.target_cols_native != 0)
            return fail();
        } else if (piece.source == nullptr || w.x_begin < 0 ||
                   w.y_begin < 0 || w.width <= 0 || w.height <= 0 ||
                   w.x_begin + w.width > source_w ||
                   w.y_begin + w.height > source_h || w.active_x < 0 ||
                   w.active_y < 0 || aw <= 0 || ah <= 0 ||
                   w.active_x + aw > w.width || w.active_y + ah > w.height)
          return fail();
        const bool single_piece_mode =
            piece.has_cached_geometry || piece.has_local_model ||
            piece.has_affine_samples;
        if (single_piece_mode) {
          // Local/cached frames are one full-width piece; a second piece
          // or an affine piece first is a malformed stream.
          if (any_piece || affine_open) return fail();
          // target_cols_native == 0 is the full-width compatibility form.
          if (piece.target_x_begin_native != 0 ||
              (piece.target_cols_native != 0 &&
               piece.target_cols_native != plan.native_width))
            return fail();
        } else {
          // Affine piece: ordered, non-overlapping native target range.
          // 0 cols resolves to the full-width compatibility piece.
          const int tx0 = piece.target_x_begin_native;
          const int tw = piece.target_cols_native == 0
                             ? plan.native_width
                             : piece.target_cols_native;
          if (tx0 < 0 || tw <= 0 || tx0 + tw > plan.native_width ||
              tx0 < affine_tx_end)
            return fail();
        }
        if (any_piece) {
          // All pieces of one frame must share transform/meta/mode.
          if (first_piece.has_local_model != piece.has_local_model ||
              first_piece.has_cached_geometry != piece.has_cached_geometry ||
              first_piece.has_affine_samples != piece.has_affine_samples ||
              std::memcmp(first_piece.affine6, piece.affine6,
                          sizeof(piece.affine6)) != 0 ||
              std::memcmp(&first_piece.meta, &piece.meta,
                          sizeof(piece.meta)) != 0)
            return fail();
        } else {
          first_piece = piece;
        }
        bool ok = false;
        if (piece.has_cached_geometry) {
          // A cached-geometry frame must carry a non-empty leaf list;
          // empty frames are the provider's `skip` contract.
          if (piece.cached_leaves == nullptr || piece.cached_leaf_count == 0)
            return fail();
          ok = kernel->accumulate_frame_cached_leaves(
              piece.affine6, w, piece.source, piece.sigma2,
              &piece.sigma2_model, piece.cached_leaves,
              piece.cached_leaf_count, piece.cached_unique_source_samples,
              fr, &piece.quality, &piece.meta);
        } else if (piece.has_affine_samples) {
          ok = kernel->accumulate_frame_affine_samples(
              piece.affine6, piece.affine_samples, piece.affine_sample_count,
              piece.sigma2_present, &piece.aligned_quality, fr, &piece.meta);
        } else if (piece.has_local_model) {
          ok = kernel->accumulate_frame_local_window(
              piece.affine6, piece.warp, w, piece.source, piece.sigma2,
              &piece.sigma2_model, fr, &piece.quality, &piece.meta);
        } else {
          if (!affine_open) {
            if (!kernel->begin_affine_frame(fr, &piece.meta))
              return fail();
            affine_open = true;
          }
          const int tw = piece.target_cols_native == 0
                             ? plan.native_width
                             : piece.target_cols_native;
          ok = kernel->accumulate_affine_piece(
              piece.affine6, piece.target_x_begin_native, tw, w,
              piece.source, piece.sigma2, &piece.sigma2_model,
              &piece.quality);
          affine_tx_end = piece.target_x_begin_native + tw;
        }
        if (!ok) {
          kernel_ok = false;
          return fail();
        }
        any_piece = true;
        return true;
      };
      if (!provider(y_begin, rows, fr, sink))
        throw std::runtime_error("FDV2_DRIVER_FRAME_PROVIDER_FAILED");
      if (!any_piece || emit_failed) {
        if (!kernel_ok && cuda) {
          fail_device("accumulate()");
          return false;
        }
        throw std::runtime_error("FDV2_DRIVER_FRAME_PROVIDER_FAILED");
      }
      if (affine_open && !kernel->finish_affine_frame(fr)) {
        if (!cuda)
          throw std::runtime_error("FDV2_DRIVER_ACCUMULATE_FAILED band=" +
                                   std::to_string(band) +
                                   " frame=" + std::to_string(fr));
        fail_device("accumulate()");
        return false;
      }
      if (!kernel_ok) {
        if (!cuda)
          throw std::runtime_error("FDV2_DRIVER_ACCUMULATE_FAILED band=" +
                                   std::to_string(band) +
                                   " frame=" + std::to_string(fr));
        fail_device("accumulate()");
        return false;
      }
      result.max_provider_enqueue_seconds = std::max(
          result.max_provider_enqueue_seconds,
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        frame_t0)
              .count());
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
      fail_device("finalize()");
      return false;
    }
    // Commit gate input (spec section 19): a nonfinite centre inside source
    // support is a hard defect.
    for (const auto &r : records)
      if (r.source_fraction > 0.0f && !std::isfinite(r.value))
        ++nonfinite_in_support;
    const auto commit_t0 = std::chrono::steady_clock::now();
    writer->commit_band(band, y_begin, rows, records,
                        plan.emit_profiles
                            ? std::span<const ForwardDrizzleV2ProfileResult>(
                                  profiles)
                            : std::span<const ForwardDrizzleV2ProfileResult>{},
                        dense);
    result.commit_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - commit_t0)
        .count();
    ++committed_this_attempt;
    ++result.bands_committed;
    if (records.capacity() + profiles.capacity() > rec_cap0)
      ++result.driver_hotpath_allocations;
    add_stats(result.totals, kernel->stats());
    result.local_samples_discarded += kernel->stats().local_samples_discarded;
    // Device-side worst frame: event deltas reported by the kernel.
    result.max_frame_seconds =
        std::max(result.max_frame_seconds,
                 kernel->stats().max_frame_seconds);

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
  const auto finish_t0 = std::chrono::steady_clock::now();
  result.generation_dir = writer->finish(gate);
  result.commit_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - finish_t0)
      .count();
  // Record the published commit hash for run provenance/checkpointing
  // (the complete-resume path reads it back via inspect()).
  {
    std::ifstream commit_in(result.generation_dir / "commit.json");
    result.commit_hash = nlohmann::json::parse(commit_in)
                             .value("commit_hash", std::string{});
  }
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
  const auto phase_t0 = std::chrono::steady_clock::now();

  const bool try_cuda =
      options.prefer_cuda && forward_drizzle_cuda_runtime_available();
  if (try_cuda) {
    std::string device_failure;
    try {
      if (attempt_backend(store_root, plan, source_width, source_height,
                          provider, true, result, options, &device_failure)) {
        result.phase_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_t0)
            .count();
        return result;
      }
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
    result.max_frame_seconds = 0.0;
    result.max_provider_enqueue_seconds = 0.0;
    result.commit_seconds = 0.0;
    result.driver_hotpath_allocations = 0;
  }

  if (!attempt_backend(store_root, plan, source_width, source_height,
                       provider, false, result, options, nullptr))
    throw std::runtime_error("FDV2_DRIVER_CPU_BACKEND_FAILED");
  result.phase_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - phase_t0)
      .count();
  return result;
}

}  // namespace tile_compile::reconstruction
