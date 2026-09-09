// Plan section 11.14 P0 --- forward-drizzle geometry instrumentation.
//
// NON-INVASIVE diagnostic counters + coarse sub-phase timers for the
// CFA-forward-drizzle geometry path (`enumerate_drizzle_stripe_leaf_cells` ->
// `sample_leaves` -> `subdivide_local` / `local_forward` ->
// `invert_local_source_to_canvas` -> `smooth_local_basis`) and its callers
// (coverage geometry, the production uniform/raw stream, the contrib-list count
// and fill passes, and the hybrid CPU-geometry path).
//
// Purpose: measure --- per phase, per frame class and per geometry VARIANT ---
// how many source samples are visited, how many top-level `sample_leaves`
// calls run, how many fixed-point inversion iterations / RBF basis evaluations
// those cost, and how that scales with the stripe count K (chunk height). This
// is what confirms or refutes the O(frames * source_pixels * stripes *
// consumers) term named in plan 11.14.1.
//
// HARD CONSTRAINT (plan 11.14.2): these counters must not change any
// compute-hash / pixel behaviour. Every field is a plain integer or double
// accumulator; increments are guarded by `registry().enabled` (a single
// well-predicted branch) and never touch floating-point geometry, control flow,
// reduction order or iteration bounds. Not thread-safe by design --- the
// forward-drizzle reference path is single-threaded today; deterministic
// parallelism is P3 and owns the atomics/merge story.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>

namespace tile_compile::reconstruction::geomstats {

// A geometry "variant" is one (consumer, pixfrac-class) enumeration context.
// pixfrac matters (plan 11.14.3): the dense footprint at pixfrac = 1 and the
// drizzle coverage droplet at pixfrac < 1 are DIFFERENT variants even though
// they share a call site, and P1/P2 dedup must not conflate them.
enum class Variant : int {
  kUnattributed = 0,
  kPrepareExclusionScan, // prepare_drizzle_frames: upfront full source sweep
  kCoverageCfa,          // sampling_geometry: CFA droplet at cfg.pixfrac
  kCoverageFootprint,    // sampling_geometry: dense footprint at pixfrac = 1
  kProductionUniformRaw, // stream_forward_drizzle_uniform_and_raw main loop
  kUniformDiagnostic,    // stream_forward_drizzle_uniform (non-production diag)
  kContribCount,         // build_frame_records: pre-count pass
  kContribFill,          // build_frame_records: fill pass
  kHybridCpuGeometry,    // build_frame_records_hybrid_local: CPU geometry pass
  kVariantCount
};

const char *variant_name(Variant v);

struct VariantCounters {
  // Enumeration structure.
  std::uint64_t enumerate_calls = 0;        // enumerate_drizzle_stripe_leaf_cells
  std::uint64_t source_rows_scanned = 0;    // sum of (source_y1 - source_y0)
  std::uint64_t source_samples_visited = 0; // inner (sx, sy) iterations

  // Per-sample work. `top_level_sample_leaves_calls` is the P2 acceptance
  // number (plan 11.14.4): it must equal eligible_local_frames * source_pixels
  // per variant and stay invariant across chunk heights. Counted ONLY at the
  // top of `sample_leaves`, never in `subdivide_local` recursion.
  std::uint64_t top_level_sample_leaves_calls = 0;
  std::uint64_t sample_leaves_discarded = 0; // returned false (whole sample)
  std::uint64_t leaves_generated = 0;        // accepted leaves pushed

  // Local-warp numeric cost.
  std::uint64_t subdivide_local_calls = 0; // every recursion level
  std::uint64_t local_forward_calls = 0;   // droplet probe warp evaluations
  std::uint64_t invert_calls = 0;          // invert_local_source_to_canvas
  // Newton steps actually executed. Each step runs exactly one
  // local_displacement_render_units -> evaluate_smooth_local_displacement ->
  // smooth_local_basis, i.e. 16 std::exp calls. So basis_evaluations ==
  // invert_iterations and exp_calls == 16 * invert_iterations.
  std::uint64_t invert_iterations = 0;

  // Output.
  std::uint64_t leaf_cells_emitted = 0; // sink() invocations

  double pixfrac = 0.0; // recorded for the variant (footprint = 1.0)

  // Coarse geometry timing for this variant: wall + CPU seconds spent inside
  // enumerate_drizzle_stripe_leaf_cells calls attributed here (includes the
  // polygon/rect area work done in the rasterize_drizzle_stripe sink).
  double geometry_wall_s = 0.0;
  double geometry_cpu_s = 0.0;
};

struct Registry {
  std::array<VariantCounters, static_cast<std::size_t>(Variant::kVariantCount)>
      v{};
  Variant current = Variant::kUnattributed;
  bool enabled = false; // set only around the instrumented phases

  // Context recorded once per phase for the report.
  int source_width = 0;
  int source_height = 0;
  int canvas_width_native = 0;
  int canvas_height_native = 0;
  int internal_scale = 0;
  int resolved_chunk_rows = 0;
  int stripe_count = 0; // K
  int prepared_frames = 0;
  int prepared_local_frames = 0;

  VariantCounters &cur() {
    return v[static_cast<std::size_t>(current)];
  }
  void reset();
};

Registry &registry();

// Stamp the per-phase geometry context (no-op unless the facility is enabled).
// `chunk_rows` is the resolved stripe height; `internal_height` the total
// internal-canvas height; K is derived as ceil(internal_height / chunk_rows).
void stamp_context(int source_width, int source_height, int canvas_width_native,
                   int canvas_height_native, int internal_scale, int chunk_rows,
                   int internal_height, int prepared_frames,
                   int prepared_local_frames);

// RAII: mark the active geometry variant (and its pixfrac) for the enclosed
// enumeration. Nesting restores the previous variant.
struct ScopedVariant {
  Variant prev;
  bool active;
  ScopedVariant(Variant variant, double pixfrac) {
    auto &r = registry();
    active = r.enabled;
    prev = r.current;
    if (active) {
      r.current = variant;
      r.v[static_cast<std::size_t>(variant)].pixfrac = pixfrac;
    }
  }
  ~ScopedVariant() {
    if (active)
      registry().current = prev;
  }
  ScopedVariant(const ScopedVariant &) = delete;
  ScopedVariant &operator=(const ScopedVariant &) = delete;
};

// RAII: accumulate wall + CPU seconds into the current variant's geometry
// timers. Cheap enough to wrap a whole enumerate call (once per frame per
// stripe); NEVER put this in a per-sample loop.
struct ScopedGeometryTimer {
  bool active;
  std::chrono::steady_clock::time_point w0;
  std::timespec c0;
  ScopedGeometryTimer() {
    active = registry().enabled;
    if (active) {
      w0 = std::chrono::steady_clock::now();
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);
    }
  }
  ~ScopedGeometryTimer() {
    if (!active)
      return;
    const auto w1 = std::chrono::steady_clock::now();
    std::timespec c1;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
    auto &c = registry().cur();
    c.geometry_wall_s +=
        std::chrono::duration<double>(w1 - w0).count();
    c.geometry_cpu_s += static_cast<double>(c1.tv_sec - c0.tv_sec) +
                        static_cast<double>(c1.tv_nsec - c0.tv_nsec) * 1e-9;
  }
  ScopedGeometryTimer(const ScopedGeometryTimer &) = delete;
  ScopedGeometryTimer &operator=(const ScopedGeometryTimer &) = delete;
};

// Enable/disable the whole facility for a phase and clear counters.
struct ScopedEnable {
  bool prev;
  explicit ScopedEnable(bool on) {
    prev = registry().enabled;
    registry().reset();
    registry().enabled = on;
  }
  ~ScopedEnable() { registry().enabled = prev; }
  ScopedEnable(const ScopedEnable &) = delete;
  ScopedEnable &operator=(const ScopedEnable &) = delete;
};

// Serialise the registry to a JSON object string (stable key order).
std::string to_json();

} // namespace tile_compile::reconstruction::geomstats
