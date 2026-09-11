// Plan section 11.14 P0 --- see the header. Registry singleton + JSON dump.
// Integer/double accumulators and string formatting only; no geometry, no FP
// contraction concerns.

#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"

#include <sstream>

namespace tile_compile::reconstruction::geomstats {

const char *variant_name(Variant v) {
  switch (v) {
  case Variant::kUnattributed:
    return "unattributed";
  case Variant::kPrepareExclusionScan:
    return "prepare_exclusion_scan";
  case Variant::kCoverageCfa:
    return "coverage_cfa";
  case Variant::kCoverageFootprint:
    return "coverage_footprint";
  case Variant::kCoverageAccumulatorReset:
    return "coverage_accumulator_reset";
  case Variant::kCoverageAccumulatorReduction:
    return "coverage_accumulator_reduction";
  case Variant::kCoverageHoleQuantile:
    return "coverage_hole_quantile";
  case Variant::kProductionUniformRaw:
    return "production_uniform_raw";
  case Variant::kUniformDiagnostic:
    return "uniform_diagnostic";
  case Variant::kContribCount:
    return "contrib_count";
  case Variant::kContribFill:
    return "contrib_fill";
  case Variant::kHybridCpuGeometry:
    return "hybrid_cpu_geometry";
  case Variant::kVariantCount:
    break;
  }
  return "unknown";
}

Registry &registry() {
  static Registry instance;
  return instance;
}

void Registry::reset() {
  v = {};
  current = Variant::kUnattributed;
  // `enabled` is managed by ScopedEnable; context fields are re-stamped by the
  // caller each phase.
  source_width = source_height = 0;
  canvas_width_native = canvas_height_native = 0;
  internal_scale = resolved_chunk_rows = stripe_count = 0;
  prepared_frames = prepared_local_frames = 0;
}

void stamp_context(int source_width, int source_height, int canvas_width_native,
                   int canvas_height_native, int internal_scale, int chunk_rows,
                   int internal_height, int prepared_frames,
                   int prepared_local_frames) {
  auto &r = registry();
  if (!r.enabled)
    return;
  r.source_width = source_width;
  r.source_height = source_height;
  r.canvas_width_native = canvas_width_native;
  r.canvas_height_native = canvas_height_native;
  r.internal_scale = internal_scale;
  r.resolved_chunk_rows = chunk_rows;
  r.stripe_count =
      chunk_rows > 0 ? (internal_height + chunk_rows - 1) / chunk_rows : 0;
  r.prepared_frames = prepared_frames;
  r.prepared_local_frames = prepared_local_frames;
}

std::string to_json() {
  const auto &r = registry();
  std::ostringstream o;
  o << "{";
  o << "\"context\":{"
    << "\"source_width\":" << r.source_width
    << ",\"source_height\":" << r.source_height
    << ",\"canvas_width_native\":" << r.canvas_width_native
    << ",\"canvas_height_native\":" << r.canvas_height_native
    << ",\"internal_scale\":" << r.internal_scale
    << ",\"resolved_chunk_rows\":" << r.resolved_chunk_rows
    << ",\"stripe_count\":" << r.stripe_count
    << ",\"prepared_frames\":" << r.prepared_frames
    << ",\"prepared_local_frames\":" << r.prepared_local_frames << "}";
  o << ",\"variants\":{";
  bool first = true;
  for (int i = 0; i < static_cast<int>(Variant::kVariantCount); ++i) {
    const auto &c = r.v[static_cast<std::size_t>(i)];
    // Skip variants that were never touched to keep the artifact compact.
    const bool touched = c.enumerate_calls || c.source_samples_visited ||
                         c.top_level_sample_leaves_calls ||
                         c.leaf_cells_emitted;
    if (!touched)
      continue;
    if (!first)
      o << ",";
    first = false;
    o << "\"" << variant_name(static_cast<Variant>(i)) << "\":{"
      << "\"pixfrac\":" << c.pixfrac
      << ",\"enumerate_calls\":" << c.enumerate_calls
      << ",\"source_rows_scanned\":" << c.source_rows_scanned
      << ",\"source_samples_visited\":" << c.source_samples_visited
      << ",\"top_level_sample_leaves_calls\":" << c.top_level_sample_leaves_calls
      << ",\"sample_leaves_discarded\":" << c.sample_leaves_discarded
      << ",\"leaves_generated\":" << c.leaves_generated
      << ",\"subdivide_local_calls\":" << c.subdivide_local_calls
      << ",\"local_forward_calls\":" << c.local_forward_calls
      << ",\"invert_calls\":" << c.invert_calls
      << ",\"invert_iterations\":" << c.invert_iterations
      << ",\"basis_evaluations\":" << c.invert_iterations
      << ",\"exp_calls\":" << (c.invert_iterations * 16u)
      << ",\"leaf_cells_emitted\":" << c.leaf_cells_emitted
      << ",\"geometry_wall_s\":" << c.geometry_wall_s
      << ",\"geometry_cpu_s\":" << c.geometry_cpu_s << "}";
  }
  o << "}}";
  return o.str();
}

} // namespace tile_compile::reconstruction::geomstats
