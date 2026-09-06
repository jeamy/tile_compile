#pragma once

// Bounded stripe coverage with the exact signal droplet kernel. The analysis
// mask is based on full frame footprints, independent of sparse CFA support.

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tile_compile::registration {

struct CoverageGateResult {
  std::array<double, 3> supported_fraction{};
  std::array<double, 3> channel_neff_p10{};
  std::array<long long, 3> channel_hole_area{};
  size_t estimated_peak_bytes = 0;
  int resolved_chunk_rows = 0;
  int workers_used = 1;
  bool passed = false;
  bool hole_check_implemented = true;
  std::vector<std::string> violations; // human-readable, one per failed check

  int valid_frame_count = 0;
  int analysis_pixels = 0;
  // Per active channel (R/G/B for OSC, L for MONO), then the conservative
  // minimum across channels (plan 14.4-style: a dense channel must not mask a
  // sparse one).
  double min_supported_fraction = 0.0;
  double min_channel_n_eff_p10 = 0.0;
  // Area (in internal-canvas pixels) of the largest 4-connected interior hole
  // in reconstruction_support_mask (an unsupported region not reachable from
  // the canvas border without crossing supported pixels). 0 if none.
  int largest_internal_hole_area_px = 0;
};

// Plan section 9.3 / 8.x: circular dither-spread diagnostic, modulo the 2-px
// Bayer supercell period, evaluated at 5 native-canvas sites (center + 4
// corners) using the Rayleigh circular-statistics estimator
// (theta = pi * (offset mod 2), sigma_circ_px = sqrt(-2 * ln(R)) / pi, R the
// mean resultant vector length over valid frames at that site). x_p10/y_p10
// are the 10th percentile (worst-of-5-sites, i.e. least phase diversity)
// across the 5 sites, per axis. Diagnostic only --- NEVER a hard gate (plan
// explicitly: direct rasterized channel coverage/n_eff is authoritative
// because a dither-mod-2 proxy can be wrong under rotation/local warps).
struct DitherSpreadCircularDiagnostic {
  double x_p10 = 0.0;
  double y_p10 = 0.0;
};

DitherSpreadCircularDiagnostic
compute_dither_spread_circular_diagnostic(const RegistrationSamplingPlan &plan);

struct GeometricCoverageResult {
  int internal_width = 0;
  int internal_height = 0;

  DitherSpreadCircularDiagnostic dither_spread_circular;

  // Present for OSC (R/G/B) or MONO (L only; R/G/B left empty).
  std::vector<uint32_t> support_count_r;
  std::vector<uint32_t> support_count_g;
  std::vector<uint32_t> support_count_b;
  std::vector<uint32_t> support_count_l;

  // Dense frame-footprint overlap; independent of channel droplet support.
  std::vector<uint8_t> analysis_common_mask;
  // mindestens ein nutzbarer Frame trägt bei, konservativ über die aktiven
  // Kanäle (plan 9.3).
  std::vector<uint8_t> reconstruction_support_mask;

  long long local_samples_total = 0;
  long long local_samples_discarded = 0;
  std::vector<std::pair<std::string, double>> excluded_frames;
  CoverageGateResult gate;
};

// num_workers is retained for source compatibility. The bounded CPU reference
// currently processes one stripe/frame at a time regardless of this hint.
// resources controls chunk_rows and memory_budget_mb (0 => 512 MiB in library;
// runner resolves 0 from runtime_limits). Production omits full channel counts.
GeometricCoverageResult compute_geometric_coverage(
    const RegistrationSamplingPlan &plan, int internal_scale, float pixfrac,
    const config::ReconstructionCoverageGateConfig &gate_cfg,
    float common_overlap_required_fraction, int num_workers = 0,
    const config::ReconstructionDrizzleConfig &resources = {},
    bool retain_channel_counts = true);

// Area (in pixels) of the largest 4-connected "interior hole" in a W x H
// boolean support mask: an unsupported (0) region that cannot reach the
// mask's border without crossing a supported (nonzero) pixel. 0 if there is
// none. Exposed publicly (used internally by compute_geometric_coverage, and
// directly unit-testable / reusable for any other boolean coverage mask).
int largest_interior_hole_area(const std::vector<uint8_t> &support_mask,
                               int width, int height);

std::string
compute_coverage_geometry_hash(const RegistrationSamplingPlan &plan,
                               const config::ReconstructionDrizzleConfig &cfg,
                               float common_fraction);

// artifacts/sampling_geometry.json payload (plan section 9.4). Independent of
// gate outcome --- always serializable, even for a failing gate.
std::string serialize_sampling_geometry_json(
    const RegistrationSamplingPlan &plan,
    const std::string &coverage_geometry_hash, const std::string &kernel,
    float pixfrac, int internal_scale, const GeometricCoverageResult &coverage);

} // namespace tile_compile::registration
