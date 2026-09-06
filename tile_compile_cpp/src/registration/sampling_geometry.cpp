#include "tile_compile/registration/sampling_geometry.hpp"

#include "tile_compile/core/types.hpp"

#include <nlohmann/json.hpp>

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>

namespace tile_compile::registration {

namespace {

using json = nlohmann::json;

double percentile_double(std::vector<double> values, double pct) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = (pct / 100.0) * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi)
    return values[lo];
  const double frac = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// Exact float order statistics with bounded RAM. Positive finite float bit
// patterns preserve numeric order. Four sequential radix passes select a rank;
// no full-canvas n_eff vector or sort copy is resident.
class DiskQuantile {
  core::AtomicOutput spool{std::filesystem::temp_directory_path() /
                           "drizzle-quantile"};
  struct CloseFile {
    void operator()(FILE *f) const { std::fclose(f); }
  };
  std::unique_ptr<FILE, CloseFile> file{
      std::fopen(spool.path().string().c_str(), "w+b")};
  size_t count = 0;
  float select(size_t rank) {
    uint32_t prefix = 0, mask = 0;
    std::array<uint32_t, 4096> buffer{};
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (std::fflush(file.get()) || std::fseek(file.get(), 0, SEEK_SET))
        throw std::runtime_error("COVERAGE_TEMP_IO");
      std::array<size_t, 256> bins{};
      size_t n;
      while ((n = std::fread(buffer.data(), sizeof(uint32_t), buffer.size(),
                             file.get())))
        for (size_t i = 0; i < n; ++i)
          if ((buffer[i] & mask) == prefix)
            ++bins[(buffer[i] >> shift) & 255];
      if (std::ferror(file.get()))
        throw std::runtime_error("COVERAGE_TEMP_IO");
      int bin = 0;
      while (bin < 255 && rank >= bins[bin]) {
        rank -= bins[bin];
        ++bin;
      }
      prefix |= static_cast<uint32_t>(bin) << shift;
      mask |= 255u << shift;
    }
    return std::bit_cast<float>(prefix);
  }

public:
  DiskQuantile() {
    if (!file)
      throw std::runtime_error("COVERAGE_TEMP_IO: cannot create spool");
  }
  void add(float v) {
    if (!std::isfinite(v) || v < 0 ||
        std::fwrite(&v, sizeof(v), 1, file.get()) != 1)
      throw std::runtime_error(
          "COVERAGE_TEMP_IO: invalid value or write failure");
    ++count;
  }
  double p10() {
    if (!count)
      return 0;
    const double rank = 0.1 * static_cast<double>(count - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank)),
                 hi = static_cast<size_t>(std::ceil(rank));
    const double a = select(lo), b = lo == hi ? a : select(hi);
    return a + (b - a) * (rank - lo);
  }
};

// Connected components on two scanlines. Components touching outside the
// independent reference footprint are boundary loss, not interior holes.
class StripeHoles {
  struct Node {
    int parent;
    long long area;
    bool exterior;
  };
  int width;
  std::vector<int> previous;
  std::vector<uint8_t> previous_ref;
  std::vector<Node> components;
  long long largest = 0;

public:
  explicit StripeHoles(int w) : width(w), previous(w, -1), previous_ref(w, 0) {}
  void row(const uint8_t *reference, const uint8_t *support) {
    std::vector<Node> nodes = components;
    const int old = nodes.size();
    nodes.reserve(old + width);
    std::vector<int> current(width, -1);
    auto root = [&](int n) {
      while (nodes[n].parent != n) {
        nodes[n].parent = nodes[nodes[n].parent].parent;
        n = nodes[n].parent;
      }
      return n;
    };
    auto join = [&](int a, int b) {
      a = root(a);
      b = root(b);
      if (a != b) {
        nodes[b].parent = a;
        nodes[a].area += nodes[b].area;
        nodes[a].exterior = nodes[a].exterior || nodes[b].exterior;
      }
    };
    for (int x = 0; x < width; ++x) {
      if (previous[x] >= 0 && !reference[x])
        nodes[root(previous[x])].exterior = true;
      if (!reference[x] || support[x])
        continue;
      int id = nodes.size();
      nodes.push_back({id, 1,
                       x == 0 || x == width - 1 || !previous_ref[x] ||
                           (x > 0 && !reference[x - 1]) ||
                           (x + 1 < width && !reference[x + 1])});
      current[x] = id;
      if (x > 0 && current[x - 1] >= 0)
        join(id, current[x - 1]);
      if (previous[x] >= 0)
        join(id, previous[x]);
    }
    std::vector<int> remap(nodes.size(), -1);
    std::vector<Node> next;
    next.reserve(width);
    for (int x = 0; x < width; ++x)
      if (current[x] >= 0) {
        int r = root(current[x]);
        if (remap[r] < 0) {
          int id = next.size();
          remap[r] = id;
          next.push_back({id, nodes[r].area, nodes[r].exterior});
        }
        current[x] = remap[r];
      }
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
      if (root(i) == i && remap[i] < 0 && !nodes[i].exterior)
        largest = std::max(largest, nodes[i].area);
    previous = std::move(current);
    components = std::move(next);
    std::copy(reference, reference + width, previous_ref.begin());
  }
  long long result() const {
    return largest;
  } // active last-row components touch the exterior
};

} // namespace

// Interior-hole detection on `support_mask` (W x H, 4-connectivity):
//   1. flood-fill from every unsupported border pixel -> "exterior"
//   2. any unsupported pixel not reached is an interior-hole pixel
//   3. connected-component-label interior-hole pixels, return the largest
//      component's area (0 if there are none)
int largest_interior_hole_area(const std::vector<uint8_t> &support_mask, int W,
                               int H) {
  if (W <= 0 || H <= 0)
    return 0;
  if (support_mask.size() != static_cast<size_t>(W) * H)
    throw std::invalid_argument("COVERAGE_MASK_SHAPE_MISMATCH");
  StripeHoles holes(W);
  std::vector<uint8_t> reference(W, 1);
  for (int y = 0; y < H; ++y)
    holes.row(reference.data(),
              support_mask.data() + static_cast<size_t>(y) * W);
  return static_cast<int>(
      std::min<long long>(holes.result(), std::numeric_limits<int>::max()));
}

// Plan section 9.3/8.x circular dither-spread diagnostic. Site coordinates
// are native-canvas (canvas_to_source is defined directly on native-canvas
// coordinates, plan section 7 --- no internal_scale involved, so this needs
// no inversion and no internal-canvas rescale). Diagnostic only, never gates.
DitherSpreadCircularDiagnostic compute_dither_spread_circular_diagnostic(
    const RegistrationSamplingPlan &plan) {
  DitherSpreadCircularDiagnostic result;
  const double W = static_cast<double>(plan.canvas_width_native);
  const double H = static_cast<double>(plan.canvas_height_native);
  if (W <= 0.0 || H <= 0.0)
    return result;

  // center + 4 corners
  const double site_x[5] = {W / 2.0, 0.0, W - 1.0, 0.0, W - 1.0};
  const double site_y[5] = {H / 2.0, 0.0, 0.0, H - 1.0, H - 1.0};

  std::vector<double> sigma_x_per_site, sigma_y_per_site;
  sigma_x_per_site.reserve(5);
  sigma_y_per_site.reserve(5);

  for (int s = 0; s < 5; ++s) {
    double sum_cos_x = 0.0, sum_sin_x = 0.0, sum_cos_y = 0.0, sum_sin_y = 0.0;
    int n = 0;
    for (const FrameSamplingTransform &f : plan.frames) {
      if (!f.valid)
        continue;
      const WarpMatrix &c2s = f.canvas_to_source;
      const double sx =
          c2s(0, 0) * site_x[s] + c2s(0, 1) * site_y[s] + c2s(0, 2);
      const double sy =
          c2s(1, 0) * site_x[s] + c2s(1, 1) * site_y[s] + c2s(1, 2);
      if (!std::isfinite(sx) || !std::isfinite(sy))
        continue;
      double mx = std::fmod(sx, 2.0);
      if (mx < 0.0)
        mx += 2.0;
      double my = std::fmod(sy, 2.0);
      if (my < 0.0)
        my += 2.0;
      const double theta_x = M_PI * mx;
      const double theta_y = M_PI * my;
      sum_cos_x += std::cos(theta_x);
      sum_sin_x += std::sin(theta_x);
      sum_cos_y += std::cos(theta_y);
      sum_sin_y += std::sin(theta_y);
      ++n;
    }
    if (n == 0) {
      sigma_x_per_site.push_back(0.0);
      sigma_y_per_site.push_back(0.0);
      continue;
    }
    double Rx = std::sqrt(sum_cos_x * sum_cos_x + sum_sin_x * sum_sin_x) / n;
    double Ry = std::sqrt(sum_cos_y * sum_cos_y + sum_sin_y * sum_sin_y) / n;
    // R in [0,1] up to floating-point slop; clamp so ln() stays finite.
    Rx = std::min(std::max(Rx, 1e-12), 1.0 - 1e-12);
    Ry = std::min(std::max(Ry, 1e-12), 1.0 - 1e-12);
    sigma_x_per_site.push_back(std::sqrt(-2.0 * std::log(Rx)) / M_PI);
    sigma_y_per_site.push_back(std::sqrt(-2.0 * std::log(Ry)) / M_PI);
  }

  result.x_p10 = percentile_double(sigma_x_per_site, 10.0);
  result.y_p10 = percentile_double(sigma_y_per_site, 10.0);
  return result;
}

GeometricCoverageResult compute_geometric_coverage(
    const RegistrationSamplingPlan &plan, int internal_scale, float pixfrac,
    const config::ReconstructionCoverageGateConfig &gate_cfg, float fraction,
    int num_workers, const config::ReconstructionDrizzleConfig &resources,
    bool retain_channel_counts) {
  using namespace reconstruction;
  (void)num_workers; // deterministic bounded reference, no per-worker canvases
  if (!std::isfinite(fraction) || fraction <= 0 || fraction > 1)
    throw std::invalid_argument("COVERAGE_INVALID_COMMON_FRACTION");
  config::ReconstructionDrizzleConfig cfg = resources;
  cfg.internal_scale = internal_scale;
  cfg.pixfrac = pixfrac;
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  const auto initial = plan_drizzle_memory(plan, cfg, 1, 0, false);
  const size_t pixels = static_cast<size_t>(initial.width) * initial.height;
  if (pixels > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("COVERAGE_GEOMETRY_TOO_LARGE");
  // Full masks are published by COMMON_OVERLAP; only these two byte planes
  // remain resident in the production caller. Optional counts are for tests.
  const size_t retained =
      pixels * (2 + (retain_channel_counts ? channels * sizeof(uint32_t) : 0));
  const size_t row_scratch =
      static_cast<size_t>(initial.width) * channels * 256;
  const auto memory = plan_drizzle_memory(
      plan, cfg, channels * (3 * sizeof(double) + sizeof(uint32_t) + 1) + 5,
      retained + row_scratch, false);
  const auto space =
      std::filesystem::space(std::filesystem::temp_directory_path());
  const size_t disk_needed =
      pixels * channels * sizeof(float) + 64 * 1024 * 1024;
  if (space.available < disk_needed)
    throw std::runtime_error(
        "COVERAGE_TEMP_SPACE: insufficient quantile spool space");
  auto prepared = prepare_drizzle_frames(plan, cfg);
  GeometricCoverageResult result;
  result.local_samples_total = prepared.diagnostics.local_model_samples_total;
  result.local_samples_discarded =
      prepared.diagnostics.local_model_samples_discarded;
  result.excluded_frames =
      prepared.diagnostics.frames_excluded_subdivision_error_rate;
  result.internal_width = memory.width;
  result.internal_height = memory.height;
  result.analysis_common_mask.assign(pixels, 0);
  result.reconstruction_support_mask.assign(pixels, 0);
  std::array<std::vector<uint32_t> *, 3> counts =
      channels == 1
          ? std::array<std::vector<uint32_t> *, 3>{&result.support_count_l,
                                                   nullptr, nullptr}
          : std::array<std::vector<uint32_t> *, 3>{&result.support_count_r,
                                                   &result.support_count_g,
                                                   &result.support_count_b};
  if (retain_channel_counts)
    for (int c = 0; c < channels; ++c)
      counts[c]->assign(pixels, 0);
  const int N = prepared.frames.size();
  const int required = std::max(
      1, static_cast<int>(std::ceil(static_cast<double>(fraction) * N)));
  std::array<std::unique_ptr<DiskQuantile>, 3> quantiles;
  std::array<std::unique_ptr<StripeHoles>, 3> holes;
  std::array<size_t, 3> supported{};
  for (int c = 0; c < channels; ++c) {
    quantiles[c] = std::make_unique<DiskQuantile>();
    holes[c] = std::make_unique<StripeHoles>(memory.width);
  }
  auto &gate = result.gate;
  gate.valid_frame_count = N;
  gate.estimated_peak_bytes = memory.estimated_peak_bytes;
  gate.resolved_chunk_rows = memory.rows;
  for (int y = 0; y < memory.height; y += memory.rows) {
    const int rows = std::min(memory.rows, memory.height - y);
    const size_t n = static_cast<size_t>(memory.width) * rows,
                 offset = static_cast<size_t>(y) * memory.width;
    std::array<std::vector<double>, 3> B, w, w2;
    std::array<std::vector<uint32_t>, 3> count;
    std::array<std::vector<uint8_t>, 3> support;
    for (int c = 0; c < channels; ++c) {
      B[c].assign(n, 0);
      w[c].assign(n, 0);
      w2[c].assign(n, 0);
      count[c].assign(n, 0);
      support[c].assign(n, 0);
    }
    std::vector<uint32_t> footprint_count(n, 0);
    std::vector<uint8_t> touched(n, 0);
    for (const auto *f : prepared.frames) {
      for (int c = 0; c < channels; ++c)
        std::fill(B[c].begin(), B[c].end(), 0);
      rasterize_drizzle_stripe(
          plan, *f, internal_scale, pixfrac, y, rows,
          [&](int, int, int c, size_t i, double k) { B[c][i] += k; });
      for (int c = 0; c < channels; ++c)
        for (size_t i = 0; i < n; ++i)
          if (B[c][i] > 0) {
            ++count[c][i];
            w[c][i] += B[c][i];
            w2[c][i] += B[c][i] * B[c][i];
          }
      std::fill(touched.begin(), touched.end(), 0);
      // Dense source pixel squares define full-frame footprints, independently
      // of CFA colour and the shrunken reconstruction droplet.
      rasterize_drizzle_stripe(
          plan, *f, internal_scale, 1.0f, y, rows,
          [&](int, int, int, size_t i, double) { touched[i] = 1; });
      for (size_t i = 0; i < n; ++i)
        footprint_count[i] += touched[i];
    }
    for (size_t i = 0; i < n; ++i) {
      const bool reference =
          footprint_count[i] >= static_cast<uint32_t>(required);
      result.analysis_common_mask[offset + i] = reference;
      if (reference)
        ++gate.analysis_pixels;
      bool all = true;
      for (int c = 0; c < channels; ++c) {
        support[c][i] = count[c][i] > 0;
        all = all && support[c][i];
        if (retain_channel_counts)
          (*counts[c])[offset + i] = count[c][i];
        if (reference) {
          supported[c] += support[c][i];
          quantiles[c]->add(
              w2[c][i] > 0 ? static_cast<float>(w[c][i] * w[c][i] / w2[c][i])
                           : 0.0f);
        }
      }
      result.reconstruction_support_mask[offset + i] = all;
    }
    for (int row = 0; row < rows; ++row)
      for (int c = 0; c < channels; ++c)
        holes[c]->row(result.analysis_common_mask.data() + offset +
                          static_cast<size_t>(row) * memory.width,
                      support[c].data() +
                          static_cast<size_t>(row) * memory.width);
  }
  gate.min_supported_fraction = 1;
  gate.min_channel_n_eff_p10 = std::numeric_limits<double>::infinity();
  const double required_neff =
      std::max(static_cast<double>(gate_cfg.min_channel_n_eff_floor),
               static_cast<double>(gate_cfg.min_channel_n_eff_fraction) * N);
  for (int c = 0; c < channels; ++c) {
    const std::string name = channels == 1 ? "L"
                             : c == 0      ? "R"
                             : c == 1      ? "G"
                                           : "B";
    const double coverage =
        gate.analysis_pixels
            ? static_cast<double>(supported[c]) / gate.analysis_pixels
            : 0;
    const double neff = quantiles[c]->p10();
    const long long hole = holes[c]->result();
    gate.supported_fraction[c] = coverage;
    gate.channel_neff_p10[c] = neff;
    gate.channel_hole_area[c] = hole;
    gate.min_supported_fraction =
        std::min(gate.min_supported_fraction, coverage);
    gate.min_channel_n_eff_p10 = std::min(gate.min_channel_n_eff_p10, neff);
    gate.largest_internal_hole_area_px =
        std::max(gate.largest_internal_hole_area_px, static_cast<int>(hole));
    if (coverage < gate_cfg.min_supported_fraction)
      gate.violations.push_back("coverage_gate.min_supported_fraction " + name +
                                ": " + std::to_string(coverage));
    if (neff < required_neff)
      gate.violations.push_back("coverage_gate.min_channel_n_eff " + name +
                                ": " + std::to_string(neff));
    if (hole > gate_cfg.max_internal_hole_area_px)
      gate.violations.push_back("coverage_gate.max_internal_hole_area_px " +
                                name + ": " + std::to_string(hole));
  }
  if (N < gate_cfg.min_frames)
    gate.violations.push_back("coverage_gate.min_frames: " + std::to_string(N));
  if (gate.analysis_pixels < gate_cfg.min_analysis_pixels)
    gate.violations.push_back(
        "coverage_gate.min_analysis_pixels: insufficient_analysis_support");
  gate.passed = gate.violations.empty();
  result.dither_spread_circular =
      compute_dither_spread_circular_diagnostic(plan);
  return result;
}

std::string
compute_coverage_geometry_hash(const RegistrationSamplingPlan &plan,
                               const config::ReconstructionDrizzleConfig &cfg,
                               float common_fraction) {
  const std::string prefix = "coverage:v2:edge-centers:exact-polygons:dense-"
                             "footprint:whole-sample-veto:" +
                             compute_plan_hash(plan) + ":" + cfg.kernel;
  std::vector<uint8_t> bytes(prefix.begin(), prefix.end());
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i)
      bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  };
  auto f32 = [&](float v) { u32(std::bit_cast<uint32_t>(v)); };
  u32(cfg.internal_scale);
  f32(cfg.pixfrac);
  f32(common_fraction);
  reconstruction::ForwardDrizzleSubdivisionParams sub;
  f32(sub.position_epsilon_internal_px);
  f32(sub.area_relative_epsilon);
  u32(sub.max_subdivision_depth);
  f32(sub.per_frame_inversion_error_rate_max);
  return core::sha256_bytes(bytes);
}

std::string
serialize_sampling_geometry_json(const RegistrationSamplingPlan &plan,
                                 const std::string &coverage_geometry_hash,
                                 const std::string &kernel, float pixfrac,
                                 int internal_scale,
                                 const GeometricCoverageResult &coverage) {
  json j;
  j["schema_version"] = 2;
  j["analysis_region"] = "dense_frame_footprint_overlap";
  j["n_eff_semantics"] = "squared_weight_sum_over_sum_squared_weights";
  j["estimated_peak_bytes"] = coverage.gate.estimated_peak_bytes;
  j["resolved_chunk_rows"] = coverage.gate.resolved_chunk_rows;
  j["workers_used"] = coverage.gate.workers_used;
  j["coverage_source"] = "forward_drizzle_geometry";
  j["kernel"] = kernel;
  j["pixfrac"] = pixfrac;
  j["internal_scale"] = internal_scale;
  j["sampling_plan_hash"] = plan.plan_hash;
  j["coverage_geometry_hash"] = coverage_geometry_hash;
  j["coverage_gate"] = {
      {"passed", coverage.gate.passed},
      {"valid_frames", coverage.gate.valid_frame_count},
      {"analysis_pixels", coverage.gate.analysis_pixels},
      {"min_supported_fraction", coverage.gate.min_supported_fraction},
      {"min_channel_n_eff_p10", coverage.gate.min_channel_n_eff_p10},
      {"hole_check_implemented", coverage.gate.hole_check_implemented},
      {"largest_internal_hole_area_px",
       coverage.gate.largest_internal_hole_area_px},
      {"violations", coverage.gate.violations},
  };
  j["local_model_samples_total"] = coverage.local_samples_total;
  j["local_model_samples_discarded"] = coverage.local_samples_discarded;
  j["frames_excluded_subdivision_error_rate"] = json::array();
  for (const auto &[id, rate] : coverage.excluded_frames)
    j["frames_excluded_subdivision_error_rate"].push_back(
        {{"frame_id", id}, {"rate", rate}});
  const int channels = plan.color_mode == ColorMode::MONO ? 1 : 3;
  for (int c = 0; c < channels; ++c) {
    const std::string name = channels == 1 ? "L"
                             : c == 0      ? "R"
                             : c == 1      ? "G"
                                           : "B";
    j["coverage_gate"]["supported_fraction"][name] =
        coverage.gate.supported_fraction[c];
    j["coverage_gate"]["geometric_uniform_neff_p10"][name] =
        coverage.gate.channel_neff_p10[c];
    j["coverage_gate"]["largest_internal_hole_area_by_channel"][name] =
        coverage.gate.channel_hole_area[c];
  }
  // Plan 9.3/8.x: diagnostic only, never gates the run.
  j["dither_spread_circular_px_diagnostic"] = {
      {"x_p10", coverage.dither_spread_circular.x_p10},
      {"y_p10", coverage.dither_spread_circular.y_p10},
  };
  return j.dump(2);
}

} // namespace tile_compile::registration
