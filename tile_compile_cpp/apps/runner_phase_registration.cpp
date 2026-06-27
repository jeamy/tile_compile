#include "runner_phase_registration.hpp"
#include "runner_shared.hpp"

#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/registration/astrometric_rescue.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"
#include "tile_compile/runner/registration_outlier_utils.hpp"

#include <opencv2/opencv.hpp>

#if __has_include(<opencv2/core/cuda.hpp>)
#include <opencv2/core/cuda.hpp>
#define TILE_COMPILE_PREWARP_HAS_CUDA 1
#else
#define TILE_COMPILE_PREWARP_HAS_CUDA 0
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace registration = tile_compile::registration;

namespace {

constexpr float kPi = 3.14159265358979323846f;

/// @brief Computes required common overlap frames.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int compute_required_common_overlap_frames(int usable_frames) {
  if (usable_frames <= 1) {
    return 1;
  }
  constexpr float kCoverageFraction = 0.02f;
  constexpr int kCoverageFloor = 2;
  constexpr int kCoverageCap = 32;
  int required = static_cast<int>(
      std::ceil(static_cast<float>(usable_frames) * kCoverageFraction));
  required = std::max(required, kCoverageFloor);
  required = std::min(required, kCoverageCap);
  required = std::min(required, usable_frames);
  return std::max(required, 1);
}

/// @brief Implements wrap angle near.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float wrap_angle_near(float angle, float reference) {
  while (angle - reference > kPi) {
    angle -= 2.0f * kPi;
  }
  while (angle - reference < -kPi) {
    angle += 2.0f * kPi;
  }
  return angle;
}

/// @brief Implements unwrap angle sequence.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<float> unwrap_angle_sequence(const std::vector<float> &angles) {
  if (angles.empty()) {
    return {};
  }
  std::vector<float> out = angles;
  for (size_t i = 1; i < out.size(); ++i) {
    out[i] = wrap_angle_near(out[i], out[i - 1]);
  }
  return out;
}

struct TemporalWarpSample {
  float fi = 0.0f;
  float ang = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  float cc = 1.0f;
};

struct ScalarPolyFit {
  VectorXf coeffs;
  float max_abs_residual = std::numeric_limits<float>::infinity();
  bool ok = false;
};

struct WarpPredictionCandidate {
  bool ok = false;
  float ang = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  float score = std::numeric_limits<float>::infinity();
  float res_ang_deg = std::numeric_limits<float>::infinity();
  float res_tx = std::numeric_limits<float>::infinity();
  float res_ty = std::numeric_limits<float>::infinity();
  int support = 0;
  float span = 0.0f;
};

enum class RegistrationProvenance : uint8_t {
  unresolved = 0,
  reference,
  direct_global,
  sequential_refined,
  sequential_rescue,
  temporal_rescue,
  seeded_ecc_rescue,
  local_reference_rescue,
  astrometric_rescue,  // §4.13 — Plate-Solving via ASTAP
  model_global_poly,
  model_local_poly,
  model_interpolated,
  model_blended,
  model_nearest_copy,
};

/// @brief Implements registration provenance name.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
const char *registration_provenance_name(RegistrationProvenance provenance) {
  switch (provenance) {
  case RegistrationProvenance::unresolved:
    return "unresolved";
  case RegistrationProvenance::reference:
    return "reference";
  case RegistrationProvenance::direct_global:
    return "direct_global";
  case RegistrationProvenance::sequential_refined:
    return "sequential_refined";
  case RegistrationProvenance::sequential_rescue:
    return "sequential_rescue";
  case RegistrationProvenance::temporal_rescue:
    return "temporal_rescue";
  case RegistrationProvenance::seeded_ecc_rescue:
    return "seeded_ecc_rescue";
  case RegistrationProvenance::local_reference_rescue:
    return "local_reference_rescue";
  case RegistrationProvenance::astrometric_rescue:
    return "astrometric_rescue";
  case RegistrationProvenance::model_global_poly:
    return "model_global_poly";
  case RegistrationProvenance::model_local_poly:
    return "model_local_poly";
  case RegistrationProvenance::model_interpolated:
    return "model_interpolated";
  case RegistrationProvenance::model_blended:
    return "model_blended";
  case RegistrationProvenance::model_nearest_copy:
    return "model_nearest_copy";
  }
  return "unknown";
}

/// @brief Implements fit weighted poly.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
ScalarPolyFit fit_weighted_poly(const std::vector<float> &xs,
                                const std::vector<float> &ys,
                                const std::vector<float> &weights,
                                int degree) {
  ScalarPolyFit out;
  const int n = static_cast<int>(xs.size());
  if (n <= 0 || ys.size() != xs.size() || weights.size() != xs.size()) {
    return out;
  }
  const int deg = std::max(0, std::min(degree, n - 1));
  Eigen::MatrixXf A(n, deg + 1); // ColMajor — correct for Eigen QR decomposition
  VectorXf b(n);
  for (int i = 0; i < n; ++i) {
    const float w = std::sqrt(std::max(weights[static_cast<size_t>(i)], 1.0e-6f));
    float xpow = 1.0f;
    for (int j = 0; j <= deg; ++j) {
      A(i, j) = w * xpow;
      xpow *= xs[static_cast<size_t>(i)];
    }
    b(i) = w * ys[static_cast<size_t>(i)];
  }
  Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(A); // Eigen::MatrixXf = ColMajor, not Matrix2Df (RowMajor)
  if (qr.rank() <= 0) {
    return out;
  }
  out.coeffs = qr.solve(b);
  if (out.coeffs.size() != deg + 1) {
    return out;
  }
  float max_abs = 0.0f;
  for (int i = 0; i < n; ++i) {
    float pred = 0.0f;
    float xpow = 1.0f;
    for (int j = 0; j < out.coeffs.size(); ++j) {
      pred += out.coeffs(j) * xpow;
      xpow *= xs[static_cast<size_t>(i)];
    }
    const float r = pred - ys[static_cast<size_t>(i)];
    max_abs = std::max(max_abs, std::fabs(r));
  }
  out.max_abs_residual = max_abs;
  out.ok = true;
  return out;
}

/// @brief Implements concatenate affine warps.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
WarpMatrix concatenate_affine_warps(const WarpMatrix &w1,
                                    const WarpMatrix &w2) {
  WarpMatrix result;
  result(0, 0) = w2(0, 0) * w1(0, 0) + w2(0, 1) * w1(1, 0);
  result(0, 1) = w2(0, 0) * w1(0, 1) + w2(0, 1) * w1(1, 1);
  result(1, 0) = w2(1, 0) * w1(0, 0) + w2(1, 1) * w1(1, 0);
  result(1, 1) = w2(1, 0) * w1(0, 1) + w2(1, 1) * w1(1, 1);
  result(0, 2) = w2(0, 0) * w1(0, 2) + w2(0, 1) * w1(1, 2) + w2(0, 2);
  result(1, 2) = w2(1, 0) * w1(0, 2) + w2(1, 1) * w1(1, 2) + w2(1, 2);
  return result;
}

} // namespace

/// @brief Runs phase registration prewarp.
/// @details Part of the global registration, rescue/modeling, common-canvas, and prewarp phase implementation; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool run_phase_registration_prewarp(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, int height, int width,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    const std::shared_ptr<RunnerFrameCache> &frame_cache,
    const std::vector<image::NormalizationScales> &norm_scales,
    const std::vector<FrameMetrics> &frame_metrics,
    const VectorXf &global_weights, const io::FitsHeader &first_header,
    core::AccelerationContext &acceleration, core::EventEmitter &emitter,
    std::ostream &log_file,
    PhaseRegistrationContext &out) {
  config::RegistrationConfig registration_cfg = cfg.registration;

  // Auto-engine: detect conditions where the configured engine would fail and
  // override with triangle_star_matching. Two triggers:
  //   1) Strong field rotation (Alt/Az mount): median |rotation| per frame
  //      exceeds auto_engine_rotation_threshold_deg.
  //   2) ECC failure on dominant bright objects: when engine is
  //      robust_phase_ecc or hybrid_phase_ecc, probe ECC CC on a small sample;
  //      if the median CC of successful ECC results is below reject_cc_min_abs
  //      (or ECC succeeds on fewer than half the probe frames), override.
  // Triggered when auto_engine=true AND the configured engine is one that
  // cannot handle rotation or dominant bright objects well.
  if (registration_cfg.auto_engine && frames.size() >= 3) {
    const bool engine_rotation_blind =
        (registration_cfg.engine == "robust_phase_ecc" ||
         registration_cfg.engine == "hybrid_phase_ecc");

    if (engine_rotation_blind && registration_cfg.allow_rotation) {
      // Probe: measure rotation angle between evenly-spaced frames using
      // a quick triangle_star_matching pass. Simultaneously probe ECC CC
      // to detect failure on dominant bright objects.
      const size_t n_probe = std::min<size_t>(4, frames.size());
      std::vector<float> probe_rotations;
      std::vector<float> probe_ecc_cc;
      int probe_ecc_attempted = 0;
      probe_rotations.reserve(n_probe);
      probe_ecc_cc.reserve(n_probe);

      const int probe_ref_idx =
          static_cast<int>(frames.size() / 2);
      Matrix2Df probe_ref;
      try {
        // Try to use cached normalized frame first
        Matrix2Df img;
        if (!frame_cache->try_load_normalized(static_cast<size_t>(probe_ref_idx), img)) {
          img = io::read_fits_pixels_float(frames[static_cast<size_t>(probe_ref_idx)]);
          image::apply_normalization_inplace(img, norm_scales[static_cast<size_t>(probe_ref_idx)],
                                             detected_mode, detected_bayer_str, 0, 0);
          frame_cache->store_normalized(static_cast<size_t>(probe_ref_idx), img);
        }
        probe_ref = build_registration_proxy(img, detected_mode, detected_bayer_str);
      } catch (...) {
        std::cerr << "[REGISTRATION] Warning: auto_engine probe reference frame processing failed" << std::endl;
      }

      if (probe_ref.size() > 0) {
        config::RegistrationConfig probe_star_cfg = registration_cfg;
        probe_star_cfg.engine = "triangle_star_matching";
        probe_star_cfg.transform_model = "affine";

        config::RegistrationConfig probe_ecc_cfg = registration_cfg;
        // probe ECC directly without fallback cascade
        probe_ecc_cfg.enable_star_pair_fallback = false;

        for (size_t pi = 0; pi < n_probe; ++pi) {
          const size_t fi =
              (n_probe <= 1)
                  ? 0
                  : (pi * (frames.size() - 1)) / (n_probe - 1);
          if (static_cast<int>(fi) == probe_ref_idx) {
            continue;
          }
          try {
            // Try to use cached normalized frame first
            Matrix2Df img;
            if (!frame_cache->try_load_normalized(fi, img)) {
              img = io::read_fits_pixels_float(frames[fi]);
              image::apply_normalization_inplace(img, norm_scales[fi],
                                                 detected_mode, detected_bayer_str, 0, 0);
              frame_cache->store_normalized(fi, img);
            }
            Matrix2Df probe_mov = build_registration_proxy(img, detected_mode, detected_bayer_str);
            if (probe_mov.size() <= 0 ||
                probe_mov.rows() != probe_ref.rows() ||
                probe_mov.cols() != probe_ref.cols()) {
              continue;
            }
            // Star-based probe for rotation measurement
            auto sfr_star = registration::register_single_frame(
                probe_mov, probe_ref, probe_star_cfg);
            if (sfr_star.reg.success) {
              const float a00 = sfr_star.reg.warp(0, 0);
              const float a01 = sfr_star.reg.warp(0, 1);
              const float angle_rad = std::atan2(-a01, a00);
              const float angle_deg =
                  std::fabs(angle_rad) * (180.0f / kPi);
              const float frame_dist = std::max(
                  1.0f,
                  std::fabs(static_cast<float>(fi) -
                            static_cast<float>(probe_ref_idx)));
              probe_rotations.push_back(angle_deg / frame_dist);
            }
            // ECC probe for CC quality check
            ++probe_ecc_attempted;
            auto sfr_ecc = registration::register_single_frame(
                probe_mov, probe_ref, probe_ecc_cfg);
            if (sfr_ecc.reg.success) {
              probe_ecc_cc.push_back(sfr_ecc.ncc_warped);
            }
          } catch (...) {
            std::cerr << "[REGISTRATION] Warning: auto_engine ECC probe failed for frame " << fi << std::endl;
          }
        }
      }

      bool override_engine = false;
      std::string override_reason;

      // Check 1: field rotation
      if (!probe_rotations.empty()) {
        std::sort(probe_rotations.begin(), probe_rotations.end());
        const float median_rot_per_frame =
            probe_rotations[probe_rotations.size() / 2];
        const float threshold =
            registration_cfg.auto_engine_rotation_threshold_deg;

        if (median_rot_per_frame >= threshold) {
          override_engine = true;
          std::ostringstream oss;
          oss << "detected field rotation " << median_rot_per_frame
              << " deg/frame (threshold=" << threshold
              << ") — Alt/Az mount / strong rotation";
          override_reason = oss.str();
        } else {
          std::cout << "[REGISTRATION] auto_engine probe: rotation="
                    << median_rot_per_frame
                    << " deg/frame < threshold=" << threshold
                    << ", keeping engine='" << registration_cfg.engine
                    << "'" << std::endl;
        }
      }

      // Check 2: ECC CC quality (dominant bright object / low texture)
      // Override if: fewer than half the probed frames succeeded with ECC,
      // OR median CC of successful ECC results is below reject_cc_min_abs.
      if (!override_engine && probe_ecc_attempted > 0) {
        const float ecc_success_rate =
            static_cast<float>(probe_ecc_cc.size()) /
            static_cast<float>(probe_ecc_attempted);
        float median_ecc_cc = 0.0f;
        if (!probe_ecc_cc.empty()) {
          std::vector<float> sorted_cc = probe_ecc_cc;
          std::sort(sorted_cc.begin(), sorted_cc.end());
          median_ecc_cc = sorted_cc[sorted_cc.size() / 2];
        }
        const float cc_threshold = registration_cfg.reject_cc_min_abs;
        if (ecc_success_rate < 0.5f || median_ecc_cc < cc_threshold) {
          override_engine = true;
          std::ostringstream oss;
          oss << "ECC quality too low (success_rate=" << ecc_success_rate
              << ", median_cc=" << median_ecc_cc
              << ", threshold=" << cc_threshold
              << ") — dominant bright object or low texture scene";
          override_reason = oss.str();
        } else {
          std::cout << "[REGISTRATION] auto_engine probe: ECC ok"
                    << " (success_rate=" << ecc_success_rate
                    << ", median_cc=" << median_ecc_cc
                    << "), keeping engine='" << registration_cfg.engine
                    << "'" << std::endl;
        }
      }

      if (override_engine) {
        const std::string old_engine = registration_cfg.engine;
        registration_cfg.engine = "triangle_star_matching";
        registration_cfg.transform_model = "affine";
        std::ostringstream msg;
        msg << "auto_engine: " << override_reason
            << " — overriding engine '" << old_engine
            << "' -> 'triangle_star_matching' + transform_model=affine";
        emitter.warning(run_id, msg.str(), log_file);
        std::cout << "[REGISTRATION] " << msg.str() << std::endl;
      }
    }
  }

  auto load_frame_normalized = [&](size_t frame_index) -> Matrix2Df {
    if (frame_cache && frame_cache->has_normalized(frame_index)) {
      return frame_cache->load_normalized(frame_index);
    }
    Matrix2Df img = io::read_fits_pixels_float(frames[frame_index]);
    image::apply_normalization_inplace(img, norm_scales[frame_index],
                                       detected_mode, detected_bayer_str, 0, 0);
    if (frame_cache && img.size() > 0) {
      frame_cache->store_normalized(frame_index, img);
    }
    return img;
  };

  auto load_registration_proxy = [&](size_t frame_index) -> Matrix2Df {
    if (frame_cache) {
      Matrix2Df proxy;
      if (frame_cache->try_load_registration_proxy(frame_index, proxy)) {
        return proxy;
      }
    }
    Matrix2Df img = load_frame_normalized(frame_index);
    Matrix2Df proxy =
        build_registration_proxy(img, detected_mode, detected_bayer_str);
    if (frame_cache && proxy.size() > 0) {
      frame_cache->store_registration_proxy(frame_index, proxy);
    }
    return proxy;
  };

  emitter.phase_start(run_id, Phase::REGISTRATION, "REGISTRATION", log_file);

  std::vector<WarpMatrix> global_frame_warps(frames.size(),
                                             registration::identity_warp());
  std::vector<float> global_frame_cc(frames.size(), 0.0f);
  std::vector<uint8_t> reg_chain_validated(frames.size(), 0);
  std::vector<int> reg_chain_depth(frames.size(), -1);
  std::vector<RegistrationProvenance> reg_provenance(
      frames.size(), RegistrationProvenance::unresolved);
  std::string global_reg_status = "skipped";
  core::json global_reg_extra;
  const int temporal_center_idx =
      frames.empty() ? 0 : static_cast<int>(frames.size() / 2);
  int global_ref_idx = temporal_center_idx;
  std::string ref_frame_strategy = "temporal_center";
  float global_reg_scale = 1.0f;
  std::vector<int> requested_anchor_indices;
  int requested_anchor_count = 1;
  // §4.1, §8.B — Konfigurierbare Blind-Chain Parameter aus Config
  const int kMaxBlindChainAnchorDepth = get_effective_chain_depth(
      static_cast<int>(frames.size()), cfg.registration);
  const float kBlindChainStrongAnchorCc = cfg.registration.blind_chain_strong_anchor_cc;
  const float kBlindChainDriftThresholdPx = cfg.registration.blind_chain_drift_threshold_px;
  // §4.13 — Astrometrische Rescue
  const bool kUseAstrometry = cfg.registration.use_astrometry;

  // Logging der effektiven Chain-Tiefe
  std::cout << "[REG-CHAIN] Using max_blind_chain_depth=" << kMaxBlindChainAnchorDepth
            << " (config=" << (cfg.registration.max_blind_chain_depth == 0 ? "auto" :
                              std::to_string(cfg.registration.max_blind_chain_depth))
            << ", N=" << frames.size() << ")" << std::endl;

  auto set_registration_state =
      [&](size_t fi, const WarpMatrix &warp, float cc, bool chain_validated,
          int chain_depth, RegistrationProvenance provenance) {
        global_frame_warps[fi] = warp;
        global_frame_cc[fi] = cc;
        reg_chain_validated[fi] = chain_validated ? 1 : 0;
        reg_chain_depth[fi] = chain_depth;
        reg_provenance[fi] = provenance;
      };

  auto can_anchor_blind_chain = [&](size_t fi) -> bool {
    if (fi >= frames.size() || global_frame_cc[fi] <= 0.0f) {
      return false;
    }
    switch (reg_provenance[fi]) {
    case RegistrationProvenance::reference:
    case RegistrationProvenance::direct_global:
    case RegistrationProvenance::sequential_refined:
    case RegistrationProvenance::temporal_rescue:
    case RegistrationProvenance::seeded_ecc_rescue:
    case RegistrationProvenance::local_reference_rescue:
    case RegistrationProvenance::astrometric_rescue:
      return true;
    case RegistrationProvenance::sequential_rescue:
      return (reg_chain_depth[fi] >= 0 &&
              reg_chain_depth[fi] < kMaxBlindChainAnchorDepth) ||
             global_frame_cc[fi] >= kBlindChainStrongAnchorCc;
    case RegistrationProvenance::model_global_poly:
    case RegistrationProvenance::model_local_poly:
    case RegistrationProvenance::model_interpolated:
    case RegistrationProvenance::model_blended:
    case RegistrationProvenance::model_nearest_copy:
    case RegistrationProvenance::unresolved:
      return false;
    }
    return false;
  };

  auto write_global_registration_artifact = [&](float artifact_reg_scale) {
    core::json j;
    j["num_frames"] = static_cast<int>(frames.size());
    j["scale"] = artifact_reg_scale;
    j["ref_frame"] = global_ref_idx;
    j["extra"] = global_reg_extra;
    j["cc"] = core::json::array();
    j["source"] = core::json::array();
    j["chain_depth"] = core::json::array();
    j["warps"] = core::json::array();
    j["dithering"] = {
        {"enabled", cfg.dithering.enabled},
        {"min_shift_px", cfg.dithering.min_shift_px},
        {"detected_fraction", 0.0},
    };
    int shifts_detected = 0;
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      const auto &w = global_frame_warps[fi];
      j["cc"].push_back(global_frame_cc[fi]);
      j["source"].push_back(registration_provenance_name(reg_provenance[fi]));
      if (reg_chain_depth[fi] >= 0) {
        j["chain_depth"].push_back(reg_chain_depth[fi]);
      } else {
        j["chain_depth"].push_back(nullptr);
      }
      const float shift_mag =
          std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
      if (cfg.dithering.enabled &&
          shift_mag >= cfg.dithering.min_shift_px) {
        shifts_detected++;
      }
      j["warps"].push_back(core::json{
          {"a00", w(0, 0)},
          {"a01", w(0, 1)},
          {"tx", w(0, 2)},
          {"a10", w(1, 0)},
          {"a11", w(1, 1)},
          {"ty", w(1, 2)},
          {"shift_px", shift_mag},
          {"source", registration_provenance_name(reg_provenance[fi])},
          {"chain_validated", reg_chain_validated[fi] != 0},
          {"chain_depth",
           reg_chain_depth[fi] >= 0 ? core::json(reg_chain_depth[fi])
                                    : core::json(nullptr)},
      });
    }
    if (!frames.empty()) {
      j["dithering"]["detected_fraction"] =
          static_cast<double>(shifts_detected) /
          static_cast<double>(frames.size());
      j["dithering"]["detected_count"] = shifts_detected;
      j["dithering"]["total_frames"] = static_cast<int>(frames.size());
    }
    core::write_text(run_dir / "artifacts" / "global_registration.json",
                     j.dump(2));
  };

  // Pick one or more anchor reference frames that are both high quality and
  // temporally distributed. Long Alt/Az sessions often cannot be matched
  // robustly against a single late-session reference frame.
  if (!frame_metrics.empty()) {
    struct RefCandidate {
      int idx = 0;
      float score = 0.0f;
      float quality = 0.0f;
    };

    std::vector<RefCandidate> candidates;
    candidates.reserve(frame_metrics.size());
    for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
      float score = (i < global_weights.size()) ? global_weights[i]
                                                : frame_metrics[i].quality_score;
      if (!std::isfinite(score)) {
        score = frame_metrics[i].quality_score;
      }
      candidates.push_back(
          {i, score, static_cast<float>(frame_metrics[i].quality_score)});
    }

    const int n = static_cast<int>(candidates.size());
    requested_anchor_count = 1;
    if (n >= 120) {
      // Scale anchors with sequence length so very long Alt/Az sessions do not
      // rely on a coarse fixed bucket like 1/3/5. Keep the count odd so one
      // anchor can always be forced to the temporal center.
      requested_anchor_count = std::max(3, (n + 79) / 80);  // ceil(n / 80)
      if ((requested_anchor_count % 2) == 0) {
        ++requested_anchor_count;
      }
      const int max_odd_anchor_count = (n % 2 == 0) ? (n - 1) : n;
      requested_anchor_count =
          std::min(requested_anchor_count, std::max(1, max_odd_anchor_count));
      requested_anchor_count = std::min(requested_anchor_count, 15);
    }
    std::vector<int> desired_positions;
    desired_positions.reserve(static_cast<size_t>(requested_anchor_count));
    if (requested_anchor_count <= 1) {
      desired_positions.push_back(temporal_center_idx);
    } else {
      for (int ai = 0; ai < requested_anchor_count; ++ai) {
        const float alpha = static_cast<float>(ai) /
                            static_cast<float>(requested_anchor_count - 1);
        desired_positions.push_back(
            static_cast<int>(std::round(alpha * static_cast<float>(n - 1))));
      }
      desired_positions[static_cast<size_t>(requested_anchor_count / 2)] =
          temporal_center_idx;
    }

    for (int ai = 0; ai < requested_anchor_count; ++ai) {
      const int target = desired_positions[static_cast<size_t>(ai)];
      const int lo =
          (ai == 0)
              ? 0
              : (desired_positions[static_cast<size_t>(ai - 1)] + target) / 2;
      const int hi =
          (ai + 1 >= requested_anchor_count)
              ? (n - 1)
              : (target +
                 desired_positions[static_cast<size_t>(ai + 1)]) /
                    2;

      bool found = false;
      RefCandidate best;
      int best_dist = std::numeric_limits<int>::max();
      for (const auto &c : candidates) {
        if (c.idx < lo || c.idx > hi) {
          continue;
        }
        const int d = std::abs(c.idx - target);
        if (!found || c.score > best.score ||
            (c.score == best.score && d < best_dist) ||
            (c.score == best.score && d == best_dist &&
             c.quality > best.quality)) {
          found = true;
          best = c;
          best_dist = d;
        }
      }

      requested_anchor_indices.push_back(found ? best.idx : target);
    }

    std::sort(requested_anchor_indices.begin(), requested_anchor_indices.end());
    requested_anchor_indices.erase(
        std::unique(requested_anchor_indices.begin(),
                    requested_anchor_indices.end()),
        requested_anchor_indices.end());

    if (requested_anchor_indices.empty()) {
      requested_anchor_indices.push_back(temporal_center_idx);
      ref_frame_strategy = "temporal_center_fallback";
    } else {
      auto candidate_score = [&](int idx) -> float {
        if (idx >= 0 && idx < static_cast<int>(global_weights.size()) &&
            std::isfinite(global_weights[idx])) {
          return global_weights[idx];
        }
        if (idx >= 0 && idx < static_cast<int>(frame_metrics.size())) {
          return static_cast<float>(frame_metrics[idx].quality_score);
        }
        return -std::numeric_limits<float>::infinity();
      };

      global_ref_idx = requested_anchor_indices.front();
      int best_dist = std::numeric_limits<int>::max();
      float best_score = -std::numeric_limits<float>::infinity();
      for (int idx : requested_anchor_indices) {
        const int d = std::abs(idx - temporal_center_idx);
        const float score = candidate_score(idx);
        if (d < best_dist || (d == best_dist && score > best_score)) {
          best_dist = d;
          best_score = score;
          global_ref_idx = idx;
        }
      }

      ref_frame_strategy =
          (requested_anchor_indices.size() > 1)
              ? "quality_segmented_multi_anchor"
              : "quality_topk_centered";
    }
  }

  if (requested_anchor_indices.empty()) {
    requested_anchor_indices.push_back(global_ref_idx);
  }

  if (global_ref_idx < 0 || global_ref_idx >= static_cast<int>(frames.size())) {
    global_ref_idx = temporal_center_idx;
    ref_frame_strategy = "temporal_center_bounds_fallback";
  }
  if (std::find(requested_anchor_indices.begin(), requested_anchor_indices.end(),
                global_ref_idx) == requested_anchor_indices.end()) {
    requested_anchor_indices.push_back(global_ref_idx);
    std::sort(requested_anchor_indices.begin(), requested_anchor_indices.end());
  }
  global_reg_extra["ref_frame"] = global_ref_idx;
  global_reg_extra["diag"]["ref_frame_center"] = temporal_center_idx;
  global_reg_extra["diag"]["ref_frame_strategy"] = ref_frame_strategy;
  global_reg_extra["diag"]["requested_ref_frames"] = requested_anchor_indices;
  global_reg_extra["diag"]["effective_engine"] = registration_cfg.engine;
  global_reg_extra["diag"]["effective_transform_model"] = registration_cfg.transform_model;
  global_reg_extra["diag"]["auto_engine_enabled"] = cfg.registration.auto_engine;
  global_reg_extra["diag"]["auto_engine_overridden"] =
      (registration_cfg.engine != cfg.registration.engine ||
       registration_cfg.transform_model != cfg.registration.transform_model);

  // Anchor indices/proxies im äußeren Scope für ASTAP am Ende verfügbar
  std::vector<int> active_anchor_indices;
  std::vector<Matrix2Df> active_anchor_proxies;
  std::vector<uint8_t> is_active_anchor_frame;

  // ASTAP-Variablen im äußeren Scope (wird nach Section 6 ausgeführt)
  int reg_astrometric_rescued = 0;
  int reg_astrometric_attempted = 0;
  int reg_astrometric_replaced_weak = 0;

  if (!frames.empty()) {
    try {
      Matrix2Df ref_full;
      ref_full = load_frame_normalized(static_cast<size_t>(global_ref_idx));
      if (ref_full.size() <= 0) {
        global_reg_status = "error";
        global_reg_extra["error"] = "ref_frame_empty";
      } else {
        Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
                                ? image::cfa_green_proxy_downsample2x2(
                                      ref_full, detected_bayer_str)
                                : registration::downsample2x2_mean(ref_full);
        global_reg_scale = 1.0f;
        if (ref_reg.rows() > 0) {
          int full_h2 = ref_full.rows() - (ref_full.rows() % 2);
          global_reg_scale =
              static_cast<float>(full_h2) / static_cast<float>(ref_reg.rows());
        }
        // Diagnostic: proxy image stats
        {
          float rmin = ref_reg.minCoeff();
          float rmax = ref_reg.maxCoeff();
          float rmean = ref_reg.mean();
          std::cout << "[REG-DIAG] ref_reg " << ref_reg.rows() << "x" << ref_reg.cols()
                    << " min=" << rmin << " max=" << rmax << " mean=" << rmean
                    << std::endl;
        }

        // ================================================================
        // SECTION 1: Direct global registration (parallel, multi-anchor)
        // ================================================================
        const int reg_workers = compute_adaptive_worker_count(
            cfg, frames.size(), frames, WorkerParallelProfile::MixedIo);
        std::cout << "[REGISTRATION] Using " << reg_workers
                  << " parallel workers for " << frames.size() << " frames"
                  << std::endl;
        std::mutex reg_log_mutex;
        std::mutex reg_progress_mutex;
        std::atomic<size_t> reg_next{0};
        std::atomic<size_t> reg_done{0};
        std::atomic<bool> reg_failed{false};
        std::string reg_error;
        const std::vector<size_t> *current_reg_targets = nullptr;
        const auto proxy_scale = [&]() {
          return (global_reg_scale > 1.0e-6f) ? (1.0f / global_reg_scale)
                                              : 1.0f;
        };
        // active_anchor_indices/proxies/is_active_anchor_frame sind jetzt im äußeren Scope
        is_active_anchor_frame.assign(frames.size(), 0);
        auto maybe_add_active_anchor = [&](int anchor_idx,
                                           const Matrix2Df *proxy_override =
                                               nullptr) -> bool {
          if (anchor_idx < 0 ||
              anchor_idx >= static_cast<int>(frames.size()) ||
              is_active_anchor_frame[static_cast<size_t>(anchor_idx)] != 0) {
            return false;
          }
          Matrix2Df proxy;
          if (proxy_override) {
            proxy = *proxy_override;
          } else {
            proxy = load_registration_proxy(static_cast<size_t>(anchor_idx));
          }
          if (proxy.size() <= 0 || proxy.rows() != ref_reg.rows() ||
              proxy.cols() != ref_reg.cols()) {
            return false;
          }
          active_anchor_indices.push_back(anchor_idx);
          active_anchor_proxies.push_back(proxy);
          is_active_anchor_frame[static_cast<size_t>(anchor_idx)] = 1;
          return true;
        };

        std::vector<Matrix2Df> requested_anchor_proxies(
            requested_anchor_indices.size());
        for (size_t i = 0; i < requested_anchor_indices.size(); ++i) {
          const int anchor_idx = requested_anchor_indices[i];
          if (anchor_idx == global_ref_idx) {
            requested_anchor_proxies[i] = ref_reg;
          } else {
            requested_anchor_proxies[i] =
                load_registration_proxy(static_cast<size_t>(anchor_idx));
          }
        }

        set_registration_state(static_cast<size_t>(global_ref_idx),
                               registration::identity_warp(), 1.0f, false, 0,
                               RegistrationProvenance::reference);
        maybe_add_active_anchor(global_ref_idx, &ref_reg);

        auto requested_anchor_pos = [&](int idx) -> int {
          for (size_t pos = 0; pos < requested_anchor_indices.size(); ++pos) {
            if (requested_anchor_indices[pos] == idx) {
              return static_cast<int>(pos);
            }
          }
          return -1;
        };
        auto nearest_active_anchor_slot = [&](int frame_idx) -> size_t {
          size_t best_slot = 0;
          int best_dist = std::numeric_limits<int>::max();
          for (size_t slot = 0; slot < active_anchor_indices.size(); ++slot) {
            const int d = std::abs(active_anchor_indices[slot] - frame_idx);
            if (d < best_dist) {
              best_dist = d;
              best_slot = slot;
            }
          }
          return best_slot;
        };

        std::vector<uint8_t> requested_anchor_resolved(
            requested_anchor_indices.size(), 0);
        const int master_anchor_pos = requested_anchor_pos(global_ref_idx);
        if (master_anchor_pos >= 0) {
          requested_anchor_resolved[static_cast<size_t>(master_anchor_pos)] = 1;
        }

        auto resolve_requested_anchor = [&](int anchor_pos,
                                            int parent_pos) -> bool {
          if (anchor_pos < 0 ||
              anchor_pos >= static_cast<int>(requested_anchor_indices.size()) ||
              parent_pos < 0 ||
              parent_pos >= static_cast<int>(requested_anchor_indices.size())) {
            return false;
          }
          if (requested_anchor_resolved[static_cast<size_t>(anchor_pos)] != 0) {
            return true;
          }
          if (requested_anchor_resolved[static_cast<size_t>(parent_pos)] == 0) {
            return false;
          }

          const int anchor_idx =
              requested_anchor_indices[static_cast<size_t>(anchor_pos)];
          const int parent_idx =
              requested_anchor_indices[static_cast<size_t>(parent_pos)];
          const Matrix2Df &anchor_proxy =
              requested_anchor_proxies[static_cast<size_t>(anchor_pos)];
          const Matrix2Df &parent_proxy =
              requested_anchor_proxies[static_cast<size_t>(parent_pos)];

          if (anchor_proxy.size() <= 0 || parent_proxy.size() <= 0 ||
              anchor_proxy.rows() != ref_reg.rows() ||
              anchor_proxy.cols() != ref_reg.cols() ||
              parent_proxy.rows() != ref_reg.rows() ||
              parent_proxy.cols() != ref_reg.cols()) {
            return false;
          }

          const auto sfr_anchor = registration::register_single_frame(
              anchor_proxy, parent_proxy, registration_cfg);
          if (!sfr_anchor.reg.success) {
            return false;
          }

          const WarpMatrix w_parent_proxy = registration::scale_translation_warp(
              global_frame_warps[static_cast<size_t>(parent_idx)],
              proxy_scale());
          const WarpMatrix w_chained =
              concatenate_affine_warps(sfr_anchor.reg.warp, w_parent_proxy);
          const Matrix2Df warped_global =
              registration::apply_warp(anchor_proxy, w_chained);
          const cv::Mat valid_mask_global =
              registration::warp_valid_mask(anchor_proxy, w_chained);
          int overlap_px = 0;
          const float ncc_global = registration::compute_ncc_masked(
              warped_global, ref_reg, valid_mask_global, &overlap_px);
          if (overlap_px <= 16) {
            return false;
          }

          set_registration_state(
              static_cast<size_t>(anchor_idx),
              registration::scale_translation_warp(w_chained, global_reg_scale),
              std::max(ncc_global, 0.01f),
              parent_idx != global_ref_idx,
              std::max(0, reg_chain_depth[static_cast<size_t>(parent_idx)]) + 1,
              RegistrationProvenance::direct_global);
          requested_anchor_resolved[static_cast<size_t>(anchor_pos)] = 1;
          maybe_add_active_anchor(anchor_idx, &anchor_proxy);
          return true;
        };

        if (master_anchor_pos >= 0) {
          for (int pos = master_anchor_pos - 1; pos >= 0; --pos) {
            int parent_pos = pos + 1;
            while (parent_pos < static_cast<int>(requested_anchor_indices.size()) &&
                   requested_anchor_resolved[static_cast<size_t>(parent_pos)] == 0) {
              ++parent_pos;
            }
            resolve_requested_anchor(pos, parent_pos);
          }
          for (int pos = master_anchor_pos + 1;
               pos < static_cast<int>(requested_anchor_indices.size()); ++pos) {
            int parent_pos = pos - 1;
            while (parent_pos >= 0 &&
                   requested_anchor_resolved[static_cast<size_t>(parent_pos)] == 0) {
              --parent_pos;
            }
            resolve_requested_anchor(pos, parent_pos);
          }
        }

        int current_reg_pass_workers = reg_workers;
        auto reg_worker = [&]() {
          while (true) {
            const size_t job = reg_next.fetch_add(1);
            const size_t job_count =
                current_reg_targets ? current_reg_targets->size() : frames.size();
            if (job >= job_count) {
              break;
            }
            const size_t fi =
                current_reg_targets ? (*current_reg_targets)[job] : job;
            try {
              if (global_frame_cc[fi] > 0.0f) {
                // Pre-resolved anchor frame.
              } else {
                Matrix2Df mov_reg = load_registration_proxy(fi);
                if (mov_reg.size() <= 0) {
                  set_registration_state(fi, registration::identity_warp(), 0.0f,
                                         false, -1,
                                         RegistrationProvenance::unresolved);
                } else {
                  // Diagnostic: first few moving frames
                  if (fi < 3) {
                    float mmin = mov_reg.minCoeff();
                    float mmax = mov_reg.maxCoeff();
                    float mmean = mov_reg.mean();
                    std::lock_guard<std::mutex> lock(reg_log_mutex);
                    std::cout << "[REG-DIAG] mov_reg[" << fi << "] "
                              << mov_reg.rows() << "x" << mov_reg.cols()
                              << " min=" << mmin << " max=" << mmax
                              << " mean=" << mmean << std::endl;
                  }

                  if (mov_reg.rows() != ref_reg.rows() ||
                      mov_reg.cols() != ref_reg.cols()) {
                    set_registration_state(fi, registration::identity_warp(), 0.0f,
                                           false, -1,
                                           RegistrationProvenance::unresolved);
                  } else {
                    const size_t anchor_slot =
                        nearest_active_anchor_slot(static_cast<int>(fi));
                    const int anchor_idx = active_anchor_indices[anchor_slot];
                    const Matrix2Df &anchor_reg =
                        active_anchor_proxies[anchor_slot];

                    auto sfr = registration::register_single_frame(
                        mov_reg, anchor_reg, registration_cfg);

                    if (sfr.reg.success) {
                      const WarpMatrix w_anchor_proxy =
                          registration::scale_translation_warp(
                              global_frame_warps[static_cast<size_t>(anchor_idx)],
                              proxy_scale());
                      const WarpMatrix w_chained =
                          concatenate_affine_warps(sfr.reg.warp, w_anchor_proxy);
                      const Matrix2Df warped_global =
                          registration::apply_warp(mov_reg, w_chained);
                      const cv::Mat valid_mask_global =
                          registration::warp_valid_mask(mov_reg, w_chained);
                      int overlap_px = 0;
                      const float ncc_global = registration::compute_ncc_masked(
                          warped_global, ref_reg, valid_mask_global,
                          &overlap_px);

                      if (overlap_px <= 16) {
                        set_registration_state(
                            fi, registration::identity_warp(), 0.0f, false, -1,
                            RegistrationProvenance::unresolved);
                      } else {
                        const bool chained_anchor = anchor_idx != global_ref_idx;
                        const int chained_depth =
                            chained_anchor
                                ? (std::max(
                                       0,
                                       reg_chain_depth[static_cast<size_t>(
                                           anchor_idx)]) +
                                   1)
                                : 0;
                        const WarpMatrix w_full =
                            registration::scale_translation_warp(
                                w_chained, global_reg_scale);
                        set_registration_state(
                            fi, w_full, std::max(ncc_global, 0.01f),
                            chained_anchor, chained_depth,
                            RegistrationProvenance::direct_global);
                      }
                    } else {
                      set_registration_state(fi, registration::identity_warp(), 0.0f,
                                             false, -1,
                                             RegistrationProvenance::unresolved);
                    }

                    // Per-frame logging
                    if (fi < 5 || fi == frames.size() - 1 || (fi % 50 == 0)) {
                      std::lock_guard<std::mutex> lock(reg_log_mutex);
                      std::cout << "[REG] frame " << fi << "/" << frames.size()
                                << " anchor=" << anchor_idx
                                << " method=" << sfr.method_used
                                << " ncc_id=" << sfr.ncc_identity
                                << " cc=" << global_frame_cc[fi] << std::endl;
                    }
                  }
                }
              }
            } catch (const std::exception &e) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = e.what();
              }
            } catch (...) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = "unknown_error";
              }
            }

            const size_t done = reg_done.fetch_add(1) + 1;
            if (done % 5 == 0 || done == job_count) {
              const float p = job_count == 0
                                  ? 1.0f
                                  : static_cast<float>(done) /
                                        static_cast<float>(job_count);
              std::lock_guard<std::mutex> lock(reg_progress_mutex);
              emitter.phase_progress(
                  run_id, Phase::REGISTRATION, p,
                  "global_reg " + std::to_string(done) + "/" +
                      std::to_string(job_count) + " workers=" +
                      std::to_string(current_reg_pass_workers),
                  log_file);
            }
          }
        };

        auto run_registration_pass =
            [&](int pass_workers, const std::vector<size_t> *targets = nullptr) {
          current_reg_pass_workers = std::max(1, pass_workers);
          current_reg_targets = targets;
          reg_next.store(0, std::memory_order_relaxed);
          reg_done.store(0, std::memory_order_relaxed);
          reg_failed.store(false, std::memory_order_relaxed);
          reg_error.clear();
          const size_t job_count =
              current_reg_targets ? current_reg_targets->size() : frames.size();
          if (job_count == 0) {
            current_reg_targets = nullptr;
            return;
          }
          if (current_reg_pass_workers > 1) {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(current_reg_pass_workers));
            for (int w = 0; w < current_reg_pass_workers; ++w) {
              workers.emplace_back(reg_worker);
            }
            for (auto &worker : workers) {
              if (worker.joinable()) {
                worker.join();
              }
            }
          } else {
            reg_worker();
          }
          if (reg_failed.load(std::memory_order_relaxed)) {
            throw std::runtime_error(reg_error.empty() ? "registration_failed"
                                                       : reg_error);
          }
          current_reg_targets = nullptr;
        };

        run_registration_pass(reg_workers);

        // Adaptive Anker-Anzahl: mehr Anker bei vielen Frames oder schlechtem Seeing
        // Alt: min(21, max(3, (N+59)/60)) -> bei 325 Frames nur 6 Anker
        // Neu: min(32, max(4, (N+29)/30)) -> bei 325 Frames 12 Anker
        const int target_active_anchor_count =
            std::max(requested_anchor_count,
                     std::min(32, std::max(4, (static_cast<int>(frames.size()) + 29) / 30)));
        const int promote_limit_per_round =
            std::clamp((static_cast<int>(frames.size()) + 159) / 160, 2, 8);
        const int max_direct_anchor_rounds =
            std::clamp((static_cast<int>(frames.size()) + 239) / 240, 3, 8);

        auto promote_strong_direct_anchors = [&]() -> std::vector<int> {
          std::vector<int> promoted_indices;
          if (static_cast<int>(active_anchor_indices.size()) >=
              target_active_anchor_count) {
            return promoted_indices;
          }
          const float min_cc =
              std::max(0.35f, cfg.registration.reject_cc_min_abs + 0.10f);
          const int min_spacing = std::max(24, static_cast<int>(frames.size()) / 20);
          std::vector<std::pair<float, int>> candidates;
          candidates.reserve(frames.size());
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            if (is_active_anchor_frame[fi] != 0 ||
                reg_provenance[fi] != RegistrationProvenance::direct_global ||
                global_frame_cc[fi] < min_cc) {
              continue;
            }
            const size_t nearest_slot =
                nearest_active_anchor_slot(static_cast<int>(fi));
            const int nearest_dist =
                std::abs(active_anchor_indices[nearest_slot] -
                         static_cast<int>(fi));
            if (nearest_dist < min_spacing) {
              continue;
            }
            const float score =
                global_frame_cc[fi] +
                0.002f * static_cast<float>(std::min(nearest_dist, 100));
            candidates.emplace_back(score, static_cast<int>(fi));
          }
          std::sort(candidates.begin(), candidates.end(),
                    [](const auto &a, const auto &b) {
                      if (a.first != b.first) {
                        return a.first > b.first;
                      }
                      return a.second < b.second;
                    });

          for (const auto &[score, idx] : candidates) {
            (void)score;
            if (maybe_add_active_anchor(idx)) {
              promoted_indices.push_back(idx);
              if (static_cast<int>(promoted_indices.size()) >=
                      promote_limit_per_round ||
                  static_cast<int>(active_anchor_indices.size()) >=
                      target_active_anchor_count) {
                break;
              }
            }
          }
          return promoted_indices;
        };

        int reg_direct_anchor_rounds = 0;
        int reg_promoted_active_anchors = 0;
        int reg_promotion_retry_frames = 0;
        for (int round = 0; round < max_direct_anchor_rounds; ++round) {
          const std::vector<int> promoted = promote_strong_direct_anchors();
          if (promoted.empty()) {
            break;
          }
          reg_promoted_active_anchors += static_cast<int>(promoted.size());
          ++reg_direct_anchor_rounds;

          std::vector<size_t> retry_targets;
          retry_targets.reserve(frames.size());
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            if (global_frame_cc[fi] > 0.0f) {
              continue;
            }
            const size_t nearest_slot =
                nearest_active_anchor_slot(static_cast<int>(fi));
            const int nearest_anchor = active_anchor_indices[nearest_slot];
            if (std::find(promoted.begin(), promoted.end(), nearest_anchor) !=
                promoted.end()) {
              retry_targets.push_back(fi);
            }
          }
          reg_promotion_retry_frames += static_cast<int>(retry_targets.size());
          run_registration_pass(reg_workers, &retry_targets);
        }

        std::vector<int> active_anchor_indices_sorted = active_anchor_indices;
        std::sort(active_anchor_indices_sorted.begin(),
                  active_anchor_indices_sorted.end());
        global_reg_extra["diag"]["active_ref_frames"] = active_anchor_indices_sorted;
        global_reg_extra["diag"]["active_ref_frame_count"] =
            static_cast<int>(active_anchor_indices_sorted.size());
        global_reg_extra["diag"]["reg_target_active_anchor_count"] =
            target_active_anchor_count;
        global_reg_extra["diag"]["reg_promote_limit_per_round"] =
            promote_limit_per_round;
        global_reg_extra["diag"]["reg_max_direct_anchor_rounds"] =
            max_direct_anchor_rounds;
        global_reg_extra["diag"]["reg_direct_anchor_rounds"] = reg_direct_anchor_rounds;
        global_reg_extra["diag"]["reg_promoted_active_anchors"] =
            reg_promoted_active_anchors;
        global_reg_extra["diag"]["reg_promotion_retry_frames"] =
            reg_promotion_retry_frames;

        // ================================================================
        // SECTION 2: Sequential refinement + blind-chain rescue
        // ================================================================
        // Sequential refinement pass:
        // register each frame against its direct temporal neighbor and chain
        // that warp to the global reference. This gives us real frame-to-frame
        // registration, while keeping the global reference as a consistency
        // check so the chain does not drift uncontrollably.
        // ----------------------------------------------------------------
        int reg_sequential_refined = 0;
        int reg_sequential_rescued = 0;
        int reg_sequential_anchor_blocked = 0;
        {
          auto try_sequential_refine = [&](size_t fi,
                                           size_t neighbor_fi) -> bool {
            if (fi >= frames.size() || neighbor_fi >= frames.size()) {
              return false;
            }
            if (global_frame_cc[neighbor_fi] <= 0.0f) {
              return false;
            }

            const Matrix2Df mov_p = load_registration_proxy(fi);
            const Matrix2Df nbr_p = load_registration_proxy(neighbor_fi);
            if (mov_p.size() <= 0 || nbr_p.size() <= 0 ||
                mov_p.rows() != ref_reg.rows() ||
                mov_p.cols() != ref_reg.cols() ||
                nbr_p.rows() != ref_reg.rows() ||
                nbr_p.cols() != ref_reg.cols()) {
              return false;
            }

            // Fast path: phase-corr + optional log-polar rotation for delta warp.
            // Sufficient for consecutive frames where drift is typically < 30px.
            // Fallback to full register_single_frame only when phase-corr indicates
            // a large shift (> 100px), which likely means a phase-corr alias.
            const Matrix2Df mov_ecc_img = registration::prepare_ecc_image(mov_p);
            const Matrix2Df nbr_ecc_img = registration::prepare_ecc_image(nbr_p);
            auto [dx, dy] = registration::phasecorr_translation(mov_ecc_img, nbr_ecc_img);

            WarpMatrix w_local = registration::identity_warp();
            if (registration_cfg.allow_rotation) {
              cv::Mat r_cv(nbr_ecc_img.rows(), nbr_ecc_img.cols(), CV_32F,
                           const_cast<float *>(nbr_ecc_img.data()));
              cv::Mat m_cv(mov_ecc_img.rows(), mov_ecc_img.cols(), CV_32F,
                           const_cast<float *>(mov_ecc_img.data()));
              const float rot_deg =
                  registration::estimate_rotation_logpolar(r_cv, m_cv);
              const float th = rot_deg * 3.14159265f / 180.0f;
              const float ct = std::cos(th);
              const float st = std::sin(th);
              const float cx = static_cast<float>(mov_p.cols()) * 0.5f;
              const float cy = static_cast<float>(mov_p.rows()) * 0.5f;
              w_local(0, 0) = ct; w_local(0, 1) = -st;
              w_local(1, 0) = st; w_local(1, 1) =  ct;
              w_local(0, 2) = dx + cx * (1.0f - ct) + cy * st;
              w_local(1, 2) = dy + cy * (1.0f - ct) - cx * st;
            } else {
              w_local(0, 2) = dx;
              w_local(1, 2) = dy;
            }

            // If phase-corr gives a suspiciously large shift, fall back to full
            // star-matching which is more robust against aliasing.
            const float shift_sq = dx * dx + dy * dy;
            if (shift_sq > 10000.0f) { // > 100px: phase-corr likely aliased
              const auto sfr_local =
                  registration::register_single_frame(mov_p, nbr_p,
                                                      registration_cfg, -0.002f);
              if (!sfr_local.reg.success) {
                return false;
              }
              w_local = sfr_local.reg.warp;
            }

            const WarpMatrix w_nbr_proxy = registration::scale_translation_warp(
                global_frame_warps[neighbor_fi], proxy_scale());
            const WarpMatrix w_chained =
                concatenate_affine_warps(w_local, w_nbr_proxy);

            const Matrix2Df warped_global =
                registration::apply_warp(mov_p, w_chained);
            const cv::Mat valid_mask_global =
                registration::warp_valid_mask(mov_p, w_chained);
            int overlap_px = 0;
            const float ncc_global = registration::compute_ncc_masked(
                warped_global, ref_reg, valid_mask_global, &overlap_px);
            if (overlap_px <= 16) {
              return false;
            }

            // Compute local NCC (warped mov vs neighbor) as proxy for local correlation.
            const Matrix2Df warped_to_nbr = registration::apply_warp(mov_p, w_local);
            const cv::Mat valid_mask_local = registration::warp_valid_mask(mov_p, w_local);
            const float ncc_local = registration::compute_ncc_masked(
                warped_to_nbr, nbr_p, valid_mask_local);

            const float current_cc = global_frame_cc[fi];
            const bool missing_direct = current_cc <= 0.0f;
            const bool clearly_better = ncc_global > current_cc + 0.005f;
            const bool comparable_but_prefer_sequential =
                !missing_direct && ncc_global >= current_cc - 0.01f &&
                ncc_local >= std::max(0.12f, current_cc * 0.25f);

            if (!missing_direct && !clearly_better &&
                !comparable_but_prefer_sequential) {
              return false;
            }

            set_registration_state(
                fi, registration::scale_translation_warp(w_chained, global_reg_scale),
                std::max(ncc_global, 0.01f), true,
                std::max(0, reg_chain_depth[neighbor_fi]) + 1,
                RegistrationProvenance::sequential_refined);
            if (missing_direct) {
              ++reg_sequential_rescued;
            } else {
              ++reg_sequential_refined;
            }
            return true;
          };

          // Hüpfende Sequential Refine: suche nächsten guten Frame (cc>0.4) wenn direkter Nachbar schlecht
          const float kGoodAnchorCc = 0.40f;
          auto find_good_neighbor_refine = [&](size_t fi, int direction) -> size_t {
            // direction: +1 für vorwärts, -1 für rückwärts
            for (int dist = 1; dist <= 5; ++dist) {
              int neighbor = static_cast<int>(fi) + direction * dist;
              if (neighbor < 0 || neighbor >= static_cast<int>(frames.size())) continue;
              if (global_frame_cc[static_cast<size_t>(neighbor)] > kGoodAnchorCc) {
                return static_cast<size_t>(neighbor);
              }
            }
            // Fallback: direkter Nachbar
            return (direction > 0) ? fi - 1 : fi + 1;
          };

          for (size_t fi = static_cast<size_t>(global_ref_idx) + 1;
               fi < frames.size(); ++fi) {
            size_t neighbor = fi - 1;
            // Wenn direkter Nachbar schlecht, suche besseren
            if (global_frame_cc[neighbor] <= kGoodAnchorCc) {
              neighbor = find_good_neighbor_refine(fi, -1);
            }
            try_sequential_refine(fi, neighbor);
          }
          for (int fi = global_ref_idx - 1; fi >= 0; --fi) {
            size_t neighbor = static_cast<size_t>(fi + 1);
            // Wenn direkter Nachbar schlecht, suche besseren
            if (global_frame_cc[neighbor] <= kGoodAnchorCc) {
              neighbor = find_good_neighbor_refine(static_cast<size_t>(fi), +1);
            }
            try_sequential_refine(static_cast<size_t>(fi), neighbor);
          }
        }

        if (reg_sequential_refined > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION sequential refinement: replaced "
              << reg_sequential_refined
              << " direct reference matches with frame-to-frame chains";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-SEQ-REFINE] " << msg.str() << std::endl;
        }

        // ----------------------------------------------------------------
        // Run outward from reference frame, register each failed frame
        // against its direct neighbor.
        // CRITICAL CHANGE: Validate against the NEIGHBOR, not global ref.
        // This allows "blind chaining" through cloudy blocks where global
        // correlation is lost but inter-frame structure is preserved.
        // ----------------------------------------------------------------
        {

          auto try_sequential = [&](size_t fi, size_t neighbor_fi) -> bool {
            if (global_frame_cc[fi] > 0.0f) return false;
            // Neighbor must be valid (registered or rescued)
            if (global_frame_cc[neighbor_fi] <= 0.0f) return false;
            if (!can_anchor_blind_chain(neighbor_fi)) {
              ++reg_sequential_anchor_blocked;
              return false;
            }

            const Matrix2Df mov_p = load_registration_proxy(fi);
            const Matrix2Df nbr_p = load_registration_proxy(neighbor_fi);
            if (mov_p.size() <= 0 || nbr_p.size() <= 0 ||
                mov_p.rows() != ref_reg.rows() || mov_p.cols() != ref_reg.cols())
              return false;

            // 1. Calculate Delta Warp (mov -> neighbor)
            // Phase correlation + log-polar rotation
            const Matrix2Df mov_ecc_img = registration::prepare_ecc_image(mov_p);
            const Matrix2Df nbr_ecc_img = registration::prepare_ecc_image(nbr_p);
            auto [dx, dy] = registration::phasecorr_translation(mov_ecc_img, nbr_ecc_img);

            WarpMatrix w_delta = registration::identity_warp();
            if (registration_cfg.allow_rotation) {
              cv::Mat r_cv(nbr_ecc_img.rows(), nbr_ecc_img.cols(), CV_32F,
                           const_cast<float *>(nbr_ecc_img.data()));
              cv::Mat m_cv(mov_ecc_img.rows(), mov_ecc_img.cols(), CV_32F,
                           const_cast<float *>(mov_ecc_img.data()));
              const float rot_deg =
                  registration::estimate_rotation_logpolar(r_cv, m_cv);
              const float th = rot_deg * 3.14159265f / 180.0f;
              const float ct = std::cos(th);
              const float st = std::sin(th);
              const float cx = static_cast<float>(mov_p.cols()) * 0.5f;
              const float cy = static_cast<float>(mov_p.rows()) * 0.5f;
              w_delta(0, 0) = ct; w_delta(0, 1) = -st;
              w_delta(1, 0) = st; w_delta(1, 1) =  ct;
              w_delta(0, 2) = dx + cx * (1.0f - ct) + cy * st;
              w_delta(1, 2) = dy + cy * (1.0f - ct) - cx * st;
            } else {
              w_delta(0, 2) = dx;
              w_delta(1, 2) = dy;
            }

            // 2. Validate Delta against NEIGHBOR (not global ref)
            // Warp mov using delta -> should match neighbor
            const Matrix2Df warped_to_nbr = registration::apply_warp(mov_p, w_delta);
            const cv::Mat valid_mask = registration::warp_valid_mask(mov_p, w_delta);
            
            // Calculate NCC between warped_mov and neighbor
            int overlap_px = 0;
            const float ncc_neighbor = registration::compute_ncc_masked(
                warped_to_nbr, nbr_p, valid_mask, &overlap_px);

            // Relaxed thresholds for "Blind Chaining"
            // If the shift is small (very likely for sequential frames), we accept lower correlation.
            // This allows bridging through cloudy blocks where NCC drops but structure is consistent.
            
            // Check shift magnitude (approximate from w_delta translation)
            const float dx_val = w_delta(0, 2);
            const float dy_val = w_delta(1, 2);
            const float shift_sq = dx_val * dx_val + dy_val * dy_val;
            const bool small_shift = shift_sq < 900.0f; // 30px limit (~0.3 deg)
            
            bool accept = false;
            if (overlap_px > 32) { // minimal overlap
                if (ncc_neighbor >= 0.3f) {
                    accept = true;
                } else if (ncc_neighbor >= 0.05f && small_shift) {
                    accept = true; // Blind chaining allowed for small shifts
                }
            }
            
            if (!accept) {
                return false;
            }

            // 3. Chain to Global Reference
            // w_global = w_neighbor_global * w_delta
            // Note: concatenate_affine_warps(w1, w2) does w2 * w1.
            // We want w_nbr_proxy * w_delta.
            const WarpMatrix w_nbr_proxy = registration::scale_translation_warp(
                global_frame_warps[neighbor_fi], proxy_scale());
            const WarpMatrix w_chained =
                concatenate_affine_warps(w_delta, w_nbr_proxy);

            // 4. Calculate Global NCC (just for record, not for validation)
            const Matrix2Df warped_global = registration::apply_warp(mov_p, w_chained);
            const cv::Mat valid_mask_global = registration::warp_valid_mask(mov_p, w_chained);
            const float ncc_global = registration::compute_ncc_masked(
                warped_global, ref_reg, valid_mask_global);

            // Keep a tiny positive CC so downstream logic sees a valid warp,
            // but anchor propagation is controlled separately by provenance
            // and chain depth.
            set_registration_state(
                fi, registration::scale_translation_warp(w_chained, global_reg_scale),
                std::max(ncc_global, 0.01f), true,
                std::max(0, reg_chain_depth[neighbor_fi]) + 1,
                RegistrationProvenance::sequential_rescue);

            return true;
          };

          // Hüpfende Sequential Rescue: suche nächsten guten Anker wenn direkter Nachbar schlecht
          const float kGoodRescueAnchorCc = 0.30f; // etwas niedriger für Rescue
          auto find_good_anchor_rescue = [&](size_t fi, int direction) -> size_t {
            for (int dist = 1; dist <= 8; ++dist) {
              int anchor = static_cast<int>(fi) + direction * dist;
              if (anchor < 0 || anchor >= static_cast<int>(frames.size())) continue;
              size_t ai = static_cast<size_t>(anchor);
              if (global_frame_cc[ai] > kGoodRescueAnchorCc && can_anchor_blind_chain(ai)) {
                return ai;
              }
            }
            // Fallback: direkter Nachbar
            return (direction > 0) ? fi - 1 : fi + 1;
          };

          // Forward pass: ref -> last frame
          for (size_t fi = static_cast<size_t>(global_ref_idx) + 1;
               fi < frames.size(); ++fi) {
            size_t anchor = fi - 1;
            // Wenn direkter Nachbar kein guter Anker, suche weiter
            if (global_frame_cc[anchor] <= kGoodRescueAnchorCc ||
                !can_anchor_blind_chain(anchor)) {
              anchor = find_good_anchor_rescue(fi, -1);
            }
            if (try_sequential(fi, anchor)) ++reg_sequential_rescued;
          }
          // Backward pass: ref -> first frame
          for (int fi = global_ref_idx - 1; fi >= 0; --fi) {
            size_t anchor = static_cast<size_t>(fi + 1);
            // Wenn direkter Nachbar kein guter Anker, suche weiter
            if (global_frame_cc[anchor] <= kGoodRescueAnchorCc ||
                !can_anchor_blind_chain(anchor)) {
              anchor = find_good_anchor_rescue(static_cast<size_t>(fi), +1);
            }
            if (try_sequential(static_cast<size_t>(fi), anchor))
              ++reg_sequential_rescued;
          }
        }

        if (reg_sequential_rescued > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION sequential phase-corr rescue: recovered "
              << reg_sequential_rescued << " frames";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-SEQ] " << msg.str() << std::endl;
        }

        // ================================================================
        // SECTION 3: Temporal / seeded-ECC / local-reference rescue
        // ================================================================
        int reg_temporal_rescued_backward = 0;
        int reg_temporal_rescued_forward = 0;
        int reg_seeded_ecc_rescued_backward = 0;
        int reg_seeded_ecc_rescued_forward = 0;
        int reg_local_reference_rescued_backward = 0;
        int reg_local_reference_rescued_forward = 0;
        auto try_temporal_rescue = [&](size_t fi, size_t anchor_fi) -> bool {
          if (fi >= frames.size() || anchor_fi >= frames.size()) {
            return false;
          }
          if (global_frame_cc[anchor_fi] <= 0.0f) {
            return false;
          }

          const Matrix2Df mov_reg = load_registration_proxy(fi);
          const Matrix2Df anchor_reg = load_registration_proxy(anchor_fi);
          if (mov_reg.size() <= 0 || anchor_reg.size() <= 0 ||
              mov_reg.rows() != ref_reg.rows() ||
              mov_reg.cols() != ref_reg.cols() ||
              anchor_reg.rows() != ref_reg.rows() ||
              anchor_reg.cols() != ref_reg.cols()) {
            return false;
          }

          const auto sfr_temp =
              registration::register_single_frame(mov_reg, anchor_reg,
                                                  registration_cfg, -0.002f);
          if (!sfr_temp.reg.success) {
            return false;
          }

          const float proxy_scale =
              (global_reg_scale > 1.0e-6f) ? (1.0f / global_reg_scale) : 1.0f;
          const WarpMatrix w_anchor_to_ref =
              registration::scale_translation_warp(global_frame_warps[anchor_fi],
                                                   proxy_scale);
          const WarpMatrix w_chained =
              concatenate_affine_warps(sfr_temp.reg.warp, w_anchor_to_ref);
          const Matrix2Df warped = registration::apply_warp(mov_reg, w_chained);
          const cv::Mat valid_mask = registration::warp_valid_mask(mov_reg, w_chained);
          int overlap_pixels = 0;
          const float ncc_identity =
              registration::compute_ncc_masked(mov_reg, ref_reg, valid_mask,
                                       &overlap_pixels);
          const float ncc_chained =
              registration::compute_ncc_masked(warped, ref_reg, valid_mask);
          if (overlap_pixels <= 16 || ncc_chained <= ncc_identity + 0.01f) {
            return false;
          }

          set_registration_state(
              fi, registration::scale_translation_warp(w_chained, global_reg_scale),
              ncc_chained, true, std::max(0, reg_chain_depth[anchor_fi]) + 1,
              RegistrationProvenance::temporal_rescue);
          return true;
        };

        auto build_bridge_seed_proxy = [&](size_t fi, WarpMatrix &seed) -> bool {
          std::vector<size_t> valid_idx;
          valid_idx.reserve(frames.size());
          for (size_t i = 0; i < frames.size(); ++i) {
            if (global_frame_cc[i] > 0.0f) {
              valid_idx.push_back(i);
            }
          }
          if (valid_idx.empty()) {
            return false;
          }

          auto proxy_warp_at = [&](size_t idx) {
            const float proxy_scale =
                (global_reg_scale > 1.0e-6f) ? (1.0f / global_reg_scale) : 1.0f;
            return registration::scale_translation_warp(global_frame_warps[idx],
                                                        proxy_scale);
          };
          auto set_from_components = [&](float ang, float tx, float ty) {
            seed(0, 0) = std::cos(ang);
            seed(0, 1) = std::sin(ang);
            seed(1, 0) = -std::sin(ang);
            seed(1, 1) = std::cos(ang);
            seed(0, 2) = tx;
            seed(1, 2) = ty;
          };

          const auto it =
              std::lower_bound(valid_idx.begin(), valid_idx.end(), fi);
          if (it != valid_idx.end() && *it == fi) {
            seed = proxy_warp_at(fi);
            return true;
          }

          if (it != valid_idx.begin() && it != valid_idx.end()) {
            const size_t left_idx = *(it - 1);
            const size_t right_idx = *it;
            const auto wl = proxy_warp_at(left_idx);
            const auto wr = proxy_warp_at(right_idx);
            const float left_f = static_cast<float>(left_idx);
            const float right_f = static_cast<float>(right_idx);
            const float denom = std::max(1.0f, right_f - left_f);
            const float alpha = (static_cast<float>(fi) - left_f) / denom;
            const float ang_l = std::atan2(wl(0, 1), wl(0, 0));
            const float ang_r = wrap_angle_near(std::atan2(wr(0, 1), wr(0, 0)),
                                                ang_l);
            set_from_components(ang_l + alpha * (ang_r - ang_l),
                                wl(0, 2) + alpha * (wr(0, 2) - wl(0, 2)),
                                wl(1, 2) + alpha * (wr(1, 2) - wl(1, 2)));
            return true;
          }

          if (it == valid_idx.begin() && valid_idx.size() >= 2) {
            const size_t s0_idx = valid_idx[0];
            const size_t s1_idx = valid_idx[1];
            const auto w0 = proxy_warp_at(s0_idx);
            const auto w1 = proxy_warp_at(s1_idx);
            const float denom = std::max(
                1.0f, static_cast<float>(s1_idx) - static_cast<float>(s0_idx));
            const float delta = static_cast<float>(fi) - static_cast<float>(s0_idx);
            const float ang0 = std::atan2(w0(0, 1), w0(0, 0));
            const float ang1 =
                wrap_angle_near(std::atan2(w1(0, 1), w1(0, 0)), ang0);
            set_from_components(ang0 + delta * (ang1 - ang0) / denom,
                                w0(0, 2) + delta * (w1(0, 2) - w0(0, 2)) / denom,
                                w0(1, 2) + delta * (w1(1, 2) - w0(1, 2)) / denom);
            return true;
          }

          if (it == valid_idx.end() && valid_idx.size() >= 2) {
            const size_t s0_idx = valid_idx[valid_idx.size() - 2];
            const size_t s1_idx = valid_idx[valid_idx.size() - 1];
            const auto w0 = proxy_warp_at(s0_idx);
            const auto w1 = proxy_warp_at(s1_idx);
            const float denom = std::max(
                1.0f, static_cast<float>(s1_idx) - static_cast<float>(s0_idx));
            const float delta = static_cast<float>(fi) - static_cast<float>(s1_idx);
            const float ang0 = std::atan2(w0(0, 1), w0(0, 0));
            const float ang1 =
                wrap_angle_near(std::atan2(w1(0, 1), w1(0, 0)), ang0);
            set_from_components(ang1 + delta * (ang1 - ang0) / denom,
                                w1(0, 2) + delta * (w1(0, 2) - w0(0, 2)) / denom,
                                w1(1, 2) + delta * (w1(1, 2) - w0(1, 2)) / denom);
            return true;
          }

          return false;
        };

        auto try_seeded_ecc_rescue = [&](size_t fi) -> bool {
          if (fi >= frames.size() || global_frame_cc[fi] > 0.0f) {
            return false;
          }

          const Matrix2Df mov_reg = load_registration_proxy(fi);
          if (mov_reg.size() <= 0 || mov_reg.rows() != ref_reg.rows() ||
              mov_reg.cols() != ref_reg.cols()) {
            return false;
          }

          WarpMatrix seed = registration::identity_warp();
          if (!build_bridge_seed_proxy(fi, seed)) {
            return false;
          }

          // Use gradient-preprocessed multi-scale ECC with the interpolated
          // seed warp — robust_phase_ecc_seeded handles large shifts/rotations
          // much better than single-scale ecc_warp on the raw proxy image.
          const auto ecc_res = registration::robust_phase_ecc_seeded(
              mov_reg, ref_reg, registration_cfg.allow_rotation, seed);
          if (!ecc_res.success) {
            return false;
          }

          const Matrix2Df warped = registration::apply_warp(mov_reg, ecc_res.warp);
          const cv::Mat valid_mask = registration::warp_valid_mask(mov_reg, ecc_res.warp);
          int overlap_pixels = 0;
          const float ncc_identity =
              registration::compute_ncc_masked(mov_reg, ref_reg, valid_mask,
                                       &overlap_pixels);
          const float ncc_warped =
              registration::compute_ncc_masked(warped, ref_reg, valid_mask);
          if (overlap_pixels <= 16 || ncc_warped <= ncc_identity + 0.005f) {
            return false;
          }

          set_registration_state(
              fi, registration::scale_translation_warp(ecc_res.warp, global_reg_scale),
              ncc_warped, true, 1,
              RegistrationProvenance::seeded_ecc_rescue);
          return true;
        };

        struct SupportFrame {
          size_t idx = 0;
          float weight = 0.0f;
        };

        auto build_support_reference =
            [&](const std::vector<SupportFrame> &support,
                Matrix2Df &local_ref) -> bool {
          if (support.size() < 2) {
            return false;
          }
          Matrix2Df accum = Matrix2Df::Zero(ref_reg.rows(), ref_reg.cols());
          Matrix2Df weight_sum = Matrix2Df::Zero(ref_reg.rows(), ref_reg.cols());
          int valid_pixels = 0;

          for (const SupportFrame &s : support) {
            Matrix2Df src = load_registration_proxy(s.idx);
            if (src.size() <= 0 || src.rows() != ref_reg.rows() ||
                src.cols() != ref_reg.cols()) {
              continue;
            }
            const float proxy_scale =
                (global_reg_scale > 1.0e-6f) ? (1.0f / global_reg_scale) : 1.0f;
            const WarpMatrix w_proxy = registration::scale_translation_warp(
                global_frame_warps[s.idx], proxy_scale);
            const Matrix2Df warped = registration::apply_warp(src, w_proxy);
            const cv::Mat valid_mask = registration::warp_valid_mask(src, w_proxy);
            for (int y = 0; y < ref_reg.rows(); ++y) {
              const float *pm = valid_mask.ptr<float>(y);
              for (int x = 0; x < ref_reg.cols(); ++x) {
                if (pm[x] <= 0.5f) {
                  continue;
                }
                accum(y, x) += s.weight * warped(y, x);
                weight_sum(y, x) += s.weight;
              }
            }
          }

          local_ref = ref_reg;
          for (int y = 0; y < ref_reg.rows(); ++y) {
            for (int x = 0; x < ref_reg.cols(); ++x) {
              if (weight_sum(y, x) > 1.0e-6f) {
                local_ref(y, x) = accum(y, x) / weight_sum(y, x);
                ++valid_pixels;
              }
            }
          }

          const int min_valid_pixels = std::max(
              1024, static_cast<int>((ref_reg.rows() * ref_reg.cols()) / 20));
          return valid_pixels >= min_valid_pixels;
        };

        auto build_local_reference = [&](size_t fi, Matrix2Df &local_ref) -> bool {
          std::vector<SupportFrame> support;
          support.reserve(8);
          for (size_t radius = 1; radius < frames.size() && support.size() < 6;
               ++radius) {
            bool added = false;
            if (fi >= radius) {
              const size_t idx = fi - radius;
              if (global_frame_cc[idx] > 0.0f) {
                const float temporal_w = 1.0f / (1.0f + static_cast<float>(radius));
                support.push_back(
                    {idx, temporal_w * std::max(0.05f, global_frame_cc[idx])});
                added = true;
              }
            }
            if (fi + radius < frames.size() && support.size() < 6) {
              const size_t idx = fi + radius;
              if (global_frame_cc[idx] > 0.0f) {
                const float temporal_w = 1.0f / (1.0f + static_cast<float>(radius));
                support.push_back(
                    {idx, temporal_w * std::max(0.05f, global_frame_cc[idx])});
                added = true;
              }
            }
            if (!added && radius > 96) {
              break;
            }
          }
          return build_support_reference(support, local_ref);
        };

        auto try_local_reference_rescue = [&](size_t fi) -> bool {
          if (fi >= frames.size() || global_frame_cc[fi] > 0.0f) {
            return false;
          }

          const Matrix2Df mov_reg = load_registration_proxy(fi);
          if (mov_reg.size() <= 0 || mov_reg.rows() != ref_reg.rows() ||
              mov_reg.cols() != ref_reg.cols()) {
            return false;
          }

          WarpMatrix seed = registration::identity_warp();
          if (!build_bridge_seed_proxy(fi, seed)) {
            return false;
          }

          Matrix2Df local_ref;
          if (!build_local_reference(fi, local_ref)) {
            return false;
          }

          const auto robust_res = registration::robust_phase_ecc_seeded(
              mov_reg, local_ref, registration_cfg.allow_rotation, seed);
          if (!robust_res.success) {
            return false;
          }

          const Matrix2Df warped =
              registration::apply_warp(mov_reg, robust_res.warp);
          const cv::Mat valid_mask =
              registration::warp_valid_mask(mov_reg, robust_res.warp);
          int overlap_pixels = 0;
          const float ncc_identity =
              registration::compute_ncc_masked(mov_reg, ref_reg, valid_mask,
                                       &overlap_pixels);
          const float ncc_warped =
              registration::compute_ncc_masked(warped, ref_reg, valid_mask);
          if (overlap_pixels <= 16 || ncc_warped <= ncc_identity + 0.003f) {
            return false;
          }

          set_registration_state(
              fi, registration::scale_translation_warp(robust_res.warp, global_reg_scale),
              ncc_warped, true, 1,
              RegistrationProvenance::local_reference_rescue);
          return true;
        };

        // Iterate rescue stages so newly recovered frames can immediately act as
        // anchors for neighboring failures in the same problematic block.
        for (int pass = 0; pass < 4; ++pass) {
          bool progress = false;

          for (int fi = global_ref_idx - 1; fi >= 0; --fi) {
            if (global_frame_cc[static_cast<size_t>(fi)] > 0.0f) {
              continue;
            }
            int anchor = fi + 1;
            while (anchor < static_cast<int>(frames.size()) &&
                   global_frame_cc[static_cast<size_t>(anchor)] <= 0.0f) {
              ++anchor;
            }
            if (anchor < static_cast<int>(frames.size()) &&
                try_temporal_rescue(static_cast<size_t>(fi),
                                    static_cast<size_t>(anchor))) {
              ++reg_temporal_rescued_backward;
              progress = true;
            }
          }

          for (size_t fi = static_cast<size_t>(global_ref_idx + 1);
               fi < frames.size(); ++fi) {
            if (global_frame_cc[fi] > 0.0f) {
              continue;
            }
            int anchor = static_cast<int>(fi) - 1;
            while (anchor >= 0 &&
                   global_frame_cc[static_cast<size_t>(anchor)] <= 0.0f) {
              --anchor;
            }
            if (anchor >= 0 &&
                try_temporal_rescue(fi, static_cast<size_t>(anchor))) {
              ++reg_temporal_rescued_forward;
              progress = true;
            }
          }

          for (int fi = global_ref_idx - 1; fi >= 0; --fi) {
            if (try_seeded_ecc_rescue(static_cast<size_t>(fi))) {
              ++reg_seeded_ecc_rescued_backward;
              progress = true;
            }
          }
          for (size_t fi = static_cast<size_t>(global_ref_idx + 1);
               fi < frames.size(); ++fi) {
            if (try_seeded_ecc_rescue(fi)) {
              ++reg_seeded_ecc_rescued_forward;
              progress = true;
            }
          }

          for (int fi = global_ref_idx - 1; fi >= 0; --fi) {
            if (try_local_reference_rescue(static_cast<size_t>(fi))) {
              ++reg_local_reference_rescued_backward;
              progress = true;
            }
          }
          for (size_t fi = static_cast<size_t>(global_ref_idx + 1);
               fi < frames.size(); ++fi) {
            if (try_local_reference_rescue(fi)) {
              ++reg_local_reference_rescued_forward;
              progress = true;
            }
          }

          if (!progress) {
            break;
          }
        }

        const int reg_temporal_rescued =
            reg_temporal_rescued_backward + reg_temporal_rescued_forward;
        const int reg_seeded_ecc_rescued =
            reg_seeded_ecc_rescued_backward + reg_seeded_ecc_rescued_forward;
        const int reg_local_reference_rescued =
            reg_local_reference_rescued_backward +
            reg_local_reference_rescued_forward;
        global_reg_extra["diag"]["reg_sequential_refined"] = reg_sequential_refined;
        global_reg_extra["diag"]["reg_sequential_rescued"] = reg_sequential_rescued;
        global_reg_extra["diag"]["reg_sequential_anchor_blocked"] =
            reg_sequential_anchor_blocked;
        global_reg_extra["diag"]["reg_temporal_rescued"] = reg_temporal_rescued;
        global_reg_extra["diag"]["reg_temporal_rescued_backward"] =
            reg_temporal_rescued_backward;
        global_reg_extra["diag"]["reg_temporal_rescued_forward"] =
            reg_temporal_rescued_forward;
        global_reg_extra["diag"]["reg_seeded_ecc_rescued"] = reg_seeded_ecc_rescued;
        global_reg_extra["diag"]["reg_seeded_ecc_rescued_backward"] =
            reg_seeded_ecc_rescued_backward;
        global_reg_extra["diag"]["reg_seeded_ecc_rescued_forward"] =
            reg_seeded_ecc_rescued_forward;
        global_reg_extra["diag"]["reg_local_reference_rescued"] =
            reg_local_reference_rescued;
        global_reg_extra["diag"]["reg_local_reference_rescued_backward"] =
            reg_local_reference_rescued_backward;
        global_reg_extra["diag"]["reg_local_reference_rescued_forward"] =
            reg_local_reference_rescued_forward;
        if (reg_temporal_rescued > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION temporal rescue: recovered "
              << reg_temporal_rescued << " frames (backward="
              << reg_temporal_rescued_backward << ", forward="
              << reg_temporal_rescued_forward << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-TEMPORAL] " << msg.str() << std::endl;
        }
        if (reg_seeded_ecc_rescued > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION seeded-ECC rescue: recovered "
              << reg_seeded_ecc_rescued << " frames (backward="
              << reg_seeded_ecc_rescued_backward << ", forward="
              << reg_seeded_ecc_rescued_forward << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-ECC] " << msg.str() << std::endl;
        }
        if (reg_local_reference_rescued > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION local-reference rescue: recovered "
              << reg_local_reference_rescued << " frames (backward="
              << reg_local_reference_rescued_backward << ", forward="
              << reg_local_reference_rescued_forward << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-LOCAL-REF] " << msg.str() << std::endl;
        }

        // ASTAP wird jetzt nach Section 6 (Model-Generierung) ausgeführt
        // Siehe "SECTION 4b: Astrometric rescue" am Ende der Registrierung

        global_reg_status = "ok";
        try {
          write_global_registration_artifact(global_reg_scale);

          if (cfg.output.write_registered_frames) {
            fs::create_directories(run_dir / cfg.output.registered_dir);
            // first_header is available from outer scope

            for (size_t fi = 0; fi < frames.size(); ++fi) {
              if (static_cast<size_t>(fi) >= global_frame_warps.size())
                continue;
              const auto &w = global_frame_warps[fi];

              Matrix2Df img = load_frame_normalized(fi);
              if (img.size() <= 0)
                continue;

              Matrix2Df out_img;
              if (detected_mode == ColorMode::OSC) {
                out_img = image::warp_cfa_mosaic_via_subplanes(
                    img, w, img.rows(), img.cols(), "reflect", "linear");
              } else {
                out_img = image::apply_global_warp(img, w, detected_mode);
              }

              std::ostringstream name;
              name << "frame_" << std::setw(4) << std::setfill('0') << fi
                   << ".fits";
              io::write_fits_float(run_dir / cfg.output.registered_dir /
                                       name.str(),
                                   out_img, first_header);
            }
          }

        } catch (...) {
        }
      }
    } catch (const std::exception &e) {
      global_reg_status = "error";
      global_reg_extra["error"] = e.what();
    } catch (...) {
      global_reg_status = "error";
      global_reg_extra["error"] = "unknown_error";
    }
  } else {
    global_reg_extra["reason"] = "no_frames";
  }

  // ================================================================
  // SECTION 5: Outlier rejection (orientation / reflection / scale / cc)
  // ================================================================
  // Reject implausible global registration outliers before downstream phases.
  // These outliers can pass NCC but still produce heavy tile/grid artifacts.
  int reg_reject_orientation_outliers = 0;
  int reg_reject_reflection_outliers = 0;
  int reg_reject_scale_outliers = 0;
  int reg_reject_cc_outliers = 0;
  int reg_reject_shift_outliers = 0;
  int reg_reject_low_cc_protected = 0;
  int reg_reject_deep_chain_outliers = 0;
  int reg_meridian_flip_detected = 0;
  core::json reg_rejected_frames = core::json::array();
  std::vector<uint8_t> reg_rejected_mask(frames.size(), 0);
  if (cfg.registration.reject_outliers) {
    std::vector<float> cc_positive;
    cc_positive.reserve(frames.size());
    std::vector<float> normal_shift_mags_positive;
    normal_shift_mags_positive.reserve(frames.size());
    std::vector<float> half_turn_shift_mags_positive;
    half_turn_shift_mags_positive.reserve(frames.size());
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f) {
        continue;
      }
      cc_positive.push_back(global_frame_cc[fi]);
      const auto &w = global_frame_warps[fi];
      const auto shift_diag = registration_shift_diagnostics(w, width, height);
      if (shift_diag.half_turn_family) {
        half_turn_shift_mags_positive.push_back(shift_diag.shift_magnitude);
      } else {
        normal_shift_mags_positive.push_back(shift_diag.shift_magnitude);
      }
    }

    auto robust_median = [](std::vector<float> values) -> float {
      if (values.empty()) {
        return 0.0f;
      }
      const size_t mid = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + static_cast<long>(mid),
                       values.end());
      float med = values[mid];
      if (values.size() % 2 == 0 && mid > 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1),
                         values.end());
        med = 0.5f * (med + values[mid - 1]);
      }
      return med;
    };

    // For long Alt/Az sessions the correlation naturally varies with overlap
    // and field rotation. A run-global MAD threshold becomes overly aggressive
    // and rejects geometrically plausible edge frames solely for being less
    // correlated than the central bulk. Keep CC rejection absolute here.
    const float cc_min_keep = cfg.registration.reject_cc_min_abs;

    const float normal_shift_median = robust_median(normal_shift_mags_positive);
    const float normal_shift_limit =
        std::max(cfg.registration.reject_shift_px_min,
                 cfg.registration.reject_shift_median_multiplier * normal_shift_median);
    const float half_turn_shift_median = robust_median(half_turn_shift_mags_positive);
    const float half_turn_shift_limit =
        std::max(cfg.registration.reject_shift_px_min,
                 cfg.registration.reject_shift_median_multiplier * half_turn_shift_median);

    // Robust rotation-trend fit: alt-az sessions have a smooth, monotonic
    // field-rotation curve. False 90°/180° star-triangle matches produce
    // rotations that deviate sharply from this smooth trend. Use only
    // high-cc frames as trend anchors so local clusters of false positives
    // (whose cc is low) cannot pollute the fit. This catches bogus warps
    // even when their shift is small enough to slip past the shift-outlier
    // guard (or when the half-turn-shift median is itself polluted by the
    // false cluster, which happens when a bad cluster is locally dense).
    constexpr float kOrientationTrendCcMin = 0.35f;
    constexpr float kOrientationTrendDeviationDeg = 15.0f;
    constexpr int kOrientationTrendMinAnchors = 8;
    std::vector<float> trend_fi_anchors;
    std::vector<float> trend_ang_anchors;
    trend_fi_anchors.reserve(frames.size());
    trend_ang_anchors.reserve(frames.size());
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] < kOrientationTrendCcMin) {
        continue;
      }
      const auto &w = global_frame_warps[fi];
      trend_fi_anchors.push_back(static_cast<float>(fi));
      trend_ang_anchors.push_back(std::atan2(w(0, 1), w(0, 0)));
    }
    const bool have_orientation_trend =
        static_cast<int>(trend_fi_anchors.size()) >= kOrientationTrendMinAnchors;
    float trend_fi_lo = 0.0f;
    float trend_fi_span = 1.0f;
    VectorXf trend_coeffs(3);
    trend_coeffs.setZero();
    if (have_orientation_trend) {
      // Meridian-flip normalization: if a subset of anchors is ~pi offset
      // from the majority (180° rotation from a meridian flip), align them
      // so the polynomial fit sees a continuous curve.  Without this, the
      // fit would try to interpolate between the two clusters and produce
      // a meaningless trend that rejects all pre-flip frames.
      const float median_ang = robust_median(trend_ang_anchors);
      constexpr float kPi = 3.14159265358979323846f;
      int flip_anchors = 0;
      for (auto &ang : trend_ang_anchors) {
        const float offset = std::remainder(ang - median_ang, 2.0f * kPi);
        if (std::fabs(offset) > 0.75f * kPi) {
          ang += (offset > 0.0f ? -kPi : kPi);
          ++flip_anchors;
        }
      }
      if (flip_anchors > 0) {
        std::cout << "[REG-FILTER] meridian-flip normalization: aligned "
                  << flip_anchors << " anchor(s) by 180° for trend fit"
                  << std::endl;
      }
      // Unwrap so the fit sees a continuous curve (no 2pi jumps).
      const std::vector<float> trend_ang_unwrapped =
          unwrap_angle_sequence(trend_ang_anchors);
      trend_fi_lo = trend_fi_anchors.front();
      trend_fi_span =
          std::max(1.0f, trend_fi_anchors.back() - trend_fi_anchors.front());
      const int na = static_cast<int>(trend_fi_anchors.size());
      Eigen::MatrixXf V(na, 3);
      VectorXf ya(na);
      for (int i = 0; i < na; ++i) {
        const float t = (trend_fi_anchors[static_cast<size_t>(i)] -
                         trend_fi_lo) /
                        trend_fi_span;
        V(i, 0) = 1.0f;
        V(i, 1) = t;
        V(i, 2) = t * t;
        ya(i) = trend_ang_unwrapped[static_cast<size_t>(i)];
      }
      trend_coeffs = V.householderQr().solve(ya);
      const float res_ang_deg =
          (V * trend_coeffs - ya).cwiseAbs().maxCoeff() * 57.29577951f;
      std::cout << "[REG-FILTER] orientation-trend fit: "
                << na << " anchors (cc>=" << kOrientationTrendCcMin
                << "), max_residual=" << res_ang_deg << " deg, tolerance="
                << kOrientationTrendDeviationDeg << " deg" << std::endl;
    }
    auto predicted_trend_angle_rad = [&](size_t fi) -> float {
      const float t = (static_cast<float>(fi) - trend_fi_lo) / trend_fi_span;
      return trend_coeffs(0) + trend_coeffs(1) * t + trend_coeffs(2) * t * t;
    };

    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f)
        continue;
      const auto &w = global_frame_warps[fi];
      const auto shift_diag = registration_shift_diagnostics(w, width, height);

      bool reject = false;
      std::vector<std::string> reject_reasons;
      // Accept both 0° and ~180° rotations (trace can be positive or negative).
      // But reject mirror/reflection solutions (det < 0), which cause
      // characteristic mirrored ghost artifacts in the final stack.
      const float det = w(0, 0) * w(1, 1) - w(0, 1) * w(1, 0);
      if (det < 0.0f) {
        reject = true;
        ++reg_reject_reflection_outliers;
        reject_reasons.push_back("reflection");
      }

      if (!reject) {
        const float scale = std::sqrt(std::fabs(det));
        if (scale < cfg.registration.reject_scale_min ||
            scale > cfg.registration.reject_scale_max) {
          reject = true;
          ++reg_reject_scale_outliers;
          reject_reasons.push_back("scale");
        }
      }

      if (!reject) {
        const float cc = global_frame_cc[fi];
        if (cc < cc_min_keep && reg_chain_validated[fi] == 0) {
          reject = true;
          ++reg_reject_cc_outliers;
          reject_reasons.push_back("low_cc");
        } else if (cc < cc_min_keep) {
          const int depth = reg_chain_depth[fi];
          if (depth > kMaxBlindChainAnchorDepth) {
            reject = true;
            ++reg_reject_deep_chain_outliers;
            reject_reasons.push_back("deep_chain_low_cc");
          } else {
            ++reg_reject_low_cc_protected;
          }
        }
      }

      // Orientation-trend check: alt-az field rotation is smooth across the
      // session, so any frame whose rotation deviates by more than
      // kOrientationTrendDeviationDeg from the polynomial trend is a false
      // star-triangle match (typically 90°/180° of the correct solution).
      // This catches frames that would otherwise slip past the shift-outlier
      // guard when the half-turn median is polluted by a locally dense
      // cluster of false solutions.
      //
      // Exception: a deviation of ~180° indicates a meridian flip, which is
      // a legitimate physical event.  The flip-corrected trend fit should
      // already align pre-flip anchors, but frames registered independently
      // may still show ~180° residual.  Accept these and let the shift
      // check (which uses half_turn_shift_limit for half-turn-family warps)
      // catch genuinely false matches.
      if (!reject && have_orientation_trend) {
        const float ang_rad = std::atan2(w(0, 1), w(0, 0));
        const float predicted_rad = predicted_trend_angle_rad(fi);
        // Wrap residual into [-pi, pi] so that a false 180° match counts as
        // ~180° deviation, not ~0°.
        float diff = ang_rad - predicted_rad;
        constexpr float kTwoPi = 6.2831853071795864f;
        while (diff > 3.14159265f) {
          diff -= kTwoPi;
        }
        while (diff < -3.14159265f) {
          diff += kTwoPi;
        }
        const float diff_deg = std::fabs(diff) * 57.29577951f;
        if (diff_deg > kOrientationTrendDeviationDeg) {
          // Check if this is a legitimate meridian flip (~180° offset).
          constexpr float kMeridianFlipToleranceDeg = 15.0f;
          const float flip_residual = std::fabs(diff_deg - 180.0f);
          if (flip_residual < kMeridianFlipToleranceDeg) {
            ++reg_meridian_flip_detected;
          } else {
            reject = true;
            ++reg_reject_orientation_outliers;
            reject_reasons.push_back("orientation_trend");
          }
        }
      }

      if (!reject) {
        const float shift_mag = shift_diag.shift_magnitude;
        const float shift_limit = shift_diag.half_turn_family
                                      ? half_turn_shift_limit
                                      : normal_shift_limit;
        if (shift_mag > shift_limit) {
          reject = true;
          ++reg_reject_shift_outliers;
          reject_reasons.push_back("shift_outlier");
        }
      }

      if (reject) {
        reg_rejected_mask[fi] = 1;
        core::json rej = {
            {"frame_index", static_cast<int>(fi)},
            {"frame_name", frames[fi].filename().string()},
            {"cc", global_frame_cc[fi]},
            {"chain_validated", reg_chain_validated[fi] != 0},
            {"reasons", reject_reasons},
            {"a00", w(0, 0)},
            {"a01", w(0, 1)},
            {"tx", w(0, 2)},
            {"a10", w(1, 0)},
            {"a11", w(1, 1)},
            {"ty", w(1, 2)},
        };
        reg_rejected_frames.push_back(rej);
        std::ostringstream msg;
        msg << "REGISTRATION outlier rejected: frame=" << fi << " ("
            << frames[fi].filename().string() << ") cc=" << global_frame_cc[fi]
            << " reasons=" << core::join(reject_reasons, ",")
            << " tx=" << w(0, 2) << " ty=" << w(1, 2);
        emitter.warning(run_id, msg.str(), log_file);
        std::cout << "[REG-FILTER] " << msg.str() << std::endl;

        set_registration_state(fi, registration::identity_warp(), 0.0f, false,
                               -1, RegistrationProvenance::unresolved);
      }
    }

    // ================================================================
    // SECTION 6: Warp prediction (field-rotation polynomial / nearest-copy)
    // ================================================================
    // Predict warps for rejected frames using a polynomial field rotation
    // model fitted to the valid registrations.  For alt-az mounts the warp
    // parameters (angle, tx, ty) follow smooth trajectories that are well
    // approximated by a low-degree polynomial over the session duration.
    // This retains ALL frames (methodology v3.2.2 §1.2) while providing
    // physically plausible geometry for frames where registration failed.
    {
      std::vector<float> vfi, vang_raw, vtx, vty, vcc;
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        if (!reg_rejected_mask[fi] && global_frame_cc[fi] > 0.0f) {
          const auto &w = global_frame_warps[fi];
          vfi.push_back(static_cast<float>(fi));
          vang_raw.push_back(std::atan2(w(0, 1), w(0, 0)));
          vtx.push_back(w(0, 2));
          vty.push_back(w(1, 2));
          vcc.push_back(global_frame_cc[fi]);
        }
      }

      const int nv = static_cast<int>(vfi.size());
      int reg_model_predicted = 0;
      int reg_model_predicted_rejected = 0;
      int reg_model_predicted_missing = 0;
      int reg_model_local_refined = 0;
      int reg_model_interpolated = 0;
      int reg_model_blended = 0;

      if (nv >= 3) {
        const std::vector<float> vang = unwrap_angle_sequence(vang_raw);
        // Normalise frame indices to [0,1] for numerical stability.
        const float fi_lo = vfi.front();
        const float fi_hi = vfi.back();
        const float fi_span = std::max(1.0f, fi_hi - fi_lo);

        // Degree-2 Vandermonde
        Eigen::MatrixXf V(nv, 3);
        VectorXf ya(nv), yx(nv), yy(nv);
        for (int i = 0; i < nv; ++i) {
          const float t = (vfi[static_cast<size_t>(i)] - fi_lo) / fi_span;
          V(i, 0) = 1.0f;
          V(i, 1) = t;
          V(i, 2) = t * t;
          ya(i) = vang[static_cast<size_t>(i)];
          yx(i) = vtx[static_cast<size_t>(i)];
          yy(i) = vty[static_cast<size_t>(i)];
        }

        auto qr = V.householderQr();
        VectorXf ca = qr.solve(ya);
        VectorXf cx = qr.solve(yx);
        VectorXf cy = qr.solve(yy);

        // Residual stats for diagnostics
        const float res_ang =
            (V * ca - ya).cwiseAbs().maxCoeff() * 57.29577951f;
        const float res_tx = (V * cx - yx).cwiseAbs().maxCoeff();
        const float res_ty = (V * cy - yy).cwiseAbs().maxCoeff();

        std::vector<TemporalWarpSample> valid_samples;
        valid_samples.reserve(static_cast<size_t>(nv));
        for (int i = 0; i < nv; ++i) {
          valid_samples.push_back(
              {vfi[static_cast<size_t>(i)], vang[static_cast<size_t>(i)],
               vtx[static_cast<size_t>(i)], vty[static_cast<size_t>(i)],
               vcc[static_cast<size_t>(i)]});
        }

        auto build_local_candidate = [&](size_t fi, int support_count)
            -> WarpPredictionCandidate {
          WarpPredictionCandidate out;
          if (valid_samples.empty()) {
            return out;
          }
          std::vector<std::pair<float, size_t>> by_dist;
          by_dist.reserve(valid_samples.size());
          for (size_t i = 0; i < valid_samples.size(); ++i) {
            by_dist.emplace_back(
                std::fabs(valid_samples[i].fi - static_cast<float>(fi)), i);
          }
          std::sort(by_dist.begin(), by_dist.end(),
                    [](const auto &a, const auto &b) {
                      if (a.first != b.first) {
                        return a.first < b.first;
                      }
                      return a.second < b.second;
                    });
          if (by_dist.empty()) {
            return out;
          }
          const int take_n =
              std::max(1, std::min(support_count, static_cast<int>(by_dist.size())));
          std::vector<size_t> chosen;
          chosen.reserve(static_cast<size_t>(take_n));
          for (int i = 0; i < take_n; ++i) {
            chosen.push_back(by_dist[static_cast<size_t>(i)].second);
          }
          std::sort(chosen.begin(), chosen.end());

          float max_abs_dist = 0.0f;
          for (size_t idx : chosen) {
            max_abs_dist = std::max(
                max_abs_dist,
                std::fabs(valid_samples[idx].fi - static_cast<float>(fi)));
          }
          max_abs_dist = std::max(max_abs_dist, 1.0f);

          std::vector<float> xs;
          std::vector<float> ws;
          std::vector<float> ys_ang;
          std::vector<float> ys_tx;
          std::vector<float> ys_ty;
          xs.reserve(chosen.size());
          ws.reserve(chosen.size());
          ys_ang.reserve(chosen.size());
          ys_tx.reserve(chosen.size());
          ys_ty.reserve(chosen.size());
          for (size_t idx : chosen) {
            const auto &s = valid_samples[idx];
            const float x = (s.fi - static_cast<float>(fi)) / max_abs_dist;
            const float u = std::min(1.0f, std::fabs(x));
            const float tricube =
                std::pow(std::max(0.0f, 1.0f - u * u * u), 3.0f);
            const float w = std::max(1.0e-3f, tricube * std::max(0.05f, s.cc));
            xs.push_back(x);
            ws.push_back(w);
            ys_ang.push_back(s.ang);
            ys_tx.push_back(s.tx);
            ys_ty.push_back(s.ty);
          }

          const int degree =
              (take_n >= 5) ? 2 : ((take_n >= 2) ? 1 : 0);
          const auto fit_ang = fit_weighted_poly(xs, ys_ang, ws, degree);
          const auto fit_tx = fit_weighted_poly(xs, ys_tx, ws, degree);
          const auto fit_ty = fit_weighted_poly(xs, ys_ty, ws, degree);
          if (!fit_ang.ok || !fit_tx.ok || !fit_ty.ok ||
              fit_ang.coeffs.size() == 0 || fit_tx.coeffs.size() == 0 ||
              fit_ty.coeffs.size() == 0) {
            return out;
          }

          out.ok = true;
          out.ang = fit_ang.coeffs(0);
          out.tx = fit_tx.coeffs(0);
          out.ty = fit_ty.coeffs(0);
          out.res_ang_deg = fit_ang.max_abs_residual * 57.29577951f;
          out.res_tx = fit_tx.max_abs_residual;
          out.res_ty = fit_ty.max_abs_residual;
          out.support = take_n;
          out.span = max_abs_dist;
          out.score = out.res_ang_deg / 0.35f + out.res_tx / 20.0f +
                      out.res_ty / 20.0f + 0.05f * max_abs_dist;
          return out;
        };

        auto build_bridge_candidate = [&](size_t fi) -> WarpPredictionCandidate {
          WarpPredictionCandidate out;
          if (valid_samples.empty()) {
            return out;
          }
          int right = -1;
          for (int i = 0; i < nv; ++i) {
            if (valid_samples[static_cast<size_t>(i)].fi >= static_cast<float>(fi)) {
              right = i;
              break;
            }
          }
          if (right >= 0 && right < nv &&
              valid_samples[static_cast<size_t>(right)].fi ==
                  static_cast<float>(fi)) {
            out.ok = true;
            out.ang = valid_samples[static_cast<size_t>(right)].ang;
            out.tx = valid_samples[static_cast<size_t>(right)].tx;
            out.ty = valid_samples[static_cast<size_t>(right)].ty;
            out.support = 1;
            out.span = 0.0f;
            out.score = 0.0f;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (right > 0 && right < nv) {
            const auto &l = valid_samples[static_cast<size_t>(right - 1)];
            const auto &r = valid_samples[static_cast<size_t>(right)];
            const float denom = std::max(1.0f, r.fi - l.fi);
            const float alpha = (static_cast<float>(fi) - l.fi) / denom;
            out.ok = true;
            out.ang = l.ang + alpha * (r.ang - l.ang);
            out.tx = l.tx + alpha * (r.tx - l.tx);
            out.ty = l.ty + alpha * (r.ty - l.ty);
            out.support = 2;
            out.span = r.fi - l.fi;
            out.score = 0.5f + 0.05f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (nv >= 2 && right == 0) {
            const auto &s0 = valid_samples[0];
            const auto &s1 = valid_samples[1];
            const float denom = std::max(1.0f, s1.fi - s0.fi);
            const float delta = static_cast<float>(fi) - s0.fi;
            out.ok = true;
            out.ang = s0.ang + delta * (s1.ang - s0.ang) / denom;
            out.tx = s0.tx + delta * (s1.tx - s0.tx) / denom;
            out.ty = s0.ty + delta * (s1.ty - s0.ty) / denom;
            out.support = 2;
            out.span = std::fabs(delta);
            out.score = 1.0f + 0.08f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          if (nv >= 2 && right < 0) {
            const auto &s0 = valid_samples[static_cast<size_t>(nv - 2)];
            const auto &s1 = valid_samples[static_cast<size_t>(nv - 1)];
            const float denom = std::max(1.0f, s1.fi - s0.fi);
            const float delta = static_cast<float>(fi) - s1.fi;
            out.ok = true;
            out.ang = s1.ang + delta * (s1.ang - s0.ang) / denom;
            out.tx = s1.tx + delta * (s1.tx - s0.tx) / denom;
            out.ty = s1.ty + delta * (s1.ty - s0.ty) / denom;
            out.support = 2;
            out.span = std::fabs(delta);
            out.score = 1.0f + 0.08f * out.span;
            out.res_ang_deg = 0.0f;
            out.res_tx = 0.0f;
            out.res_ty = 0.0f;
            return out;
          }

          return out;
        };

        // Bounds of the valid-anchor shift hull — predictions that fall far
        // outside are geometrically implausible (e.g. the polynomial is
        // extrapolating across a locally dense rejected cluster with no
        // nearby valid anchors). In that case we must NOT set a bogus
        // model warp: the resulting frame would land far off-canvas and
        // create ghost patches at the canvas borders during stacking.
        const float anchor_tx_min =
            *std::min_element(vtx.begin(), vtx.end());
        const float anchor_tx_max =
            *std::max_element(vtx.begin(), vtx.end());
        const float anchor_ty_min =
            *std::min_element(vty.begin(), vty.end());
        const float anchor_ty_max =
            *std::max_element(vty.begin(), vty.end());
        const float anchor_tx_range = anchor_tx_max - anchor_tx_min;
        const float anchor_ty_range = anchor_ty_max - anchor_ty_min;
        const float kPredictionMarginPx = 50.0f;
        const float kPredictionMarginFrac = 0.25f;
        const float tx_margin =
            std::max(kPredictionMarginPx,
                     kPredictionMarginFrac * anchor_tx_range);
        const float ty_margin =
            std::max(kPredictionMarginPx,
                     kPredictionMarginFrac * anchor_ty_range);
        const float tx_lo = anchor_tx_min - tx_margin;
        const float tx_hi = anchor_tx_max + tx_margin;
        const float ty_lo = anchor_ty_min - ty_margin;
        const float ty_hi = anchor_ty_max + ty_margin;
        int reg_model_predicted_out_of_bounds = 0;

        // Predict warps for rejected frames and frames with cc=0
        // (completely failed registration, not caught by outlier filter)
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          const bool is_rejected = reg_rejected_mask[fi] != 0;
          const bool is_missing_registration = global_frame_cc[fi] <= 0.0f;
          if (!is_rejected && !is_missing_registration) {
            continue;
          }

          const float t = (static_cast<float>(fi) - fi_lo) / fi_span;
          WarpPredictionCandidate global_candidate;
          global_candidate.ok = true;
          global_candidate.ang = ca(0) + ca(1) * t + ca(2) * t * t;
          global_candidate.tx = cx(0) + cx(1) * t + cx(2) * t * t;
          global_candidate.ty = cy(0) + cy(1) * t + cy(2) * t * t;
          global_candidate.score =
              res_ang / 0.35f + res_tx / 20.0f + res_ty / 20.0f + 5.0f;

          WarpPredictionCandidate best_local;
          for (int support_count : {6, 8, 12, 16, 24, 32, 48}) {
            if (support_count > nv) {
              continue;
            }
            const auto cand = build_local_candidate(fi, support_count);
            if (!cand.ok || cand.score >= best_local.score) {
              continue;
            }
            best_local = cand;
          }
          const auto bridge_candidate = build_bridge_candidate(fi);

          WarpPredictionCandidate chosen = global_candidate;
          RegistrationProvenance chosen_provenance =
              RegistrationProvenance::model_global_poly;
          const bool outside_valid_span =
              static_cast<float>(fi) < fi_lo || static_cast<float>(fi) > fi_hi;
          if (!outside_valid_span && best_local.ok && bridge_candidate.ok) {
            chosen.ok = true;
            const float local_conf = 1.0f / (1.0f + std::max(0.0f, best_local.score));
            const float bridge_conf =
                1.0f / (1.0f + std::max(0.0f, bridge_candidate.score));
            const float norm = std::max(1.0e-6f, local_conf + bridge_conf);
            const float wl = local_conf / norm;
            const float wb = bridge_conf / norm;
            chosen.ang = bridge_candidate.ang +
                         wl * (wrap_angle_near(best_local.ang, bridge_candidate.ang) -
                               bridge_candidate.ang);
            chosen.tx = wl * best_local.tx + wb * bridge_candidate.tx;
            chosen.ty = wl * best_local.ty + wb * bridge_candidate.ty;
            chosen.score = std::min(best_local.score, bridge_candidate.score);
            chosen_provenance = RegistrationProvenance::model_blended;
            ++reg_model_blended;
          } else if (!outside_valid_span && best_local.ok) {
            chosen = best_local;
            chosen_provenance = RegistrationProvenance::model_local_poly;
            ++reg_model_local_refined;
          } else if (bridge_candidate.ok) {
            chosen = bridge_candidate;
            chosen_provenance = RegistrationProvenance::model_interpolated;
            ++reg_model_interpolated;
          }

          // Reject predictions that fall outside the anchor shift hull: these
          // are extrapolations across a locally dense rejected cluster where
          // the polynomial has no nearby valid anchors and produces wildly
          // off-canvas warps. Leave such frames as `unresolved` so they are
          // skipped in prewarp (see below) and do not expand the canvas nor
          // pollute the stack with ghost patches at the borders.
          if (chosen.tx < tx_lo || chosen.tx > tx_hi ||
              chosen.ty < ty_lo || chosen.ty > ty_hi) {
            set_registration_state(fi, registration::identity_warp(), 0.0f,
                                   false, -1,
                                   RegistrationProvenance::unresolved);
            ++reg_model_predicted_out_of_bounds;
            continue;
          }

          WarpMatrix w;
          w(0, 0) = std::cos(chosen.ang);
          w(0, 1) = std::sin(chosen.ang);
          w(1, 0) = -std::sin(chosen.ang);
          w(1, 1) = std::cos(chosen.ang);
          w(0, 2) = chosen.tx;
          w(1, 2) = chosen.ty;

          // Small positive cc → included in prewarp but lower than valid
          // frames. Downstream tile-level quality metrics handle weighting.
          set_registration_state(fi, w, 1.0e-4f, false, -1,
                                 chosen_provenance);
          ++reg_model_predicted;
          if (is_rejected) {
            ++reg_model_predicted_rejected;
          } else {
            ++reg_model_predicted_missing;
          }
        }
        if (reg_model_predicted_out_of_bounds > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION field-rotation model: "
              << reg_model_predicted_out_of_bounds
              << " predictions dropped as out-of-bounds extrapolations"
              << " (tx hull=[" << anchor_tx_min << "," << anchor_tx_max
              << "] ty hull=[" << anchor_ty_min << "," << anchor_ty_max
              << "], margin_frac=" << kPredictionMarginFrac << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
        global_reg_extra["diag"]["reg_model_predicted_out_of_bounds"] =
            reg_model_predicted_out_of_bounds;

        {
          std::ostringstream msg;
          msg << "REGISTRATION field-rotation model: predicted "
              << reg_model_predicted << " non-valid frames from " << nv
              << " valid warps (max residual: angle="
              << std::fixed << std::setprecision(2) << res_ang
              << "deg tx=" << res_tx << "px ty=" << res_ty << "px"
              << ", rejected=" << reg_model_predicted_rejected
              << ", missing_registration=" << reg_model_predicted_missing
              << ", local_refined=" << reg_model_local_refined
              << ", interpolated=" << reg_model_interpolated
              << ", blended=" << reg_model_blended << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
      } else if (nv >= 1) {
        // Too few points for polynomial — copy nearest valid warp.
        const float nc_tx_min = *std::min_element(vtx.begin(), vtx.end());
        const float nc_tx_max = *std::max_element(vtx.begin(), vtx.end());
        const float nc_ty_min = *std::min_element(vty.begin(), vty.end());
        const float nc_ty_max = *std::max_element(vty.begin(), vty.end());
        constexpr float kNCMarginPx = 50.0f;
        const float nc_tx_lo = nc_tx_min - kNCMarginPx;
        const float nc_tx_hi = nc_tx_max + kNCMarginPx;
        const float nc_ty_lo = nc_ty_min - kNCMarginPx;
        const float nc_ty_hi = nc_ty_max + kNCMarginPx;
        int nc_out_of_bounds = 0;
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!reg_rejected_mask[fi] && global_frame_cc[fi] > 0.0f) {
            continue;
          }
          // Find nearest valid frame
          int best = -1;
          int best_dist = std::numeric_limits<int>::max();
          for (int k = 0; k < nv; ++k) {
            int d = std::abs(static_cast<int>(fi) -
                             static_cast<int>(vfi[static_cast<size_t>(k)]));
            if (d < best_dist) {
              best_dist = d;
              best = static_cast<int>(vfi[static_cast<size_t>(k)]);
            }
          }
          if (best >= 0) {
            const auto &bw = global_frame_warps[static_cast<size_t>(best)];
            if (bw(0, 2) < nc_tx_lo || bw(0, 2) > nc_tx_hi ||
                bw(1, 2) < nc_ty_lo || bw(1, 2) > nc_ty_hi) {
              set_registration_state(fi, registration::identity_warp(), 0.0f,
                                     false, -1,
                                     RegistrationProvenance::unresolved);
              ++nc_out_of_bounds;
              continue;
            }
            set_registration_state(
                fi, bw, 1.0e-4f,
                false, -1, RegistrationProvenance::model_nearest_copy);
            ++reg_model_predicted;
          }
        }
        if (nc_out_of_bounds > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION nearest-copy: " << nc_out_of_bounds
              << " warp(s) dropped as out-of-bounds";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
        if (reg_model_predicted > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION nearest-copy fallback: predicted "
              << reg_model_predicted << " rejected frames from " << nv
              << " valid warp(s)";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-MODEL] " << msg.str() << std::endl;
        }
      }

      global_reg_extra["diag"]["reg_model_predicted"] = reg_model_predicted;
      global_reg_extra["diag"]["reg_model_predicted_rejected"] =
          reg_model_predicted_rejected;
      global_reg_extra["diag"]["reg_model_predicted_missing"] =
          reg_model_predicted_missing;
      global_reg_extra["diag"]["reg_model_local_refined"] = reg_model_local_refined;
      global_reg_extra["diag"]["reg_model_interpolated"] = reg_model_interpolated;
      global_reg_extra["diag"]["reg_model_blended"] = reg_model_blended;
    }

    // ================================================================
    // SECTION 4b: Astrometric rescue (ASTAP plate-solving) - nach Section 6
    // ================================================================
    // §4.13 — Astrometrische Rescue als letztes Mittel für Model-Frames
    // mit niedrigem CC. Läuft nach Model-Generierung um auch interpolierte
    // Frames zu retten.
    if (kUseAstrometry) {
      const float astrometry_cc_threshold = std::clamp(
          cfg.registration.reject_cc_min_abs * 0.5f, 0.05f, 0.15f);
      const int astrometry_chain_depth_threshold =
          std::max(12, kMaxBlindChainAnchorDepth * 2);
      auto should_try_astrometry = [&](size_t fi) -> bool {
        if (fi >= frames.size() || is_active_anchor_frame[fi] != 0 ||
            reg_provenance[fi] == RegistrationProvenance::astrometric_rescue) {
          return false;
        }
        const bool unresolved = global_frame_cc[fi] <= 0.0f;
        const bool weak_chained =
            reg_chain_validated[fi] != 0 &&
            global_frame_cc[fi] < astrometry_cc_threshold;
        const bool extreme_chain =
            reg_chain_depth[fi] >= astrometry_chain_depth_threshold;
        // Also try astrometry for model-interpolated frames with low CC
        // These frames failed direct registration and got interpolated warps
        const bool weak_model =
            global_frame_cc[fi] > 0.0f &&
            global_frame_cc[fi] < astrometry_cc_threshold &&
            (reg_provenance[fi] == RegistrationProvenance::model_global_poly ||
             reg_provenance[fi] == RegistrationProvenance::model_local_poly ||
             reg_provenance[fi] == RegistrationProvenance::model_interpolated ||
             reg_provenance[fi] == RegistrationProvenance::model_blended ||
             reg_provenance[fi] == RegistrationProvenance::model_nearest_copy);
        return unresolved || weak_chained || extreme_chain || weak_model;
      };
      auto nearest_astrometry_anchor_slot = [&](int frame_idx) -> size_t {
        size_t best_slot = 0;
        int best_dist = std::numeric_limits<int>::max();
        for (size_t slot = 0; slot < active_anchor_indices.size(); ++slot) {
          const int d = std::abs(active_anchor_indices[slot] - frame_idx);
          if (d < best_dist) {
            best_dist = d;
            best_slot = slot;
          }
        }
        return best_slot;
      };

      if (registration::is_astap_available(cfg.astrometry.astap_bin,
                                            cfg.astrometry.astap_data_dir)) {
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!should_try_astrometry(fi)) {
            continue;
          }

          const size_t anchor_slot =
              nearest_astrometry_anchor_slot(static_cast<int>(fi));
          const int ref_anchor_idx = active_anchor_indices[anchor_slot];
          const Matrix2Df &ref_proxy_astro = active_anchor_proxies[anchor_slot];
          const std::string ref_fits_path =
              frames[static_cast<size_t>(ref_anchor_idx)].string();
          ++reg_astrometric_attempted;

          const Matrix2Df mov_proxy_astro = load_registration_proxy(fi);
          const std::string mov_fits_path = frames[fi].string();

          auto astro_res =
              registration::try_astrometric_rescue_from_paths(
                  mov_fits_path, ref_fits_path,
                  mov_proxy_astro, ref_proxy_astro,
                  cfg.astrometry.astap_bin,
                  cfg.astrometry.astap_data_dir,
                  global_reg_scale,
                  static_cast<float>(cfg.astrometry.search_radius),
                  0.20f);

          if (astro_res.success) {
            const float prev_cc = global_frame_cc[fi];
            const bool weak_or_unresolved =
                prev_cc <= 0.0f || prev_cc < astrometry_cc_threshold;
            const bool replace_existing =
                prev_cc <= 0.0f ||
                astro_res.correlation > prev_cc + 0.02f ||
                reg_chain_depth[fi] >= astrometry_chain_depth_threshold;
            if (!replace_existing) {
              continue;
            }

            set_registration_state(
                fi,
                registration::scale_translation_warp(astro_res.warp,
                                                     global_reg_scale),
                astro_res.correlation, true, 0,
                RegistrationProvenance::astrometric_rescue);
            ++reg_astrometric_rescued;
            if (weak_or_unresolved && prev_cc > 0.0f) {
              ++reg_astrometric_replaced_weak;
            }
          }
        }

        if (reg_astrometric_rescued > 0) {
          std::ostringstream msg;
          msg << "REGISTRATION astrometric rescue: recovered "
              << reg_astrometric_rescued
              << " frames via plate-solving (attempted="
              << reg_astrometric_attempted << ", weak_replaced="
              << reg_astrometric_replaced_weak << ")";
          emitter.warning(run_id, msg.str(), log_file);
          std::cout << "[REG-ASTROMETRY] " << msg.str() << std::endl;
        }
      } else {
        std::cout << "[REG-ASTROMETRY] ASTAP not available, skipping astrometric rescue" << std::endl;
      }
    }
    global_reg_extra["diag"]["reg_astrometric_rescued"] = reg_astrometric_rescued;
    global_reg_extra["diag"]["reg_astrometric_attempted"] = reg_astrometric_attempted;
    global_reg_extra["diag"]["reg_astrometric_replaced_weak"] =
        reg_astrometric_replaced_weak;
  }
  if (reg_reject_orientation_outliers > 0 ||
      reg_reject_reflection_outliers > 0 ||
      reg_reject_scale_outliers > 0 ||
      reg_reject_cc_outliers > 0 ||
      reg_reject_shift_outliers > 0 ||
      reg_reject_low_cc_protected > 0 ||
      reg_reject_deep_chain_outliers > 0 ||
      reg_meridian_flip_detected > 0) {
    std::cout << "[REG-FILTER] rejected outlier warps: orientation="
              << reg_reject_orientation_outliers
              << " reflection=" << reg_reject_reflection_outliers
              << " scale=" << reg_reject_scale_outliers
              << " cc=" << reg_reject_cc_outliers
              << " shift=" << reg_reject_shift_outliers
              << " deep_chain=" << reg_reject_deep_chain_outliers
              << " low_cc_protected=" << reg_reject_low_cc_protected
              << " meridian_flip_detected=" << reg_meridian_flip_detected
              << std::endl;
  }
  global_reg_extra["diag"]["reg_reject_orientation_outliers"] =
      reg_reject_orientation_outliers;
  global_reg_extra["diag"]["reg_reject_reflection_outliers"] =
      reg_reject_reflection_outliers;
  global_reg_extra["diag"]["reg_reject_scale_outliers"] = reg_reject_scale_outliers;
  global_reg_extra["diag"]["reg_reject_cc_outliers"] = reg_reject_cc_outliers;
  global_reg_extra["diag"]["reg_reject_shift_outliers"] = reg_reject_shift_outliers;
  global_reg_extra["diag"]["reg_reject_low_cc_protected"] = reg_reject_low_cc_protected;
  global_reg_extra["diag"]["reg_reject_deep_chain_outliers"] = reg_reject_deep_chain_outliers;
  global_reg_extra["diag"]["reg_meridian_flip_detected"] = reg_meridian_flip_detected;
  global_reg_extra["reg_rejected_frames"] = static_cast<int>(reg_rejected_frames.size());
  global_reg_extra["diag"]["reg_rejected_frames"] = reg_rejected_frames;

  // All frames now have warps (valid registration or polynomial prediction).
  // Tile-level quality metrics handle downstream weighting (v3.2.2 §1.2).
  int n_cc_positive = 0;
  int n_cc_zero = 0;
  int n_cc_negative = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (global_frame_cc[fi] > 0.0f) {
      ++n_cc_positive;
    } else if (global_frame_cc[fi] < 0.0f) {
      ++n_cc_negative;
    } else {
      ++n_cc_zero;
    }
  }
  std::cout << "[REG] cc>0: " << n_cc_positive << ", cc==0: " << n_cc_zero
            << ", cc<0: " << n_cc_negative << std::endl;
  global_reg_extra["frames_cc_positive"] = n_cc_positive;
  global_reg_extra["frames_cc_zero"] = n_cc_zero;
  global_reg_extra["frames_cc_negative"] = n_cc_negative;
  core::json reg_source_counts = core::json::object();
  int reg_max_chain_depth = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    const std::string source = registration_provenance_name(reg_provenance[fi]);
    reg_source_counts[source] = reg_source_counts.value(source, 0) + 1;
    if (reg_chain_depth[fi] > reg_max_chain_depth) {
      reg_max_chain_depth = reg_chain_depth[fi];
    }
  }
  global_reg_extra["diag"]["reg_source_counts"] = reg_source_counts;
  global_reg_extra["diag"]["reg_max_chain_depth"] = reg_max_chain_depth;
  global_reg_extra["diag"]["reg_blind_chain_anchor_depth_limit"] =
      kMaxBlindChainAnchorDepth;
  global_reg_extra["diag"]["reg_blind_chain_anchor_strong_cc"] =
      kBlindChainStrongAnchorCc;

  if (global_reg_status == "ok") {
    try {
      write_global_registration_artifact(global_reg_scale);
    } catch (...) {
    }
  }

  emitter.phase_end(run_id, Phase::REGISTRATION, global_reg_status,
                    global_reg_extra, log_file);

  // Export model-predicted mask so the pipeline can apply a weight penalty.
  out.model_predicted_mask.assign(frames.size(), 0);
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    const auto p = reg_provenance[fi];
    if (p == RegistrationProvenance::model_interpolated ||
        p == RegistrationProvenance::model_blended ||
        p == RegistrationProvenance::model_global_poly ||
        p == RegistrationProvenance::model_local_poly ||
        p == RegistrationProvenance::model_nearest_copy) {
      out.model_predicted_mask[fi] = 1;
    }
  }

  emitter.phase_start(run_id, Phase::PREWARP, "PREWARP", log_file);

  // ================================================================
  // SECTION 7: Canvas bounds computation
  // ================================================================
  // Compute bounding box for field rotation: output canvas must be large enough
  // to contain all rotated frames (Alt/Az mounts near pole).
  // Skip frames that are `unresolved` (either never registered or whose model
  // prediction was rejected as out-of-bounds extrapolation). Including them
  // would either use identity warps (harmless) or — before the bounds guard
  // existed — bogus model warps that inflated the canvas by thousands of
  // pixels and produced ghost patches at the borders during stacking.
  std::vector<WarpMatrix> bbox_warps;
  bbox_warps.reserve(frames.size());
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (reg_provenance[fi] == RegistrationProvenance::unresolved) {
      continue;
    }
    bbox_warps.push_back(global_frame_warps[fi]);
  }
  WarpBounds bbox = compute_warps_bounds(width, height, bbox_warps);
  
  // Round canvas to even dimensions for CFA (Bayer) compatibility.
  // warp_cfa_mosaic_via_subplanes works on half-resolution subplanes, so
  // canvas must be even in both dimensions to avoid size mismatch in store().
  int canvas_width = (bbox.width() + 1) & ~1;   // round up to even
  int canvas_height = (bbox.height() + 1) & ~1; // round up to even
  
  // Offset to shift all frames into positive coordinate space
  int offset_x = -bbox.min_x;
  int offset_y = -bbox.min_y;
  
  // For OSC mode: ensure offsets are even to preserve Bayer pattern alignment.
  // If offset is odd, the entire CFA mosaic shifts by one pixel and R/G swap.
  if (detected_mode == ColorMode::OSC) {
    if ((offset_x & 1) != 0) {
      offset_x = (offset_x + 1) & ~1;  // round up to even
      canvas_width += 1;
      canvas_width = (canvas_width + 1) & ~1;  // keep canvas even
    }
    if ((offset_y & 1) != 0) {
      offset_y = (offset_y + 1) & ~1;  // round up to even
      canvas_height += 1;
      canvas_height = (canvas_height + 1) & ~1;  // keep canvas even
    }
  }

  const auto prewarp_acceleration =
      acceleration.selection_for(core::AccelerationPhase::prewarp);
  const core::AccelerationOps prewarp_ops(
      acceleration, core::AccelerationPhase::prewarp);
  const auto prewarp_input_batch =
      core::make_device_frame_batch(frames.size(), height, width, 1);
  const auto prewarp_output_batch =
      core::make_device_frame_batch(frames.size(), canvas_height, canvas_width,
                                    1);
  {
    std::ostringstream msg;
    msg << "PREWARP acceleration "
        << core::acceleration_selection_summary(prewarp_acceleration);
    if (!prewarp_acceleration.request_honored &&
        !prewarp_acceleration.fallback_reason.empty()) {
      emitter.warning(run_id, msg.str(), log_file);
    }
    std::cout << "[PREWARP] " << msg.str() << std::endl;
  }
  
  // Apply offset correction to all warps
  if (offset_x != 0 || offset_y != 0) {
    const float ox = static_cast<float>(offset_x);
    const float oy = static_cast<float>(offset_y);
    for (auto& w : global_frame_warps) {
      // Compose destination-space translation q = p + offset into an inverse-map
      // warp M (src = M * p) => M' = M * T(-offset).
      w(0, 2) -= w(0, 0) * ox + w(0, 1) * oy;
      w(1, 2) -= w(1, 0) * ox + w(1, 1) * oy;
    }
  }
  
  // Log canvas expansion for field rotation
  if (canvas_width > width || canvas_height > height) {
    std::ostringstream msg;
    msg << "Field rotation detected: expanding canvas from " << width << "x" << height
        << " to " << canvas_width << "x" << canvas_height
        << " (bbox: [" << bbox.min_x << "," << bbox.min_y << "] to ["
        << bbox.max_x << "," << bbox.max_y << "], offset: ["
        << offset_x << "," << offset_y << "])";
    emitter.warning(run_id, msg.str(), log_file);
    std::cout << "[PREWARP] " << msg.str() << std::endl;
  }

  // Pre-warp all frames at full resolution before tile extraction.
  // Applying rotation warps to small tile ROIs is fundamentally broken:
  // warpAffine needs source pixels outside the tile boundary that don't
  // exist, causing CFA pattern corruption (colored tile rectangles).
  //
  // Disk-backed: frames are written as raw float binaries and mmap'd on
  // demand, so RAM usage is bounded by OS page cache rather than N*W*H*4.
  DiskCacheFrameStore prewarped_frames(
      run_dir / ".prewarped_cache", frames.size(), canvas_height, canvas_width);
  std::vector<uint8_t> frame_has_data(frames.size(), 0);
  const size_t canvas_px =
      static_cast<size_t>(std::max(0, canvas_height)) *
      static_cast<size_t>(std::max(0, canvas_width));
  const int prewarp_workers = compute_adaptive_worker_count(
      cfg, frames.size(), frames, WorkerParallelProfile::IoHeavy);
  std::vector<std::vector<uint16_t>> worker_overlap_coverage(
      static_cast<size_t>(std::max(1, prewarp_workers)));
  std::cout << "[PREWARP] Using " << prewarp_workers
            << " parallel workers for " << frames.size() << " frames"
            << std::endl;
  std::mutex prewarp_log_mutex;
  std::mutex prewarp_progress_mutex;
  std::atomic<size_t> prewarp_next{0};
  std::atomic<size_t> prewarp_done{0};
  std::atomic<int> n_frames_with_data{0};
  std::atomic<bool> prewarp_failed{false};
  std::string prewarp_error;

#if TILE_COMPILE_PREWARP_HAS_CUDA
  std::vector<cv::cuda::Stream> prewarp_streams;
  if (prewarp_acceleration.using_gpu && prewarp_workers > 1) {
    prewarp_streams.resize(static_cast<size_t>(prewarp_workers));
  }
#endif

  auto prewarp_worker = [&](int worker_index) {
    std::vector<uint16_t> local_overlap_coverage(canvas_px, 0);
    while (true) {
      const size_t fi = prewarp_next.fetch_add(1);
      if (fi >= frames.size()) {
        break;
      }
      // Skip frames whose registration is unresolved (never registered, or
      // rejected as outlier with no usable model prediction). frame_has_data
      // stays 0 for these, so downstream phases exclude them from stacking
      // entirely rather than placing them at identity on the canvas.
      if (reg_provenance[fi] == RegistrationProvenance::unresolved) {
        prewarp_done.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      try {
        Matrix2Df img = load_frame_normalized(fi);
        if (img.size() <= 0) {
          continue;
        }
        if (cfg.stacking.per_frame_cosmetic_correction) {
          if (detected_mode == ColorMode::OSC) {
            img = image::cosmetic_correction_cfa(
                img, cfg.stacking.per_frame_cosmetic_correction_sigma, true, 0,
                0);
          } else {
            img = image::cosmetic_correction(
                img, cfg.stacking.per_frame_cosmetic_correction_sigma, true);
          }
        }
        const auto &w = global_frame_warps[fi];
        Matrix2Df warped;
        std::vector<uint8_t> warped_valid_mask;
        bool warped_has_data = false;
        prewarp_ops.warp_affine_frame(std::move(img), w, detected_mode,
                                      canvas_height, canvas_width, offset_x,
                                      offset_y, warped, &warped_valid_mask,
                                      &warped_has_data
#if TILE_COMPILE_PREWARP_HAS_CUDA
                                      , prewarp_streams.empty() ? nullptr : &prewarp_streams[static_cast<size_t>(worker_index)]
#endif
                                      );
        if (warped.size() > 0) {
          prewarped_frames.store(fi, warped);
          const bool stored = prewarped_frames.has_data(fi);
          if (stored && warped_has_data) {
            frame_has_data[fi] = 1;
            n_frames_with_data.fetch_add(1, std::memory_order_relaxed);
            if (warped_valid_mask.size() == canvas_px) {
              for (size_t pi = 0; pi < canvas_px; ++pi) {
                if (warped_valid_mask[pi] != 0 &&
                    local_overlap_coverage[pi] <
                        std::numeric_limits<uint16_t>::max()) {
                  ++local_overlap_coverage[pi];
                }
              }
            } else {
              const float *warped_ptr = warped.data();
              for (size_t pi = 0; pi < canvas_px; ++pi) {
                if (std::isfinite(warped_ptr[pi]) &&
                    local_overlap_coverage[pi] <
                        std::numeric_limits<uint16_t>::max()) {
                  ++local_overlap_coverage[pi];
                }
              }
            }
          }
        }
      } catch (const std::exception &e) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = e.what();
        }
      } catch (...) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = "unknown_error";
        }
      }

      const size_t done = prewarp_done.fetch_add(1) + 1;
      if (done % 5 == 0 || done == frames.size()) {
        std::lock_guard<std::mutex> lock(prewarp_progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::PREWARP, static_cast<int>(done),
            static_cast<int>(frames.size()), "prewarp workers=" +
                                              std::to_string(prewarp_workers),
            "frames", log_file);
      }
    }
    worker_overlap_coverage[static_cast<size_t>(worker_index)] =
        std::move(local_overlap_coverage);
  };

  if (prewarp_workers > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(prewarp_workers));
    for (int w = 0; w < prewarp_workers; ++w) {
      workers.emplace_back(prewarp_worker, w);
    }
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    prewarp_worker(0);
  }

  if (prewarp_failed.load(std::memory_order_relaxed)) {
    core::json extra = {
        {"error", prewarp_error.empty() ? "unknown_error" : prewarp_error},
        {"acceleration",
         core::acceleration_selection_to_json(prewarp_acceleration)},
        {"device_frame_batch_input",
         core::device_frame_batch_to_json(prewarp_input_batch)},
        {"device_frame_batch_output",
         core::device_frame_batch_to_json(prewarp_output_batch)},
    };
    emitter.phase_end(run_id, Phase::PREWARP, "error", extra, log_file);
    std::cerr << "Error during PREWARP: "
              << (prewarp_error.empty() ? "unknown_error" : prewarp_error)
              << std::endl;
    emitter.run_end(run_id, false, "error", log_file);
    return false;
  }

  out.n_usable_frames = n_frames_with_data.load(std::memory_order_relaxed);
  const int required_common_frames =
      compute_required_common_overlap_frames(out.n_usable_frames);
  out.canvas_width = canvas_width;
  out.canvas_height = canvas_height;
  out.tile_offset_x = offset_x;
  out.tile_offset_y = offset_y;
  out.overlap_coverage_count.assign(canvas_px, 0);
  for (const auto &local_overlap_coverage : worker_overlap_coverage) {
    if (local_overlap_coverage.size() != canvas_px) {
      continue;
    }
    for (size_t pi = 0; pi < canvas_px; ++pi) {
      const uint32_t accum =
          static_cast<uint32_t>(out.overlap_coverage_count[pi]) +
          static_cast<uint32_t>(local_overlap_coverage[pi]);
      out.overlap_coverage_count[pi] = static_cast<uint16_t>(
          std::min<uint32_t>(accum, std::numeric_limits<uint16_t>::max()));
    }
  }
  out.common_valid_mask.assign(canvas_px, static_cast<uint8_t>(0));
  for (size_t pi = 0; pi < canvas_px; ++pi) {
    if (static_cast<int>(out.overlap_coverage_count[pi]) >=
        required_common_frames) {
      out.common_valid_mask[pi] = static_cast<uint8_t>(1);
    }
  }

  core::json prewarp_extra = {
      {"num_frames", static_cast<int>(frames.size())},
      {"num_frames_with_data", out.n_usable_frames},
      {"canvas_width", canvas_width},
      {"canvas_height", canvas_height},
      {"tile_offset_x", offset_x},
      {"tile_offset_y", offset_y},
      {"workers", prewarp_workers},
      {"common_overlap_mode", "inline_prewarp_coverage"},
      {"required_common_frames", required_common_frames},
      {"acceleration",
       core::acceleration_selection_to_json(prewarp_acceleration)},
      {"device_frame_batch_input",
       core::device_frame_batch_to_json(prewarp_input_batch)},
      {"device_frame_batch_output",
       core::device_frame_batch_to_json(prewarp_output_batch)},
  };
  emitter.phase_end(run_id, Phase::PREWARP, "ok", prewarp_extra, log_file);

  out.frame_has_data = std::move(frame_has_data);
  out.prewarped_frames = std::move(prewarped_frames);
  out.min_valid_frames = required_common_frames;
  return true;
}

} // namespace tile_compile::runner
