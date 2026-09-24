#pragma once

// QualityFrameWeightPlan --- milestone M3 (plan section 11.9). Records the
// per-frame effective quality factor G_eff(f) and its three inputs, so that
//   G_eff(f) = G_quality(f)
//            * model_prediction_factor(f)
//            * registration_residual_factor(f)
// is computed exactly once, before pixel reconstruction, and never a second
// time inside the reconstructor (plan 11.9: "Eine doppelte Anwendung der
// Registrierungsfaktoren in Pipeline und Rekonstruktor ist ausgeschlossen").
//
// The two registration factors are taken verbatim from the
// RegistrationSamplingPlan (where runner_phase_registration already computed
// and persisted them); GLOBAL_QUALITY only reads them here, never recomputes
// them.

#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/global_quality.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include <string>
#include <vector>

namespace tile_compile::reconstruction {

struct QualityFrameWeight {
  std::string frame_id;
  float g_quality = 0.0f;
  float model_prediction_factor = 0.0f;
  float registration_residual_factor = 0.0f;
  float g_eff = 0.0f;  // product of the three above
};

struct QualityFrameWeightPlan {
  static constexpr int kSchemaVersion = 1;

  std::string source_identity_hash;       // from the sampling plan
  std::string sampling_plan_hash;         // from the sampling plan
  std::string source_quality_config_hash; // proxy version + GlobalQualityConfig

  std::vector<QualityFrameWeight> frames;

  std::string plan_hash;  // canonical, over every field above
};

// `g_quality` must be positionally aligned with `sampling_plan.frames`
// (one entry per frame, same order --- the plan's frames are already
// source_index-ascending). g_eff = g_quality * model_prediction_factor *
// registration_residual_factor, with the two factors read straight from the
// sampling plan. `plan_hash` is filled in.
QualityFrameWeightPlan build_quality_frame_weight_plan(
    const registration::RegistrationSamplingPlan& sampling_plan,
    const VectorXf& g_quality, const std::string& source_quality_config_hash);

// Canonical hash of proxy_version=1 plus every numeric field of the config
// that changes G_quality(f). Belongs to the source-quality hash domain
// (plan 18.3): a change here invalidates G_quality and all Q-profiles.
std::string compute_source_quality_config_hash(const GlobalQualityConfig& cfg);

// Canonical, byte-exact hash over the whole plan (fixed field order, little
// endian, IEEE-754 float bit patterns) --- same scheme as
// registration_sampling_plan's compute_plan_hash().
std::string compute_quality_frame_weight_plan_hash(const QualityFrameWeightPlan& plan);

std::string serialize_quality_frame_weight_plan(const QualityFrameWeightPlan& plan);
bool parse_quality_frame_weight_plan(const std::string& json,
                                     QualityFrameWeightPlan& out, std::string& error);

}  // namespace tile_compile::reconstruction
