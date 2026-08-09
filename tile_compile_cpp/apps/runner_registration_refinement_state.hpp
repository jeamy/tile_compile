#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/registration/global_registration.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tile_compile::runner::detail {

struct RegistrationResidualStats {
  int ref_stars = 0;
  int frame_stars = 0;
  int matched_stars = 0;
  float median_px = 0.0f;
  float p90_px = 0.0f;
  float rms_px = 0.0f;
  float max_px = 0.0f;
  float weight_factor = 1.0f;
  bool applicable = false;
};

struct AffineRefinementFrameStats {
  bool attempted = false;
  bool applied = false;
  std::string reason = "disabled";
  registration::AffineStarRefinementResult fit;
  float ncc_before = 0.0f;
  float ncc_after = 0.0f;
  float overlap_ratio = 0.0f;
};

struct SmoothLocalRefinementFrameStats {
  bool attempted = false;
  bool applied = false;
  std::string reason = "disabled";
  registration::SmoothLocalRefinementResult fit;
  float ncc_before = 0.0f;
  float ncc_after = 0.0f;
  float overlap_ratio = 0.0f;
};

struct RefinementAggregateState {
  int residual_applicable = 0;
  int residual_damped = 0;
  int affine_attempted = 0;
  int affine_applied = 0;
  int affine_rejected = 0;
  int local_attempted = 0;
  int local_applied = 0;
  int local_rejected = 0;
  std::vector<float> residual_medians;
  std::vector<float> residual_p90s;
  std::vector<float> residual_factors;
};

struct RefinementRollbackSnapshot {
  WarpMatrix warp = WarpMatrix::Zero();
  int residual_applicable = 0;
  int residual_damped = 0;
  int affine_attempted = 0;
  int affine_applied = 0;
  int affine_rejected = 0;
  int local_attempted = 0;
  int local_applied = 0;
  int local_rejected = 0;
  size_t residual_medians_size = 0;
  size_t residual_p90s_size = 0;
  size_t residual_factors_size = 0;
};

inline RefinementRollbackSnapshot make_refinement_rollback_snapshot(
    const WarpMatrix &warp, const RefinementAggregateState &aggregate) {
  return {warp,
          aggregate.residual_applicable,
          aggregate.residual_damped,
          aggregate.affine_attempted,
          aggregate.affine_applied,
          aggregate.affine_rejected,
          aggregate.local_attempted,
          aggregate.local_applied,
          aggregate.local_rejected,
          aggregate.residual_medians.size(),
          aggregate.residual_p90s.size(),
          aggregate.residual_factors.size()};
}

inline void rollback_refinement_frame(
    const RefinementRollbackSnapshot &snapshot, WarpMatrix &warp,
    float &residual_weight_factor, RegistrationResidualStats &residual_stats,
    AffineRefinementFrameStats &affine,
    SmoothLocalRefinementFrameStats &local,
    RefinementAggregateState &aggregate) {
  warp = snapshot.warp;
  residual_weight_factor = 1.0f;
  residual_stats = RegistrationResidualStats{};
  aggregate.residual_applicable = snapshot.residual_applicable;
  aggregate.residual_damped = snapshot.residual_damped;
  aggregate.affine_attempted =
      snapshot.affine_attempted + (affine.attempted ? 1 : 0);
  aggregate.affine_applied = snapshot.affine_applied;
  aggregate.affine_rejected =
      snapshot.affine_rejected + (affine.attempted ? 1 : 0);
  aggregate.local_attempted =
      snapshot.local_attempted + (local.attempted ? 1 : 0);
  aggregate.local_applied = snapshot.local_applied;
  aggregate.local_rejected =
      snapshot.local_rejected + (local.attempted ? 1 : 0);
  aggregate.residual_medians.resize(snapshot.residual_medians_size);
  aggregate.residual_p90s.resize(snapshot.residual_p90s_size);
  aggregate.residual_factors.resize(snapshot.residual_factors_size);
  affine.applied = false;
  affine.reason = "exception";
  local.applied = false;
  local.fit.model.valid = false;
  local.reason = "exception";
}

struct RefinementProxyCache {
  Matrix2Df proxy;
  std::vector<registration::StarPoint> stars;

  template <typename Detector>
  bool commit_candidate(bool accepted, Matrix2Df candidate,
                        Detector &&detector) {
    if (!accepted) {
      return false;
    }
    proxy = std::move(candidate);
    stars = std::forward<Detector>(detector)(proxy);
    return true;
  }
};

} // namespace tile_compile::runner::detail
