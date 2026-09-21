#include "tile_compile/registration/warp_prediction_gate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tile_compile::registration {

namespace {

constexpr float kRadToDeg = 57.29577951308232f;
constexpr float kPi = 3.14159265358979323846f;

float wrap_to_pi(float a) {
  while (a > kPi) {
    a -= 2.0f * kPi;
  }
  while (a < -kPi) {
    a += 2.0f * kPi;
  }
  return a;
}

float median_of(std::vector<float> v) {
  if (v.empty()) {
    return 0.0f;
  }
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
  if (v.size() % 2 == 1) {
    return v[mid];
  }
  const float hi = v[mid];
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid - 1), v.end());
  return 0.5f * (hi + v[mid - 1]);
}

}  // namespace

WarpPredictionGateResult evaluate_warp_prediction(
    float fi, float pred_ang_rad, float pred_tx, float pred_ty,
    std::span<const WarpPredictionAnchor> anchors) {
  WarpPredictionGateResult result;
  const int n = static_cast<int>(anchors.size());

  int l = -1;
  int r = -1;
  for (int i = 0; i < n; ++i) {
    if (anchors[static_cast<size_t>(i)].fi <= fi) {
      l = i;
    }
    if (anchors[static_cast<size_t>(i)].fi > fi) {
      r = i;
      break;
    }
  }
  if (l < 0 && r < 0) {
    result.reason = WarpPredictionGateReason::no_anchor;
    return result;
  }

  // Per-consecutive-pair rates over the full anchor sequence.
  std::vector<float> rate_ang;
  std::vector<float> rate_shift;
  rate_ang.reserve(static_cast<size_t>(std::max(0, n - 1)));
  rate_shift.reserve(static_cast<size_t>(std::max(0, n - 1)));
  for (int i = 0; i + 1 < n; ++i) {
    const auto &a = anchors[static_cast<size_t>(i)];
    const auto &b = anchors[static_cast<size_t>(i + 1)];
    const float dfi = std::max(1.0f, b.fi - a.fi);
    rate_ang.push_back(std::fabs(b.ang_rad - a.ang_rad) / dfi);
    rate_shift.push_back(std::hypot(b.tx - a.tx, b.ty - a.ty) / dfi);
  }
  const float global_rate_ang = median_of(rate_ang);
  const float global_rate_shift = median_of(rate_shift);

  // Local rate on one side of an anchor: max over the up-to
  // kWarpGateLocalRatePairs consecutive pairs adjacent to the anchor on that
  // side (pairs ending at `anchor_idx` for a left anchor, pairs starting at
  // `anchor_idx` for a right anchor). Falls back to the global rate.
  auto side_rates = [&](int anchor_idx, bool left_side, float &ra,
                        float &rs) {
    float la = 0.0f;
    float ls = 0.0f;
    bool any = false;
    const int lo = left_side ? std::max(0, anchor_idx - kWarpGateLocalRatePairs)
                             : anchor_idx;
    const int hi = left_side ? anchor_idx - 1
                             : std::min(n - 2, anchor_idx +
                                                   kWarpGateLocalRatePairs - 1);
    for (int i = lo; i <= hi; ++i) {
      la = std::max(la, rate_ang[static_cast<size_t>(i)]);
      ls = std::max(ls, rate_shift[static_cast<size_t>(i)]);
      any = true;
    }
    ra = std::max(global_rate_ang, any ? la : 0.0f);
    rs = std::max(global_rate_shift, any ? ls : 0.0f);
  };

  // Single-sided: extrapolation past the anchor span.
  if (l < 0 || r < 0) {
    const int anchor_idx = (l >= 0) ? l : r;
    const float dist = std::fabs(fi - anchors[static_cast<size_t>(anchor_idx)].fi);
    result.anchor_distance = static_cast<int>(std::lround(dist));
    if (dist > static_cast<float>(kWarpGateMaxExtrapolationFrames)) {
      result.reason = WarpPredictionGateReason::extrapolation_too_long;
      return result;
    }
  }

  struct SideEval {
    int anchor_idx = -1;
    float angle_dev_deg = 0.0f;
    float angle_allowed_deg = 0.0f;
    float shift_dev_px = 0.0f;
    float shift_allowed_px = 0.0f;
    float dist = 0.0f;
    float angle_ratio = 0.0f;
    float shift_ratio = 0.0f;
    WarpPredictionGateReason fail = WarpPredictionGateReason::ok;
  };

  auto eval_side = [&](int anchor_idx, bool left_side) -> SideEval {
    SideEval e;
    e.anchor_idx = anchor_idx;
    const auto &a = anchors[static_cast<size_t>(anchor_idx)];
    e.dist = std::fabs(fi - a.fi);
    float ra = 0.0f;
    float rs = 0.0f;
    side_rates(anchor_idx, left_side, ra, rs);
    e.angle_allowed_deg =
        std::min(kWarpGateAngleHardCapDeg,
                 kWarpGateAngleAbsTolDeg +
                     kWarpGateRateMult * ra * kRadToDeg * e.dist);
    e.angle_dev_deg =
        std::fabs(wrap_to_pi(pred_ang_rad - a.ang_rad)) * kRadToDeg;
    e.angle_ratio = e.angle_dev_deg / std::max(1.0e-6f, e.angle_allowed_deg);
    if (e.angle_dev_deg > e.angle_allowed_deg) {
      e.fail = WarpPredictionGateReason::angle_discontinuity;
      return e;
    }
    e.shift_allowed_px =
        std::min(kWarpGateShiftHardCapPx,
                 kWarpGateShiftAbsTolPx + kWarpGateRateMult * rs * e.dist);
    e.shift_dev_px = std::hypot(pred_tx - a.tx, pred_ty - a.ty);
    e.shift_ratio = e.shift_dev_px / std::max(1.0e-6f, e.shift_allowed_px);
    if (e.shift_dev_px > e.shift_allowed_px) {
      e.fail = WarpPredictionGateReason::shift_discontinuity;
    }
    return e;
  };

  auto fill_from = [&](const SideEval &e) {
    result.angle_dev_deg = e.angle_dev_deg;
    result.angle_allowed_deg = e.angle_allowed_deg;
    result.shift_dev_px = e.shift_dev_px;
    result.shift_allowed_px = e.shift_allowed_px;
    result.anchor_distance = static_cast<int>(std::lround(e.dist));
  };

  // Angle check has priority over shift check; check l before r.
  SideEval el;
  SideEval er;
  if (l >= 0) {
    el = eval_side(l, true);
    if (el.fail != WarpPredictionGateReason::ok) {
      result.reason = el.fail;
      fill_from(el);
      return result;
    }
  }
  if (r >= 0) {
    er = eval_side(r, false);
    if (er.fail != WarpPredictionGateReason::ok) {
      result.reason = er.fail;
      fill_from(er);
      return result;
    }
  }

  result.ok = true;
  result.reason = WarpPredictionGateReason::ok;
  if (l >= 0 && r >= 0) {
    fill_from((el.angle_ratio + el.shift_ratio >=
               er.angle_ratio + er.shift_ratio)
                  ? el
                  : er);
  } else if (l >= 0) {
    fill_from(el);
  } else {
    fill_from(er);
  }
  return result;
}

const char *warp_prediction_gate_reason_name(WarpPredictionGateReason reason) {
  switch (reason) {
    case WarpPredictionGateReason::ok:
      return "ok";
    case WarpPredictionGateReason::no_anchor:
      return "no_anchor";
    case WarpPredictionGateReason::extrapolation_too_long:
      return "extrapolation_too_long";
    case WarpPredictionGateReason::angle_discontinuity:
      return "angle_discontinuity";
    case WarpPredictionGateReason::shift_discontinuity:
      return "shift_discontinuity";
  }
  return "unknown";
}

}  // namespace tile_compile::registration
