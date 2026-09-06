#include "tile_compile/reconstruction/quality_frame_weight_plan.hpp"

#include "tile_compile/core/utils.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <set>

namespace tile_compile::reconstruction {

using json = nlohmann::json;

namespace {

// Byte-exact canonical encoder --- same convention as
// registration_sampling_plan.cpp's ByteSink (little endian, IEEE-754 float
// bit patterns, NaN payload normalized).
struct ByteSink {
  std::vector<uint8_t> bytes;
  void u32(uint32_t v) {
    bytes.push_back(static_cast<uint8_t>(v & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
  }
  void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
  void u64(uint64_t v) {
    u32(static_cast<uint32_t>(v & 0xffffffffu));
    u32(static_cast<uint32_t>((v >> 32) & 0xffffffffu));
  }
  void f32(float v) {
    if (std::isnan(v)) v = std::numeric_limits<float>::quiet_NaN();
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    u32(bits);
  }
  void b(bool v) { bytes.push_back(v ? 1 : 0); }
  void str(const std::string& s) {
    u64(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};

void validate_weights(const QualityFrameWeightPlan &plan) {
  std::set<std::string> ids;
  if (plan.source_identity_hash.empty() || plan.sampling_plan_hash.empty() ||
      plan.source_quality_config_hash.empty())
    throw std::invalid_argument("quality_frame_weight_plan missing provenance");
  for (const auto &f : plan.frames) {
    if (f.frame_id.empty() || !ids.insert(f.frame_id).second)
      throw std::invalid_argument("quality_frame_weight_plan duplicate/empty frame_id");
    for (float v : {f.g_quality, f.model_prediction_factor,
                    f.registration_residual_factor, f.g_eff})
      if (!std::isfinite(v) || v < 0 || v > 1)
        throw std::invalid_argument("quality_frame_weight_plan invalid factor");
    if (f.g_eff != f.g_quality * f.model_prediction_factor * f.registration_residual_factor)
      throw std::invalid_argument("quality_frame_weight_plan inconsistent g_eff");
  }
}

}  // namespace

std::string compute_source_quality_config_hash(const GlobalQualityConfig& cfg) {
  ByteSink s;
  s.str("source_quality_config:proxy_version=1");
  s.f32(cfg.w_bg);
  s.f32(cfg.w_noise);
  s.f32(cfg.w_grad);
  s.f32(cfg.w_fwhm);
  s.f32(cfg.w_roundness);
  s.f32(cfg.w_star_count);
  s.f32(cfg.clamp_lo);
  s.f32(cfg.clamp_hi);
  s.b(cfg.adaptive_weights);
  s.f32(cfg.weight_exponent_scale);
  s.i32(cfg.star_max_corners);
  s.i32(cfg.star_patch_radius);
  return core::sha256_bytes(s.bytes);
}

std::string compute_quality_frame_weight_plan_hash(const QualityFrameWeightPlan& plan) {
  ByteSink s;
  s.str("quality_frame_weight_plan:v1");
  s.i32(QualityFrameWeightPlan::kSchemaVersion);
  s.str(plan.source_identity_hash);
  s.str(plan.sampling_plan_hash);
  s.str(plan.source_quality_config_hash);
  s.u64(plan.frames.size());
  for (const auto& f : plan.frames) {
    s.str(f.frame_id);
    s.f32(f.g_quality);
    s.f32(f.model_prediction_factor);
    s.f32(f.registration_residual_factor);
    s.f32(f.g_eff);
  }
  return core::sha256_bytes(s.bytes);
}

QualityFrameWeightPlan build_quality_frame_weight_plan(
    const registration::RegistrationSamplingPlan& sampling_plan, const VectorXf& g_quality,
    const std::string& source_quality_config_hash) {
  if (static_cast<size_t>(g_quality.size()) != sampling_plan.frames.size()) {
    throw std::invalid_argument(
        "QUALITY_FRAME_WEIGHT_PLAN: g_quality size does not match sampling plan frame count");
  }
  QualityFrameWeightPlan plan;
  plan.source_identity_hash = sampling_plan.source_identity_hash;
  plan.sampling_plan_hash = sampling_plan.plan_hash;
  plan.source_quality_config_hash = source_quality_config_hash;
  plan.frames.reserve(sampling_plan.frames.size());
  for (size_t i = 0; i < sampling_plan.frames.size(); ++i) {
    const auto& sf = sampling_plan.frames[i];
    QualityFrameWeight w;
    w.frame_id = sf.frame_id;
    w.g_quality = g_quality[static_cast<int>(i)];
    // Read verbatim from the sampling plan --- never recomputed here.
    w.model_prediction_factor = sf.model_prediction_factor;
    w.registration_residual_factor = sf.registration_residual_factor;
    w.g_eff = w.g_quality * w.model_prediction_factor * w.registration_residual_factor;
    plan.frames.push_back(std::move(w));
  }
  validate_weights(plan);
  plan.plan_hash = compute_quality_frame_weight_plan_hash(plan);
  return plan;
}

std::string serialize_quality_frame_weight_plan(const QualityFrameWeightPlan& plan) {
  json j;
  j["schema_version"] = QualityFrameWeightPlan::kSchemaVersion;
  j["source_identity_hash"] = plan.source_identity_hash;
  j["sampling_plan_hash"] = plan.sampling_plan_hash;
  j["source_quality_config_hash"] = plan.source_quality_config_hash;
  j["plan_hash"] = plan.plan_hash;
  j["frames"] = json::array();
  for (const auto& f : plan.frames) {
    j["frames"].push_back({
        {"frame_id", f.frame_id},
        {"g_quality", f.g_quality},
        {"model_prediction_factor", f.model_prediction_factor},
        {"registration_residual_factor", f.registration_residual_factor},
        {"g_eff", f.g_eff},
    });
  }
  return j.dump(2);
}

bool parse_quality_frame_weight_plan(const std::string& text, QualityFrameWeightPlan& out,
                                     std::string& error) {
  try {
    const json j = json::parse(text);
    if (j.value("schema_version", -1) != QualityFrameWeightPlan::kSchemaVersion) {
      error = "unsupported quality_frame_weight_plan schema_version";
      return false;
    }
    QualityFrameWeightPlan plan;
    plan.source_identity_hash = j.at("source_identity_hash").get<std::string>();
    plan.sampling_plan_hash = j.at("sampling_plan_hash").get<std::string>();
    plan.source_quality_config_hash = j.at("source_quality_config_hash").get<std::string>();
    plan.plan_hash = j.at("plan_hash").get<std::string>();
    if (!j.at("frames").is_array())
      throw std::invalid_argument("quality_frame_weight_plan frames must be an array");
    for (const auto& jf : j.at("frames")) {
      QualityFrameWeight f;
      f.frame_id = jf.at("frame_id").get<std::string>();
      f.g_quality = jf.at("g_quality").get<float>();
      f.model_prediction_factor = jf.at("model_prediction_factor").get<float>();
      f.registration_residual_factor = jf.at("registration_residual_factor").get<float>();
      f.g_eff = jf.at("g_eff").get<float>();
      plan.frames.push_back(std::move(f));
    }
    // Fail closed on a tampered / truncated artifact: the stored hash must
    // match a fresh recompute, and g_eff must be the exact product.
    if (plan.plan_hash != compute_quality_frame_weight_plan_hash(plan)) {
      error = "quality_frame_weight_plan hash mismatch";
      return false;
    }
    for (const auto& f : plan.frames) {
      const float expect = f.g_quality * f.model_prediction_factor * f.registration_residual_factor;
      if (std::abs(expect - f.g_eff) > 1e-6f * (1.0f + std::abs(f.g_eff))) {
        error = "quality_frame_weight_plan g_eff is not the product of its factors";
        return false;
      }
    }
    validate_weights(plan);
    error.clear();
    out = std::move(plan);
    return true;
  } catch (const std::exception& e) {
    error = std::string("quality_frame_weight_plan parse error: ") + e.what();
    return false;
  }
}

}  // namespace tile_compile::reconstruction
