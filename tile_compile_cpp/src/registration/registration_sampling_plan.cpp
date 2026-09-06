#include <set>
#include "tile_compile/registration/registration_sampling_plan.hpp"

#include "tile_compile/core/utils.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <limits>

namespace tile_compile::registration {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// enum <-> string
// ---------------------------------------------------------------------------

std::string sampling_warp_convention_to_string(SamplingWarpConvention c) {
  switch (c) {
    case SamplingWarpConvention::canvas_to_source:
      return "canvas_to_source";
  }
  return "canvas_to_source";
}

SamplingWarpConvention string_to_sampling_warp_convention(const std::string& s) {
  // Only one convention exists; anything else is a hard error at parse time.
  if (s == "canvas_to_source") return SamplingWarpConvention::canvas_to_source;
  return SamplingWarpConvention::canvas_to_source;
}

// ---------------------------------------------------------------------------
// affine inversion (plan section 7.2)
// ---------------------------------------------------------------------------

WarpMatrix opencv_to_edge_sampling_map(const WarpMatrix& map) {
  WarpMatrix result=map;
  result(0,2)+=0.5f-0.5f*(map(0,0)+map(0,1));
  result(1,2)+=0.5f-0.5f*(map(1,0)+map(1,1));
  return result;
}

bool invert_affine_2x3(const WarpMatrix& canvas_to_source,
                       float det_min, float det_max,
                       WarpMatrix& out) {
  const float a = canvas_to_source(0, 0);
  const float b = canvas_to_source(0, 1);
  const float c = canvas_to_source(0, 2);
  const float d = canvas_to_source(1, 0);
  const float e = canvas_to_source(1, 1);
  const float f = canvas_to_source(1, 2);

  for (float v : {a, b, c, d, e, f}) {
    if (!std::isfinite(v)) return false;
  }

  const float det = a * e - b * d;
  if (!std::isfinite(det)) return false;
  // Orientation-preserving only: registration warps for stacking never reflect,
  // so a non-positive determinant or one outside the scale-reject bounds
  // (det_min = reject_scale_min^2, det_max = reject_scale_max^2) is rejected.
  if (det < det_min || det > det_max) return false;

  const float inv_det = 1.0f / det;
  const float ai = e * inv_det;
  const float bi = -b * inv_det;
  const float di = -d * inv_det;
  const float ei = a * inv_det;
  const float ci = -(ai * c + bi * f);
  const float fi = -(di * c + ei * f);

  for (float v : {ai, bi, ci, di, ei, fi}) {
    if (!std::isfinite(v)) return false;
  }

  out(0, 0) = ai; out(0, 1) = bi; out(0, 2) = ci;
  out(1, 0) = di; out(1, 1) = ei; out(1, 2) = fi;
  return true;
}

// ---------------------------------------------------------------------------
// local (non-affine) source->canvas inversion (plan section 7.3)
// ---------------------------------------------------------------------------

namespace {

// d(q) in native-canvas render units. Mirrors render_smooth_local_displacement:
//   model_coordinate = (q - offset) * scale ;  displacement_render = d_model / scale
// Returns {0,0} when there is no active local model (caller should use the
// affine inverse in that case).
bool local_displacement_render_units(const FrameSamplingTransform& frame,
                                     float qx, float qy,
                                     float& out_dx, float& out_dy) {
  out_dx = 0.0f;
  out_dy = 0.0f;
  if (frame.has_smooth_local_model && !frame.smooth_local_model.valid) return false;
  if (!frame.has_smooth_local_model) {
    return true;  // no correction here == using the model, not a silent fallback
  }
  const float scale = frame.model_coordinate_scale;
  if (!(scale > 0.0f) || !std::isfinite(scale)) {
    return false;  // a local model without a valid coordinate scale is broken
  }
  const float qmx = (qx - frame.model_offset_x) * scale;
  const float qmy = (qy - frame.model_offset_y) * scale;
  const cv::Point2f dm =
      evaluate_smooth_local_displacement(frame.smooth_local_model, qmx, qmy);
  const float inv_scale = 1.0f / scale;
  out_dx = dm.x * inv_scale;
  out_dy = dm.y * inv_scale;
  return std::isfinite(out_dx) && std::isfinite(out_dy);
}

}  // namespace

bool invert_local_source_to_canvas(const FrameSamplingTransform& frame,
                                   float sx, float sy,
                                   int canvas_width_native,
                                   int canvas_height_native,
                                   const LocalInversionParams& params,
                                   float& out_qx, float& out_qy) {
  if (!frame.source_to_canvas_affine_valid) return false;

  // u = inverse(W_global)(s) == affine source_to_canvas applied to (sx, sy).
  const WarpMatrix& s2c = frame.source_to_canvas;
  const float ux = s2c(0, 0) * sx + s2c(0, 1) * sy + s2c(0, 2);
  const float uy = s2c(1, 0) * sx + s2c(1, 1) * sy + s2c(1, 2);
  if (!std::isfinite(ux) || !std::isfinite(uy)) return false;

  const float margin = params.safety_margin_px;
  auto out_of_bounds = [&](float x, float y) {
    return x < -margin || y < -margin ||
           x > static_cast<float>(canvas_width_native) + margin ||
           y > static_cast<float>(canvas_height_native) + margin;
  };
  if (out_of_bounds(ux, uy)) return false;

  float qx = ux;
  float qy = uy;
  bool converged = false;
  const int max_iter = params.max_iter > 0 ? params.max_iter : 1;
  for (int n = 0; n < max_iter; ++n) {
    float dx = 0.0f;
    float dy = 0.0f;
    if (!local_displacement_render_units(frame, qx, qy, dx, dy)) return false;
    const float qx_new = ux - dx;
    const float qy_new = uy - dy;
    if (!std::isfinite(qx_new) || !std::isfinite(qy_new)) return false;
    if (out_of_bounds(qx_new, qy_new)) return false;
    const float step = std::max(std::abs(qx_new - qx), std::abs(qy_new - qy));
    qx = qx_new;
    qy = qy_new;
    if (step < params.tol_px) {
      converged = true;
      break;
    }
  }
  if (!converged) return false;
  out_qx = qx;
  out_qy = qy;
  return true;
}

// ---------------------------------------------------------------------------
// JSON serialization (plan section 7.4)
// ---------------------------------------------------------------------------

namespace {

json warp_to_json(const WarpMatrix& m) {
  return json::array({m(0, 0), m(0, 1), m(0, 2),
                      m(1, 0), m(1, 1), m(1, 2)});
}

bool warp_from_json(const json& j, WarpMatrix& out, std::string& error) {
  if (!j.is_array() || j.size() != 6) {
    error = "warp matrix must be an array of 6 numbers";
    return false;
  }
  out(0, 0) = j[0].get<float>(); out(0, 1) = j[1].get<float>();
  out(0, 2) = j[2].get<float>(); out(1, 0) = j[3].get<float>();
  out(1, 1) = j[4].get<float>(); out(1, 2) = j[5].get<float>();
  return true;
}

json smooth_model_to_json(const SmoothLocalWarpModel& m) {
  json cx = json::array();
  json cy = json::array();
  for (int i = 0; i < m.coeff_x.size(); ++i) cx.push_back(m.coeff_x[i]);
  for (int i = 0; i < m.coeff_y.size(); ++i) cy.push_back(m.coeff_y[i]);
  return json{{"valid", m.valid},
              {"image_rows", m.image_rows},
              {"image_cols", m.image_cols},
              {"coeff_x", std::move(cx)},
              {"coeff_y", std::move(cy)}};
}

bool smooth_model_from_json(const json& j, SmoothLocalWarpModel& out,
                            std::string& error) {
  if (!j.is_object()) {
    error = "smooth_local_model must be an object";
    return false;
  }
  out.valid = j.value("valid", false);
  out.image_rows = j.value("image_rows", 0);
  out.image_cols = j.value("image_cols", 0);
  const auto& cx = j.at("coeff_x");
  const auto& cy = j.at("coeff_y");
  if (!cx.is_array() || !cy.is_array() ||
      static_cast<int>(cx.size()) != out.coeff_x.size() ||
      static_cast<int>(cy.size()) != out.coeff_y.size()) {
    error = "smooth_local_model coeff arrays have wrong length";
    return false;
  }
  for (int i = 0; i < out.coeff_x.size(); ++i) out.coeff_x[i] = cx[i].get<float>();
  for (int i = 0; i < out.coeff_y.size(); ++i) out.coeff_y[i] = cy[i].get<float>();
  return true;
}

}  // namespace

std::string serialize_to_json_string(const RegistrationSamplingPlan& plan) {
  json frames = json::array();
  for (const auto& f : plan.frames) {
    frames.push_back(json{
        {"frame_id", f.frame_id},
        {"source_index", f.source_index},
        {"valid", f.valid},
        {"canvas_to_source", warp_to_json(f.canvas_to_source)},
        {"source_to_canvas", warp_to_json(f.source_to_canvas)},
        {"source_to_canvas_affine_valid", f.source_to_canvas_affine_valid},
        {"has_smooth_local_model", f.has_smooth_local_model},
        {"smooth_local_model", smooth_model_to_json(f.smooth_local_model)},
        {"model_coordinate_scale", f.model_coordinate_scale},
        {"model_offset_x", f.model_offset_x},
        {"model_offset_y", f.model_offset_y},
        {"registration_residual_factor", f.registration_residual_factor},
        {"residual_applicable", f.residual_applicable},
        {"model_prediction_factor", f.model_prediction_factor},
        {"model_predicted", f.model_predicted},
        {"chain_depth", f.chain_depth},
        {"provenance", f.provenance}});
  }

  json root = {
      {"schema_version", RegistrationSamplingPlan::kSchemaVersion},
      {"warp_convention", sampling_warp_convention_to_string(plan.convention)},
      {"source_width", plan.source_width},
      {"source_height", plan.source_height},
      {"canvas_width_native", plan.canvas_width_native},
      {"canvas_height_native", plan.canvas_height_native},
      {"canvas_offset_x_native", plan.canvas_offset_x_native},
      {"canvas_offset_y_native", plan.canvas_offset_y_native},
      {"internal_scale", plan.internal_scale},
      {"output_scale", plan.output_scale},
      {"color_mode", color_mode_to_string(plan.color_mode)},
      {"bayer_pattern", bayer_pattern_to_string(plan.bayer_pattern)},
      {"cfa_origin_x", plan.cfa_origin_x},
      {"cfa_origin_y", plan.cfa_origin_y},
      {"source_identity_hash", plan.source_identity_hash},
      {"plan_hash", plan.plan_hash},
      {"frames", std::move(frames)}};
  return root.dump(2);
}

static bool parse_plan_fields(const std::string& json_text,
                            RegistrationSamplingPlan& out,
                            std::string& error) {
  json root;
  try {
    root = json::parse(json_text);
  } catch (const std::exception& e) {
    error = std::string("json parse error: ") + e.what();
    return false;
  }
  if (!root.is_object()) {
    error = "root must be an object";
    return false;
  }
  if (root.value("schema_version", -1) != RegistrationSamplingPlan::kSchemaVersion) {
    error = "unsupported schema_version";
    return false;
  }
  const std::string conv = root.value("warp_convention", std::string());
  if (conv != "canvas_to_source") {
    error = "unsupported warp_convention: " + conv;
    return false;
  }
  out = RegistrationSamplingPlan{};
  out.convention = SamplingWarpConvention::canvas_to_source;
  out.source_width = root.value("source_width", 0);
  out.source_height = root.value("source_height", 0);
  out.canvas_width_native = root.value("canvas_width_native", 0);
  out.canvas_height_native = root.value("canvas_height_native", 0);
  out.canvas_offset_x_native = root.value("canvas_offset_x_native", 0);
  out.canvas_offset_y_native = root.value("canvas_offset_y_native", 0);
  out.internal_scale = root.value("internal_scale", 1);
  out.output_scale = root.value("output_scale", 1);

  const std::string cm = root.value("color_mode", std::string("MONO"));
  if (cm == "OSC") out.color_mode = ColorMode::OSC;
  else if (cm == "RGB") out.color_mode = ColorMode::RGB;
  else if(cm=="MONO") out.color_mode = ColorMode::MONO;
  else {error="unsupported color_mode";return false;}

  out.bayer_pattern =
      string_to_bayer_pattern(root.value("bayer_pattern", std::string("UNKNOWN")));
  out.cfa_origin_x = root.value("cfa_origin_x", 0);
  out.cfa_origin_y = root.value("cfa_origin_y", 0);
  out.source_identity_hash = root.at("source_identity_hash").get<std::string>();
  out.plan_hash = root.value("plan_hash", std::string());

  const auto frames_it = root.find("frames");
  if (frames_it == root.end() || !frames_it->is_array()) {
    error = "frames must be an array";
    return false;
  }
  out.frames.reserve(frames_it->size());
  for (const auto& jf : *frames_it) {
    FrameSamplingTransform f;
    f.frame_id = jf.value("frame_id", std::string());
    f.source_index = jf.value("source_index", static_cast<std::size_t>(0));
    f.valid = jf.value("valid", false);
    if (!warp_from_json(jf.at("canvas_to_source"), f.canvas_to_source, error))
      return false;
    if (!warp_from_json(jf.at("source_to_canvas"), f.source_to_canvas, error))
      return false;
    f.source_to_canvas_affine_valid =
        jf.value("source_to_canvas_affine_valid", false);
    f.has_smooth_local_model = jf.value("has_smooth_local_model", false);
    if (!smooth_model_from_json(jf.at("smooth_local_model"), f.smooth_local_model,
                                error))
      return false;
    f.model_coordinate_scale = jf.value("model_coordinate_scale", 1.0f);
    f.model_offset_x = jf.value("model_offset_x", 0.0f);
    f.model_offset_y = jf.value("model_offset_y", 0.0f);
    f.registration_residual_factor =
        jf.value("registration_residual_factor", 1.0f);
    f.residual_applicable = jf.value("residual_applicable", true);
    f.model_prediction_factor = jf.value("model_prediction_factor", 1.0f);
    f.model_predicted = jf.value("model_predicted", false);
    f.chain_depth = jf.value("chain_depth", 0);
    f.provenance = jf.value("provenance", std::string());
    out.frames.push_back(std::move(f));
  }
  return true;
}

bool parse_from_json_string(const std::string& text,RegistrationSamplingPlan& out,std::string& error) {
  try {
    const auto root=json::parse(text);
    const auto shape=json::parse(serialize_to_json_string(RegistrationSamplingPlan{}));
    for(auto it=shape.begin();it!=shape.end();++it)
      if(!root.contains(it.key())) {error="missing field: "+it.key();return false;}
    RegistrationSamplingPlan frame_shape;
    frame_shape.frames.emplace_back();
    const auto required_frame=json::parse(serialize_to_json_string(frame_shape)).at("frames").at(0);
    for(const auto& f:root.at("frames")) for(auto it=required_frame.begin();it!=required_frame.end();++it)
      if(!f.contains(it.key())) {error="missing frame field: "+it.key();return false;}
    RegistrationSamplingPlan parsed;
    if(!parse_plan_fields(text,parsed,error)) return false;
    if(parsed.source_width<=0 || parsed.source_height<=0 || parsed.canvas_width_native<=0 || parsed.canvas_height_native<=0 ||
       (parsed.internal_scale!=1 && parsed.internal_scale!=2) || parsed.output_scale<1 || parsed.output_scale>parsed.internal_scale ||
       parsed.color_mode==ColorMode::RGB || (parsed.color_mode==ColorMode::OSC && parsed.bayer_pattern==BayerPattern::UNKNOWN)) {
      error="invalid sampling geometry or color contract";return false;
    }
    std::set<size_t> ids;std::set<std::string> names;
    for(const auto& f:parsed.frames) {
      if(f.frame_id.empty() || !ids.insert(f.source_index).second || !names.insert(f.frame_id).second ||
         !f.canvas_to_source.allFinite() || !f.source_to_canvas.allFinite() ||
         !f.smooth_local_model.coeff_x.allFinite() || !f.smooth_local_model.coeff_y.allFinite() ||
         !std::isfinite(f.model_offset_x) || !std::isfinite(f.model_offset_y) ||
         !std::isfinite(f.model_coordinate_scale) ||
         !(f.registration_residual_factor>=0 && f.registration_residual_factor<=1) ||
         !(f.model_prediction_factor>=0 && f.model_prediction_factor<=1)) {
        error="invalid or duplicate frame contract";return false;
      }
      if(f.valid) {
        WarpMatrix inverse;
        if(!f.source_to_canvas_affine_valid ||
           !invert_affine_2x3(f.canvas_to_source,1e-12f,1e12f,inverse) ||
           !inverse.isApprox(f.source_to_canvas,1e-5f) ||
           (f.has_smooth_local_model && (!f.smooth_local_model.valid || f.model_coordinate_scale<=0 ||
             f.smooth_local_model.image_rows<=0 || f.smooth_local_model.image_cols<=0))) {
          error="invalid checked inverse or local model";return false;
        }
      }
    }
    if(parsed.plan_hash!=compute_plan_hash(parsed)) {error="sampling plan hash mismatch";return false;}
    out=std::move(parsed);error.clear();return true;
  } catch(const std::exception& e) {error=e.what();return false;}
}

// ---------------------------------------------------------------------------
// canonical hash (plan section 7.4)
// ---------------------------------------------------------------------------

namespace {

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
    // bit-exact IEEE-754; normalize the two NaN payloads so a NaN never makes
    // the hash unstable.
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
  void warp(const WarpMatrix& m) {
    f32(m(0, 0)); f32(m(0, 1)); f32(m(0, 2));
    f32(m(1, 0)); f32(m(1, 1)); f32(m(1, 2));
  }
};

}  // namespace

std::string compute_plan_hash(const RegistrationSamplingPlan& plan) {
  ByteSink s;
  // A fixed tag + schema version so an encoding change is itself a hash change.
  s.str("registration_sampling_plan:v2:edge-centers");
  s.str(plan.source_identity_hash);
  s.i32(RegistrationSamplingPlan::kSchemaVersion);
  s.i32(static_cast<int32_t>(plan.convention));

  s.i32(plan.source_width);
  s.i32(plan.source_height);
  s.i32(plan.canvas_width_native);
  s.i32(plan.canvas_height_native);
  s.i32(plan.canvas_offset_x_native);
  s.i32(plan.canvas_offset_y_native);
  s.i32(static_cast<int32_t>(plan.color_mode));
  s.i32(static_cast<int32_t>(plan.bayer_pattern));
  s.i32(plan.cfa_origin_x);
  s.i32(plan.cfa_origin_y);

  // NOTE: internal_scale / output_scale are deliberately excluded (drizzle
  // geometry hash domain, plan section 18.3). chain_depth, model_predicted and
  // provenance are excluded as diagnostic-only (their weight effect lives in
  // model_prediction_factor).
  s.u64(plan.frames.size());
  for (const auto& f : plan.frames) {
    s.str(f.frame_id);
    s.u64(f.source_index);
    s.b(f.valid);
    s.warp(f.canvas_to_source);
    s.warp(f.source_to_canvas);
    s.b(f.source_to_canvas_affine_valid);
    s.b(f.has_smooth_local_model);
    s.b(f.smooth_local_model.valid);
    s.i32(f.smooth_local_model.image_rows);
    s.i32(f.smooth_local_model.image_cols);
    for (int i = 0; i < f.smooth_local_model.coeff_x.size(); ++i)
      s.f32(f.smooth_local_model.coeff_x[i]);
    for (int i = 0; i < f.smooth_local_model.coeff_y.size(); ++i)
      s.f32(f.smooth_local_model.coeff_y[i]);
    s.f32(f.model_coordinate_scale);
    s.f32(f.model_offset_x);
    s.f32(f.model_offset_y);
    s.f32(f.registration_residual_factor);
    s.b(f.residual_applicable);
    s.f32(f.model_prediction_factor);
  }
  return core::sha256_bytes(s.bytes);
}

}  // namespace tile_compile::registration
