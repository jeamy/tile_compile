#include "services/preprocessing_service.hpp"

#include <algorithm>
#include <cmath>

namespace tile_compile::preprocessing_service {

namespace {

nlohmann::json compact_frames(const nlohmann::json& frames, size_t limit = 256) {
    nlohmann::json out = nlohmann::json::array();
    if (!frames.is_array()) return out;
    const size_t n = std::min(frames.size(), limit);
    for (size_t i = 0; i < n; ++i) out.push_back(frames[i]);
    return out;
}

} // namespace

std::vector<std::string> phase_order() {
    return {
        "INPUT_SCAN",
        "CALIBRATION",
        "CFA_CHANNEL_PREP",
        "REFERENCE_SELECTION",
        "REGISTRATION",
        "QUALITY_ANALYSIS",
        "FRAME_FILTERING",
        "STACKING",
        "ASTROMETRY",
        "BGE",
        "PCC",
        "HYPERMETRIC_STRETCH",
        "REPORT",
    };
}

nlohmann::json default_config() {
    return {
        {"mode", "linear_prestack"},
        {"lights_dir", ""},
        {"bias_dir", ""},
        {"darks_dir", ""},
        {"flats_dir", ""},
        {"darkflats_dir", ""},
        {"input_mode", "auto"},
        {"raw_formats", "tile_compile"},
        {"bayer_pattern", "auto"},
        {"cfa_mode", "tile_compile"},
        {"mono_mode", "auto"},
        {"registration_reference", "best_quality"},
        {"calibration", {
            {"use_bias", false},
            {"use_dark", false},
            {"use_flat", false},
            {"bias_use_master", false},
            {"dark_use_master", false},
            {"flat_use_master", false},
            {"darkflat_use_master", false},
            {"dark_auto_select", true},
            {"dark_match_use_temp", false},
            {"dark_match_exposure_tolerance_percent", 8.0},
            {"dark_match_temp_tolerance_c", 3.0},
            {"bias_master", ""},
            {"dark_master", ""},
            {"flat_master", ""},
            {"darkflat_master", ""},
            {"pattern", "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz"},
        }},
        {"rejection", {
            {"method", "sigma"},
            {"low", 3.0},
            {"high", 3.0},
            {"max_iters", 3},
            {"min_fraction", 0.4},
        }},
        {"quality_filter", {
            {"mode", "auto"},
            {"min_stars", 30},
            {"max_fwhm_sigma", 2.0},
            {"max_eccentricity", 0.65},
            {"min_correlation", 0.75},
        }},
        {"stacking", {
            {"normalization", "addscale"},
            {"weighting", "quality"},
            {"cosmetic_correction", false},
            {"cosmetic_correction_sigma", 5.0},
            {"per_frame_cosmetic_correction", false},
            {"per_frame_cosmetic_correction_sigma", 5.0},
        }},
        {"postprocess", {
            {"astrometry", true},
            {"bge", true},
            {"pcc", true},
            {"hypermetric_stretch", true},
        }},
        {"tile", {
            {"size_factor", 32},
            {"min_size", 64},
            {"max_divisor", 6},
            {"overlap_fraction", 0.25},
            {"star_min_count", 10},
            {"star_soft_count", 10},
        }},
        {"hypermetric_stretch", {
            {"require_successful_pcc", true},
            {"mode", "ready_to_use"},
            {"sensor_profile", "rec709"},
            {"fallback_profile", "rec709"},
            {"adaptive_anchor", true},
            {"target_bg", 0.15},
            {"protect_b", 6.0},
            {"convergence_power", 3.5},
            {"log_d_mode", "auto"},
            {"fixed_log_d", 2.0},
            {"color_strategy", "fixed"},
            {"fixed_color_strategy", 0.0},
            {"color_grip", 1.0},
            {"shadow_convergence", 0.0},
            {"linear_expansion", 0.0},
            {"write_channels", false},
            {"output_rgb", "stacked_rgb_hms.fits"},
        }},
        {"report", {
            {"detailed", true},
            {"formats", nlohmann::json::array({"json", "markdown", "html"})},
        }},
        {"runtime_limits", {
            {"parallel_workers", 4},
            {"memory_budget", 512},
            {"acceleration_backend", "auto"},
        }},
    };
}

nlohmann::json parameter_groups() {
    return nlohmann::json::array({
        {{"id", "input"}, {"label", "Input"}, {"paths", {"lights_dir", "bias_dir", "darks_dir", "flats_dir", "darkflats_dir", "input_mode", "raw_formats"}}},
        {{"id", "calibration"}, {"label", "Calibration"}, {"paths", {"calibration.use_bias", "calibration.use_dark", "calibration.use_flat", "bias_dir", "darks_dir", "flats_dir", "darkflats_dir", "calibration.bias_use_master", "calibration.dark_use_master", "calibration.flat_use_master", "calibration.darkflat_use_master", "calibration.dark_auto_select", "calibration.dark_match_use_temp", "calibration.dark_match_exposure_tolerance_percent", "calibration.dark_match_temp_tolerance_c", "calibration.bias_master", "calibration.dark_master", "calibration.flat_master", "calibration.darkflat_master", "calibration.pattern"}}},
        {{"id", "cfa_mono"}, {"label", "CFA / Mono"}, {"paths", {"input_mode", "bayer_pattern", "cfa_mode", "mono_mode"}}},
        {{"id", "registration"}, {"label", "Registration"}, {"paths", {"registration_reference"}}},
        {{"id", "quality_filter"}, {"label", "Quality Filter"}, {"paths", {"quality_filter.mode", "quality_filter.min_stars", "quality_filter.max_fwhm_sigma", "quality_filter.max_eccentricity", "quality_filter.min_correlation", "quality_filter.manual_overrides"}}},
        {{"id", "stacking"}, {"label", "Stacking"}, {"paths", {"rejection.method", "rejection.low", "rejection.high", "rejection.max_iters", "rejection.min_fraction", "stacking.normalization", "stacking.weighting", "stacking.cosmetic_correction", "stacking.cosmetic_correction_sigma", "stacking.per_frame_cosmetic_correction", "stacking.per_frame_cosmetic_correction_sigma"}}},
        {{"id", "postprocess"}, {"label", "Postprocess"}, {"paths", {"postprocess.astrometry", "postprocess.bge", "postprocess.pcc", "postprocess.hypermetric_stretch"}}},
        {{"id", "bge_tile"}, {"label", "BGE / Tile"}, {"paths", {"bge.method", "bge.autobge.num_sample_points", "bge.autobge.poly_degree", "bge.autobge.rbf_smooth", "bge.autobge.downsample_scale", "bge.autobge.patch_size", "bge.autobge.patch_estimator", "bge.autobge.stretch_mode", "bge.autobge.stretch_target_median", "bge.autobge.border_margin", "bge.autobge.bright_exclusion_fraction", "bge.autobge.gradient_descent_max_iters", "bge.autobge.random_seed", "bge.autobge.normalize_between_stages", "bge.autobge.apply_guards", "bge.autobge.mono_mode", "bge.sample_quantile", "bge.sample_estimator", "bge.min_sample_bg_value", "bge.structure_thresh_percentile", "bge.min_tiles_per_cell", "bge.min_valid_sample_fraction_for_apply", "bge.min_valid_samples_for_apply", "bge.mask.star_dilate_px", "bge.mask.sat_dilate_px", "bge.grid.N_g", "bge.grid.G_min_px", "bge.grid.G_max_fraction", "bge.grid.insufficient_cell_strategy", "bge.fit.method", "bge.fit.robust_loss", "bge.fit.huber_delta", "bge.fit.irls_max_iterations", "bge.fit.irls_tolerance", "bge.fit.polynomial_order", "bge.fit.rbf_phi", "bge.fit.rbf_mu_factor", "bge.fit.rbf_lambda", "bge.fit.rbf_epsilon", "bge.autotune.enabled", "bge.autotune.max_evals", "bge.autotune.holdout_fraction", "bge.autotune.alpha_flatness", "bge.autotune.beta_roughness", "bge.autotune.strategy", "bge.tile_weight_lambda_structure", "tile.size_factor", "tile.min_size", "tile.max_divisor", "tile.overlap_fraction", "tile.star_min_count", "tile.star_soft_count"}}},
        {{"id", "hypermetric_stretch"}, {"label", "HyperMetric Stretch"}, {"paths", {"hypermetric_stretch.require_successful_pcc", "hypermetric_stretch.mode", "hypermetric_stretch.sensor_profile", "hypermetric_stretch.fallback_profile", "hypermetric_stretch.adaptive_anchor", "hypermetric_stretch.target_bg", "hypermetric_stretch.protect_b", "hypermetric_stretch.convergence_power", "hypermetric_stretch.log_d_mode", "hypermetric_stretch.fixed_log_d", "hypermetric_stretch.color_strategy", "hypermetric_stretch.fixed_color_strategy", "hypermetric_stretch.color_grip", "hypermetric_stretch.shadow_convergence", "hypermetric_stretch.linear_expansion", "hypermetric_stretch.write_channels", "hypermetric_stretch.output_rgb"}}},
        {{"id", "report"}, {"label", "Report"}, {"paths", {"report.detailed", "report.formats"}}},
        {{"id", "runtime_limits"}, {"label", "Runtime Limits"}, {"paths", {"runtime_limits.parallel_workers", "runtime_limits.memory_budget", "runtime_limits.acceleration_backend"}}},
    });
}

nlohmann::json normalize_scan_result(const nlohmann::json& raw,
                                     const std::string& input_path) {
    const nlohmann::json frames = raw.contains("frames") && raw["frames"].is_array()
        ? raw["frames"]
        : nlohmann::json::array();
    const std::string color_mode = raw.value("color_mode", "UNKNOWN");
    std::string input_mode = "unknown";
    if (color_mode == "OSC" || color_mode == "RGB") input_mode = "cfa_osc";
    else if (color_mode == "MONO") input_mode = "mono";

    nlohmann::json consistency_warnings = nlohmann::json::array();
    bool requires_confirmation = raw.value("requires_user_confirmation", false);

    if (frames.size() > 1) {
        int ref_w = -1, ref_h = -1;
        double ref_exp = -1.0;
        int ref_gain = -1;
        std::string ref_channel_mode;

        for (const auto& f : frames) {
            const int fw = f.value("image_width", f.value("width", -1));
            const int fh = f.value("image_height", f.value("height", -1));
            const double fexp = f.value("exposure", f.value("exposure_s", -1.0));
            const int fgain = f.value("gain", -1);
            const std::string fcm = f.value("channel_mode", f.value("color_mode", std::string()));

            if (ref_w < 0 && fw > 0) { ref_w = fw; ref_h = fh; }
            else if (fw > 0 && (fw != ref_w || fh != ref_h)) {
                consistency_warnings.push_back({
                    {"code", "INCONSISTENT_DIMENSIONS"},
                    {"message", "Frames have different image dimensions"},
                });
                requires_confirmation = true;
                break;
            }

            if (ref_exp < 0.0 && fexp >= 0.0) ref_exp = fexp;
            else if (fexp >= 0.0 && std::abs(fexp - ref_exp) > 0.5) {
                consistency_warnings.push_back({
                    {"code", "INCONSISTENT_EXPOSURE"},
                    {"message", "Frames have different exposure times"},
                });
                requires_confirmation = true;
            }

            if (ref_gain < 0 && fgain >= 0) ref_gain = fgain;
            else if (fgain >= 0 && fgain != ref_gain) {
                consistency_warnings.push_back({
                    {"code", "INCONSISTENT_GAIN"},
                    {"message", "Frames have different gain values"},
                });
                requires_confirmation = true;
            }

            if (ref_channel_mode.empty() && !fcm.empty()) ref_channel_mode = fcm;
            else if (!fcm.empty() && fcm != ref_channel_mode) {
                consistency_warnings.push_back({
                    {"code", "INCONSISTENT_CHANNEL_MODE"},
                    {"message", "Frames have mixed channel modes (CFA/Mono)"},
                });
                requires_confirmation = true;
            }
        }
    }

    nlohmann::json warnings = raw.contains("warnings") ? raw["warnings"] : nlohmann::json::array();
    for (const auto& cw : consistency_warnings) {
        bool already_present = false;
        const std::string cw_code = cw.value("code", std::string());
        for (const auto& w : warnings) {
            if (w.value("code", std::string()) == cw_code) { already_present = true; break; }
        }
        if (!already_present) warnings.push_back(cw);
    }

    nlohmann::json normalized = {
        {"ok", raw.value("ok", true)},
        {"input_path", raw.value("input_path", input_path)},
        {"frames_detected", raw.value("frames_detected", static_cast<int>(frames.size()))},
        {"image_width", raw.value("image_width", 0)},
        {"image_height", raw.value("image_height", 0)},
        {"color_mode", color_mode},
        {"input_mode", input_mode},
        {"bayer_pattern", raw.contains("bayer_pattern") ? raw["bayer_pattern"] : nlohmann::json(nullptr)},
        {"requires_user_confirmation", requires_confirmation},
        {"warnings", warnings},
        {"errors", raw.contains("errors") ? raw["errors"] : nlohmann::json::array()},
        {"frames", compact_frames(frames)},
        {"frames_total", frames.size()},
        {"frames_truncated", frames.size() > 256},
    };
    if (raw.contains("color_mode_candidates")) {
        normalized["color_mode_candidates"] = raw["color_mode_candidates"];
    }
    return normalized;
}

fs::path run_dir_for_job(const fs::path& runs_dir, const std::string& job_id) {
    return runs_dir / ("preprocessing_" + job_id);
}

nlohmann::json read_status_from_job(const nlohmann::json& job_json) {
    const std::string state = job_json.value("state", "unknown");
    const nlohmann::json data = (job_json.contains("data") && job_json["data"].is_object())
        ? job_json["data"]
        : nlohmann::json::object();

    const std::string current_phase_str = (data.contains("current_phase") && data["current_phase"].is_string())
        ? data["current_phase"].get<std::string>()
        : std::string();

    const nlohmann::json eff_cfg = (data.contains("effective_config") && data["effective_config"].is_object())
        ? data["effective_config"]
        : nlohmann::json::object();
    const nlohmann::json pp = (eff_cfg.contains("postprocess") && eff_cfg["postprocess"].is_object())
        ? eff_cfg["postprocess"]
        : nlohmann::json::object();
    const bool astrometry = pp.value("astrometry", true);
    const bool bge        = pp.value("bge", true);
    const bool pcc        = pp.value("pcc", true);
    const bool stretch    = pp.value("hypermetric_stretch", true);

    const auto is_skipped = [&](const std::string& phase) {
        if (phase == "ASTROMETRY"        && !astrometry) return true;
        if (phase == "BGE"               && !bge)        return true;
        if (phase == "PCC"               && !pcc)        return true;
        if (phase == "HYPERMETRIC_STRETCH" && !stretch)  return true;
        return false;
    };

    const auto ordered = phase_order();
    bool found_current = false;

    nlohmann::json phases = nlohmann::json::array();
    for (const auto& phase : ordered) {
        std::string status = "pending";
        double pct = 0.0;

        if (is_skipped(phase)) {
            status = "skipped";
        } else if (state == "ok") {
            status = "ok";
            pct = 1.0;
        } else if (state == "running") {
            if (current_phase_str.empty() || current_phase_str == phase) {
                status = "running";
                pct = job_json.value("progress", 0.0);
                found_current = true;
            } else if (!found_current) {
                status = "ok";
                pct = 1.0;
            }
        } else if (state == "error") {
            if (!current_phase_str.empty() && current_phase_str == phase) status = "failed";
            else if (current_phase_str.empty() || !found_current) {
                if (!found_current) status = "ok"; else status = "pending";
            }
            if (phase == current_phase_str) found_current = true;
        } else if (state == "cancelled") {
            if (!current_phase_str.empty() && current_phase_str == phase) status = "aborted";
            else if (!found_current) status = "ok";
            if (phase == current_phase_str) found_current = true;
        }

        phases.push_back({{"phase", phase}, {"status", status}, {"pct", pct}});
    }

    nlohmann::json cur_phase_json = nullptr;
    if (state == "running") {
        cur_phase_json = current_phase_str.empty() ? nlohmann::json(ordered.front()) : nlohmann::json(current_phase_str);
    } else if ((state == "error" || state == "cancelled") && !current_phase_str.empty()) {
        cur_phase_json = current_phase_str;
    }

    return {
        {"status", state},
        {"current_phase", cur_phase_json},
        {"progress", job_json.value("progress", 0.0)},
        {"phases", phases},
        {"job", job_json},
    };
}

} // namespace tile_compile::preprocessing_service
