#include "services/pi/pi_parameter_catalog.hpp"

#include <sstream>

namespace tile_compile::pi {
namespace {

const nlohmann::json* get_dotted_ptr(const nlohmann::json& root, const std::string& dotted_path) {
    const nlohmann::json* cur = &root;
    std::istringstream iss(dotted_path);
    std::string part;
    while (std::getline(iss, part, '.')) {
        if (!cur->is_object() || !cur->contains(part)) return nullptr;
        cur = &(*cur)[part];
    }
    return cur;
}

} // namespace

nlohmann::json curated_parameter_metadata(const std::string& path) {
    if (path == "pcc.max_residual_rms") {
        return {
            {"cpp_default", 0.35},
            {"unit", "robust PCC residual RMS"},
            {"phase", "PCC"},
            {"semantic", "Rejects unstable or noisy PCC fits."},
            {"diagnostic_only", false},
            {"requires_evidence", {"pcc.status", "pcc.residual_rms", "pcc.stars_used"}},
            {"hard_rules", {
                "Do not recommend below observed successful pcc.residual_rms unless PCC failed.",
                "Do not claim a schema maximum if schema_max is null.",
                "Do not claim a schema recommended value if recommended_value is null."
            }}
        };
    }
    if (path == "pcc.k_max") {
        return {
            {"cpp_default", 3.20},
            {"safe_min", 1.0},
            {"unit", "linear PCC apply strength cap"},
            {"phase", "PCC"},
            {"semantic", "Caps the linear PCC apply strength; it is not an atmospheric extinction coefficient."},
            {"diagnostic_only", false},
            {"requires_evidence", {"pcc.status", "pcc.matrix_diag", "pcc.condition_number"}},
            {"hard_rules", {
                "Do not recommend values below 1.0 because the PCC apply path clamps the raw cap to at least 1.0.",
                "Do not describe this parameter as a physical atmospheric extinction coefficient."
            }}
        };
    }
    if (path == "aqmh.pyramid.base_window_px") {
        return {
            {"cpp_default", 4},
            {"unit", "pixels"},
            {"phase", "AQMH_MAPS"},
            {"semantic", "Base local quality window for the smallest AQMH scale."},
            {"diagnostic_only", false},
            {"requires_evidence", {"aqmh.map_statistics", "image_quality.local_quality_noise"}},
            {"hard_rules", {
                "Do not claim a 16-256 schema range unless schema_min/schema_max explicitly provide it.",
                "Do not infer an invalid value merely because it is smaller than stellar FWHM."
            }}
        };
    }
    if (path == "aqmh.diagnostics.r_morph_canvas_px") {
        return {
            {"cpp_default", 6},
            {"unit", "canvas pixels"},
            {"phase", "AQMH_DIAGNOSTICS"},
            {"semantic", "Morphological radius for AQMH diagnostic masks and region extraction."},
            {"diagnostic_only", true},
            {"requires_evidence", {"aqmh.diagnostics.regions"}},
            {"hard_rules", {
                "Do not present diagnostic-only changes as direct reconstruction-quality improvements."
            }}
        };
    }
    if (path == "validation.max_background_rms_increase_percent") {
        return {
            {"cpp_default", 0.0},
            {"disabled_value", 0.0},
            {"unit", "percent"},
            {"phase", "VALIDATION"},
            {"semantic", "Optional background RMS degradation guard; 0.0 disables this check."},
            {"diagnostic_only", false},
            {"requires_evidence", {"validation.background_rms_increase_percent", "validation.background_rms_ok"}},
            {"hard_rules", {
                "Do not claim 0.0 means any RMS increase disables processing; 0.0 means no check."
            }}
        };
    }
    if (path == "registration.enable_local_background_subtraction") {
        return {
            {"cpp_default", false},
            {"unit", "boolean"},
            {"phase", "REGISTRATION"},
            {"semantic", "Local background subtraction before star detection; useful when registration evidence shows gradients hurt star detection."},
            {"diagnostic_only", false},
            {"requires_evidence", {"registration.failure_reasons", "scan_metrics.sky_gradient"}},
            {"hard_rules", {
                "Do not claim the schema default is true.",
                "Without registration diagnostics, mark recommendations review_required with low confidence."
            }}
        };
    }
    return nlohmann::json::object();
}

nlohmann::json build_parameter_catalog(const SchemaPathMap& schema_paths,
                                       const nlohmann::json& base_config) {
    nlohmann::json catalog = nlohmann::json::object();
    for (const auto& [path, schema_node] : schema_paths) {
        if (!schema_node.is_object()) continue;
        const std::string schema_type = schema_node.contains("type") && schema_node["type"].is_string()
            ? schema_node["type"].get<std::string>()
            : std::string();
        if (schema_type == "object" || schema_type == "array") continue;

        nlohmann::json meta = {
            {"path", path},
            {"recommended_value", nullptr},
            {"diagnostic_only", false},
            {"metadata_scope", "schema_leaf"},
            {"semantic", schema_node.contains("description") && schema_node["description"].is_string()
                ? schema_node["description"].get<std::string>()
                : std::string("No curated semantic metadata is available for this parameter.")},
            {"requires_evidence", nlohmann::json::array()},
            {"hard_rules", {
                "Do not claim schema defaults, maxima, recommended values or units unless this catalog entry explicitly provides them.",
                "If current_value is absent, say the current value is not present in the base config.",
                "Recommendations without parameter-specific evidence must be review_required with low confidence."
            }}
        };
        nlohmann::json curated = curated_parameter_metadata(path);
        if (!curated.empty()) {
            for (auto it = curated.begin(); it != curated.end(); ++it) {
                meta[it.key()] = it.value();
            }
            meta["metadata_scope"] = "curated";
        }
        meta["path"] = path;
        if (schema_node.contains("type")) meta["type"] = schema_node["type"];
        if (schema_node.contains("enum")) meta["schema_enum"] = schema_node["enum"];
        if (schema_node.contains("default")) meta["schema_default"] = schema_node["default"];
        if (schema_node.contains("minimum")) meta["schema_min"] = schema_node["minimum"];
        if (schema_node.contains("exclusiveMinimum")) meta["schema_exclusive_min"] = schema_node["exclusiveMinimum"];
        if (schema_node.contains("maximum")) meta["schema_max"] = schema_node["maximum"];
        else meta["schema_max"] = nullptr;
        if (schema_node.contains("description") && schema_node["description"].is_string()) {
            meta["description"] = schema_node["description"];
        }
        if (const nlohmann::json* cur = get_dotted_ptr(base_config, path)) {
            meta["current_value"] = *cur;
        } else {
            meta["current_value_present"] = false;
        }
        catalog[path] = std::move(meta);
    }
    return catalog;
}

} // namespace tile_compile::pi
