#include "services/pi/pi_feature_vector.hpp"

#include <cmath>
#include <limits>

namespace tile_compile::pi {
namespace {

void add_aggregate_stats(nlohmann::json& numeric,
                         const nlohmann::json& aggregate,
                         const char* metric_key,
                         const char* feature_prefix) {
    if (!aggregate.contains(metric_key) || !aggregate[metric_key].is_object()) return;
    const nlohmann::json& stats = aggregate[metric_key];
    for (const char* stat_key : {"median", "p10", "p90", "mean"}) {
        if (stats.contains(stat_key) && stats[stat_key].is_number()) {
            numeric[std::string(feature_prefix) + "_" + stat_key] = stats[stat_key];
        }
    }
}

} // namespace

nlohmann::json build_scan_feature_vector(const nlohmann::json& scan_metrics,
                                         const nlohmann::json& scan_result) {
    nlohmann::json numeric = nlohmann::json::object();
    nlohmann::json categorical = nlohmann::json::object();

    if (scan_metrics.is_object() && scan_metrics.contains("aggregate") && scan_metrics["aggregate"].is_object()) {
        const nlohmann::json& aggregate = scan_metrics["aggregate"];
        add_aggregate_stats(numeric, aggregate, "sky_gradient", "sky_gradient");
        add_aggregate_stats(numeric, aggregate, "fwhm", "fwhm");
        add_aggregate_stats(numeric, aggregate, "roundness", "roundness");
        add_aggregate_stats(numeric, aggregate, "noise", "noise");
        add_aggregate_stats(numeric, aggregate, "background", "background");
        add_aggregate_stats(numeric, aggregate, "gradient_energy", "gradient_energy");
        if (aggregate.contains("star_count") && aggregate["star_count"].is_object()) {
            const nlohmann::json& sc = aggregate["star_count"];
            if (sc.contains("median") && sc["median"].is_number()) numeric["star_count_median"] = sc["median"];
            if (sc.contains("mean") && sc["mean"].is_number()) numeric["star_count_mean"] = sc["mean"];
        }
        if (aggregate.contains("frames_ok") && aggregate["frames_ok"].is_number()) {
            numeric["frames_ok"] = aggregate["frames_ok"];
        }
    }

    if (scan_result.is_object()) {
        if (scan_result.contains("frames_detected") && scan_result["frames_detected"].is_number()) {
            numeric["frame_count"] = scan_result["frames_detected"];
        } else if (scan_result.contains("frames_total") && scan_result["frames_total"].is_number()) {
            numeric["frame_count"] = scan_result["frames_total"];
        }
        if (scan_result.contains("color_mode") && scan_result["color_mode"].is_string()) {
            categorical["color_mode"] = scan_result["color_mode"];
        }
        if (scan_result.contains("bayer_pattern") && scan_result["bayer_pattern"].is_string()) {
            categorical["bayer_pattern"] = scan_result["bayer_pattern"];
        }
    }

    return {
        {"schema_version", kFeatureVectorSchemaVersion},
        {"domain", "scan"},
        {"numeric", numeric},
        {"categorical", categorical}
    };
}

double feature_vector_distance(const nlohmann::json& a, const nlohmann::json& b) {
    double sum_sq = 0.0;
    const nlohmann::json a_num = a.value("numeric", nlohmann::json::object());
    const nlohmann::json b_num = b.value("numeric", nlohmann::json::object());
    int compared = 0;
    for (auto it = a_num.begin(); it != a_num.end(); ++it) {
        if (!it.value().is_number()) continue;
        if (!b_num.contains(it.key()) || !b_num[it.key()].is_number()) continue;
        const double diff = it.value().get<double>() - b_num[it.key()].get<double>();
        sum_sq += diff * diff;
        ++compared;
    }
    // Comparing on zero shared numeric features is not "distance 0" (identical) — treat it as
    // maximally dissimilar so an empty/mismatched feature set never wins a nearest-neighbor vote
    // by accident.
    if (compared == 0) return std::numeric_limits<double>::infinity();

    constexpr double kCategoricalMismatchPenalty = 4.0;
    const nlohmann::json a_cat = a.value("categorical", nlohmann::json::object());
    const nlohmann::json b_cat = b.value("categorical", nlohmann::json::object());
    for (auto it = a_cat.begin(); it != a_cat.end(); ++it) {
        if (!b_cat.contains(it.key())) continue;
        if (it.value() != b_cat[it.key()]) sum_sq += kCategoricalMismatchPenalty * kCategoricalMismatchPenalty;
    }

    return std::sqrt(sum_sq);
}

} // namespace tile_compile::pi
