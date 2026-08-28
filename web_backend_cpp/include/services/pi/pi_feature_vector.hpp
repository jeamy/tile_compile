#pragma once

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

inline constexpr const char* kFeatureVectorSchemaVersion = "pi.feature-vector.v1";

// Projects scan_metrics/scan_result into the flat, versioned feature-vector schema from
// docs/PI/pi_local_learning_plan_de.md, Abschnitt 4.1. Grounded in the real
// scan_metrics.aggregate shape (tile_compile_cpp/apps/cli_main.cpp: agg_stats() ->
// {min,max,mean,median,p10,p90,count} per metric key: background, noise, gradient_energy,
// sky_gradient, fwhm, roundness, star_count) — not invented field names.
nlohmann::json build_scan_feature_vector(const nlohmann::json& scan_metrics,
                                         const nlohmann::json& scan_result);

// Distance between two feature vectors of the SAME domain/schema: sum of squared differences over
// numeric keys present in both (missing keys on either side are skipped, not penalized — schema is
// additive per Abschnitt 4.1, older/newer records may not share every field) plus a fixed penalty
// per mismatching categorical key. Deliberately simple (no learned scaling/weights) — this backs
// the nearest-neighbor model (Abschnitt 4.2) explicitly named as the low-data fallback, not a
// trained metric.
double feature_vector_distance(const nlohmann::json& a, const nlohmann::json& b);

} // namespace tile_compile::pi
