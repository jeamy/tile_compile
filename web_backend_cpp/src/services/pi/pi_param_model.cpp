#include "services/pi/pi_param_model.hpp"

#include "app_state.hpp"
#include "services/pi/pi_feature_vector.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace tile_compile::pi {
namespace {

std::optional<nlohmann::json> read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    nlohmann::json parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

nlohmann::json read_jsonl_file(const fs::path& path) {
    nlohmann::json items = nlohmann::json::array();
    std::ifstream in(path);
    if (!in) return items;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        nlohmann::json parsed = nlohmann::json::parse(line, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) items.push_back(std::move(parsed));
    }
    return items;
}

// Highest v<N> subdirectory of target_dir that contains both metadata.json and
// reference_points.jsonl, or empty if none qualifies. Deliberately just "highest number found" —
// no separate active-version pointer file; Schritt 6 (retraining/versioning) can add rollback
// logic on top without changing this directory layout.
fs::path latest_qualifying_version_dir(const fs::path& target_dir) {
    std::error_code ec;
    if (!fs::is_directory(target_dir, ec)) return {};
    int best_version = -1;
    fs::path best_dir;
    for (const auto& entry : fs::directory_iterator(target_dir, ec)) {
        if (ec || !entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.empty() || name[0] != 'v') continue;
        int version = 0;
        try {
            version = std::stoi(name.substr(1));
        } catch (...) {
            continue;
        }
        std::error_code exists_ec;
        if (!fs::is_regular_file(entry.path() / "metadata.json", exists_ec)) continue;
        if (!fs::is_regular_file(entry.path() / "reference_points.jsonl", exists_ec)) continue;
        if (version > best_version) {
            best_version = version;
            best_dir = entry.path();
        }
    }
    return best_dir;
}

} // namespace

// Schritt 6 (docs/PI/pi_local_learning_plan_de.md, Abschnitt 4.3): pins a scan-domain model to the
// config_schema it was trained against, so a model copied between installs (the whole point of
// pi_models/ living in the portable install root, not a hidden per-OS app-data dir) never applies a
// value the current schema doesn't recognize. The training-side script (scripts/pi_retrain_models.py)
// computes this the same way — raw bytes of the schema file — so the two sides agree without needing
// to share code.
std::string compute_file_sha256(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream contents;
    contents << in.rdbuf();
    const std::string bytes = contents.str();
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned char byte : digest) hex << std::setw(2) << static_cast<int>(byte);
    return hex.str();
}

fs::path default_pi_models_dir(const std::shared_ptr<AppState>& state) {
    return state->runtime.project_root / "pi_models";
}

fs::path pi_models_dir(const std::shared_ptr<AppState>& state) {
    if (const char* raw = std::getenv("TILE_COMPILE_PI_MODELS_DIR")) {
        if (*raw) return fs::path(raw);
    }
    return default_pi_models_dir(state);
}

namespace {
fs::path target_dir_for(const std::shared_ptr<AppState>& state,
                        const std::string& domain,
                        const std::string& target_path) {
    return pi_models_dir(state) / domain / target_path;
}
} // namespace

ParamShadowPrediction predict_param_nn(const std::shared_ptr<AppState>& state,
                                       const std::string& domain,
                                       const std::string& target_path,
                                       const nlohmann::json& feature_vector) {
    ParamShadowPrediction result;
    if (target_path.empty()) {
        result.reason = "empty_target_path";
        return result;
    }

    const fs::path target_dir = target_dir_for(state, domain, target_path);
    const fs::path version_dir = latest_qualifying_version_dir(target_dir);
    if (version_dir.empty()) {
        result.reason = "no_model";
        return result;
    }

    const auto metadata = read_json_file(version_dir / "metadata.json");
    if (!metadata.has_value()) {
        result.reason = "unreadable_metadata";
        return result;
    }
    result.model_version = version_dir.filename().string();

    if (domain == "scan") {
        const std::string trained_schema_sha256 = metadata->value("config_schema_sha256", std::string());
        if (!trained_schema_sha256.empty()) {
            const std::string current_schema_sha256 = compute_file_sha256(state->runtime.schema_path);
            if (current_schema_sha256.empty()) {
                // Cannot verify — fail closed rather than silently trust an unpinned/unverifiable
                // model, consistent with "never apply, only ever fall back" on any uncertainty here.
                result.reason = "schema_unreadable_cannot_verify_pin";
                return result;
            }
            if (current_schema_sha256 != trained_schema_sha256) {
                result.reason = "config_schema_mismatch";
                return result;
            }
        }
        // Absent config_schema_sha256 (e.g. a hand-seeded test fixture) is treated as unpinned, not
        // as a mismatch — the pin is a safety net added in Schritt 6, not a hard requirement that
        // would break every reference set written before it existed.
    }

    const nlohmann::json reference_points = read_jsonl_file(version_dir / "reference_points.jsonl");
    result.n_reference_points = static_cast<int>(reference_points.size());
    if (reference_points.empty()) {
        result.reason = "empty_reference_set";
        return result;
    }

    // k=5, or fewer if the reference set is smaller — Abschnitt 4.2 names ~20 examples as the point
    // a nearest-neighbor model becomes useful at all; below that this still runs, just with low
    // confidence, which is visible in the shadow log rather than hidden.
    constexpr int kMaxK = 5;
    std::vector<std::pair<double, const nlohmann::json*>> scored;
    scored.reserve(reference_points.size());
    for (const auto& point : reference_points) {
        if (!point.contains("feature_vector") || !point.contains("value")) continue;
        const double distance = feature_vector_distance(feature_vector, point["feature_vector"]);
        if (!std::isfinite(distance)) continue;  // no shared numeric features with this point
        scored.emplace_back(distance, &point);
    }
    if (scored.empty()) {
        result.reason = "no_comparable_reference_points";
        return result;
    }
    std::sort(scored.begin(), scored.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
    const int k = std::min(static_cast<int>(scored.size()), kMaxK);
    result.k_used = k;

    // Inverse-distance weighting either way; which combination rule depends on whether the k
    // nearest points' "value" are all numbers (regression — Schritt 5's live-edit params, e.g.
    // brightness.midtones) or not (classification — Schritt 3's scan paths, e.g. bge.method). A
    // reference set must not mix the two for the same target_path (undefined which branch wins if
    // the k-nearest happen to be mixed; not guarded against here, that is a training-side
    // invariant, not a runtime one).
    bool all_numeric = true;
    for (int i = 0; i < k; ++i) {
        if (!(*scored[static_cast<size_t>(i)].second)["value"].is_number()) { all_numeric = false; break; }
    }

    if (all_numeric) {
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        double weighted_sq_diff_accum = 0.0;  // filled in a second pass, needs the mean first
        std::vector<double> weights(static_cast<size_t>(k));
        std::vector<double> values(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) {
            const double distance = scored[static_cast<size_t>(i)].first;
            const double weight = 1.0 / (1.0 + distance);
            const double value = (*scored[static_cast<size_t>(i)].second)["value"].get<double>();
            weights[static_cast<size_t>(i)] = weight;
            values[static_cast<size_t>(i)] = value;
            weighted_sum += weight * value;
            total_weight += weight;
        }
        if (total_weight <= 0.0) {
            result.reason = "vote_failed";
            return result;
        }
        const double predicted = weighted_sum / total_weight;
        for (int i = 0; i < k; ++i) {
            const double diff = values[static_cast<size_t>(i)] - predicted;
            weighted_sq_diff_accum += weights[static_cast<size_t>(i)] * diff * diff;
        }
        const double weighted_std = std::sqrt(weighted_sq_diff_accum / total_weight);
        // Confidence proxy for regression: neighbors that agree tightly -> high confidence. No
        // principled calibration behind this (no training data exists yet to calibrate against) —
        // documented as a proxy, not a probability.
        result.available = true;
        result.predicted_value = predicted;
        result.confidence = 1.0 / (1.0 + weighted_std);
        return result;
    }

    std::map<std::string, double> vote_weight;
    double total_weight = 0.0;
    for (int i = 0; i < k; ++i) {
        const double distance = scored[static_cast<size_t>(i)].first;
        const double weight = 1.0 / (1.0 + distance);
        const std::string value_key = (*scored[static_cast<size_t>(i)].second)["value"].dump();
        vote_weight[value_key] += weight;
        total_weight += weight;
    }
    std::string best_key;
    double best_weight = -1.0;
    for (const auto& [key, weight] : vote_weight) {
        if (weight > best_weight) {
            best_weight = weight;
            best_key = key;
        }
    }
    if (best_key.empty() || total_weight <= 0.0) {
        result.reason = "vote_failed";
        return result;
    }

    result.available = true;
    result.predicted_value = nlohmann::json::parse(best_key, nullptr, false);
    result.confidence = best_weight / total_weight;  // fraction of k-NN vote mass agreeing
    return result;
}

namespace {

std::string now_iso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream out;
    out << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

// actual_value may be null (no ground truth known for this call, e.g. Schritt 5 when the field
// simply wasn't part of this operation's params) — the entry is still logged for visibility into
// how often that happens, just without an agreement verdict.
void log_one_shadow_prediction(const std::shared_ptr<AppState>& state,
                               const std::string& domain,
                               const std::string& target_path,
                               const nlohmann::json& feature_vector,
                               bool actual_known,
                               const nlohmann::json& actual_value) {
    const ParamShadowPrediction prediction = predict_param_nn(state, domain, target_path, feature_vector);

    nlohmann::json entry = {
        {"schema_version", "pi.param-shadow-prediction.v1"},
        {"domain", domain},
        {"target_path", target_path},
        {"logged_at", now_iso()},
        {"feature_vector", feature_vector},
        {"model_available", prediction.available},
    };
    if (prediction.available) {
        entry["model_prediction"] = prediction.predicted_value;
        entry["model_confidence"] = prediction.confidence;
        entry["model_version"] = prediction.model_version;
        entry["n_reference_points"] = prediction.n_reference_points;
        entry["k_used"] = prediction.k_used;
    } else {
        entry["model_unavailable_reason"] = prediction.reason;
    }
    entry["actual_known"] = actual_known;
    if (actual_known) entry["actual_value"] = actual_value;
    if (prediction.available && actual_known) {
        if (prediction.predicted_value.is_number() && actual_value.is_number()) {
            // Tolerance-based agreement for continuous targets — exact equality on floats would
            // almost never fire and say nothing useful.
            constexpr double kAgreementTolerance = 0.05;
            entry["agrees_with_actual"] =
                std::abs(prediction.predicted_value.get<double>() - actual_value.get<double>()) <= kAgreementTolerance;
        } else {
            entry["agrees_with_actual"] = (prediction.predicted_value == actual_value);
        }
    }

    const fs::path target_dir = target_dir_for(state, domain, target_path);
    std::error_code ec;
    fs::create_directories(target_dir, ec);
    std::ofstream out(target_dir / "shadow_predictions.jsonl", std::ios::app);
    if (out) out << entry.dump() << '\n';
}

} // namespace

void log_scan_param_shadow_predictions(const std::shared_ptr<AppState>& state,
                                       const nlohmann::json& scan_metrics,
                                       const nlohmann::json& scan_result,
                                       const nlohmann::json& validated_updates) {
    try {
        const nlohmann::json feature_vector = build_scan_feature_vector(scan_metrics, scan_result);
        // PoC target paths per Abschnitt 4.2/7 — deliberately just these two, not every
        // config_schema path, until the pipeline has proven itself on a small, well-understood set.
        for (const char* target_path : {"bge.method", "normalization.mode"}) {
            nlohmann::json llm_value = nullptr;
            bool llm_recommended = false;
            if (validated_updates.is_array()) {
                for (const auto& update : validated_updates) {
                    if (!update.is_object() || update.value("path", std::string()) != target_path) continue;
                    if (update.contains("value")) {
                        llm_value = update["value"];
                        llm_recommended = true;
                    }
                    break;
                }
            }
            log_one_shadow_prediction(state, "scan", target_path, feature_vector, llm_recommended, llm_value);
        }
    } catch (const std::exception&) {
        // best-effort observability; must never break scan analysis itself
    }
}

void log_live_edit_param_shadow_predictions(const std::shared_ptr<AppState>& state,
                                            const std::string& op_type,
                                            const nlohmann::json& feature_vector_before,
                                            const nlohmann::json& actual_params) {
    try {
        if (op_type.empty() || !actual_params.is_object()) return;
        // Only numeric fields — predict_param_nn's regression branch (see above) needs a numeric
        // reference set, and non-numeric live-edit params (e.g. denoise.luminance: bool,
        // chroma_denoise.mode: string) are not part of this PoC's scope.
        for (auto it = actual_params.begin(); it != actual_params.end(); ++it) {
            if (!it.value().is_number()) continue;
            const std::string target_path = op_type + "." + it.key();
            log_one_shadow_prediction(state, "live_edit", target_path, feature_vector_before,
                                      /*actual_known=*/true, it.value());
        }
    } catch (const std::exception&) {
        // best-effort observability; must never break the live-edit operation itself
    }
}

} // namespace tile_compile::pi
