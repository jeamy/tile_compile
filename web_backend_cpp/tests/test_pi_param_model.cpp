#include "services/pi/pi_param_model.hpp"
#include "services/pi/pi_feature_vector.hpp"

#include "app_state.hpp"
#include "backend_test_harness.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>

using nlohmann::json;

namespace {

json make_scan_metrics(double sky_gradient_median, double fwhm_median) {
    return {
        {"aggregate", {
            {"sky_gradient", {{"median", sky_gradient_median}, {"p10", sky_gradient_median * 0.8}, {"p90", sky_gradient_median * 1.2}, {"mean", sky_gradient_median}}},
            {"fwhm", {{"median", fwhm_median}, {"p10", fwhm_median * 0.9}, {"p90", fwhm_median * 1.1}, {"mean", fwhm_median}}}
        }}
    };
}

json make_scan_result(const std::string& color_mode) {
    return {{"frames_detected", 100}, {"color_mode", color_mode}, {"bayer_pattern", "GBRG"}};
}

void write_reference_points(const std::filesystem::path& dir, const std::vector<std::pair<json, json>>& points) {
    std::filesystem::create_directories(dir);
    std::ofstream metadata(dir / "metadata.json");
    metadata << json{{"schema_version", "pi.param-model-metadata.v1"}, {"target_path", "bge.method"},
                     {"n_samples", static_cast<int>(points.size())}}.dump(2);
    metadata.close();

    std::ofstream points_out(dir / "reference_points.jsonl");
    for (const auto& [feature_vector, value] : points) {
        points_out << json{{"feature_vector", feature_vector}, {"value", value}}.dump() << "\n";
    }
}

} // namespace

int main() {
    try {
        // --- build_scan_feature_vector: grounded field names, values pass through ---
        const json fv = tile_compile::pi::build_scan_feature_vector(
            make_scan_metrics(0.0155, 2.3), make_scan_result("OSC"));
        expect_equal(fv["schema_version"].get<std::string>(), "pi.feature-vector.v1", "feature vector schema");
        expect_equal(fv["domain"].get<std::string>(), "scan", "feature vector domain");
        expect_true(fv["numeric"].contains("sky_gradient_median"), "sky_gradient_median present");
        expect_equal(fv["numeric"]["sky_gradient_median"].get<double>(), 0.0155, "sky_gradient_median value");
        expect_equal(fv["categorical"]["color_mode"].get<std::string>(), "OSC", "color_mode categorical");
        expect_true(!fv["numeric"].contains("nonexistent_field"), "no invented fields");

        // --- feature_vector_distance: identical vectors -> 0, no shared numeric keys -> infinity ---
        expect_equal(tile_compile::pi::feature_vector_distance(fv, fv), 0.0, "distance to self is 0");
        const json empty_fv = {{"numeric", json::object()}, {"categorical", json::object()}};
        expect_true(!std::isfinite(tile_compile::pi::feature_vector_distance(fv, empty_fv)),
                   "distance with no shared numeric features is infinite, not 0");

        // --- predict_param_nn: no model present -> available=false, honest reason ---
        const auto dir = std::filesystem::temp_directory_path() /
            ("tile_compile_pi_param_model_test_" + std::to_string(getpid()));
        std::filesystem::remove_all(dir);
        auto state = std::make_shared<AppState>();
        state->runtime.project_root = dir;

        const auto no_model = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", fv);
        expect_true(!no_model.available, "no model available before any is written (bootstrap state)");
        expect_equal(no_model.reason, std::string("no_model"), "no_model reason");

        // --- predict_param_nn: seeded reference points -> correct weighted-NN vote ---
        const json close_fv = tile_compile::pi::build_scan_feature_vector(
            make_scan_metrics(0.0150, 2.2), make_scan_result("OSC"));
        const json far_fv = tile_compile::pi::build_scan_feature_vector(
            make_scan_metrics(0.5, 20.0), make_scan_result("OSC"));
        write_reference_points(dir / "pi_models" / "scan" / "bge.method" / "v1", {
            {close_fv, "none"},
            {close_fv, "none"},
            {far_fv, "autobge"},
        });

        const auto predicted = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", fv);
        expect_true(predicted.available, "prediction available once reference points exist");
        expect_equal(predicted.predicted_value.get<std::string>(), std::string("none"),
                    "nearest neighbors (2x close, same value) outvote the one far point");
        expect_true(predicted.confidence > 0.5, "confidence favors the winning value");
        expect_equal(static_cast<long>(predicted.n_reference_points), 3L, "reference point count reported");
        expect_equal(predicted.model_version, std::string("v1"), "model version reported");

        // Higher version must win over v1 when both qualify.
        write_reference_points(dir / "pi_models" / "scan" / "bge.method" / "v2", {
            {far_fv, "autobge"}, {far_fv, "autobge"}, {far_fv, "autobge"},
        });
        const auto predicted_v2 = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", far_fv);
        expect_equal(predicted_v2.model_version, std::string("v2"), "highest qualifying version is used");

        // --- predict_param_nn: config_schema_sha256 pinning (Schritt 6) ---
        {
            std::filesystem::create_directories(dir / "tile_compile_cpp");
            const auto schema_path = dir / "tile_compile_cpp" / "tile_compile.schema.yaml";
            std::ofstream schema_out(schema_path);
            schema_out << "fixture: schema\n";
            schema_out.close();
            state->runtime.schema_path = schema_path;
            const std::string real_hash = tile_compile::pi::compute_file_sha256(schema_path);
            expect_true(!real_hash.empty(), "schema fixture file hash computed");

            write_reference_points(dir / "pi_models" / "scan" / "bge.method" / "v3", {
                {close_fv, "none"}, {close_fv, "none"}, {far_fv, "autobge"},
            });
            // Inject config_schema_sha256 into the just-written v3 metadata (write_reference_points()
            // doesn't set it — this test controls it directly to check both the match and mismatch
            // paths precisely).
            const auto v3_metadata_path = dir / "pi_models" / "scan" / "bge.method" / "v3" / "metadata.json";

            std::ofstream mismatched(v3_metadata_path);
            mismatched << json{{"schema_version", "pi.param-model-metadata.v1"}, {"target_path", "bge.method"},
                               {"config_schema_sha256", "0000000000000000000000000000000000000000000000000000000000000000"}}.dump();
            mismatched.close();
            const auto mismatch_result = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", fv);
            expect_true(!mismatch_result.available, "prediction rejected when config_schema_sha256 does not match");
            expect_equal(mismatch_result.reason, std::string("config_schema_mismatch"), "mismatch reason reported");

            std::ofstream matching(v3_metadata_path);
            matching << json{{"schema_version", "pi.param-model-metadata.v1"}, {"target_path", "bge.method"},
                             {"config_schema_sha256", real_hash}}.dump();
            matching.close();
            const auto match_result = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", fv);
            expect_true(match_result.available, "prediction succeeds when config_schema_sha256 matches");

            // Point at a schema path that doesn't exist -> fail closed, not "trust it anyway".
            state->runtime.schema_path = dir / "does_not_exist.yaml";
            const auto unreadable_result = tile_compile::pi::predict_param_nn(state, "scan", "bge.method", fv);
            expect_true(!unreadable_result.available, "prediction rejected when current schema is unreadable");
            expect_equal(unreadable_result.reason, std::string("schema_unreadable_cannot_verify_pin"),
                        "fail-closed reason reported");

            // Restore: subsequent steps below rely on bge.method predictions succeeding again, and
            // this pinning sub-test must not leave that broken for them.
            state->runtime.schema_path = schema_path;
        }

        // --- log_scan_param_shadow_predictions: writes an entry per PoC path, never throws ---
        tile_compile::pi::log_scan_param_shadow_predictions(
            state, make_scan_metrics(0.0155, 2.3), make_scan_result("OSC"),
            json::array({{{"path", "bge.method"}, {"value", "none"}}}));
        const auto shadow_log_path = dir / "pi_models" / "scan" / "bge.method" / "shadow_predictions.jsonl";
        expect_true(std::filesystem::is_regular_file(shadow_log_path), "shadow prediction log written");
        {
            std::ifstream in(shadow_log_path);
            std::string line;
            bool found_agreement_entry = false;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                auto entry = json::parse(line, nullptr, false);
                if (!entry.is_discarded() && entry.value("model_available", false) &&
                    entry.value("actual_known", false)) {
                    found_agreement_entry = true;
                    expect_true(entry.contains("agrees_with_actual"), "agreement field present when both sides have a value");
                }
            }
            expect_true(found_agreement_entry, "at least one logged entry compares model vs LLM");
        }

        // normalization.mode was also logged even though no model exists for it (bootstrap state).
        const auto normalization_log_path = dir / "pi_models" / "scan" / "normalization.mode" / "shadow_predictions.jsonl";
        expect_true(std::filesystem::is_regular_file(normalization_log_path),
                   "shadow log written for the second PoC path even with no model");

        // --- predict_param_nn: numeric reference values -> regression, not classification ---
        // (Schritt 5, docs/PI/pi_local_learning_plan_de.md: live-edit params are continuous, e.g.
        // brightness.midtones — the vote branch used for scan's bge.method must not fire here.)
        const json live_edit_fv_a = {{"numeric", {{"mean_luma", 0.20}, {"std_luma", 0.05}}}, {"categorical", json::object()}};
        const json live_edit_fv_b = {{"numeric", {{"mean_luma", 0.21}, {"std_luma", 0.05}}}, {"categorical", json::object()}};
        const json live_edit_fv_query = {{"numeric", {{"mean_luma", 0.205}, {"std_luma", 0.05}}}, {"categorical", json::object()}};
        write_reference_points(dir / "pi_models" / "live_edit" / "brightness.midtones" / "v1", {
            {live_edit_fv_a, 0.10}, {live_edit_fv_b, 0.20},
        });
        const auto regression = tile_compile::pi::predict_param_nn(
            state, "live_edit", "brightness.midtones", live_edit_fv_query);
        expect_true(regression.available, "regression prediction available");
        expect_true(regression.predicted_value.is_number(),
                   "numeric reference values produce a numeric prediction, not a vote result");
        // Query is equidistant between 0.10 and 0.20 in mean_luma -> weighted mean should land
        // between the two reference values, not collapse to either one (that would indicate the
        // classification vote branch fired instead of the regression branch).
        expect_true(regression.predicted_value.get<double>() > 0.10 &&
                   regression.predicted_value.get<double>() < 0.20,
                   "regression prediction is a genuine weighted mean between the two neighbors");

        // --- log_live_edit_param_shadow_predictions: one entry per numeric param field ---
        tile_compile::pi::log_live_edit_param_shadow_predictions(
            state, "brightness", live_edit_fv_query,
            {{"midtones", 0.18}, {"shadows", 0.0}});
        const auto live_edit_log_path =
            dir / "pi_models" / "live_edit" / "brightness.midtones" / "shadow_predictions.jsonl";
        expect_true(std::filesystem::is_regular_file(live_edit_log_path),
                   "live-edit shadow log written for brightness.midtones");
        {
            std::ifstream in(live_edit_log_path);
            std::string line;
            bool found = false;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                auto entry = json::parse(line, nullptr, false);
                if (entry.is_discarded()) continue;
                expect_equal(entry.value("domain", std::string()), std::string("live_edit"), "domain field is live_edit");
                if (entry.value("model_available", false) && entry.value("actual_known", false)) {
                    found = true;
                    expect_equal(entry["actual_value"].get<double>(), 0.18, "actual applied value logged");
                }
            }
            expect_true(found, "at least one live-edit entry compares model prediction vs actual applied value");
        }
        // shadows field also gets its own log file (op_type.field granularity), even without a model.
        const auto shadows_log_path =
            dir / "pi_models" / "live_edit" / "brightness.shadows" / "shadow_predictions.jsonl";
        expect_true(std::filesystem::is_regular_file(shadows_log_path),
                   "shadow log written per numeric field, not just the first one");

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
