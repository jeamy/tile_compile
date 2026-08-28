#include "services/pi/pi_live_edit_recorder.hpp"

#include "app_state.hpp"
#include "backend_test_harness.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_storage_paths.hpp"

#include <cstdio>
#include <filesystem>
#include <opencv2/core.hpp>
#include <unistd.h>

using nlohmann::json;

namespace {

cv::Mat make_synthetic_fits(float b, float g, float r, int rows = 8, int cols = 8) {
    return cv::Mat(rows, cols, CV_32FC3, cv::Scalar(b, g, r));
}

json find_by_op_type(const json& items, const std::string& op_type) {
    for (const auto& item : items) {
        if (item.value("op_type", std::string()) == op_type) return item;
    }
    return nullptr;
}

} // namespace

int main() {
    try {
        // --- build_live_edit_feature_vector: known synthetic image -> known stats ---
        const cv::Mat flat_mid_gray = make_synthetic_fits(0.5f, 0.5f, 0.5f);
        const json fv = tile_compile::pi::build_live_edit_feature_vector(flat_mid_gray);
        expect_equal(fv["schema_version"].get<std::string>(), "pi.feature-vector.v1", "feature vector schema");
        expect_equal(fv["domain"].get<std::string>(), "live_edit", "feature vector domain");
        expect_equal(fv["numeric"]["mean_luma"].get<double>(), 0.5, "flat mid-gray mean_luma", 1e-3);
        expect_equal(fv["numeric"]["std_luma"].get<double>(), 0.0, "flat image has zero std_luma", 1e-6);
        expect_equal(fv["numeric"]["hist_black_clip_frac"].get<double>(), 0.0, "no black clipping in mid-gray image");
        expect_equal(fv["numeric"]["hist_white_clip_frac"].get<double>(), 0.0, "no white clipping in mid-gray image");
        expect_equal(fv["numeric"]["color_balance_rg"].get<double>(), 1.0, "equal R/G on neutral gray", 1e-6);

        const cv::Mat near_black = make_synthetic_fits(0.001f, 0.001f, 0.001f);
        const json fv_black = tile_compile::pi::build_live_edit_feature_vector(near_black);
        expect_equal(fv_black["numeric"]["hist_black_clip_frac"].get<double>(), 1.0,
                    "near-black image is fully black-clipped");

        // --- record_live_edit_session_outcome: terminal state, not every step ---
        const auto dir = std::filesystem::temp_directory_path() /
            ("tile_compile_pi_live_edit_test_" + std::to_string(getpid()));
        std::filesystem::remove_all(dir);
        auto state = std::make_shared<AppState>();
        state->runtime.project_root = dir;
        // pi_storage_dir() defaults to runtime.runs_dir/.pi_memory; point runs_dir at dir too so the
        // test doesn't depend on backend_runtime's own default-resolution logic.
        state->runtime.runs_dir = dir / "runs";

        // Session story: brightness applied then adjusted twice (3 stack entries, same op-type,
        // only the terminal one should count as one record) and survives to close; sharpen applied
        // then fully undone before close (must be recorded as retained=false, not silently dropped).
        const std::vector<json> final_undo_stack = {
            {{"type", "brightness"}, {"params", {{"midtones", 0.15}}}, {"source", "adjust"}},
        };
        const json edit_history = json::array({
            {{"action", "apply"}, {"operation", {{"type", "brightness"}, {"params", {{"midtones", 0.05}}}, {"source", "chat"}}}},
            {{"action", "adjust"}, {"direction", "increase"}, {"operation", {{"type", "brightness"}, {"params", {{"midtones", 0.15}}}, {"source", "adjust"}}}},
            {{"action", "apply"}, {"operation", {{"type", "sharpen"}, {"params", {{"amount", 0.3}}}, {"source", "chat"}}}},
            {{"action", "undo"}, {"operation", {{"type", "sharpen"}, {"params", {{"amount", 0.3}}}, {"source", "chat"}}}},
        });

        tile_compile::pi::record_live_edit_session_outcome(
            state, "test_run_id", flat_mid_gray, final_undo_stack, edit_history);

        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        const json items = store.list(1000);
        expect_equal(static_cast<long>(items.size()), 2L,
                    "one record for the surviving op-type, one for the fully-undone one — not one per stack entry/edit_history event");

        const json brightness_record = find_by_op_type(items, "brightness");
        expect_true(!brightness_record.is_null(), "brightness candidate recorded");
        expect_equal(brightness_record["type"].get<std::string>(), std::string("live_edit_operation"),
                    "live-edit candidates use the live_edit_operation type (Abschnitt 5)");
        expect_true(brightness_record["outcome"]["retained"].get<bool>(), "surviving op-type is retained=true");
        expect_equal(brightness_record["config_updates"][0]["value"]["midtones"].get<double>(), 0.15,
                    "terminal value recorded (0.15, the final adjust step), not the first apply (0.05)");

        const json sharpen_record = find_by_op_type(items, "sharpen");
        expect_true(!sharpen_record.is_null(), "sharpen candidate recorded even though fully undone");
        expect_true(!sharpen_record["outcome"]["retained"].get<bool>(), "fully-undone op-type is retained=false");

        // Empty session (nothing ever applied) must not create any candidates.
        std::filesystem::remove_all(dir);
        state->runtime.runs_dir = dir / "runs";
        tile_compile::pi::record_live_edit_session_outcome(
            state, "empty_run_id", flat_mid_gray, {}, json::array());
        tile_compile::pi::PiMemoryStore empty_store(tile_compile::pi::pi_storage_dir(state));
        expect_equal(static_cast<long>(empty_store.list(1000).size()), 0L,
                    "an empty session (nothing ever applied) records nothing");

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
