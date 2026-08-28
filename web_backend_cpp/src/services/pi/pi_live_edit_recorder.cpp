#include "services/pi/pi_live_edit_recorder.hpp"

#include "app_state.hpp"
#include "services/pi/pi_feature_vector.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_storage_paths.hpp"

#include <map>
#include <opencv2/imgproc.hpp>
#include <set>

namespace tile_compile::pi {

nlohmann::json build_live_edit_feature_vector(const cv::Mat& fits) {
    nlohmann::json numeric = nlohmann::json::object();
    if (!fits.empty() && fits.channels() >= 3 && fits.depth() == CV_32F) {
        cv::Mat gray;
        cv::cvtColor(fits, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mean_bgr, std_bgr;
        cv::meanStdDev(fits, mean_bgr, std_bgr);
        cv::Scalar mean_gray, std_gray;
        cv::meanStdDev(gray, mean_gray, std_gray);

        const double total = static_cast<double>(gray.total());
        if (total > 0) {
            numeric["hist_black_clip_frac"] = cv::countNonZero(gray < 0.01) / total;
            numeric["hist_white_clip_frac"] = cv::countNonZero(gray > 0.99) / total;
        }
        numeric["mean_luma"] = mean_gray[0];
        numeric["std_luma"] = std_gray[0];

        // OpenCV Mat channel order is BGR.
        const double mean_b = mean_bgr[0];
        const double mean_g = mean_bgr[1];
        const double mean_r = mean_bgr[2];
        if (mean_g > 1e-6) {
            numeric["color_balance_rg"] = mean_r / mean_g;
            numeric["color_balance_bg"] = mean_b / mean_g;
        }
    }

    return {
        {"schema_version", kFeatureVectorSchemaVersion},
        {"domain", "live_edit"},
        {"numeric", numeric},
        {"categorical", nlohmann::json::object()}
    };
}

namespace {

nlohmann::json build_live_edit_candidate(const std::string& run_id,
                                         const nlohmann::json& feature_vector,
                                         const nlohmann::json& op,
                                         bool retained) {
    const std::string op_type = op.value("type", std::string());
    return {
        {"type", "live_edit_operation"},
        {"source", "live_image_chat"},
        {"privacy_class", "metadata_only"},
        {"op_type", op_type},
        {"config_updates", nlohmann::json::array({
            {{"path", op_type}, {"value", op.value("params", nlohmann::json::object())}}
        })},
        {"context_signature", {
            {"schema_version", "pi.context_signature.v1"},
            {"domain", "live_edit"},
            {"feature_vector", feature_vector}
        }},
        {"scope", {
            {"applies_when", nlohmann::json::array({"similar_live_edit_feature_vector"})},
            {"does_not_apply_when", nlohmann::json::array({"different_image_statistics"})},
            {"confidence", retained ? 0.6 : 0.3}
        }},
        {"evidence", {
            {"run_id", run_id},
            {"source", "live_image_chat"},
            {"op_source", op.value("source", std::string())}
        }},
        {"outcome", {
            {"stage", "live_edit_session_closed"},
            {"retained", retained},
            {"verified", true},
            {"validation_valid", retained}
        }}
    };
}

} // namespace

void record_live_edit_session_outcome(const std::shared_ptr<AppState>& state,
                                      const std::string& run_id,
                                      const cv::Mat& original_fits,
                                      const std::vector<nlohmann::json>& final_undo_stack,
                                      const nlohmann::json& edit_history) {
    try {
        if (run_id.empty()) return;

        // Last occurrence per type wins — repeated adjust_step increments each push their own
        // stack entry with identical params (LiveImageSessionStore::apply_adjust rebuilds the
        // stack from last_adjust_step each time), so this naturally collapses to the terminal value
        // without needing to special-case the adjust trajectory.
        std::map<std::string, nlohmann::json> terminal_by_type;
        for (const auto& entry : final_undo_stack) {
            if (!entry.is_object() || !entry.contains("type") || !entry["type"].is_string()) continue;
            terminal_by_type[entry["type"].get<std::string>()] = entry;
        }
        if (terminal_by_type.empty() &&
            !(edit_history.is_array() && !edit_history.empty())) {
            return;  // nothing was ever applied in this session — no signal to record
        }

        std::set<std::string> ever_applied_types;
        if (edit_history.is_array()) {
            for (const auto& event : edit_history) {
                if (!event.is_object()) continue;
                const std::string action = event.value("action", std::string());
                if (action != "apply" && action != "adjust") continue;
                const nlohmann::json op = event.value("operation", nlohmann::json::object());
                if (op.is_object() && op.contains("type") && op["type"].is_string()) {
                    ever_applied_types.insert(op["type"].get<std::string>());
                }
            }
        }

        const nlohmann::json feature_vector = build_live_edit_feature_vector(original_fits);
        PiMemoryStore store(pi_storage_dir(state));

        for (const auto& [type, op] : terminal_by_type) {
            store.append_candidate(build_live_edit_candidate(run_id, feature_vector, op, /*retained=*/true));
        }
        for (const auto& type : ever_applied_types) {
            if (terminal_by_type.count(type)) continue;  // survived to close — already recorded above
            store.append_candidate(build_live_edit_candidate(
                run_id, feature_vector, {{"type", type}, {"params", nlohmann::json::object()}},
                /*retained=*/false));
        }
    } catch (const std::exception&) {
        // best-effort observability; must never break session close
    }
}

} // namespace tile_compile::pi
