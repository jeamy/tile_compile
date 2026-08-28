#include "services/pi/pi_outcome_recorder.hpp"

#include "app_state.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_storage_paths.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

namespace tile_compile::pi {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

fs::path marker_path(const fs::path& run_dir) {
    return run_dir / "artifacts" / "pi_outcome_recorded.json";
}

// Only "matched" and "no_provenance" are permanent: a run's provenance file is written exactly
// once at run start and never appears later, so if it is missing now it never will be. Every
// other reason (no memory candidate yet, quality not written yet, transient I/O error) must stay
// retryable — otherwise the first poll after completion permanently decides "no data for this
// run" and no later fix, including a fix to the matching logic itself, can ever re-harvest it.
bool marker_is_terminal(const json& marker) {
    if (marker.value("matched", false)) return true;
    const std::string reason = marker.value("reason", std::string());
    return reason == "no_provenance";
}

std::optional<json> read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    json parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

void write_marker(const fs::path& run_dir, const json& marker) {
    std::error_code ec;
    fs::create_directories(run_dir / "artifacts", ec);
    std::ofstream out(marker_path(run_dir), std::ios::out | std::ios::trunc);
    if (out) out << marker.dump(2);
}

} // namespace

void record_run_outcome_if_needed(const std::shared_ptr<AppState>& state,
                                  const std::string& run_id,
                                  const fs::path& run_dir) {
    // Fast path: only short-circuit on a genuinely terminal outcome (see marker_is_terminal()).
    // Cheap on purpose either way — this is called from a status-poll route that may fire every
    // second, and a small local JSON read is negligible next to the full memory-store scan below.
    if (const auto existing_marker = read_json_file(marker_path(run_dir)); existing_marker.has_value()) {
        if (marker_is_terminal(*existing_marker)) return;
    }

    const auto provenance = read_json_file(run_dir / "artifacts" / "pi_run_provenance.json");
    if (!provenance.has_value()) {
        write_marker(run_dir, {{"matched", false}, {"reason", "no_provenance"}});
        return;
    }
    const std::string prior_revision_id = provenance->value("prior_active_config_revision_id", std::string());
    const std::string config_sha256 = provenance->value("config_sha256", std::string());

    const auto quality = read_json_file(run_dir / "artifacts" / "pi_run_quality.json");
    if (!quality.has_value()) {
        write_marker(run_dir, {{"matched", false}, {"reason", "no_quality"}});
        return;
    }

    try {
        PiMemoryStore store(pi_storage_dir(state));
        const json all_items = store.list(100000);

        // Primary key: revision lineage (apply-time revision_id == the config revision active
        // when this run started). Content-hash equality across the apply-time and run-start YAML
        // serializers is not guaranteed (different serializers; effective_config_yaml() injects
        // color_mode/astap paths on the run-start side) and is kept only as a secondary,
        // best-effort fallback for cases where revision tracking itself is unavailable.
        json matched_item;
        bool found = false;
        std::string match_kind;
        if (!prior_revision_id.empty()) {
            for (const auto& item : all_items) {
                const std::string item_revision = item.value("revision_id", std::string());
                if (!item_revision.empty() && item_revision == prior_revision_id) {
                    matched_item = item;
                    found = true;
                    match_kind = "revision_id";
                    break;
                }
            }
        }
        if (!found && !config_sha256.empty()) {
            for (const auto& item : all_items) {
                const std::string item_sha = item.value("config_sha256", std::string());
                if (!item_sha.empty() && item_sha == config_sha256) {
                    matched_item = item;
                    found = true;
                    match_kind = "config_sha256";
                    break;
                }
            }
        }
        if (!found) {
            write_marker(run_dir, {{"matched", false}, {"reason", "no_memory_candidate"},
                                   {"prior_active_config_revision_id", prior_revision_id},
                                   {"config_sha256", config_sha256}});
            return;
        }

        // Carry the existing outcome forward (applied_paths, validation_valid at apply time, ...)
        // and add the measured result — list()/attach_outcome() replace the whole "outcome" object
        // per memory_id (see PiMemoryStore::list()), so losing those fields here would be silent.
        json outcome = matched_item.value("outcome", json::object());
        outcome["stage"] = "run_completed";
        outcome["verified"] = true;
        outcome["run_id"] = run_id;
        outcome["match_kind"] = match_kind;
        // NOT quality->value("mean_weight", nullptr): nlohmann deduces the value<T>() template
        // parameter from the default argument's type (std::nullptr_t here), and get<std::nullptr_t>()
        // on an actual JSON number throws type_error.302 ("type must be null, but is number") —
        // caught this empirically via the M31 smoke test (docs/PI/pi_local_learning_plan_de.md,
        // Abschnitt 0.1), same class of bug already fixed once in runner_phase_local_metrics.cpp.
        outcome["mean_weight"] = quality->contains("mean_weight") ? (*quality)["mean_weight"] : json(nullptr);
        outcome["valid_frame_fraction"] = quality->contains("valid_frame_fraction")
            ? (*quality)["valid_frame_fraction"] : json(nullptr);
        // No same-frames baseline exists yet to diff against (docs/PI/pi_local_learning_plan_de.md,
        // Abschnitt 0.3/9): record the raw measured value, not a fabricated delta. Delta
        // computation is left to the offline training step, which can compare records sharing a
        // similar context_signature once enough of them exist.
        outcome["quality_delta"] = nullptr;
        outcome["comparison_kind"] = "unpaired";
        const bool has_mean_weight = quality->contains("mean_weight") && !(*quality)["mean_weight"].is_null();
        outcome["validation_valid"] = has_mean_weight ? true : outcome.value("validation_valid", false);

        const std::string memory_id = matched_item.value("memory_id", std::string());
        store.attach_outcome(memory_id, outcome, "pi_outcome_recorder",
                             "Schritt 1c: automatischer Join von Run-Provenance und AQMH-Run-Quality");

        // Schritt 2 (docs/PI/pi_local_learning_plan_de.md, Abschnitt 5/7): shadow-mode only —
        // evaluate and log what auto-promotion WOULD decide, never apply it. A failure here must
        // not take down outcome recording, which already succeeded above; it is its own try/catch.
        try {
            const json decision = store.evaluate_auto_promotion(memory_id);
            store.log_auto_promotion_shadow_decision(decision);
        } catch (const std::exception&) {
            // best-effort observability; not worth failing the whole recorder over
        }

        write_marker(run_dir, {{"matched", true}, {"memory_id", memory_id}, {"match_kind", match_kind}});
    } catch (const std::exception& e) {
        write_marker(run_dir, {{"matched", false}, {"reason", "error"}, {"error", e.what()}});
    }
}

} // namespace tile_compile::pi
