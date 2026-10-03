#include "services/pi/pi_run_learning_store.hpp"
#include "services/pi/pi_database.hpp"
#include "services/pi/pi_decision_state.hpp"
#include <openssl/evp.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace tile_compile::pi;
using nlohmann::json;
namespace fs = std::filesystem;
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void write(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path()); std::ofstream(path) << text;
}
int main() {
    const auto root = fs::temp_directory_path() / ("pi_learning_test_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const auto run = root / "run";
        const std::string yaml = "data:\n  color_mode: OSC\ncalibration:\n  use_dark: true\n";
        write(run / "config.yaml", yaml);
        write(run / "artifacts" / "effective_config.json", json{
            {"source_config_sha256", sha256_prefixed(yaml).substr(7)},
            {"expanded_yaml", yaml + "normalization:\n  enabled: true\n"}
        }.dump());
        write(root / "lights" / "a.fits", "RAW_PIXELS_MUST_NOT_BE_ARCHIVED");
        write(root / "darks" / "d.fits", "RAW_DARK_PIXELS_MUST_NOT_BE_ARCHIVED");
        write(run / "outputs" / "stacked_rgb.fits", "RESULT_FITS_PIXELS_MUST_NOT_BE_ARCHIVED");
        write(run / "artifacts" / "pi_run_chat_history.json", "PRIVATE_CHAT_MUST_NOT_BE_ARCHIVED");
        write(run / "artifacts" / "pi_run_provenance.json", json{{"original_input_dir", (root / "lights").string()},
            {"effective_input_dir", (root / "lights").string()}, {"started_at", "start"}, {"config_revision_id", "rev_1"}}.dump());
        write(run / "artifacts" / "run_provenance.json", json{{"build", {{"version", "test"}}}, {"input_manifest", {
            {"entry_count", 1}, {"entries", json::array({{{"path", (root / "lights" / "a.fits").string()},
                {"size_bytes", 34}, {"sha256", "existing-light-hash"}, {"acquisition", {{"EXPTIME", 120}}}}})}}}}.dump());
        write(run / "artifacts" / "global_metrics.json", "{\"metrics\":[{\"noise\":0.1,\"fwhm\":2.4}]}");
        write(run / "artifacts" / "stats.json", "{\"status\":\"ok\",\"summary\":{\"frames\":1}}");
        json scan_event = {{"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}};
        scan_event["calibration"]["applied"] = true;
        scan_event["calibration"]["steps"]["dark"]["input_manifest"]["entries"] = json::array({
            {{"path", (root / "darks" / "d.fits").string()}, {"acquisition", {{"CCD-TEMP", -10}}}}
        });
        write(run / "logs" / "run_events.jsonl", scan_event.dump() + "\nmalformed line\n");
        // A linked artifact outside the run must not be followed.
        write(root / "external.json", "{\"secret\":\"EXTERNAL_SECRET\"}");
        fs::create_symlink(root / "external.json", run / "artifacts" / "pcc.json");
        std::string uid;
        {
            auto db = PiDatabase::open(root / "store");
            db->meta_set("new_v4_data", "preserved");
            db->execute("DROP TABLE run_learning_previews");
            db->execute("DROP TABLE run_learning_state");
            db->execute("DROP TABLE run_learning_snapshots");
            db->execute("PRAGMA user_version = 4");
        }
        {
            PiRunLearningStore store(root / "store");
            check(PiDatabase::open(root / "store")->schema_version() == kPiDatabaseSchemaVersion, "schema v4 upgrade");
            check(PiDatabase::open(root / "store")->meta_get("new_v4_data") == "preserved", "new SQLite data preserved during upgrade");
            const auto first = store.capture(run, "original", "completion", "completed");
            uid = first["run_uid"];
            check(first["config"]["effective"]["calibration"]["use_dark"] == true, "effective config retained");
            check(first["config"]["parser_defaults_included"] == true && first["config"]["effective"]["normalization"]["enabled"] == true,
                  "expanded runner defaults retained with matching hash");
            check(!first["config"]["submitted"].contains("normalization"), "submitted YAML kept separately");
            check(first["source"]["light_manifest"]["entries"][0]["acquisition"]["EXPTIME"] == 120, "light acquisition retained");
            check(first["source"]["calibration"]["steps"]["dark"]["input_manifest"]["entries"][0]["acquisition"]["CCD-TEMP"] == -10, "dark acquisition retained");
            check(first["phase_events"].size() == 1, "native runner phase events retained");
            check(first["validation_state"] == "unreviewed" && first["quality_delta"].is_null(), "success not a quality/promotion claim");
            const auto serialized = first.dump();
            for (const char* sentinel : {"RAW_PIXELS_MUST_NOT_BE_ARCHIVED", "RAW_DARK_PIXELS_MUST_NOT_BE_ARCHIVED",
                                         "RESULT_FITS_PIXELS_MUST_NOT_BE_ARCHIVED", "PRIVATE_CHAT_MUST_NOT_BE_ARCHIVED", "EXTERNAL_SECRET"})
                check(serialized.find(sentinel) == std::string::npos, "raw pixels, chats and external artifacts not archived");
            check(!first["capture_issues"].empty(), "external symlink diagnosed");
            const auto repeated = store.capture(run, "original", "completion", "completed");
            check(first["snapshot_id"] == repeated["snapshot_id"], "identical capture idempotent");
            check(store.history(uid).size() == 1, "no duplicate snapshots");
            check(store.set_excluded(uid, true, "test_run"), "explicit exclusion");
            write(run / "artifacts" / "stats.json", "{\"status\":\"ok\",\"summary\":{\"frames\":2}}");
            const auto updated = store.capture(run, "original", "stats_refresh", "completed");
            check(store.history(uid).size() == 2, "changed metrics create immutable revision");
            check(updated["excluded_from_learning"] == true, "recapture cannot erase user exclusion");
            const std::string base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j1ioAAAAASUVORK5CYII=";
            std::vector<unsigned char> png(base64.size());
            int size = EVP_DecodeBlock(png.data(), reinterpret_cast<const unsigned char*>(base64.data()), static_cast<int>(base64.size()));
            if (base64.back() == '=') --size;
            png.resize(static_cast<std::size_t>(size));
            store.save_preview(uid, updated["snapshot_id"], png);
            check(store.preview(uid).value() == png, "PNG preview round trip");
            bool rejected = false;
            try { store.save_preview(uid, updated["snapshot_id"], {1,2,3}); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "non-PNG preview rejected");
            auto oversized = png;
            oversized.resize(2 * 1024 * 1024 + 1);
            rejected = false;
            try { store.save_preview(uid, updated["snapshot_id"], oversized); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "PNG byte limit enforced");
            auto too_wide = png;
            too_wide[18] = 8; too_wide[19] = 0;
            rejected = false;
            try { store.save_preview(uid, updated["snapshot_id"], too_wide); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "PNG dimension limit enforced");
            write(run / "config.yaml", "data:\n  color_mode: MONO\n");
            const auto resumed = store.capture(run, "original", "resume_start", "running");
            check(resumed["config"]["parser_defaults_included"] == false && resumed["config"]["expanded_config_stale"] == true,
                  "stale expanded config cannot override resumed YAML");
            check(resumed["config"]["effective"]["data"]["color_mode"] == "MONO", "actual resumed config archived");
            store.mark_artifacts_state(uid, "deletion_pending");
            fs::remove_all(run);
            store.mark_artifacts_deleted(uid);
            const auto retained = store.get(uid).value();
            check(retained["artifacts_state"] == "deleted", "deletion marker");
            check(retained["config"]["yaml"].is_string(), "config remains after file deletion");
            check(retained["artifacts"]["artifacts/stats.json"]["data"]["summary"]["frames"] == 2, "statistics remain after deletion");
            check(retained["excluded_from_learning"] == true, "deletion does not change eligibility");
            check(store.preview(uid).has_value(), "PNG remains after deletion");
            check(fs::exists(root / "lights" / "a.fits") && fs::exists(root / "darks" / "d.fits"), "raw sources stay on disk");
            check(store.list()[0]["run_uid"].get<std::string>() == uid, "deleted run in archive list");
            check(store.set_excluded(uid, false, ""), "explicit re-enable");
        }
        {
            PiRunLearningStore reopened(root / "store");
            check(reopened.get(uid)->at("artifacts_state") == "deleted", "archive survives reopen");
        }
        {
            auto db = PiDatabase::open(root / "v5_store");
            db->meta_set("v5_archive_marker", "keep");
            db->execute("DROP TABLE run_learning_previews");
            db->execute("CREATE TABLE run_learning_previews(run_uid TEXT PRIMARY KEY REFERENCES run_index(run_uid), "
                        "snapshot_id TEXT NOT NULL REFERENCES run_learning_snapshots(snapshot_id), png_base64 TEXT NOT NULL)");
            db->execute("PRAGMA user_version = 5");
        }
        {
            auto db = PiDatabase::open(root / "v5_store");
            check(db->schema_version() == kPiDatabaseSchemaVersion && db->meta_get("v5_archive_marker") == "keep", "schema v5 data retained");
            db->query("SELECT source_artifact FROM run_learning_previews");
        }
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; fs::remove_all(root); return 1;
    }
}
