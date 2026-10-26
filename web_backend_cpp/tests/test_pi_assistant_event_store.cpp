#include "services/pi/pi_assistant_event_store.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace tile_compile::pi;
using nlohmann::json;
static void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("pi_assistant_fixture_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {
            auto db = PiDatabase::open(dir);
            for (const auto* uid : {"uid_a", "uid_b"}) db->execute(
                "INSERT INTO run_index(run_uid, config_sha256, started_at, json) VALUES(?, '', '', '{}')", {uid});
            // Simulate an existing v6 database; only this owned fixture is modified.
            db->execute("DROP TABLE analysis_proposals");
            db->execute("DROP TABLE analysis_revisions");
            db->execute("DROP TABLE analysis_contexts");
            db->execute("DROP TABLE assistant_thread_events");
            db->execute("DROP TABLE retention_deletion_journal");
            db->execute("PRAGMA user_version = 6");
        }
        {
            PiAssistantEventStore store(dir);
            auto db = PiDatabase::open(dir);
            require(db->schema_version() == kPiDatabaseSchemaVersion, "v6 upgrade");
            require(db->query("SELECT run_uid FROM run_index").size() == 2, "upgrade preserves identities");
            const json result = {{"state", {{"phase", "complete"}}}, {"advice", json::array()}};
            const auto event = store.append("event_a", "uid_a", result);
            require(event["context_id"] == "run:uid_a", "context binding");
            require(event["evaluation_method"] == "backend_rules", "no model evidence implied");
            require(store.append("event_a", "uid_a", result) == event, "idempotent retry");
            const auto model_event = store.append("event_model", "uid_a",
                {{"model_selection", {{"status", "ok"}, {"used_for_recommendation", true}}}});
            require(model_event["evaluation_method"] == "backend_rules+jev_model_choice",
                    "audited model choice is distinguished from deterministic rules");
            const auto advice = store.append("advice_a", "uid_a", {{"advice", {{"suggestions", json::array({{{"candidate_id", "candidate_x"}}})}}}});
            const auto dismissed = store.record_post_run_dismissal("dismiss_a", "uid_a", "candidate_x",
                advice["event_id"], "patch_source_a", true);
            require(dismissed["result"]["source_patch_id"] == "patch_source_a", "dismissal keeps source patch link");
            require(store.post_run_dismissals("uid_a").size() == 1, "active dismissal is durable");
            require(store.list("uid_a").size() == 3, "dismissal events are not shown as advice cards");
            store.record_post_run_dismissal("restore_a", "uid_a", "candidate_x", advice["event_id"],
                "patch_source_a", false);
            require(store.post_run_dismissals("uid_a").empty(), "restore appends an undo event");
            require(store.list("uid_b").empty(), "foreign context isolation");
            bool conflict = false;
            try { store.append("event_a", "uid_b", result); } catch (const std::invalid_argument&) { conflict = true; }
            require(conflict, "same event ID cannot move contexts");
            conflict = false;
            try { store.append("event_a", "uid_a", {{"changed", true}}); } catch (const std::invalid_argument&) { conflict = true; }
            require(conflict, "same event ID cannot change content");
            bool blocked = false;
            try { db->execute("UPDATE assistant_thread_events SET json = '{}' WHERE event_id = 'event_a'"); }
            catch (const std::exception&) { blocked = true; }
            require(blocked, "immutable snapshots");
            store.append("event_b", "uid_a", result);
            require(store.list("uid_a", 1).size() == 1, "bounded view");
            for (int i = 0; i < 3; ++i) store.append("large_" + std::to_string(i), "uid_b", {{"text", std::string(3 * 1024 * 1024, 'x')}});
            require(store.list("uid_b").size() == 2, "8 MiB aggregate window");
            require(db->query("SELECT event_id FROM assistant_thread_events WHERE run_uid = 'uid_b'").size() == 3, "window does not delete old data");
        }
        {
            PiAssistantEventStore reopened(dir);
            require(reopened.list("uid_a").size() == 4, "cards survive database reopen; dismissal events remain hidden");
            require(reopened.post_run_dismissals("uid_a").empty(), "restored dismissal remains restored after reopen");
        }
        std::filesystem::remove_all(dir); // Owned fixture only.
        std::cout << "Assistant event store tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
