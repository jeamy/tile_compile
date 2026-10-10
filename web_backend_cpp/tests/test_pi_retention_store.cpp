#include "services/pi/pi_database.hpp"
#include "services/pi/pi_decision_record_store.hpp"
#include "services/pi/pi_retention_store.hpp"
#include "services/pi/pi_jev_store.hpp"

#include "backend_test_harness.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include <sqlite3.h>

using nlohmann::json;

int main() {
    try {
        using namespace tile_compile::pi;
        const auto dir = std::filesystem::temp_directory_path() /
            ("tile_compile_pi_retention_test_" + std::to_string(getpid()));
        std::filesystem::remove_all(dir);

        PiRetentionStore retention(dir);
        const auto policy = retention.status();
        expect_equal(policy["schema_version"].get<std::string>(), "pi.retention-policy.v1", "policy schema");
        expect_true(policy["approved"].get<bool>(), "approved policy is persisted");
        expect_true(!policy["backup_retention_enforced"].get<bool>(), "backup retention is not claimed before the first backup exists");
        expect_equal(policy["conversation_idle_retention_days"].get<long>(), 90L, "conversation retention policy");

        auto db = PiDatabase::open(dir);
        const auto backup_file = dir / "test_backup.sqlite";
        db->backup_to(backup_file);
        expect_true(std::filesystem::exists(backup_file), "SQLite Backup API produces a snapshot");
        db->execute("INSERT INTO memories(memory_id, json) VALUES(?, ?)",
                    {std::string("backup_probe"), json{{"id", "backup_probe"}}.dump()});
        const auto verify_dir = dir / "backup_verify";
        std::filesystem::create_directories(verify_dir);
        std::filesystem::copy_file(backup_file, verify_dir / kPiDatabaseFileName);
        auto verified_backup = PiDatabase::open(verify_dir);
        expect_true(verified_backup->query("SELECT 1 FROM memories WHERE memory_id = ?", {std::string("backup_probe")}).empty(),
                    "SQLite Backup API snapshot is consistent and readable");
        const std::int64_t now = 1'800'000'000;
        const std::int64_t old_preview = now - 31 * 24 * 60 * 60;
        const std::string old_date = "2026-01-01T00:00:00Z";
        db->execute("INSERT INTO action_previews(preview_id, action_plan_id, expires_at, state, json) VALUES(?, ?, ?, ?, ?)",
                    {std::string("old_preview"), std::string("plan_old"), old_preview,
                     std::string("dismissed"), json{{"created_at_epoch", old_preview}}.dump()});
        db->execute("INSERT INTO action_previews(preview_id, action_plan_id, expires_at, state, json) VALUES(?, ?, ?, ?, ?)",
                    {std::string("new_preview"), std::string("plan_new"), now,
                     std::string("pending"), json{{"created_at_epoch", now}}.dump()});
        db->execute("INSERT INTO jev_documents(proposal_id, part, json) VALUES(?, ?, ?)",
                    {std::string("old_jev"), std::string("status"), json{{"created_at", old_date}}.dump()});
        db->execute("INSERT INTO jev_documents(proposal_id, part, json) VALUES(?, ?, ?)",
                    {std::string("old_jev"), std::string("request"), json{{"prompt", "private"}}.dump()});
        db->execute("INSERT INTO jev_documents(proposal_id, part, json) VALUES(?, ?, ?)",
                    {std::string("old_jev"), std::string("response"), json{{"answer", "private"}}.dump()});
        db->execute("INSERT INTO jev_documents(proposal_id, part, json) VALUES(?, ?, ?)",
                    {std::string("old_jev"), std::string("proposal"), json{{"choice", "keep_current"}}.dump()});

        const auto chat_dir = dir / "context_chat";
        std::filesystem::create_directories(chat_dir);
        const auto expired_chat = chat_dir / "expired.json";
        const auto current_chat = chat_dir / "current.json";
        { std::ofstream(expired_chat) << "{}"; std::ofstream(current_chat) << "{}"; }
        std::filesystem::last_write_time(expired_chat,
            std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 91));

        PiDecisionRecordStore records(dir);
        auto record = json::object();
        record["schema_version"] = kDecisionRecordSchemaVersion;
        record["kind"] = "config_reject";
        record["actor"] = "user";
        record["decision_id"] = "old_record";
        record["created_at"] = old_date;
        record["origin"] = "live";
        record["privacy_class"] = "metadata_plus_user_text";
        record["context_ref"] = {{"context_id", "ctx"}, {"run_uid", "run"}};
        record["rationale"] = json::object();
        record["rationale"]["user"] = {
            {"reason_codes", json::array({"other"})},
            {"catalog_version", "v1"}, {"text", "because"}, {"no_reason_given", false}
        };
        record["rationale"]["basis"] = json::array();
        records.append(record);

        const auto result = retention.maintain(now);
        expect_true(result["ok"].get<bool>(), "maintenance succeeds");
        expect_true(result.contains("backup") && result["backup"]["ok"].get<bool>(), "maintenance creates a managed SQLite backup");
        expect_true(retention.status()["backup_retention_enforced"].get<bool>(), "backup expiry is reported enforced");
        expect_equal(result["counts"]["previews"].get<long>(), 1L, "expired preview removed");
        expect_equal(result["counts"]["jev_raw_parts"].get<long>(), 2L, "expired raw Jev parts removed");
        expect_equal(result["counts"]["redacted_records"].get<long>(), 1L, "expired user text redacted");
        expect_equal(result["counts"]["conversation_files"].get<long>(), 1L, "idle conversation removed");
        expect_true(db->query("SELECT 1 FROM action_previews WHERE preview_id = ?", {std::string("old_preview")}).empty(),
                     "old preview no longer stored");
        expect_true(!db->query("SELECT 1 FROM action_previews WHERE preview_id = ?", {std::string("new_preview")}).empty(),
                    "recent preview retained");
        expect_true(!PiJevStore(dir / "decisions").get("old_jev", "request").has_value(), "raw request removed");
        expect_true(PiJevStore(dir / "decisions").get("old_jev", "proposal").has_value(), "proposal snapshot retained");
        expect_true(!std::filesystem::exists(expired_chat), "expired chat deleted");
        expect_true(std::filesystem::exists(current_chat), "recent chat retained");
        const auto redacted = records.get("old_record");
        expect_equal(redacted["rationale"]["user"]["text"].get<std::string>(), "", "aged text redacted");
        expect_equal(retention.status()["last_maintenance_status"].get<std::string>(), "success", "maintenance status");
        expect_equal(retention.maintain(now)["counts"]["redacted_records"].get<long>(), 0L,
                     "repeated cleanup is idempotent");

        db->execute("INSERT INTO run_index(run_uid, config_sha256, started_at, json) VALUES(?, ?, ?, ?)",
                    {std::string("forget_uid"), std::string("sha256:test"), std::string("2026-01-01T00:00:00Z"), json::object().dump()});
        db->execute("INSERT INTO run_aliases(run_key, run_uid) VALUES(?, ?)",
                    {std::string("run-name"), std::string("forget_uid")});
        auto forget_record = json::object();
        forget_record["schema_version"] = kDecisionRecordSchemaVersion;
        forget_record["kind"] = "config_apply";
        forget_record["actor"] = "user";
        forget_record["context_ref"] = {{"context_id", "run:forget_uid"}, {"run_uid", "forget_uid"}};
        forget_record["rationale"] = {{"user", json::object()}, {"basis", json::array()}};
        const auto forget_decision = records.append(forget_record);
        db->execute("INSERT INTO assistant_thread_events(event_id, run_uid, created_at, json) VALUES(?, ?, ?, ?)",
                    {std::string("event_forget"), std::string("forget_uid"), now, json::object().dump()});
        db->execute("INSERT INTO memories(memory_id, json) VALUES(?, ?)",
                    {std::string("memory_to_reset"), json{{"id", "memory_to_reset"}}.dump()});
        const auto managed_backup = retention.create_backup(now);
        const std::string managed_backup_id = managed_backup["backup_id"].get<std::string>();
        db->execute("INSERT INTO run_learning_snapshots(snapshot_id, run_uid, content_sha256, created_at, json, summary_json) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    {std::string("snapshot_forget"), std::string("forget_uid"), std::string("sha256:snapshot"), now,
                     json::object().dump(), json::object().dump()});
        db->execute("INSERT INTO run_learning_state(run_uid, artifacts_state, excluded, exclusion_code, latest_snapshot_id) "
                    "VALUES(?, ?, 0, '', ?)", {std::string("forget_uid"), std::string("retained"), std::string("snapshot_forget")});
        bool refused_without_confirmation = false;
        try { retention.forget_run("forget_uid", false, now); } catch (const std::invalid_argument&) { refused_without_confirmation = true; }
        expect_true(refused_without_confirmation, "forget requires explicit confirmation");
        const auto forgotten = retention.forget_run("forget_uid", true, now);
        expect_true(forgotten["ok"].get<bool>(), "confirmed forget succeeds");
        expect_true(records.get(forget_decision["decision_id"].get<std::string>()).is_null(), "run decision removed");
        expect_true(db->query("SELECT 1 FROM run_learning_snapshots WHERE run_uid = ?", {std::string("forget_uid")}).empty(),
                    "run learning snapshots removed");
        expect_true(db->query("SELECT 1 FROM assistant_thread_events WHERE run_uid = ?", {std::string("forget_uid")}).empty(),
                    "run assistant cards removed");
        expect_equal(pi_sql_text(db->query("SELECT operation FROM retention_deletion_journal WHERE context_uid = ?",
                                           {std::string("forget_uid")})[0][0]), std::string("forget_run"), "content-free forget audit retained");

        bool refused_empty_reset = false;
        try { retention.reset({}, true, now); } catch (const std::invalid_argument&) { refused_empty_reset = true; }
        expect_true(refused_empty_reset, "reset requires explicit categories");
        const auto reset = retention.reset({"memories", "previews"}, true, now + 1);
        expect_true(reset["ok"].get<bool>(), "confirmed category reset succeeds");
        expect_true(db->query("SELECT 1 FROM memories WHERE memory_id = ?", {std::string("memory_to_reset")}).empty(),
                    "selected memory category erased");
        expect_true(db->query("SELECT 1 FROM jev_documents WHERE proposal_id = ?", {std::string("old_jev")}).size() == 2,
                    "unselected Jev category retained");
        expect_true(db->query("SELECT 1 FROM action_previews WHERE preview_id = ?", {std::string("new_preview")}).empty(),
                    "selected preview category erased");

        sqlite3* reader = nullptr;
        expect_true(sqlite3_open_v2(db->path().string().c_str(), &reader, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nullptr) == SQLITE_OK,
                    "open independent SQLite reader");
        sqlite3_busy_timeout(reader, 1000);
        expect_true(sqlite3_exec(reader, "BEGIN; SELECT count(*) FROM memories;", nullptr, nullptr, nullptr) == SQLITE_OK,
                    "hold an independent WAL read snapshot");
        bool checkpoint_failure_reported = false;
        try { (void)retention.maintain(now + 1); }
        catch (const std::exception&) { checkpoint_failure_reported = true; }
        expect_true(checkpoint_failure_reported, "checkpoint contention is not reported as successful cleanup");
        expect_equal(retention.status()["last_maintenance_status"].get<std::string>(), "failed", "checkpoint failure is visible");
        sqlite3_exec(reader, "ROLLBACK", nullptr, nullptr, nullptr);
        sqlite3_close(reader);

        db->execute("INSERT INTO run_index(run_uid, config_sha256, started_at, json) VALUES(?, ?, ?, ?)",
                    {std::string("crash_recovery_uid"), std::string("sha256:pending"), std::string("2026-01-01T00:00:00Z"), json::object().dump()});
        {
            std::ofstream journal(dir / "pi_retention_deletion_journal.jsonl", std::ios::app);
            journal << json{{"op", "forget_run"}, {"operation_id", "pending_forget_recovery"},
                            {"run_uid", "crash_recovery_uid"}, {"created_at", "2026-01-02T00:00:00Z"}}.dump() << "\n";
        }
        PiRetentionStore startup_recovery(dir);
        expect_true(db->query("SELECT 1 FROM run_index WHERE run_uid = ?", {std::string("crash_recovery_uid")}).empty(),
                    "startup replays a durable deletion marker after a simulated pre-commit crash");

        bool restore_confirmation_required = false;
        try { retention.restore_backup(managed_backup_id, false, now + 2); }
        catch (const std::invalid_argument&) { restore_confirmation_required = true; }
        expect_true(restore_confirmation_required, "restore requires explicit confirmation");
        const auto restored = retention.restore_backup(managed_backup_id, true, now + 2);
        expect_true(restored["reconciliation"].get<std::string>() == "complete", "restore reconciles tombstones before returning");
        expect_true(db->query("SELECT 1 FROM run_index WHERE run_uid = ?", {std::string("forget_uid")}).empty(),
                    "restore cannot resurrect a forgotten run");
        expect_true(db->query("SELECT 1 FROM memories WHERE memory_id = ?", {std::string("memory_to_reset")}).empty(),
                    "restore cannot resurrect a reset category");
        expect_equal(retention.status()["restore_reconciliation"].get<std::string>(), "reconciled", "restore status is reported");
        {
            std::ofstream marker(dir / "pi_restore.in_progress.json");
            marker << json{{"schema_version", "pi.restore-in-progress.v1"}, {"backup_id", managed_backup_id}}.dump();
        }
        PiRetentionStore interrupted_restore_recovery(dir);
        expect_true(!std::filesystem::exists(dir / "pi_restore.in_progress.json"),
                    "startup completes a restore interrupted after its durable marker");
        expect_true(db->query("SELECT 1 FROM run_index WHERE run_uid = ?", {std::string("crash_recovery_uid")}).empty(),
                    "interrupted restore recovery replays pending tombstones before availability");

        const auto journal_path = dir / "pi_retention_deletion_journal.jsonl";
        const auto journal_backup = dir / "journal_saved_for_failure_test.jsonl";
        std::filesystem::rename(journal_path, journal_backup);
        bool missing_journal_refused = false;
        try { retention.restore_backup(managed_backup_id, true, now + 3); }
        catch (const std::runtime_error&) { missing_journal_refused = true; }
        std::filesystem::rename(journal_backup, journal_path);
        expect_true(missing_journal_refused, "restore fails closed if external journal is unavailable");

        const auto corrupt_backup = retention.create_backup(now + 5);
        const auto corrupt_backup_path = dir / "pi_backups" / ("pi_backup_" + corrupt_backup["backup_id"].get<std::string>() + ".sqlite");
        { std::ofstream corrupt(corrupt_backup_path, std::ios::binary | std::ios::trunc); corrupt << "not a sqlite database"; }
        bool corrupt_backup_refused = false;
        try { retention.restore_backup(corrupt_backup["backup_id"].get<std::string>(), true, now + 6); }
        catch (const std::exception&) { corrupt_backup_refused = true; }
        expect_true(corrupt_backup_refused, "corrupt backup is rejected before active database replacement");
        expect_true(db->query("SELECT 1 FROM memories WHERE memory_id = ?", {std::string("memory_to_reset")}).empty(),
                    "failed staged restore leaves active database unchanged");

        const auto expiry_candidate = retention.create_backup(now + 3);
        const auto expiry_id = expiry_candidate["backup_id"].get<std::string>();
        const auto expiry_path = dir / "pi_backups" / ("pi_backup_" + expiry_id + ".sqlite");
        std::filesystem::last_write_time(expiry_path,
            std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 14));
        (void)retention.create_backup(now + 4);
        expect_true(!std::filesystem::exists(expiry_path), "backup at the 14-day boundary is expired during maintenance");

        const auto managed_backup_path = dir / "pi_backups" / ("pi_backup_" + managed_backup_id + ".sqlite");
        std::filesystem::last_write_time(managed_backup_path,
            std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 15));
        bool expired_backup_refused = false;
        try { retention.restore_backup(managed_backup_id, true, now + 4); }
        catch (const std::invalid_argument&) { expired_backup_refused = true; }
        expect_true(expired_backup_refused, "restore refuses a backup older than 14 days");

        // VACUUM failure after the WAL checkpoint: cleanup already committed stays, but the run must report failure.
        const std::int64_t vacuum_now = now + 7;
        db->execute("INSERT INTO action_previews(preview_id, action_plan_id, expires_at, state, json) VALUES(?, ?, ?, ?, ?)",
                    {std::string("vacuum_expired_preview"), std::string("plan_vacuum"), vacuum_now - 31 * 24 * 60 * 60,
                     std::string("dismissed"), json{{"created_at_epoch", vacuum_now - 31 * 24 * 60 * 60}}.dump()});
        PiRetentionStore::set_vacuum_hook_for_testing([] { throw std::runtime_error("injected VACUUM failure"); });
        bool vacuum_failure_reported = false;
        try { (void)retention.maintain(vacuum_now); }
        catch (const std::runtime_error&) { vacuum_failure_reported = true; }
        PiRetentionStore::set_vacuum_hook_for_testing(nullptr);
        expect_true(vacuum_failure_reported, "VACUUM failure is reported to the caller");
        const auto vacuum_status = retention.status();
        expect_equal(vacuum_status["last_maintenance_status"].get<std::string>(), "failed", "VACUUM failure is visible in status");
        expect_true(vacuum_status["last_maintenance_error"].get<std::string>().find("injected VACUUM failure") != std::string::npos,
                    "VACUUM failure message is recorded");
        expect_true(db->query("SELECT 1 FROM action_previews WHERE preview_id = ?", {std::string("vacuum_expired_preview")}).empty(),
                    "cleanup committed before the VACUUM failure is not rolled back");
        expect_true(retention.maintain(vacuum_now)["ok"].get<bool>(), "repeated maintenance succeeds after VACUUM failure");
        expect_equal(retention.status()["last_maintenance_status"].get<std::string>(), "success", "successful retry clears failed status");

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
