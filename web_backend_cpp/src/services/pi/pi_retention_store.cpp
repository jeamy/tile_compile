#include "services/pi/pi_retention_store.hpp"

#include "services/pi/pi_decision_record_store.hpp"

#include <openssl/evp.h>
#include <algorithm>
#include <functional>

#include <chrono>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <set>
#include <mutex>
#include <stdexcept>
#include <fstream>
#include <regex>
#include <cstring>
#include <cerrno>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#define O_CREAT _O_CREAT
#define O_APPEND _O_APPEND
#define O_EXCL _O_EXCL
#endif
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tile_compile::pi {
namespace {

constexpr std::int64_t kDay = 24 * 60 * 60;
std::string iso_utc(std::int64_t epoch);

std::recursive_mutex& retention_operation_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

std::string sha256_hex(const std::string& value) {
    unsigned char output[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(value.data(), value.size(), output, &size, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("cannot hash PI context identifier");
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size * 2);
    for (unsigned int i = 0; i < size; ++i) {
        result += digits[output[i] >> 4];
        result += digits[output[i] & 15];
    }
    return result;
}

std::string safe_alias(const std::string& alias) {
    std::string safe;
    for (unsigned char ch : alias) {
        safe.push_back((std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') ? static_cast<char>(ch) : '_');
    }
    if (safe.empty()) safe = "run";
    return safe;
}

int native_open(const std::filesystem::path& path, int flags, int mode = 0600) {
#ifdef _WIN32
    return ::_wopen(path.c_str(), flags, mode);
#else
    return ::open(path.c_str(), flags, mode);
#endif
}

std::ptrdiff_t native_write(int fd, const char* data, std::size_t size) {
#ifdef _WIN32
    const unsigned int count = static_cast<unsigned int>(std::min<std::size_t>(size, 0x7fffffffU));
    return ::_write(fd, data, count);
#else
    return ::write(fd, data, size);
#endif
}

int native_sync(int fd) {
#ifdef _WIN32
    return ::_commit(fd);
#else
    return ::fsync(fd);
#endif
}

int native_close(int fd) {
#ifdef _WIN32
    return ::_close(fd);
#else
    return ::close(fd);
#endif
}

int native_pid() {
#ifdef _WIN32
    return ::_getpid();
#else
    return ::getpid();
#endif
}

void sync_directory(const std::filesystem::path& path);

std::filesystem::path deletion_journal_path(const std::filesystem::path& dir) {
    return dir / "pi_retention_deletion_journal.jsonl";
}

void append_deletion_marker(const std::filesystem::path& dir, const nlohmann::json& marker) {
    const auto path = deletion_journal_path(dir);
    std::error_code status_ec;
    const auto existing_status = std::filesystem::symlink_status(path, status_ec);
    if (!status_ec && existing_status.type() != std::filesystem::file_type::not_found &&
        !std::filesystem::is_regular_file(existing_status))
        throw std::runtime_error("external PI deletion journal is not a regular file");
    if (status_ec && status_ec != std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot inspect external PI deletion journal");
    const std::string line = marker.dump() + "\n";
    const int fd = native_open(path, O_WRONLY | O_CREAT | O_APPEND
#ifndef _WIN32
                               | O_CLOEXEC | O_NOFOLLOW
#endif
                               , 0600);
    if (fd < 0) throw std::runtime_error("cannot open external PI deletion journal: " + std::string(std::strerror(errno)));
    std::size_t written = 0;
    while (written < line.size()) {
        const auto n = native_write(fd, line.data() + written, line.size() - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot write external PI deletion journal: " + std::string(error)); }
        written += static_cast<std::size_t>(n);
    }
    if (native_sync(fd) != 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot sync external PI deletion journal: " + std::string(error)); }
    if (native_close(fd) != 0) throw std::runtime_error("cannot close external PI deletion journal");
    sync_directory(dir);
}

void sync_directory(const std::filesystem::path& path) {
#ifdef _WIN32
    (void)path;
#else
    const int fd = native_open(path, O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("cannot open PI retention directory for sync: " + std::string(std::strerror(errno)));
    if (native_sync(fd) != 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot sync PI retention directory: " + std::string(error)); }
    native_close(fd);
#endif
}

void write_durable_file(const std::filesystem::path& path, const std::string& contents) {
    const int fd = native_open(path, O_WRONLY | O_CREAT | O_EXCL
#ifndef _WIN32
                               | O_CLOEXEC | O_NOFOLLOW
#endif
                               , 0600);
    if (fd < 0) throw std::runtime_error("cannot create durable PI retention file: " + std::string(std::strerror(errno)));
    std::size_t written = 0;
    while (written < contents.size()) {
        const auto n = native_write(fd, contents.data() + written, contents.size() - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot write durable PI retention file: " + std::string(error)); }
        written += static_cast<std::size_t>(n);
    }
    if (native_sync(fd) != 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot sync durable PI retention file: " + std::string(error)); }
    if (native_close(fd) != 0) throw std::runtime_error("cannot close durable PI retention file");
}

nlohmann::json read_deletion_journal(const std::filesystem::path& dir) {
    const auto path = deletion_journal_path(dir);
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || status.type() == std::filesystem::file_type::not_found)
        throw std::runtime_error("external PI deletion journal is missing; refusing backup restore");
    if (ec || !std::filesystem::is_regular_file(status))
        throw std::runtime_error("external PI deletion journal is not a regular file");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read external PI deletion journal");
    nlohmann::json entries = nlohmann::json::array();
    std::string line;
    while (std::getline(input, line)) {
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (!parsed.is_object() || !parsed.contains("op") || !parsed["op"].is_string())
            throw std::runtime_error("external PI deletion journal is corrupt; refusing backup restore");
        entries.push_back(std::move(parsed));
    }
    if (input.bad()) throw std::runtime_error("failed while reading external PI deletion journal");
    return entries;
}

int remove_central_conversations(const std::filesystem::path& storage_dir) {
    int removed = 0;
    for (const char* name : {"context_chat", "run_chat", "live_image_chat"}) {
        const auto dir = storage_dir / name;
        std::error_code ec;
        const bool present = std::filesystem::exists(dir, ec);
        if (ec) throw std::runtime_error("cannot inspect conversation directory during retention operation");
        if (!present) continue;
        if (!std::filesystem::is_directory(dir, ec) || ec) throw std::runtime_error("unsafe conversation directory during retention operation");
        for (std::filesystem::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
            if (ec) throw std::runtime_error("cannot enumerate conversations during retention operation");
            const auto st = it->symlink_status(ec);
            if (ec) throw std::runtime_error("cannot inspect conversation during retention operation");
            if (std::filesystem::is_regular_file(st) && it->path().extension() == ".json") {
                if (!std::filesystem::remove(it->path(), ec) || ec) throw std::runtime_error("cannot remove conversation during retention operation");
                ++removed;
            }
        }
    }
    return removed;
}

void reconcile_deletions(const std::shared_ptr<PiDatabase>& db,
                         const std::filesystem::path& storage_dir,
                         const nlohmann::json& entries) {
    for (const auto& entry : entries) {
        const std::string op = entry.value("op", std::string());
        if (op == "redact") {
            const std::string id = entry.value("decision_id", std::string());
            if (id.empty()) throw std::runtime_error("invalid redaction tombstone");
            if (!db->query("SELECT 1 FROM decision_records WHERE decision_id = ?", {id}).empty()) {
                PiDecisionRecordStore records(db->dir());
                if (db->query("SELECT 1 FROM decision_links WHERE decision_id = ? AND type = 'redact'", {id}).empty())
                    records.redact(id, entry.value("reason", std::string("restore_reconciliation")));
            }
        } else if (op == "forget_run") {
            const std::string uid = entry.value("run_uid", std::string());
            const std::string operation_id = entry.value("operation_id", std::string());
            if (uid.empty() || operation_id.empty()) throw std::runtime_error("invalid run-forget tombstone");
            if (!db->query("SELECT 1 FROM retention_deletion_journal WHERE operation_id = ?", {operation_id}).empty()) continue;
            PiDatabase::Tx tx(*db);
            db->meta_set("pi.retention.operation", "restore_reconciliation");
            db->execute("DELETE FROM decision_links WHERE decision_id IN (SELECT decision_id FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':')", {uid, uid, uid});
            db->execute("DELETE FROM decision_reasons WHERE decision_id IN (SELECT decision_id FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':')", {uid, uid, uid});
            db->execute("DELETE FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':'", {uid, uid, uid});
            db->execute("DELETE FROM assistant_thread_events WHERE run_uid = ?", {uid});
            db->execute("DELETE FROM run_learning_previews WHERE run_uid = ?", {uid});
            db->execute("DELETE FROM run_learning_state WHERE run_uid = ?", {uid});
            db->execute("DELETE FROM run_learning_snapshots WHERE run_uid = ?", {uid});
            db->execute("DELETE FROM run_aliases WHERE run_uid = ?", {uid});
            db->execute("DELETE FROM run_index WHERE run_uid = ?", {uid});
            db->execute("INSERT OR IGNORE INTO retention_deletion_journal(operation_id, context_uid, operation, created_at) VALUES(?, ?, 'forget_run', ?)",
                        {operation_id, uid, entry.value("created_at", iso_utc(0))});
            db->meta_set("pi.retention.operation", "");
            tx.commit();
        } else if (op == "pi_reset") {
            const auto categories = entry.value("categories", nlohmann::json::array());
            const std::string operation_id = entry.value("operation_id", std::string());
            if (!categories.is_array() || operation_id.empty()) throw std::runtime_error("invalid PI-reset tombstone");
            if (!db->query("SELECT 1 FROM retention_deletion_journal WHERE operation_id = ?", {operation_id}).empty()) continue;
            if (std::find(categories.begin(), categories.end(), nlohmann::json("conversations")) != categories.end())
                remove_central_conversations(storage_dir);
            PiDatabase::Tx tx(*db);
            db->meta_set("pi.retention.operation", "restore_reconciliation");
            for (const auto& category_value : categories) {
                if (!category_value.is_string()) throw std::runtime_error("invalid PI-reset category tombstone");
                const auto category = category_value.get<std::string>();
                if (category == "decision_records") {
                    db->execute("DELETE FROM decision_links"); db->execute("DELETE FROM decision_reasons"); db->execute("DELETE FROM decision_records");
                } else if (category == "memories") {
                    for (const char* table : {"memory_reviews", "memory_outcomes", "memory_shadow", "memory_dedupe_backup", "memories"}) db->execute(std::string("DELETE FROM ") + table);
                } else if (category == "jev") {
                    db->execute("DELETE FROM jev_events"); db->execute("DELETE FROM jev_documents");
                } else if (category == "previews") db->execute("DELETE FROM action_previews");
                else if (category == "run_learning") {
                    db->execute("DELETE FROM assistant_thread_events"); db->execute("DELETE FROM run_learning_previews");
                    db->execute("DELETE FROM run_learning_state"); db->execute("DELETE FROM run_learning_snapshots");
                    db->execute("DELETE FROM run_aliases"); db->execute("DELETE FROM run_index");
                } else if (category != "conversations") throw std::runtime_error("unknown PI-reset tombstone category");
            }
            db->execute("INSERT OR IGNORE INTO retention_deletion_journal(operation_id, context_uid, operation, created_at, categories) VALUES(?, 'all', 'pi_reset', ?, ?)",
                        {operation_id, entry.value("created_at", iso_utc(0)), categories.dump()});
            db->meta_set("pi.retention.operation", "");
            tx.commit();
        } else throw std::runtime_error("unknown external PI deletion-journal operation");
    }
}

std::function<void()>& vacuum_hook() {
    static std::function<void()> hook;
    return hook;
}

void compact_database(const std::shared_ptr<PiDatabase>& db) {
    const auto checkpoint = db->query("PRAGMA wal_checkpoint(TRUNCATE)");
    if (checkpoint.empty() || pi_sql_int(checkpoint[0][0]) != 0)
        throw std::runtime_error("PI database WAL checkpoint is busy or incomplete");
    if (vacuum_hook()) vacuum_hook()();
    db->execute("VACUUM");
}

std::string iso_utc(std::int64_t epoch) {
    const std::time_t t = static_cast<std::time_t>(epoch);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

} // namespace

void PiRetentionStore::set_vacuum_hook_for_testing(std::function<void()> hook) {
    vacuum_hook() = std::move(hook);
}

nlohmann::json PiRetentionStore::policy() {
    return {
        {"schema_version", "pi.retention-policy.v1"},
        {"policy_version", 1},
        {"approved", true},
        {"preview_retention_days", 30},
        {"jev_raw_retention_days", 90},
        {"user_text_retention_days", 30},
        {"conversation_idle_retention_days", 90},
        {"backup_retention_days", 14},
        {"decision_metadata", "until_explicit_reset_or_context_forget"},
        {"accepted_memories", "until_review_or_explicit_reset"},
        {"raw_fits_archived", false}
    };
}

PiRetentionStore::PiRetentionStore(std::filesystem::path storage_dir)
    : _storage_dir(std::move(storage_dir)), _db(PiDatabase::open(_storage_dir)) {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    const auto journal = deletion_journal_path(_storage_dir);
    std::error_code journal_ec;
    const auto journal_status = std::filesystem::symlink_status(journal, journal_ec);
    if (!journal_ec && journal_status.type() != std::filesystem::file_type::not_found &&
        !std::filesystem::is_regular_file(journal_status))
        throw std::runtime_error("external PI deletion journal is not a regular file");
    if (journal_ec && journal_ec != std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot inspect external PI deletion journal");
    const int fd = native_open(journal, O_WRONLY | O_CREAT | O_APPEND
#ifndef _WIN32
                               | O_CLOEXEC | O_NOFOLLOW
#endif
                               , 0600);
    if (fd < 0) throw std::runtime_error("cannot initialize external PI deletion journal: " + std::string(std::strerror(errno)));
    if (native_sync(fd) != 0) { const auto error = std::strerror(errno); native_close(fd); throw std::runtime_error("cannot sync external PI deletion journal: " + std::string(error)); }
    native_close(fd);
    sync_directory(_storage_dir);
    const auto restore_marker = _storage_dir / "pi_restore.in_progress.json";
    std::error_code marker_ec;
    const auto marker_status = std::filesystem::symlink_status(restore_marker, marker_ec);
    if (marker_ec && marker_ec != std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot inspect PI restore recovery marker");
    if (!marker_ec && marker_status.type() != std::filesystem::file_type::not_found) {
        if (!std::filesystem::is_regular_file(marker_status)) throw std::runtime_error("unsafe PI restore recovery marker");
        std::ifstream marker_input(restore_marker);
        nlohmann::json marker;
        try { marker_input >> marker; } catch (...) { throw std::runtime_error("PI restore recovery marker is corrupt; PI storage is unavailable"); }
        if (!marker.is_object() || !marker.contains("backup_id") || !marker["backup_id"].is_string())
            throw std::runtime_error("PI restore recovery marker is invalid; PI storage is unavailable");
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        restore_backup(marker["backup_id"].get<std::string>(), true, now);
    }
    reconcile_deletions(_db, _storage_dir, read_deletion_journal(_storage_dir));
    _db->meta_set("pi.retention.policy", policy().dump());
}

nlohmann::json PiRetentionStore::status() const {
    auto result = policy();
    result["last_maintenance_at"] = _db->meta_get("pi.retention.last_maintenance_at");
    result["last_maintenance_status"] = _db->meta_get("pi.retention.last_maintenance_status", "never_run");
    result["last_maintenance_error"] = _db->meta_get("pi.retention.last_maintenance_error");
    result["backup_management"] = "sqlite_backup_api_14d";
    result["backup_last_status"] = _db->meta_get("pi.backup.last_status", "never_run");
    result["backup_retention_enforced"] = result["backup_last_status"] == "success";
    result["restore_reconciliation"] = _db->meta_get("pi.restore.last_status", "not_yet_run");
    result["backup_count"] = _db->meta_get("pi.backup.count", "0");
    result["last_backup_at"] = _db->meta_get("pi.backup.last_at");
    return result;
}

nlohmann::json PiRetentionStore::create_backup(std::int64_t now) const {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    if (now < 0) throw std::invalid_argument("backup time must be non-negative");
    _db->meta_set("pi.backup.last_status", "running");
    const auto backup_dir = _storage_dir / "pi_backups";
    std::error_code ec;
    std::filesystem::create_directories(backup_dir, ec);
    if (ec) throw std::runtime_error("cannot create PI backup directory: " + ec.message());
    const auto backup_dir_status = std::filesystem::symlink_status(backup_dir, ec);
    if (ec || !std::filesystem::is_directory(backup_dir_status) || std::filesystem::is_symlink(backup_dir_status))
        throw std::runtime_error("PI backup directory is unsafe");
    const auto id = std::to_string(now) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto final_path = backup_dir / ("pi_backup_" + id + ".sqlite");
    const auto temp_path = backup_dir / (".pi_backup_" + id + ".tmp");
    const auto manifest_path = backup_dir / ("pi_backup_" + id + ".manifest.json");
    const auto manifest_temp = backup_dir / (".pi_backup_" + id + ".manifest.tmp");
    try {
        _db->backup_to(temp_path);
        std::filesystem::rename(temp_path, final_path);
        const auto journal = read_deletion_journal(_storage_dir);
        const nlohmann::json manifest = {{"schema_version", "pi.backup-manifest.v1"},
                                         {"backup_id", id}, {"created_at", iso_utc(now)},
                                         {"journal_entries", journal.size()}};
        write_durable_file(manifest_temp, manifest.dump());
        std::filesystem::rename(manifest_temp, manifest_path);
        sync_directory(backup_dir);
    } catch (...) {
        std::filesystem::remove(temp_path, ec);
        std::filesystem::remove(manifest_temp, ec);
        throw;
    }
    const auto cutoff = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 14);
    std::int64_t count = 0;
    bool removed_expired = false;
    for (std::filesystem::directory_iterator it(backup_dir, ec), end; it != end; it.increment(ec)) {
        if (ec) throw std::runtime_error("cannot enumerate PI backups: " + ec.message());
        const auto st = it->symlink_status(ec);
        if (ec == std::errc::no_such_file_or_directory) { ec.clear(); continue; }
        if (ec) throw std::runtime_error("cannot inspect PI backup: " + ec.message());
        if (!std::filesystem::is_regular_file(st) || it->path().extension() != ".sqlite" || it->path().filename().string().rfind("pi_backup_", 0) != 0) continue;
        const auto modified = it->last_write_time(ec);
        if (ec) throw std::runtime_error("cannot inspect PI backup age: " + ec.message());
        if (modified <= cutoff) {
            if (!std::filesystem::remove(it->path(), ec) || ec) throw std::runtime_error("cannot expire PI backup: " + it->path().string());
            const auto manifest = backup_dir / (it->path().stem().string() + ".manifest.json");
            std::filesystem::remove(manifest, ec);
            if (ec) throw std::runtime_error("cannot expire PI backup manifest: " + ec.message());
            removed_expired = true;
        } else ++count;
    }
    if (removed_expired) sync_directory(backup_dir);
    _db->meta_set("pi.backup.count", std::to_string(count));
    _db->meta_set("pi.backup.last_at", iso_utc(now));
    _db->meta_set("pi.backup.last_status", "success");
    return {{"ok", true}, {"backup_id", id}, {"created_at", iso_utc(now)}, {"retained_count", count}};
}

nlohmann::json PiRetentionStore::restore_backup(const std::string& backup_id, bool confirmed,
                                                std::int64_t now) const {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    if (!confirmed) throw std::invalid_argument("confirmed=true is required to restore a PI backup");
    if (now < 0) throw std::invalid_argument("restore time must be non-negative");
    static const std::regex valid_id(R"(^[0-9]+_[0-9]+$)");
    if (!std::regex_match(backup_id, valid_id)) throw std::invalid_argument("invalid managed backup_id");
    const auto backup_dir = _storage_dir / "pi_backups";
    std::error_code ec;
    const auto backup_dir_status = std::filesystem::symlink_status(backup_dir, ec);
    if (ec || !std::filesystem::is_directory(backup_dir_status) || std::filesystem::is_symlink(backup_dir_status))
        throw std::invalid_argument("managed PI backup directory is unavailable or unsafe");
    const auto source = backup_dir / ("pi_backup_" + backup_id + ".sqlite");
    const auto source_status = std::filesystem::symlink_status(source, ec);
    if (ec || !std::filesystem::is_regular_file(source_status)) throw std::invalid_argument("managed PI backup not found or unsafe");
    const auto backup_modified = std::filesystem::last_write_time(source, ec);
    if (ec || backup_modified <= std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 14))
        throw std::invalid_argument("PI backup is expired");
    const auto entries = read_deletion_journal(_storage_dir);
    const auto manifest_path = backup_dir / ("pi_backup_" + backup_id + ".manifest.json");
    const auto manifest_status = std::filesystem::symlink_status(manifest_path, ec);
    if (ec || !std::filesystem::is_regular_file(manifest_status)) throw std::runtime_error("PI backup manifest is missing or unsafe; refusing restore");
    std::ifstream manifest_input(manifest_path);
    nlohmann::json manifest;
    try { manifest_input >> manifest; } catch (...) { throw std::runtime_error("PI backup manifest is corrupt; refusing restore"); }
    if (!manifest.is_object() || manifest.value("schema_version", std::string()) != "pi.backup-manifest.v1" ||
        manifest.value("backup_id", std::string()) != backup_id || !manifest.contains("journal_entries") ||
        !manifest["journal_entries"].is_number_integer() || manifest["journal_entries"].get<std::int64_t>() < 0)
        throw std::runtime_error("PI backup manifest is invalid; refusing restore");
    const auto journal_offset = manifest["journal_entries"].get<std::size_t>();
    if (journal_offset > entries.size()) throw std::runtime_error("external deletion journal is incomplete; refusing restore");
    nlohmann::json replay_entries = nlohmann::json::array();
    for (std::size_t i = journal_offset; i < entries.size(); ++i) replay_entries.push_back(entries[i]);
    const auto stage_dir = _storage_dir / (".pi_restore_stage_" + std::to_string(native_pid()) + "_" + backup_id);
    if (std::filesystem::exists(stage_dir, ec)) throw std::runtime_error("PI restore staging path already exists");
    std::filesystem::create_directories(stage_dir, ec);
    if (ec) throw std::runtime_error("cannot create PI restore staging directory: " + ec.message());
    try {
        std::filesystem::copy_file(source, stage_dir / kPiDatabaseFileName,
                                   std::filesystem::copy_options::none, ec);
        if (ec) throw std::runtime_error("cannot stage PI backup: " + ec.message());
        {
            auto staged = PiDatabase::open(stage_dir);
            if (staged->schema_version() != kPiDatabaseSchemaVersion)
                throw std::runtime_error("PI backup schema is unsupported; restore refused");
            const auto integrity = staged->query("PRAGMA quick_check");
            if (integrity.empty() || pi_sql_text(integrity.front().front()) != "ok" ||
                !staged->query("PRAGMA foreign_key_check").empty())
                throw std::runtime_error("PI backup integrity check failed; restore refused");
            reconcile_deletions(staged, _storage_dir, replay_entries);
            const auto preview_cutoff = now - 30 * kDay;
            staged->execute("DELETE FROM action_previews WHERE json_valid(json) AND json_type(json, '$.created_at_epoch') = 'integer' AND CAST(json_extract(json, '$.created_at_epoch') AS INTEGER) <= ?", {preview_cutoff});
            const auto jev_cutoff = iso_utc(now - 90 * kDay);
            const auto expired_jev = staged->query("SELECT d.proposal_id, d.part FROM jev_documents d JOIN jev_documents s ON s.proposal_id=d.proposal_id AND s.part='status' WHERE d.part IN ('request','response') AND json_valid(s.json) AND json_type(s.json,'$.created_at')='text' AND json_extract(s.json,'$.created_at') <= ?", {jev_cutoff});
            for (const auto& row : expired_jev) staged->execute("DELETE FROM jev_documents WHERE proposal_id=? AND part=?", {pi_sql_text(row[0]), pi_sql_text(row[1])});
            compact_database(staged);
            const auto restore_marker = _storage_dir / "pi_restore.in_progress.json";
            bool recovery_marker_exists = std::filesystem::exists(restore_marker, ec);
            if (ec) throw std::runtime_error("cannot inspect PI restore recovery marker");
            if (recovery_marker_exists) {
                std::ifstream marker_input(restore_marker);
                nlohmann::json marker;
                try { marker_input >> marker; } catch (...) { throw std::runtime_error("existing PI restore marker is corrupt"); }
                if (marker.value("backup_id", std::string()) != backup_id) throw std::runtime_error("another PI restore is pending");
            } else {
                write_durable_file(restore_marker, nlohmann::json{{"schema_version", "pi.restore-in-progress.v1"}, {"backup_id", backup_id}}.dump());
                sync_directory(_storage_dir);
            }
            _db->set_restore_blocked(true);
            try {
                _db->restore_from(staged->path());
                std::filesystem::remove(restore_marker, ec);
                if (ec) throw std::runtime_error("cannot clear PI restore recovery marker: " + ec.message());
                sync_directory(_storage_dir);
                _db->set_restore_blocked(false);
            } catch (...) {
                _db->set_restore_blocked(true);
                throw;
            }
        }
        compact_database(_db);
        _db->meta_set("pi.restore.last_status", "reconciled");
        _db->meta_set("pi.restore.last_at", iso_utc(now));
        _db->meta_set("pi.backup.last_status", "success");
        _db->meta_set("pi.backup.last_at", manifest.value("created_at", std::string()));
    } catch (const std::exception& e) {
        try {
            _db->meta_set("pi.restore.last_status", "failed");
            _db->meta_set("pi.restore.last_error", e.what());
        } catch (...) {}
        std::filesystem::remove_all(stage_dir, ec);
        throw;
    }
    std::filesystem::remove_all(stage_dir, ec);
    if (ec) throw std::runtime_error("PI restored but staging cleanup failed: " + ec.message());
    return {{"ok", true}, {"backup_id", backup_id}, {"restored_at", iso_utc(now)},
            {"deletion_markers_replayed", replay_entries.size()}, {"reconciliation", "complete"}};
}

nlohmann::json PiRetentionStore::maintain(std::int64_t now) const {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    if (now < 0) throw std::invalid_argument("maintenance time must be non-negative");
    const std::string now_iso = iso_utc(now);
    const std::string text_cutoff = iso_utc(now - 30 * kDay);
    const std::int64_t preview_cutoff = now - 30 * kDay;
    const std::string jev_cutoff = iso_utc(now - 90 * kDay);
    nlohmann::json counts = {{"previews", 0}, {"jev_raw_parts", 0}, {"redacted_records", 0}, {"conversation_files", 0}};

    try {
        _db->execute("PRAGMA secure_delete=ON");
        PiDatabase::Tx tx(*_db);

        // created_at_epoch is part of the server-generated preview envelope. Bad/missing timestamps
        // are retained rather than guessed, keeping cleanup fail-closed.
        counts["previews"] = _db->execute(
            "DELETE FROM action_previews WHERE json_valid(json) "
            "AND json_type(json, '$.created_at_epoch') = 'integer' "
            "AND CAST(json_extract(json, '$.created_at_epoch') AS INTEGER) <= ?",
            {preview_cutoff});

        // Request/response are the only raw provider payloads. Keep proposal/candidate/state snapshots.
        // A proposal without a valid status timestamp is retained rather than age-guessed.
        const auto jev_rows = _db->query(
            "SELECT d.proposal_id, d.part FROM jev_documents d "
            "JOIN jev_documents s ON s.proposal_id = d.proposal_id AND s.part = 'status' "
            "WHERE d.part IN ('request','response') AND json_valid(s.json) "
            "AND json_type(s.json, '$.created_at') = 'text' "
            "AND json_extract(s.json, '$.created_at') <= ?",
            {jev_cutoff});
        for (const auto& row : jev_rows) {
            counts["jev_raw_parts"] = counts["jev_raw_parts"].get<int>() +
                _db->execute("DELETE FROM jev_documents WHERE proposal_id = ? AND part = ?",
                             {pi_sql_text(row[0]), pi_sql_text(row[1])});
        }

        // Redaction is an overlay audit event plus an immediate JSON update. Keep record metadata/reason codes.
        PiDecisionRecordStore records(_storage_dir);
        const auto record_rows = _db->query(
            "SELECT decision_id FROM decision_records WHERE created_at <= ? "
            "AND json_valid(json) AND COALESCE(json_extract(json, '$.rationale.user.text'), '') != '' "
            "AND NOT EXISTS (SELECT 1 FROM decision_links l WHERE l.decision_id = decision_records.decision_id AND l.type = 'redact')",
            {text_cutoff});
        for (const auto& row : record_rows) {
            const auto decision_id = pi_sql_text(row[0]);
            append_deletion_marker(_storage_dir, {{"op", "redact"}, {"decision_id", decision_id}, {"reason", "retention_30d"}});
            records.redact(decision_id, "retention_30d");
            counts["redacted_records"] = counts["redacted_records"].get<int>() + 1;
        }

        tx.commit();

        // Application-owned central conversation histories are flat JSON files. Use file modification
        // time as last activity; never recurse, follow symlinks, or inspect/delete run artifacts.
        const auto file_cutoff = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 90);
        for (const char* name : {"context_chat", "run_chat", "live_image_chat"}) {
            const auto dir = _storage_dir / name;
            std::error_code ec;
            const bool present = std::filesystem::exists(dir, ec);
            if (ec) throw std::runtime_error("cannot inspect PI conversation directory: " + dir.string());
            if (!present) continue;
            if (!std::filesystem::is_directory(dir, ec) || ec)
                throw std::runtime_error("cannot inspect PI conversation directory: " + dir.string());
            for (std::filesystem::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
                if (ec) throw std::runtime_error("cannot enumerate PI conversation directory: " + ec.message());
                const auto status = it->symlink_status(ec);
                if (ec) throw std::runtime_error("cannot inspect PI conversation entry: " + ec.message());
                if (!std::filesystem::is_regular_file(status) || it->path().extension() != ".json") continue;
                const auto modified = it->last_write_time(ec);
                if (ec) throw std::runtime_error("cannot read PI conversation activity time: " + ec.message());
                if (modified <= file_cutoff) {
                    if (!std::filesystem::remove(it->path(), ec) || ec)
                        throw std::runtime_error("cannot expire PI conversation file: " + it->path().string());
                    counts["conversation_files"] = counts["conversation_files"].get<int>() + 1;
                }
            }
        }

        _db->meta_set("pi.retention.last_maintenance_at", now_iso);
        _db->meta_set("pi.retention.last_maintenance_status", "success");
        _db->meta_set("pi.retention.last_maintenance_error", "");

        // Checkpoint and compact active DB pages after expiring/redacting data. This is not a claim
        // about external backups, filesystem snapshots, or physical media-level erasure.
        compact_database(_db);
        const auto backup = create_backup(now);
        return {{"ok", true}, {"policy_version", 1}, {"maintenance_at", now_iso}, {"counts", counts}, {"backup", backup}};
    } catch (const std::exception& e) {
        try {
            _db->meta_set("pi.retention.last_maintenance_at", now_iso);
            _db->meta_set("pi.retention.last_maintenance_status", "failed");
            _db->meta_set("pi.retention.last_maintenance_error", e.what());
        } catch (...) {}
        throw;
    }
}

nlohmann::json PiRetentionStore::forget_run(const std::string& uid, bool confirmed, std::int64_t now) const {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    if (!confirmed) throw std::invalid_argument("confirmed=true is required to forget a run context");
    if (uid.empty()) throw std::invalid_argument("run_uid is required");
    if (now < 0) throw std::invalid_argument("forget time must be non-negative");

    _db->execute("PRAGMA secure_delete=ON");
    const auto known = _db->query("SELECT run_uid FROM run_index WHERE run_uid = ?", {uid});
    if (known.empty()) throw std::invalid_argument("unknown run_uid");
    std::vector<std::string> aliases;
    for (const auto& row : _db->query("SELECT run_key FROM run_aliases WHERE run_uid = ?", {uid}))
        aliases.push_back(pi_sql_text(row[0]));

    // Remove only centrally-owned chat files derived from this exact UID/known aliases. Never touch
    // historical artifacts inside run directories or follow symlinks.
    auto remove_owned = [&](const std::filesystem::path& file) {
        std::error_code ec;
        const auto status = std::filesystem::symlink_status(file, ec);
        if (ec == std::errc::no_such_file_or_directory || status.type() == std::filesystem::file_type::not_found) return;
        if (ec) throw std::runtime_error("cannot inspect PI conversation file: " + file.string());
        if (!std::filesystem::is_regular_file(status)) return;
        if (!std::filesystem::remove(file, ec) || ec)
            throw std::runtime_error("cannot remove PI conversation file: " + file.string());
    };
    remove_owned(_storage_dir / "context_chat" / (sha256_hex("run:" + uid) + ".json"));
    for (const auto& alias : aliases) {
        if (safe_alias(alias).size() > 180) continue;
        const auto hash = static_cast<unsigned long long>(std::hash<std::string>{}(alias));
        remove_owned(_storage_dir / "run_chat" /
                     (safe_alias(alias) + "_" + std::to_string(hash) + ".json"));
        remove_owned(_storage_dir / "live_image_chat" /
                     (safe_alias(alias) + "_" + std::to_string(hash) + ".json"));
    }

    const std::string operation_id = "forget_" + sha256_hex(uid + ":" + std::to_string(now));
    append_deletion_marker(_storage_dir, {{"op", "forget_run"}, {"operation_id", operation_id}, {"run_uid", uid}, {"created_at", iso_utc(now)}});
    PiDatabase::Tx tx(*_db);
    _db->meta_set("pi.retention.operation", "forget_run:" + uid);
    _db->execute("DELETE FROM decision_links WHERE decision_id IN "
                 "(SELECT decision_id FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':')",
                 {uid, uid, uid});
    _db->execute("DELETE FROM decision_reasons WHERE decision_id IN "
                 "(SELECT decision_id FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':')",
                 {uid, uid, uid});
    const int deleted_records = _db->execute(
        "DELETE FROM decision_records WHERE run_uid = ? OR substr(image_id,1,length(?)) = ? || ':'",
        {uid, uid, uid});
    const int deleted_events = _db->execute("DELETE FROM assistant_thread_events WHERE run_uid = ?", {uid});
    _db->execute("DELETE FROM run_learning_previews WHERE run_uid = ?", {uid});
    _db->execute("DELETE FROM run_learning_state WHERE run_uid = ?", {uid});
    const int deleted_snapshots = _db->execute("DELETE FROM run_learning_snapshots WHERE run_uid = ?", {uid});
    _db->execute("DELETE FROM run_aliases WHERE run_uid = ?", {uid});
    _db->execute("DELETE FROM run_index WHERE run_uid = ?", {uid});
    _db->execute("INSERT INTO retention_deletion_journal(operation_id, context_uid, operation, created_at) "
                 "VALUES(?, ?, 'forget_run', ?) ON CONFLICT(operation_id) DO NOTHING",
                 {operation_id, uid, iso_utc(now)});
    _db->meta_set("pi.retention.operation", "");
    tx.commit();

    compact_database(_db);
    /* compacted above */
    return {{"ok", true}, {"operation_id", operation_id}, {"run_uid", uid},
            {"deleted_records", deleted_records}, {"deleted_assistant_events", deleted_events},
            {"deleted_run_snapshots", deleted_snapshots}, {"artifacts_deleted", false}};
}

nlohmann::json PiRetentionStore::reset(const std::vector<std::string>& categories, bool confirmed,
                                       std::int64_t now) const {
    std::lock_guard<std::recursive_mutex> operation_lock(retention_operation_mutex());
    if (!confirmed) throw std::invalid_argument("confirmed=true is required for PI reset");
    if (now < 0) throw std::invalid_argument("reset time must be non-negative");
    static const std::set<std::string> allowed = {
        "memories", "jev", "decision_records", "previews", "conversations", "run_learning"};
    const std::set<std::string> selected(categories.begin(), categories.end());
    if (selected.empty()) throw std::invalid_argument("at least one data category must be selected");
    for (const auto& category : selected) {
        if (!allowed.count(category)) throw std::invalid_argument("unknown PI reset category: " + category);
    }

    _db->execute("PRAGMA secure_delete=ON");
    int deleted_rows = 0;
    std::string selected_json = nlohmann::json(selected).dump();
    const std::string operation_id = "reset_" + sha256_hex(std::to_string(now) + ":" + selected_json);
    append_deletion_marker(_storage_dir, {{"op", "pi_reset"}, {"operation_id", operation_id}, {"categories", nlohmann::json(selected)}, {"created_at", iso_utc(now)}});
    int deleted_conversations = selected.count("conversations") ? remove_central_conversations(_storage_dir) : 0;
    PiDatabase::Tx tx(*_db);
    _db->meta_set("pi.retention.operation", "pi_reset");
    if (selected.count("decision_records")) {
        deleted_rows += _db->execute("DELETE FROM decision_links");
        deleted_rows += _db->execute("DELETE FROM decision_reasons");
        deleted_rows += _db->execute("DELETE FROM decision_records");
    }
    if (selected.count("memories")) {
        for (const char* table : {"memory_reviews", "memory_outcomes", "memory_shadow", "memory_dedupe_backup", "memories"})
            deleted_rows += _db->execute(std::string("DELETE FROM ") + table);
    }
    if (selected.count("jev")) {
        deleted_rows += _db->execute("DELETE FROM jev_events");
        deleted_rows += _db->execute("DELETE FROM jev_documents");
    }
    if (selected.count("previews")) deleted_rows += _db->execute("DELETE FROM action_previews");
    if (selected.count("run_learning")) {
        deleted_rows += _db->execute("DELETE FROM assistant_thread_events");
        deleted_rows += _db->execute("DELETE FROM run_learning_previews");
        deleted_rows += _db->execute("DELETE FROM run_learning_state");
        deleted_rows += _db->execute("DELETE FROM run_learning_snapshots");
        deleted_rows += _db->execute("DELETE FROM run_aliases");
        deleted_rows += _db->execute("DELETE FROM run_index");
    }
    _db->meta_set("pi.retention.operation", "");
    _db->execute("INSERT INTO retention_deletion_journal(operation_id, context_uid, operation, created_at, categories) "
                 "VALUES(?, 'all', 'pi_reset', ?, ?) ON CONFLICT(operation_id) DO NOTHING",
                 {operation_id, iso_utc(now), selected_json});
    tx.commit();

    compact_database(_db);
    /* compacted above */
    return {{"ok", true}, {"operation_id", operation_id}, {"categories", selected},
            {"deleted_rows", deleted_rows}, {"deleted_conversation_files", deleted_conversations},
            {"run_artifacts_deleted", false}};
}

} // namespace tile_compile::pi
