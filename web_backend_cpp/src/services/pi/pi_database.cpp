#include "services/pi/pi_database.hpp"

#include <sqlite3.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace tile_compile::pi {
namespace {

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, std::weak_ptr<PiDatabase>>& registry() {
    static std::map<std::string, std::weak_ptr<PiDatabase>> r;
    return r;
}

std::string utc_iso_now() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
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

[[noreturn]] void throw_sqlite(sqlite3* db, const std::string& what) {
    throw std::runtime_error("PI database " + what + ": " + (db ? sqlite3_errmsg(db) : "no connection"));
}

struct Stmt {
    sqlite3* db{nullptr};
    sqlite3_stmt* st{nullptr};
    Stmt(sqlite3* d, const std::string& sql, const std::vector<PiSqlValue>& params) : db(d) {
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) throw_sqlite(db, "prepare");
        int idx = 1;
        for (const auto& p : params) {
            int rc = SQLITE_OK;
            if (std::holds_alternative<std::nullptr_t>(p)) {
                rc = sqlite3_bind_null(st, idx);
            } else if (std::holds_alternative<std::int64_t>(p)) {
                rc = sqlite3_bind_int64(st, idx, std::get<std::int64_t>(p));
            } else {
                const auto& s = std::get<std::string>(p);
                rc = sqlite3_bind_text(st, idx, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
            }
            if (rc != SQLITE_OK) throw_sqlite(db, "bind");
            ++idx;
        }
    }
    ~Stmt() { if (st) sqlite3_finalize(st); }
    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
};

PiSqlValue column_value(sqlite3_stmt* st, int i) {
    switch (sqlite3_column_type(st, i)) {
        case SQLITE_INTEGER: return static_cast<std::int64_t>(sqlite3_column_int64(st, i));
        case SQLITE_NULL: return nullptr;
        default: {
            const unsigned char* t = sqlite3_column_text(st, i);
            const int n = sqlite3_column_bytes(st, i);
            return std::string(reinterpret_cast<const char*>(t ? t : reinterpret_cast<const unsigned char*>("")),
                               static_cast<std::size_t>(n));
        }
    }
}

const char* kSchemaV1 = R"SQL(
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Memories (vormals memories_v2.jsonl, memory_reviews_v2.jsonl, memory_outcomes_v2.jsonl,
-- memory_auto_promotion_shadow_v1.jsonl). Append-only Ereignistabellen; Merge-Logik im Store.
CREATE TABLE IF NOT EXISTS memories (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_memory_id ON memories(memory_id);

CREATE TABLE IF NOT EXISTS memory_reviews (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_reviews_memory_id ON memory_reviews(memory_id);

CREATE TABLE IF NOT EXISTS memory_outcomes (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_outcomes_memory_id ON memory_outcomes(memory_id);

CREATE TABLE IF NOT EXISTS memory_shadow (
    seq  INTEGER PRIMARY KEY AUTOINCREMENT,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_dedupe_backup (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    backup_id TEXT NOT NULL,
    json      TEXT NOT NULL
);

-- Decision Records (docs/PI/DECISIONTRACE/pi_decision_trace_plan_de.md)
CREATE TABLE IF NOT EXISTS decision_records (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        TEXT NOT NULL UNIQUE,
    idempotency_key    TEXT,
    kind               TEXT NOT NULL,
    actor              TEXT NOT NULL,
    parent_decision_id TEXT,
    context_id         TEXT,
    run_uid            TEXT,
    image_id           TEXT,
    created_at         TEXT NOT NULL,
    json               TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS decision_records_idem
    ON decision_records(idempotency_key) WHERE idempotency_key IS NOT NULL AND idempotency_key != '';
CREATE INDEX IF NOT EXISTS decision_records_kind    ON decision_records(kind);
CREATE INDEX IF NOT EXISTS decision_records_parent  ON decision_records(parent_decision_id);
CREATE INDEX IF NOT EXISTS decision_records_context ON decision_records(context_id);
CREATE INDEX IF NOT EXISTS decision_records_run     ON decision_records(run_uid);
CREATE INDEX IF NOT EXISTS decision_records_image   ON decision_records(image_id);
CREATE INDEX IF NOT EXISTS decision_records_created ON decision_records(created_at);

CREATE TABLE IF NOT EXISTS decision_reasons (
    decision_id TEXT NOT NULL REFERENCES decision_records(decision_id),
    code        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_reasons_code ON decision_reasons(code);
CREATE INDEX IF NOT EXISTS decision_reasons_decision ON decision_reasons(decision_id);

CREATE TABLE IF NOT EXISTS decision_links (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    link_id     TEXT NOT NULL UNIQUE,
    decision_id TEXT NOT NULL REFERENCES decision_records(decision_id),
    type        TEXT NOT NULL,
    data        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_links_decision ON decision_links(decision_id);

-- Append-only: Records werden nie geloescht; nur die JSON-Spalte darf sich aendern (Redaktion).
CREATE TRIGGER IF NOT EXISTS decision_records_no_delete BEFORE DELETE ON decision_records
BEGIN SELECT RAISE(ABORT, 'decision_records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS decision_records_immutable
BEFORE UPDATE OF decision_id, idempotency_key, kind, actor, parent_decision_id, context_id, run_uid,
                 image_id, created_at ON decision_records
BEGIN SELECT RAISE(ABORT, 'decision_records key columns are immutable'); END;
CREATE TRIGGER IF NOT EXISTS decision_links_no_delete BEFORE DELETE ON decision_links
BEGIN SELECT RAISE(ABORT, 'decision_links are append-only'); END;
CREATE TRIGGER IF NOT EXISTS decision_links_no_update BEFORE UPDATE ON decision_links
BEGIN SELECT RAISE(ABORT, 'decision_links are append-only'); END;
)SQL";

} // namespace

nlohmann::json pi_sql_json(const PiSqlValue& value) {
    if (!std::holds_alternative<std::string>(value)) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(std::get<std::string>(value), nullptr, false);
    if (parsed.is_discarded()) return nlohmann::json::object();
    return parsed;
}

std::string pi_sql_text(const PiSqlValue& value) {
    if (std::holds_alternative<std::string>(value)) return std::get<std::string>(value);
    if (std::holds_alternative<std::int64_t>(value)) return std::to_string(std::get<std::int64_t>(value));
    return "";
}

std::int64_t pi_sql_int(const PiSqlValue& value) {
    if (std::holds_alternative<std::int64_t>(value)) return std::get<std::int64_t>(value);
    if (std::holds_alternative<std::string>(value)) {
        try { return std::stoll(std::get<std::string>(value)); } catch (...) { return 0; }
    }
    return 0;
}

PiDatabase::PiDatabase(std::filesystem::path dir, std::filesystem::path path)
    : _dir(std::move(dir)), _path(std::move(path)) {}

PiDatabase::~PiDatabase() {
    if (_db) sqlite3_close(_db);
}

std::shared_ptr<PiDatabase> PiDatabase::open(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw std::runtime_error("failed to create PI storage directory: " + ec.message());
    const std::filesystem::path path = dir / kPiDatabaseFileName;
    const std::string key = std::filesystem::absolute(path).lexically_normal().string();

    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(key);
    if (it != registry().end()) {
        if (auto existing = it->second.lock()) {
            if (std::filesystem::exists(path)) return existing;  // Datei geloescht -> neu oeffnen
        }
    }
    std::shared_ptr<PiDatabase> db(new PiDatabase(dir, path));
    const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
    if (sqlite3_open_v2(path.string().c_str(), &db->_db, flags, nullptr) != SQLITE_OK) {
        const std::string msg = db->_db ? sqlite3_errmsg(db->_db) : "open failed";
        throw std::runtime_error("failed to open PI database " + path.string() + ": " + msg);
    }
    sqlite3_busy_timeout(db->_db, 5000);
    db->execute("PRAGMA journal_mode=WAL");
    db->execute("PRAGMA synchronous=NORMAL");
    db->execute("PRAGMA foreign_keys=ON");
    db->init_schema();
    db->import_legacy_jsonl();
    registry()[key] = db;
    return db;
}

int PiDatabase::execute(const std::string& sql, const std::vector<PiSqlValue>& params) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    if (params.empty()) {
        char* err = nullptr;
        if (sqlite3_exec(_db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            const std::string msg = err ? err : "exec failed";
            sqlite3_free(err);
            throw std::runtime_error("PI database exec: " + msg);
        }
        return sqlite3_changes(_db);
    }
    Stmt s(_db, sql, params);
    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(s.st)) == SQLITE_ROW) {}
    if (rc != SQLITE_DONE) throw_sqlite(_db, "step");
    return sqlite3_changes(_db);
}

std::vector<PiSqlRow> PiDatabase::query(const std::string& sql, const std::vector<PiSqlValue>& params) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    Stmt s(_db, sql, params);
    std::vector<PiSqlRow> rows;
    const int cols = sqlite3_column_count(s.st);
    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(s.st)) == SQLITE_ROW) {
        PiSqlRow row;
        row.reserve(static_cast<std::size_t>(cols));
        for (int i = 0; i < cols; ++i) row.push_back(column_value(s.st, i));
        rows.push_back(std::move(row));
    }
    if (rc != SQLITE_DONE) throw_sqlite(_db, "step");
    return rows;
}

std::int64_t PiDatabase::last_insert_rowid() {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    return static_cast<std::int64_t>(sqlite3_last_insert_rowid(_db));
}

PiDatabase::Tx::Tx(PiDatabase& db) : _db(db), _lock(db._mutex) {
    // Tiefe ueber sqlite3_get_autocommit: nur die aeusserste Tx beginnt wirklich.
    _outermost = sqlite3_get_autocommit(db._db) != 0;
    if (_outermost) db.execute("BEGIN IMMEDIATE");
}

PiDatabase::Tx::~Tx() {
    if (_outermost && !_done) {
        try { _db.execute("ROLLBACK"); } catch (...) {}
    }
}

void PiDatabase::Tx::commit() {
    if (_done) return;
    _done = true;
    if (_outermost) _db.execute("COMMIT");
}

std::string PiDatabase::meta_get(const std::string& key, const std::string& fallback) {
    auto rows = query("SELECT value FROM meta WHERE key = ?", {key});
    return rows.empty() ? fallback : pi_sql_text(rows[0][0]);
}

void PiDatabase::meta_set(const std::string& key, const std::string& value) {
    execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            {key, value});
}

int PiDatabase::schema_version() {
    auto rows = query("PRAGMA user_version");
    return rows.empty() ? 0 : static_cast<int>(pi_sql_int(rows[0][0]));
}

void PiDatabase::init_schema() {
    Tx tx(*this);
    const int version = schema_version();
    if (version > kPiDatabaseSchemaVersion) {
        throw std::runtime_error("PI database schema version " + std::to_string(version) +
                                 " is newer than supported " + std::to_string(kPiDatabaseSchemaVersion));
    }
    if (version < 1) {
        execute(kSchemaV1);
        execute("PRAGMA user_version = 1");
    }
    tx.commit();
}

void PiDatabase::import_legacy_jsonl() {
    if (!meta_get("jsonl_import_v1").empty()) return;
    struct Source { const char* file; const char* table; bool has_memory_id; };
    static const Source sources[] = {
        {"memories_v2.jsonl", "memories", true},
        {"memory_reviews_v2.jsonl", "memory_reviews", true},
        {"memory_outcomes_v2.jsonl", "memory_outcomes", true},
        {"memory_auto_promotion_shadow_v1.jsonl", "memory_shadow", false},
    };
    nlohmann::json counts = nlohmann::json::object();
    Tx tx(*this);
    for (const auto& src : sources) {
        long imported = 0;
        std::ifstream in(_dir / src.file);
        if (in) {
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                auto parsed = nlohmann::json::parse(line, nullptr, false);
                if (parsed.is_discarded() || !parsed.is_object()) continue;
                if (src.has_memory_id) {
                    const std::string id = parsed.contains("memory_id") && parsed["memory_id"].is_string()
                        ? parsed["memory_id"].get<std::string>() : std::string();
                    execute(std::string("INSERT INTO ") + src.table + "(memory_id, json) VALUES(?, ?)",
                            {id, parsed.dump()});
                } else {
                    execute(std::string("INSERT INTO ") + src.table + "(json) VALUES(?)", {parsed.dump()});
                }
                ++imported;
            }
        }
        counts[src.table] = imported;
    }
    meta_set("jsonl_import_v1", nlohmann::json({{"at", utc_iso_now()}, {"counts", counts}}).dump());
    tx.commit();
}

} // namespace tile_compile::pi
