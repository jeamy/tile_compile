#pragma once
// Gemeinsame SQLite-Datenbank fuer den PI-Speicher (Memories und Decision Records).
//
// Eine Datei `pi_store_v2.sqlite` im PI-Storage-Verzeichnis. Eine Verbindung je Datenbankdatei und
// Prozess (Registry, geteilt ueber shared_ptr), WAL-Modus, serialisiert ueber einen rekursiven Mutex,
// damit Transaktionen mehrerer Threads nicht ineinandergreifen. Keine Altbestandsmigration:
// alte JSONL-Dateien und pi_store_v1.sqlite werden nicht eingelesen.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>

struct sqlite3;

namespace tile_compile::pi {

inline constexpr const char* kPiDatabaseFileName = "pi_store_v2.sqlite";
inline constexpr int kPiDatabaseSchemaVersion = 7;

// SQL-Parameter/-Spalte: NULL, Ganzzahl oder Text.
using PiSqlValue = std::variant<std::nullptr_t, std::int64_t, std::string>;
using PiSqlRow = std::vector<PiSqlValue>;

class PiDatabase : public std::enable_shared_from_this<PiDatabase> {
public:
    // Oeffnet (oder liefert die gecachte) Datenbank im Verzeichnis `dir`. Legt Verzeichnis,
    // das aktuelle Schema frisch an. Wirft std::runtime_error bei Fehlern.
    static std::shared_ptr<PiDatabase> open(const std::filesystem::path& dir);

    ~PiDatabase();
    PiDatabase(const PiDatabase&) = delete;
    PiDatabase& operator=(const PiDatabase&) = delete;

    const std::filesystem::path& path() const { return _path; }
    const std::filesystem::path& dir() const { return _dir; }

    // Fuehrt ein Statement ohne Ergebniszeilen aus; liefert die Anzahl geaenderter Zeilen.
    int execute(const std::string& sql, const std::vector<PiSqlValue>& params = {});
    std::vector<PiSqlRow> query(const std::string& sql, const std::vector<PiSqlValue>& params = {});
    std::int64_t last_insert_rowid();

    // RAII-Transaktion (BEGIN IMMEDIATE). Verschachtelte Tx zaehlen nur hoch; Rollback bei
    // Destruktion ohne commit(). Haelt den Datenbank-Mutex fuer die Lebensdauer.
    class Tx {
    public:
        explicit Tx(PiDatabase& db);
        ~Tx();
        Tx(const Tx&) = delete;
        Tx& operator=(const Tx&) = delete;
        void commit();
    private:
        PiDatabase& _db;
        std::unique_lock<std::recursive_mutex> _lock;
        bool _outermost{false};
        bool _done{false};
    };

    // meta-Schluessel/Wert (z. B. Marker, gecachte Indizes).
    std::string meta_get(const std::string& key, const std::string& fallback = "");
    void meta_set(const std::string& key, const std::string& value);

    int schema_version();

private:
    PiDatabase(std::filesystem::path dir, std::filesystem::path path);
    void init_schema();

    std::filesystem::path _dir;
    std::filesystem::path _path;
    sqlite3* _db{nullptr};
    std::recursive_mutex _mutex;
};

// Parst eine JSON-Spalte tolerant (kaputter Inhalt -> leeres Objekt).
nlohmann::json pi_sql_json(const PiSqlValue& value);
std::string pi_sql_text(const PiSqlValue& value);
std::int64_t pi_sql_int(const PiSqlValue& value);

} // namespace tile_compile::pi
