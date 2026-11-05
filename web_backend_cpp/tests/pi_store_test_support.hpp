#pragma once
// Test-Hilfe: schreibt Fixture-Zeilen (wie frueher in die JSONL-Dateien) direkt in die SQLite-
// Tabellen des PI-Speichers. Gedacht fuer Route-Tests, bei denen das Backend als eigener Prozess
// auf demselben Verzeichnis arbeitet (WAL: mehrere Prozesse sind erlaubt).
//
//   PiStoreSeed out(memory_dir, "memories");   // oder "memory_reviews", "memory_outcomes"
//   out << nlohmann::json{...}.dump() << "\n";

#include "services/pi/pi_database.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

class PiStoreSeed {
public:
    PiStoreSeed(const std::filesystem::path& dir, std::string table)
        : _db(tile_compile::pi::PiDatabase::open(dir)), _table(std::move(table)) {}
    ~PiStoreSeed() { flush(); }

    PiStoreSeed& operator<<(const std::string& text) { return append(text); }
    PiStoreSeed& operator<<(const char* text) { return append(text); }

private:
    PiStoreSeed& append(const std::string& text) {
        _buffer += text;
        flush();
        return *this;
    }
    void flush() {
        std::size_t pos = 0;
        while ((pos = _buffer.find('\n')) != std::string::npos) {
            const std::string line = _buffer.substr(0, pos);
            _buffer.erase(0, pos + 1);
            if (line.empty()) continue;
            auto parsed = nlohmann::json::parse(line, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object()) continue;
            const std::string id = parsed.contains("memory_id") && parsed["memory_id"].is_string()
                ? parsed["memory_id"].get<std::string>() : std::string();
            _db->execute("INSERT INTO " + _table + "(memory_id, json) VALUES(?, ?)", {id, parsed.dump()});
        }
    }

    std::shared_ptr<tile_compile::pi::PiDatabase> _db;
    std::string _table;
    std::string _buffer;
};
