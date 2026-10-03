#include "services/pi/pi_run_index.hpp"
#include <openssl/rand.h>
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace tile_compile::pi {
namespace {
std::string new_uid() {
    unsigned char bytes[16];
    if (RAND_bytes(bytes, sizeof(bytes)) != 1) throw std::runtime_error("Cannot generate run UID");
    static constexpr char hex[] = "0123456789abcdef";
    std::string uid = "run_";
    for (const auto b : bytes) { uid += hex[b >> 4]; uid += hex[b & 15]; }
    return uid;
}
bool valid_uid(const std::string& uid) {
    return uid.size() == 36 && uid.rfind("run_", 0) == 0 &&
        std::all_of(uid.begin() + 4, uid.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
}
PiRunIndex::PiRunIndex(const std::filesystem::path& storage_dir) : _db(PiDatabase::open(storage_dir)) {}
std::string PiRunIndex::generate_uid() { return new_uid(); }
std::string PiRunIndex::run_key(const std::filesystem::path& path) {
    if (path.empty()) throw std::invalid_argument("Empty run path");
    auto key = std::filesystem::weakly_canonical(std::filesystem::absolute(path)).generic_string();
#ifdef _WIN32
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return key;
}
std::optional<nlohmann::json> PiRunIndex::get(const std::string& uid) {
    auto rows = _db->query("SELECT json FROM run_index WHERE run_uid = ?", {uid});
    if (rows.empty()) return std::nullopt;
    auto object = pi_sql_json(rows[0][0]);
    object["run_keys"] = nlohmann::json::array();
    for (const auto& row : _db->query("SELECT run_key FROM run_aliases WHERE run_uid = ? ORDER BY run_key", {uid}))
        object["run_keys"].push_back(pi_sql_text(row[0]));
    return object;
}
nlohmann::json PiRunIndex::resolve(const std::filesystem::path& run_dir, const std::string& provenance_uid,
                                  const std::string& config_sha256, const std::string& started_at) {
    if (!provenance_uid.empty() && !valid_uid(provenance_uid)) throw std::invalid_argument("Invalid provenance run UID");
    const auto key = run_key(run_dir);
    PiDatabase::Tx tx(*_db);
    const auto aliases = _db->query("SELECT run_uid FROM run_aliases WHERE run_key = ?", {key});
    if (!aliases.empty()) {
        const auto uid = pi_sql_text(aliases[0][0]);
        if (!provenance_uid.empty() && provenance_uid != uid) throw std::runtime_error("Run identity conflict");
        auto result = get(uid).value();
        result["run_key"] = key;
        tx.commit();
        return result;
    }
    const auto uid = provenance_uid.empty() ? new_uid() : provenance_uid;
    if (!get(uid)) {
        nlohmann::json object = {{"schema_version", "pi.run-index.v1"}, {"run_uid", uid},
            {"config_sha256", config_sha256}, {"started_at", started_at},
            {"identity_source", provenance_uid.empty() ? "central_assignment" : "provenance"}};
        _db->execute("INSERT INTO run_index(run_uid, config_sha256, started_at, json) VALUES(?, ?, ?, ?)",
                     {uid, config_sha256, started_at, object.dump()});
    }
    _db->execute("INSERT INTO run_aliases(run_key, run_uid) VALUES(?, ?)", {key, uid});
    auto result = get(uid).value();
    result["run_key"] = key;
    tx.commit();
    return result;
}
nlohmann::json PiRunIndex::candidates(const std::string& config_sha256, const std::string& started_at) {
    nlohmann::json result = nlohmann::json::array();
    if (config_sha256.empty() || started_at.empty()) return result;
    for (const auto& row : _db->query("SELECT json FROM run_index WHERE config_sha256 = ? AND started_at = ? ORDER BY run_uid",
                                      {config_sha256, started_at})) result.push_back(pi_sql_json(row[0]));
    return result; // Suggest candidates only; never attach an alias by fingerprint.
}
} // namespace tile_compile::pi
