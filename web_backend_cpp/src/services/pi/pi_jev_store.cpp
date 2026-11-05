#include "services/pi/pi_jev_store.hpp"
#include <stdexcept>

namespace tile_compile::pi {
namespace {
void validate_key(const std::string& id, const std::string& part) {
    if (id.empty() || part.empty()) throw std::invalid_argument("Empty Jev store key");
}
}
PiJevStore::PiJevStore(const std::filesystem::path& decisions_dir)
    : _db(PiDatabase::open(decisions_dir.parent_path())) {}

std::optional<nlohmann::json> PiJevStore::get(const std::string& id, const std::string& part) const {
    validate_key(id, part);
    auto rows = _db->query("SELECT json FROM jev_documents WHERE proposal_id = ? AND part = ?", {id, part});
    if (rows.empty()) return std::nullopt;
    return pi_sql_json(rows[0][0]);
}
void PiJevStore::put(const std::string& id, const std::string& part, const nlohmann::json& data) const {
    validate_key(id, part);
    _db->execute("INSERT INTO jev_documents(proposal_id, part, json) VALUES(?, ?, ?) "
                 "ON CONFLICT(proposal_id, part) DO UPDATE SET json = excluded.json", {id, part, data.dump()});
}
void PiJevStore::event(const std::string& id, const nlohmann::json& data) const {
    validate_key(id, "event");
    _db->execute("INSERT INTO jev_events(proposal_id, json) VALUES(?, ?)", {id, data.dump()});
}
nlohmann::json PiJevStore::events(const std::string& id) const {
    nlohmann::json result = nlohmann::json::array();
    for (const auto& row : _db->query("SELECT json FROM jev_events WHERE proposal_id = ? ORDER BY seq", {id}))
        result.push_back(pi_sql_json(row[0]));
    return result;
}
} // namespace tile_compile::pi
