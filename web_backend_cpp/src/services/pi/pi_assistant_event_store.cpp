#include "services/pi/pi_assistant_event_store.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "time_utils.hpp"

namespace tile_compile::pi {
PiAssistantEventStore::PiAssistantEventStore(const std::filesystem::path& storage_dir)
    : _db(PiDatabase::open(storage_dir)) {}

std::optional<nlohmann::json> PiAssistantEventStore::get(const std::string& id) const {
    const auto rows = _db->query("SELECT json FROM assistant_thread_events WHERE event_id = ?", {id});
    if (rows.empty()) return std::nullopt;
    return pi_sql_json(rows[0][0]);
}

nlohmann::json PiAssistantEventStore::append(const std::string& id, const std::string& uid, const nlohmann::json& result) {
    if (id.empty() || id.size() > 200 || uid.empty() || !result.is_object())
        throw std::invalid_argument("Invalid assistant event");
    if (result.dump().size() > 4 * 1024 * 1024) throw std::invalid_argument("Assistant card exceeds 4 MiB");
    PiDatabase::Tx tx(*_db);
    if (const auto old = get(id)) {
        if (old->value("run_uid", std::string()) != uid || old->at("result") != result)
            throw std::invalid_argument("Assistant event ID already has different content");
        tx.commit();
        return *old;
    }
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    nlohmann::json event = {{"schema_version", "pi.assistant-event.v1"}, {"event_id", id},
        {"run_uid", uid}, {"context_id", "run:" + uid}, {"provider", "jev"},
        {"kind", "jev_post_run"}, {"evaluation_method", "backend_rules"},
        {"created_at", utc_now_iso()}, {"created_at_epoch_ms", now}, {"result", result}};
    _db->execute("INSERT INTO assistant_thread_events(event_id, run_uid, created_at, json) VALUES(?, ?, ?, ?)",
                 {id, uid, static_cast<std::int64_t>(now), event.dump()});
    tx.commit();
    return event;
}

nlohmann::json PiAssistantEventStore::list(const std::string& uid, int limit) const {
    nlohmann::json items = nlohmann::json::array();
    const auto rows = _db->query("WITH recent AS (SELECT json, created_at, rowid AS ordinal FROM assistant_thread_events "
        "WHERE run_uid = ? ORDER BY created_at DESC, rowid DESC LIMIT ?), bounded AS "
        "(SELECT *, SUM(LENGTH(CAST(json AS BLOB))) OVER (ORDER BY created_at DESC, ordinal DESC) AS bytes FROM recent) "
        "SELECT json FROM bounded WHERE bytes <= 8388608 ORDER BY created_at DESC, ordinal DESC",
        {uid, std::clamp(limit, 1, 100)});
    for (auto it = rows.rbegin(); it != rows.rend(); ++it) items.push_back(pi_sql_json((*it)[0]));
    return items;
}
} // namespace tile_compile::pi
