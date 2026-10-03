#include "services/pi/pi_preview_store.hpp"

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <limits>
#include <stdexcept>

namespace tile_compile::pi {
namespace {
std::string hex(const unsigned char* bytes, std::size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size * 2);
    for (std::size_t i = 0; i < size; ++i) {
        result += digits[bytes[i] >> 4];
        result += digits[bytes[i] & 15];
    }
    return result;
}
std::string digest(const nlohmann::json& value) {
    const auto bytes = value.dump(); // Default JSON object keys are sorted.
    unsigned char output[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), output, &size, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot hash PI preview");
    return hex(output, size);
}
}

PiPreviewStore::PiPreviewStore(const std::filesystem::path& dir) : _db(PiDatabase::open(dir)) {}

std::string PiPreviewStore::action_plan_id(const nlohmann::json& plan) {
    if (!plan.is_object()) throw std::invalid_argument("Plan must be an object");
    return "plan_" + digest(plan);
}

nlohmann::json PiPreviewStore::create(const nlohmann::json& plan, const nlohmann::json& base_config,
                                     std::int64_t now, std::int64_t ttl_seconds) {
    if (!base_config.is_object() || now < 0 || ttl_seconds <= 0 ||
        now > std::numeric_limits<std::int64_t>::max() - ttl_seconds)
        throw std::invalid_argument("Invalid preview config, timestamp or TTL");
    const auto plan_id = action_plan_id(plan);
    unsigned char random[16];
    if (RAND_bytes(random, sizeof(random)) != 1) throw std::runtime_error("Cannot generate preview ID");
    const auto id = "preview_" + hex(random, sizeof(random));
    nlohmann::json object = {
        {"schema_version", "pi.action-preview.v1"}, {"preview_id", id},
        {"action_plan_id", plan_id}, {"plan_sha256", "sha256:" + digest(plan)},
        {"config_sha256", "sha256:" + digest(base_config)},
        {"created_at_epoch", now}, {"expires_at_epoch", now + ttl_seconds},
        {"state", "pending"}, {"plan", plan}
    };
    _db->execute("INSERT INTO action_previews(preview_id, action_plan_id, expires_at, state, json) "
                 "VALUES(?, ?, ?, 'pending', ?)", {id, plan_id, now + ttl_seconds, object.dump()});
    return object;
}

std::optional<nlohmann::json> PiPreviewStore::get(const std::string& preview_id, std::int64_t now) {
    const auto rows = _db->query("SELECT json, state, expires_at FROM action_previews WHERE preview_id = ?", {preview_id});
    if (rows.empty()) return std::nullopt;
    auto object = pi_sql_json(rows[0][0]);
    const auto state = pi_sql_text(rows[0][1]);
    object["state"] = state == "pending" && now >= pi_sql_int(rows[0][2]) ? "expired" : state;
    return object;
}

bool PiPreviewStore::transition(const std::string& preview_id, const std::string& state) {
    if (state != "applied" && state != "dismissed") throw std::invalid_argument("Invalid preview state");
    PiDatabase::Tx tx(*_db);
    const auto rows = _db->query("SELECT state FROM action_previews WHERE preview_id = ?", {preview_id});
    if (rows.empty()) return false;
    const auto previous = pi_sql_text(rows[0][0]);
    if (previous != "pending" && previous != state) return false;
    _db->execute("UPDATE action_previews SET state = ? WHERE preview_id = ?", {state, preview_id});
    tx.commit();
    return true;
}
} // namespace tile_compile::pi
