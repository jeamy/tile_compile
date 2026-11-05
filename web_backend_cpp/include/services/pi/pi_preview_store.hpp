#pragma once

#include "services/pi/pi_database.hpp"
#include <optional>

namespace tile_compile::pi {

// Standalone persistence only: expiry is a read-time state, not a decision event.
class PiPreviewStore {
public:
    explicit PiPreviewStore(const std::filesystem::path& dir);
    static std::string action_plan_id(const nlohmann::json& plan);
    nlohmann::json create(const nlohmann::json& plan, const nlohmann::json& base_config,
                          std::int64_t now, std::int64_t ttl_seconds);
    std::optional<nlohmann::json> get(const std::string& preview_id, std::int64_t now);
    // Returns false for unknown IDs or incompatible terminal states; retries are idempotent.
    bool transition(const std::string& preview_id, const std::string& state);
    // Apply intent, stored in the preview JSON while the column stays "pending" (no schema change): begin_apply records
    // the config hash before the write, record_saved the hash and path after it, complete clears the intent. get()
    // reports a preview with an intent as "applying"; a crash inside that window leaves it for reconciliation.
    bool begin_apply(const std::string& preview_id, const std::string& base_sha256, const nlohmann::json& preview);
    bool record_saved(const std::string& preview_id, const std::string& saved_sha256, const std::string& saved_path);
    bool release_apply(const std::string& preview_id);
    bool complete(const std::string& preview_id, const nlohmann::json& result);
private:
    std::shared_ptr<PiDatabase> _db;
};

} // namespace tile_compile::pi
