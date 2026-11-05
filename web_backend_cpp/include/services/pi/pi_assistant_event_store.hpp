#pragma once
#include "services/pi/pi_database.hpp"
#include <optional>

namespace tile_compile::pi {
// Durable provider cards, separate from policy-gated Decision Records and chat text.
class PiAssistantEventStore {
public:
    explicit PiAssistantEventStore(const std::filesystem::path& storage_dir);
    nlohmann::json append(const std::string& id, const std::string& run_uid, const nlohmann::json& result);
    nlohmann::json record_post_run_dismissal(const std::string& id, const std::string& run_uid,
        const std::string& candidate_id, const std::string& advice_event_id,
        const std::string& source_patch_id, bool dismissed);
    nlohmann::json post_run_dismissals(const std::string& run_uid) const;
    std::optional<nlohmann::json> get(const std::string& id) const;
    nlohmann::json list(const std::string& run_uid, int limit = 100) const;
private:
    std::shared_ptr<PiDatabase> _db;
};
} // namespace tile_compile::pi
