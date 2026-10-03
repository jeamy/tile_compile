#pragma once
#include "services/pi/pi_database.hpp"
#include <optional>

namespace tile_compile::pi {
// Durable provider cards, separate from policy-gated Decision Records and chat text.
class PiAssistantEventStore {
public:
    explicit PiAssistantEventStore(const std::filesystem::path& storage_dir);
    nlohmann::json append(const std::string& id, const std::string& run_uid, const nlohmann::json& result);
    std::optional<nlohmann::json> get(const std::string& id) const;
    nlohmann::json list(const std::string& run_uid, int limit = 100) const;
private:
    std::shared_ptr<PiDatabase> _db;
};
} // namespace tile_compile::pi
