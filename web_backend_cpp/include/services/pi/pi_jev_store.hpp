#pragma once
#include "services/pi/pi_database.hpp"
#include <optional>

namespace tile_compile::pi {

// Proposal snapshots and events share the PI database; no legacy file import.
class PiJevStore {
public:
    explicit PiJevStore(const std::filesystem::path& decisions_dir);
    std::optional<nlohmann::json> get(const std::string& id, const std::string& part) const;
    void put(const std::string& id, const std::string& part, const nlohmann::json& data) const;
    void event(const std::string& id, const nlohmann::json& data) const;
    nlohmann::json events(const std::string& id) const;
    const std::shared_ptr<PiDatabase>& database() const { return _db; }
private:
    std::shared_ptr<PiDatabase> _db;
};
} // namespace tile_compile::pi
