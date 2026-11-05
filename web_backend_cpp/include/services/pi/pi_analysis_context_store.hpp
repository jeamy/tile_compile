#pragma once
#include "services/pi/pi_decision_service.hpp"

namespace tile_compile::pi {
// Authoritative scan identities/revisions; not a Decision Record or transcript store.
class PiAnalysisContextStore {
public:
    explicit PiAnalysisContextStore(const std::filesystem::path& storage_dir);
    nlohmann::json resolve(const AdviceRequest& request);
    std::optional<nlohmann::json> get(const std::string& uid) const;
    nlohmann::json revisions(const std::string& uid) const;
    void link(const std::string& uid, const std::string& revision, const std::string& proposal);
    nlohmann::json proposals(const std::string& uid) const;
private:
    std::shared_ptr<PiDatabase> db_;
};
} // namespace tile_compile::pi
