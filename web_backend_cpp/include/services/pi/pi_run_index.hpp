#pragma once
#include "services/pi/pi_database.hpp"
#include <optional>
#include <stdexcept>

namespace tile_compile::pi {
class PiRunIdentityConflict : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class PiRunIndex {
public:
    explicit PiRunIndex(const std::filesystem::path& storage_dir);
    // Exact normalized paths and an explicit provenance UID are authoritative.
    // Historical runs receive a central UID only; their artifacts are not modified.
    nlohmann::json resolve(const std::filesystem::path& run_dir, const std::string& provenance_uid = "",
                          const std::string& config_sha256 = "", const std::string& started_at = "");
    // Explicit user relink of a known identity. Keeps old aliases; never merges two UIDs.
    nlohmann::json relink(const std::string& run_uid, const std::filesystem::path& target,
                          const std::string& target_provenance_uid = "");
    std::optional<nlohmann::json> get(const std::string& run_uid);
    nlohmann::json candidates(const std::string& config_sha256, const std::string& started_at);
    static std::string run_key(const std::filesystem::path& path);
    static std::string generate_uid();
private:
    std::shared_ptr<PiDatabase> _db;
};
} // namespace tile_compile::pi
