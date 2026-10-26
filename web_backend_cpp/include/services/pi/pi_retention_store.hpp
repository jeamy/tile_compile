#pragma once

#include "services/pi/pi_database.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::pi {

class PiRetentionStore {
public:
    explicit PiRetentionStore(std::filesystem::path storage_dir);

    nlohmann::json status() const;
    // Applies the fixed v1 policy to SQLite-backed PI data. Safe to repeat after failure/restart.
    nlohmann::json maintain(std::int64_t now_epoch_seconds) const;
    nlohmann::json create_backup(std::int64_t now_epoch_seconds) const;
    nlohmann::json restore_backup(const std::string& backup_id, bool confirmed,
                                  std::int64_t now_epoch_seconds) const;
    // Permanently removes PI data bound to a known run UID, preserving only a content-free tombstone.
    nlohmann::json forget_run(const std::string& run_uid, bool confirmed, std::int64_t now_epoch_seconds) const;
    nlohmann::json reset(const std::vector<std::string>& categories, bool confirmed,
                         std::int64_t now_epoch_seconds) const;

    static nlohmann::json policy();

    // Test seam: runs after the WAL checkpoint and before VACUUM. A throwing hook simulates a VACUUM failure.
    static void set_vacuum_hook_for_testing(std::function<void()> hook);

private:
    std::filesystem::path _storage_dir;
    std::shared_ptr<PiDatabase> _db;
};

} // namespace tile_compile::pi
