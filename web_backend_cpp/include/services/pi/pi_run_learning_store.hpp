#pragma once
#include "services/pi/pi_database.hpp"
#include <optional>

namespace tile_compile::pi {

// Local run archive, separate from metadata-only memory exports. Never copies FITS pixels.
class PiRunLearningStore {
public:
    explicit PiRunLearningStore(const std::filesystem::path& storage_dir);
    nlohmann::json capture(const std::filesystem::path& run_dir, const std::string& run_id,
                          const std::string& stage, const std::string& status);
    std::optional<nlohmann::json> get(const std::string& run_uid);
    nlohmann::json history(const std::string& run_uid, int limit = 50);
    nlohmann::json history_summaries(const std::string& run_uid, int limit = 50);
    std::optional<nlohmann::json> snapshot(const std::string& run_uid, const std::string& snapshot_id);
    nlohmann::json list(int limit = 100);
    bool mark_artifacts_state(const std::string& run_uid, const std::string& state);
    bool mark_artifacts_deleted(const std::string& run_uid);
    bool set_excluded(const std::string& run_uid, bool excluded, const std::string& reason_code);
    // PNG only, <=1024 pixels per edge and <=2 MiB. Associated with the exact snapshot.
    void save_preview(const std::string& run_uid, const std::string& snapshot_id,
                      const std::vector<unsigned char>& png, const std::string& source_artifact = "");
    std::optional<std::vector<unsigned char>> preview(const std::string& run_uid);
private:
    std::shared_ptr<PiDatabase> _db;
};
} // namespace tile_compile::pi
