#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

/// @brief Stored snapshot of a YAML configuration revision.
/// @details Revisions are kept both for UI edits and per-run configuration history so clients
/// can list, inspect, and restore previous configuration text.
struct ConfigRevision {
    std::string revision_id;
    std::string path;
    std::string source;
    std::string created_at;
    std::optional<std::string> run_id;
    std::string yaml_text;
};

/// @brief Serializes a configuration revision for REST responses.
nlohmann::json config_revision_to_json(const ConfigRevision& r);

/// @brief Thread-safe in-memory store for live configuration editor revisions.
/// @details Complements the per-run on-disk revision helpers below; it tracks recent edits made
/// through the backend before or while a run is created.
class ConfigRevisionStore {
public:
    /// @brief Adds a revision snapshot and returns the generated revision id.
    std::string add(const fs::path& path,
                    const std::string& yaml_text,
                    const std::string& source = "save",
                    const std::optional<std::string>& run_id = std::nullopt);
    /// @brief Fetches a revision by id.
    std::optional<ConfigRevision> get(const std::string& revision_id) const;
    /// @brief Lists revisions in insertion order.
    std::vector<ConfigRevision> list() const;
    /// @brief Returns the number of stored in-memory revisions.
    int count() const;
    /// @brief Returns the newest revision id, or an empty string when the store is empty.
    std::string latest_id() const;

private:
    mutable std::mutex _mutex;
    std::vector<ConfigRevision> _revisions;
    int _counter{0};
};

/// @brief Returns the directory used for on-disk revisions inside a run directory.
fs::path run_config_revisions_dir(const fs::path& run_dir);

/// @brief Writes a per-run configuration revision and updates the run revision index.
std::string add_run_config_revision(const fs::path& run_dir,
                                    const std::string& yaml_text,
                                    const std::string& source = "run_config",
                                    const std::optional<std::string>& run_id = std::nullopt);

/// @brief Reads all persisted configuration revisions for a run.
std::vector<ConfigRevision> list_run_config_revisions(const fs::path& run_dir);

/// @brief Loads one persisted run configuration revision by id.
std::optional<ConfigRevision> get_run_config_revision(const fs::path& run_dir,
                                                      const std::string& revision_id);
