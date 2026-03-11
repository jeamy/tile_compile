#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct ConfigRevision {
    std::string revision_id;
    std::string path;
    std::string source;
    std::string created_at;
    std::optional<std::string> run_id;
    std::string yaml_text;
};

nlohmann::json config_revision_to_json(const ConfigRevision& r);

class ConfigRevisionStore {
public:
    std::string add(const fs::path& path,
                    const std::string& yaml_text,
                    const std::string& source = "save",
                    const std::optional<std::string>& run_id = std::nullopt);
    std::optional<ConfigRevision> get(const std::string& revision_id) const;
    std::vector<ConfigRevision> list() const;
    int count() const;
    std::string latest_id() const;

private:
    mutable std::mutex _mutex;
    std::vector<ConfigRevision> _revisions;
    int _counter{0};
};
