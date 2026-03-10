#pragma once
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <nlohmann/json.hpp>

struct ConfigRevision {
    std::string revision_id;
    std::string source;
    std::string yaml_text;
    std::string timestamp;
};

nlohmann::json config_revision_to_json(const ConfigRevision& r);

class ConfigRevisionStore {
public:
    std::string add(const std::string& yaml_text, const std::string& source = "save");
    std::optional<ConfigRevision> get(const std::string& revision_id) const;
    std::vector<ConfigRevision> list() const;
    int count() const;
    std::string latest_id() const;

private:
    mutable std::mutex _mutex;
    std::vector<ConfigRevision> _revisions;
    int _counter{0};
};
