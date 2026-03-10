#include "services/config_revisions.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

nlohmann::json config_revision_to_json(const ConfigRevision& r) {
    return {
        {"revision_id", r.revision_id},
        {"path",        r.path},
        {"created_at",  r.created_at},
        {"source",      r.source},
        {"run_id",      r.run_id.has_value() ? nlohmann::json(*r.run_id) : nlohmann::json(nullptr)},
        {"has_snapshot", !r.yaml_text.empty()},
    };
}

static std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string ConfigRevisionStore::add(const fs::path& path,
                                     const std::string& yaml_text,
                                     const std::string& source,
                                     const std::optional<std::string>& run_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    ConfigRevision r;
    r.revision_id = "cfg_" + std::to_string(++_counter);
    r.path = path.string();
    r.source = source;
    r.created_at = now_iso();
    r.run_id = run_id;
    r.yaml_text = yaml_text;
    _revisions.push_back(r);
    return r.revision_id;
}

std::optional<ConfigRevision> ConfigRevisionStore::get(const std::string& revision_id) const {
    std::lock_guard<std::mutex> lk(_mutex);
    for (auto it = _revisions.rbegin(); it != _revisions.rend(); ++it)
        if (it->revision_id == revision_id) return *it;
    return std::nullopt;
}

std::vector<ConfigRevision> ConfigRevisionStore::list() const {
    std::lock_guard<std::mutex> lk(_mutex);
    auto copy = _revisions;
    std::reverse(copy.begin(), copy.end());
    return copy;
}

int ConfigRevisionStore::count() const {
    std::lock_guard<std::mutex> lk(_mutex);
    return (int)_revisions.size();
}

std::string ConfigRevisionStore::latest_id() const {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_revisions.empty()) return "";
    return _revisions.back().revision_id;
}
