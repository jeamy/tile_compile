#include "services/config_revisions.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

nlohmann::json config_revision_to_json(const ConfigRevision& r) {
    return {
        {"revision_id", r.revision_id},
        {"source",      r.source},
        {"timestamp",   r.timestamp},
    };
}

static std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string ConfigRevisionStore::add(const std::string& yaml_text, const std::string& source) {
    std::lock_guard<std::mutex> lk(_mutex);
    ConfigRevision r;
    r.revision_id = "rev_" + std::to_string(++_counter);
    r.source      = source;
    r.yaml_text   = yaml_text;
    r.timestamp   = now_iso();
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
