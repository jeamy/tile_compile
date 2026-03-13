#include "services/config_revisions.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {

using json = nlohmann::json;

std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string now_compact_utc() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
    return oss.str();
}

fs::path run_config_revisions_index_path(const fs::path& run_dir) {
    return run_config_revisions_dir(run_dir) / "index.json";
}

json read_run_revision_index(const fs::path& run_dir) {
    const fs::path index_path = run_config_revisions_index_path(run_dir);
    std::ifstream in(index_path);
    if (!in) return json::array();
    json parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_array()) return json::array();
    return parsed;
}

bool write_run_revision_index(const fs::path& run_dir, const json& index) {
    std::error_code ec;
    fs::create_directories(run_config_revisions_dir(run_dir), ec);
    std::ofstream out(run_config_revisions_index_path(run_dir), std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << index.dump(2);
    return static_cast<bool>(out);
}

std::string next_run_revision_id(const fs::path& run_dir) {
    const fs::path dir = run_config_revisions_dir(run_dir);
    const std::string stamp = now_compact_utc();
    for (int suffix = 0; suffix < 1000; ++suffix) {
        std::ostringstream oss;
        oss << "run_cfg_" << stamp;
        if (suffix > 0) oss << "_" << suffix;
        const std::string candidate = oss.str();
        if (!fs::exists(dir / (candidate + ".yaml"))) return candidate;
    }
    return "run_cfg_" + stamp + "_overflow";
}

ConfigRevision revision_from_index_entry(const json& item, const fs::path& run_dir) {
    ConfigRevision revision;
    revision.revision_id = item.value("revision_id", std::string());
    std::string file_name = item.value("file_name", revision.revision_id + ".yaml");
    revision.path = (run_config_revisions_dir(run_dir) / file_name).string();
    revision.source = item.value("source", std::string("run_config"));
    revision.created_at = item.value("created_at", std::string());
    if (item.contains("run_id") && item["run_id"].is_string()) {
        revision.run_id = item["run_id"].get<std::string>();
    }
    return revision;
}

} // namespace

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

fs::path run_config_revisions_dir(const fs::path& run_dir) {
    return run_dir / "artifacts" / "config_revisions";
}

std::string add_run_config_revision(const fs::path& run_dir,
                                    const std::string& yaml_text,
                                    const std::string& source,
                                    const std::optional<std::string>& run_id) {
    if (yaml_text.empty()) return "";

    std::error_code ec;
    fs::create_directories(run_config_revisions_dir(run_dir), ec);

    const std::string revision_id = next_run_revision_id(run_dir);
    const fs::path yaml_path = run_config_revisions_dir(run_dir) / (revision_id + ".yaml");
    std::ofstream yaml_out(yaml_path, std::ios::out | std::ios::trunc);
    if (!yaml_out) return "";
    yaml_out << yaml_text;
    yaml_out.close();
    if (!yaml_out) return "";

    json index = read_run_revision_index(run_dir);
    index.push_back({
        {"revision_id", revision_id},
        {"file_name", yaml_path.filename().string()},
        {"source", source},
        {"created_at", now_iso()},
        {"run_id", run_id.has_value() ? json(*run_id) : json(nullptr)},
    });
    if (!write_run_revision_index(run_dir, index)) return "";
    return revision_id;
}

std::vector<ConfigRevision> list_run_config_revisions(const fs::path& run_dir) {
    std::vector<ConfigRevision> revisions;
    json index = read_run_revision_index(run_dir);
    if (!index.is_array()) return revisions;
    revisions.reserve(index.size());
    for (const auto& item : index) {
        if (!item.is_object()) continue;
        ConfigRevision revision = revision_from_index_entry(item, run_dir);
        if (!revision.revision_id.empty()) revisions.push_back(std::move(revision));
    }
    std::reverse(revisions.begin(), revisions.end());
    return revisions;
}

std::optional<ConfigRevision> get_run_config_revision(const fs::path& run_dir,
                                                      const std::string& revision_id) {
    if (revision_id.empty()) return std::nullopt;
    json index = read_run_revision_index(run_dir);
    if (!index.is_array()) return std::nullopt;
    for (auto it = index.rbegin(); it != index.rend(); ++it) {
        if (!it->is_object()) continue;
        if (it->value("revision_id", std::string()) != revision_id) continue;
        ConfigRevision revision = revision_from_index_entry(*it, run_dir);
        std::ifstream in(revision.path);
        if (in) {
            revision.yaml_text.assign((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
        }
        return revision;
    }
    return std::nullopt;
}
