#include "services/pi/pi_decision_outcome.hpp"

#include "services/pi/pi_decision_policy.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <vector>

namespace tile_compile::pi {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

json yaml_to_json(const YAML::Node& n) {
    switch (n.Type()) {
        case YAML::NodeType::Null: return nullptr;
        case YAML::NodeType::Map: {
            json o = json::object();
            for (const auto& kv : n) o[kv.first.as<std::string>()] = yaml_to_json(kv.second);
            return o;
        }
        case YAML::NodeType::Sequence: {
            json a = json::array();
            for (const auto& v : n) a.push_back(yaml_to_json(v));
            return a;
        }
        case YAML::NodeType::Scalar: {
            const std::string s = n.Scalar();
            if (n.Tag() == "!") return s;  // quoted scalar stays a string
            if (s == "true" || s == "True" || s == "TRUE") return true;
            if (s == "false" || s == "False" || s == "FALSE") return false;
            if (s == "null" || s == "~") return nullptr;
            static const std::regex kInt(R"(^[-+]?[0-9]+$)");
            static const std::regex kFloat(R"(^[-+]?([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][-+]?[0-9]+)?$)");
            if (std::regex_match(s, kInt)) return std::stoll(s);
            if (std::regex_match(s, kFloat)) return std::stod(s);
            return s;
        }
        default: return nullptr;
    }
}

std::optional<json> read_json(const fs::path& p) {
    std::ifstream in(p);
    if (!in) return std::nullopt;
    json j = json::parse(in, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    return j;
}

void write_json_atomic(const fs::path& p, const json& j) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        out << j.dump(2);
    }
    fs::rename(tmp, p);
}

bool safe_run_id(const std::string& id) {
    return !id.empty() && id.find('/') == std::string::npos && id.find('\\') == std::string::npos && id.find("..") == std::string::npos;
}

} // namespace

json yaml_text_to_json(const std::string& yaml_text) { return yaml_to_json(YAML::Load(yaml_text)); }

json load_run_config_yaml(const fs::path& run_dir) {
    const fs::path p = run_dir / "config.yaml";
    if (!fs::is_regular_file(p)) throw std::runtime_error("run config.yaml not found");
    return yaml_to_json(YAML::LoadFile(p.string()));
}

json record_jev_outcome_if_needed(const fs::path& decisions_dir, const std::string& run_id, const fs::path& run_dir,
                                  const RunConfigLoader& loader) {
    if (!safe_run_id(run_id)) return {{"terminal", true}, {"reason", "invalid_run_id"}};
    const fs::path marker_path = decisions_dir / "_run_markers" / (run_id + ".json");
    if (const auto existing = read_json(marker_path); existing && existing->value("terminal", false)) return *existing;

    auto write_marker = [&](const json& m) {
        try { write_json_atomic(marker_path, m); } catch (const std::exception&) {}
        return m;
    };

    const auto provenance = read_json(run_dir / "artifacts" / "pi_run_provenance.json");
    if (!provenance) return write_marker({{"run_id", run_id}, {"terminal", true}, {"reason", "no_provenance"}});
    const std::string started_at = provenance->value("started_at", std::string());
    if (started_at.empty()) return write_marker({{"run_id", run_id}, {"terminal", true}, {"reason", "no_provenance"}});

    // Applied proposals that predate the run.
    std::vector<std::pair<std::string, json>> proposals;
    std::error_code ec;
    if (fs::is_directory(decisions_dir, ec)) {
        std::vector<fs::path> dirs;
        for (const auto& e : fs::directory_iterator(decisions_dir, ec))
            if (e.is_directory() && e.path().filename().string().rfind("_", 0) != 0) dirs.push_back(e.path());
        std::sort(dirs.begin(), dirs.end());
        for (const auto& d : dirs) {
            const auto prop = read_json(d / "proposal.json");
            if (!prop || prop->value("status", std::string()) != "applied_to_draft") continue;
            if (!prop->contains("updates") || !(*prop)["updates"].is_array() || (*prop)["updates"].empty()) continue;
            const std::string applied_at = prop->value("applied_at", std::string());
            // ISO-8601 UTC strings of identical shape compare correctly as text.
            if (applied_at.empty() || applied_at.size() != started_at.size() || applied_at > started_at) continue;
            proposals.emplace_back(d.filename().string(), *prop);
        }
    }
    if (proposals.empty()) return write_marker({{"run_id", run_id}, {"terminal", true}, {"reason", "no_applied_proposals"}});

    json run_config;
    try {
        run_config = loader(run_dir);
    } catch (const std::exception& e) {
        return write_marker({{"run_id", run_id}, {"terminal", false}, {"reason", "config_unreadable"}, {"error", e.what()}});
    }

    json recorded = json::array(), skipped = json::array();
    try {
        for (const auto& [dir_name, prop] : proposals) {
            json checks = json::array();
            int present = 0;
            for (const auto& u : prop["updates"]) {
                const std::string path = u.value("path", std::string());
                const json actual = config_get(run_config, path);
                const bool ok = u.contains("value") && json_values_equal(actual, u["value"]);
                if (ok) ++present;
                checks.push_back({{"path", path}, {"expected", u.value("value", json(nullptr))}, {"actual", actual}, {"value_present", ok}});
            }
            const int total = static_cast<int>(prop["updates"].size());
            const std::string attribution = present == total ? "paths_present" : (present > 0 ? "paths_partial" : "paths_absent");
            if (attribution == "paths_absent") {
                skipped.push_back({{"proposal_id", dir_name}, {"attribution", attribution}});
                continue;
            }
            const fs::path outcome_path = decisions_dir / dir_name / "outcome.json";
            json outcome = read_json(outcome_path).value_or(json{{"schema_version", "pi.decision-outcome.v1"},
                                                                  {"proposal_id", dir_name}, {"runs", json::array()}});
            bool have = false;
            for (const auto& r : outcome["runs"]) have = have || r.value("run_id", std::string()) == run_id;
            if (!have) {
                outcome["runs"].push_back({{"run_id", run_id},
                                           {"config_revision_id", provenance->value("config_revision_id", std::string())},
                                           {"attribution", attribution},
                                           {"path_checks", checks},
                                           {"comparison_kind", "unpaired"},
                                           {"quality_delta", nullptr}});
                write_json_atomic(outcome_path, outcome);
            }
            recorded.push_back({{"proposal_id", dir_name}, {"attribution", attribution}});
        }
    } catch (const std::exception& e) {
        return write_marker({{"run_id", run_id}, {"terminal", false}, {"reason", "error"}, {"error", e.what()}});
    }
    return write_marker({{"run_id", run_id}, {"terminal", true}, {"reason", "recorded"}, {"recorded", recorded}, {"skipped", skipped}});
}

} // namespace tile_compile::pi
