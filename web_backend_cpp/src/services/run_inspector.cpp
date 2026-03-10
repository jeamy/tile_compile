#include "services/run_inspector.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

nlohmann::json read_run_status(const fs::path& run_dir) {
    fs::path status_file = run_dir / "status.json";
    if (!fs::exists(status_file)) return {{"status", "unknown"}};
    std::ifstream f(status_file);
    if (!f) return {{"status", "unknown"}};
    try {
        return nlohmann::json::parse(f);
    } catch (...) {
        return {{"status", "unknown"}};
    }
}

std::vector<nlohmann::json> discover_runs(const fs::path& runs_dir, int limit) {
    std::vector<nlohmann::json> result;
    if (!fs::is_directory(runs_dir)) return result;

    std::vector<fs::path> dirs;
    for (auto& entry : fs::directory_iterator(runs_dir)) {
        if (entry.is_directory()) dirs.push_back(entry.path());
    }
    std::sort(dirs.begin(), dirs.end(), [](const fs::path& a, const fs::path& b) {
        return fs::last_write_time(a) > fs::last_write_time(b);
    });

    int count = 0;
    for (auto& d : dirs) {
        if (count >= limit) break;
        auto status = read_run_status(d);
        std::string run_id = d.filename().string();
        nlohmann::json item = {
            {"run_id",        run_id},
            {"name",          run_id},
            {"status",        status.value("status", "unknown")},
            {"current_phase", status.value("current_phase", nullptr)},
            {"progress",      status.value("progress", 0.0)},
            {"run_dir",       d.string()},
        };
        result.push_back(item);
        ++count;
    }
    return result;
}

std::string read_run_logs(const fs::path& run_dir, int tail) {
    fs::path log_file = run_dir / "runner.log";
    if (!fs::exists(log_file)) return "";
    std::ifstream f(log_file);
    if (!f) return "";
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(line);
    int start = (int)lines.size() - tail;
    if (start < 0) start = 0;
    std::ostringstream oss;
    for (int i = start; i < (int)lines.size(); ++i)
        oss << lines[i] << "\n";
    return oss.str();
}

nlohmann::json list_run_artifacts(const fs::path& run_dir) {
    nlohmann::json items = nlohmann::json::array();
    if (!fs::is_directory(run_dir)) return items;

    static const std::vector<std::string> ARTIFACT_EXTS = {
        ".json", ".jsonl", ".html", ".yaml", ".yml", ".png", ".fits", ".log"
    };

    std::function<void(const fs::path&, const std::string&)> scan =
        [&](const fs::path& dir, const std::string& prefix) {
            for (auto& entry : fs::directory_iterator(dir)) {
                std::string name = entry.path().filename().string();
                std::string rel = prefix.empty() ? name : prefix + "/" + name;
                if (entry.is_directory()) {
                    scan(entry.path(), rel);
                } else if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    for (auto& e : ARTIFACT_EXTS) {
                        if (ext == e) {
                            items.push_back({
                                {"path",     rel},
                                {"name",     name},
                                {"size",     (int64_t)fs::file_size(entry.path())},
                            });
                            break;
                        }
                    }
                }
            }
        };
    scan(run_dir, "");
    return items;
}
