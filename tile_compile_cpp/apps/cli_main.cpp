#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fitsio.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <openssl/sha.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string get_executable_dir() {
    return fs::current_path().string();
}

static std::string default_gui_state_path() {
    return (fs::path(get_executable_dir()) / "tile_compile_gui_state.json").string();
}

static void print_json(const json& j) {
    std::cout << j.dump(2) << std::endl;
}

static std::string read_file_text(const fs::path& p) {
    std::ifstream ifs(p);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static bool write_file_text(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p);
    if (!ofs) return false;
    ofs << content;
    return true;
}

static std::string read_stdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

// ============================================================================
// get-schema
// ============================================================================
int cmd_get_schema() {
    std::cout << tile_compile::config::get_schema_json() << std::endl;
    return 0;
}

// ============================================================================
// load-gui-state [--path <path>]
// ============================================================================
int cmd_load_gui_state(const std::string& path_arg) {
    fs::path p = path_arg.empty() ? default_gui_state_path() : path_arg;
    
    json state = json::object();
    if (fs::exists(p) && fs::is_regular_file(p)) {
        try {
            std::string raw = read_file_text(p);
            state = json::parse(raw);
            if (!state.is_object()) state = json::object();
        } catch (...) {
            state = json::object();
        }
    }
    
    json result;
    result["ok"] = true;
    result["path"] = p.string();
    result["state"] = state;
    print_json(result);
    return 0;
}

// ============================================================================
// save-gui-state [--path <path>] [--stdin | <json>]
// ============================================================================
int cmd_save_gui_state(const std::string& path_arg, const std::string& json_text, bool use_stdin) {
    fs::path p = path_arg.empty() ? default_gui_state_path() : path_arg;
    
    std::string raw = use_stdin ? read_stdin() : json_text;
    if (raw.empty()) {
        std::cerr << "save-gui-state requires JSON text either as argument or via --stdin\n";
        return 2;
    }
    
    json obj;
    try {
        obj = json::parse(raw);
    } catch (const std::exception& e) {
        std::cerr << "save-gui-state: failed to parse JSON: " << e.what() << "\n";
        return 2;
    }
    
    if (!obj.is_object()) {
        std::cerr << "save-gui-state: state must be a JSON object\n";
        return 2;
    }
    
    if (!write_file_text(p, obj.dump(2) + "\n")) {
        std::cerr << "save-gui-state: failed to write file: " << p.string() << "\n";
        return 1;
    }
    
    json result;
    result["ok"] = true;
    result["path"] = p.string();
    result["saved"] = true;
    print_json(result);
    return 0;
}

// ============================================================================
// load-config <path>
// ============================================================================
int cmd_load_config(const std::string& path) {
    fs::path p(path);
    if (!fs::exists(p)) {
        json result;
        result["ok"] = false;
        result["error"] = "File not found: " + path;
        print_json(result);
        return 1;
    }
    
    std::string yaml_text = read_file_text(p);
    json result;
    result["path"] = path;
    result["yaml"] = yaml_text;
    print_json(result);
    return 0;
}

// ============================================================================
// save-config <path> [--stdin | <yaml>]
// ============================================================================
int cmd_save_config(const std::string& path, const std::string& yaml_text, bool use_stdin) {
    std::string content = use_stdin ? read_stdin() : yaml_text;
    if (content.empty()) {
        std::cerr << "save-config requires YAML text either as argument or via --stdin\n";
        return 2;
    }
    
    if (!write_file_text(path, content)) {
        std::cerr << "save-config: failed to write file: " << path << "\n";
        return 1;
    }
    
    json result;
    result["path"] = path;
    result["saved"] = true;
    print_json(result);
    return 0;
}

// ============================================================================
// validate-config --path <path> | --yaml <yaml> | --stdin
// ============================================================================
int cmd_validate_config(const std::string& path, const std::string& yaml_arg, bool use_stdin, bool strict_exit) {
    std::string yaml_text;
    if (!path.empty()) {
        yaml_text = read_file_text(path);
    } else if (use_stdin) {
        yaml_text = read_stdin();
    } else {
        yaml_text = yaml_arg;
    }
    
    json result;
    result["valid"] = false;
    result["errors"] = json::array();
    result["warnings"] = json::array();
    if (!path.empty()) result["path"] = path;
    
    try {
        YAML::Node node = YAML::Load(yaml_text);
        tile_compile::config::Config cfg = tile_compile::config::Config::from_yaml(node);
        cfg.validate();
        result["valid"] = true;
    } catch (const std::exception& e) {
        result["errors"].push_back(e.what());
    }
    
    print_json(result);
    if (strict_exit) {
        return result["valid"].get<bool>() ? 0 : 1;
    }
    return 0;
}

// ============================================================================
// scan <input_path> [--frames-min N] [--with-checksums]
// ============================================================================
static std::vector<fs::path> find_fits_files(const fs::path& dir) {
    std::vector<fs::path> files;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return files;
    
    std::regex fits_regex(R"(.*\.(fit|fits|fts)$)", std::regex::icase);
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (std::regex_match(name, fits_regex)) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int cmd_scan(const std::string& input_path, int frames_min, bool with_checksums) {
    fs::path p(input_path);
    
    json result;
    result["ok"] = false;
    result["input_path"] = input_path;
    result["frames_detected"] = 0;
    result["frames"] = json::array();
    result["errors"] = json::array();
    result["warnings"] = json::array();
    result["color_mode"] = "UNKNOWN";
    result["bayer_pattern"] = "";
    result["color_mode_candidates"] = json::array();
    result["requires_user_confirmation"] = false;
    
    if (!fs::exists(p)) {
        result["errors"].push_back("Input path does not exist: " + input_path);
        print_json(result);
        return 0;
    }
    
    if (!fs::is_directory(p)) {
        result["errors"].push_back("Input path is not a directory: " + input_path);
        print_json(result);
        return 0;
    }
    
    auto files = find_fits_files(p);
    result["frames_detected"] = static_cast<int>(files.size());
    
    // Detect color mode from FITS headers
    std::map<std::string, int> bayerpat_counts;
    std::string first_bayerpat;
    bool bayerpat_consistent = true;
    
    for (const auto& f : files) {
        json frame;
        frame["path"] = f.string();
        frame["filename"] = f.filename().string();
        frame["size_bytes"] = static_cast<int64_t>(fs::file_size(f));
        
        // Try to read BAYERPAT from FITS header (simplified - would need cfitsio)
        // For now, we'll detect based on filename patterns or assume MONO
        // This is a placeholder - real implementation needs FITS header reading
        
        result["frames"].push_back(frame);
    }
    
    // Determine color mode
    if (!files.empty()) {
        // For now, default to MONO with user confirmation
        // Real implementation would read FITS headers
        result["color_mode"] = "MONO";
        result["color_mode_candidates"].push_back("MONO");
        result["color_mode_candidates"].push_back("RGB");
        result["color_mode_candidates"].push_back("RGGB");
        result["color_mode_candidates"].push_back("BGGR");
        result["color_mode_candidates"].push_back("GRBG");
        result["color_mode_candidates"].push_back("GBRG");
        result["requires_user_confirmation"] = true;
        result["warnings"].push_back("Color mode detection not fully implemented - please confirm manually");
    }
    
    if (static_cast<int>(files.size()) < frames_min) {
        result["errors"].push_back("Not enough frames: found " + std::to_string(files.size()) + 
                                   ", minimum required: " + std::to_string(frames_min));
    }
    
    if (result["errors"].empty()) {
        result["ok"] = true;
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// list-runs <runs_dir>
// ============================================================================
int cmd_list_runs(const std::string& runs_dir) {
    fs::path p(runs_dir);
    
    json result;
    result["runs_dir"] = runs_dir;
    result["runs"] = json::array();
    
    if (!fs::exists(p) || !fs::is_directory(p)) {
        print_json(result);
        return 0;
    }
    
    for (const auto& entry : fs::directory_iterator(p)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            // Check if it looks like a run directory (has run.log or events.jsonl)
            if (fs::exists(entry.path() / "run.log") || fs::exists(entry.path() / "events.jsonl")) {
                json run;
                run["name"] = name;
                run["path"] = entry.path().string();
                
                // Try to get modification time
                auto ftime = fs::last_write_time(entry.path());
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                auto time_t_val = std::chrono::system_clock::to_time_t(sctp);
                std::ostringstream oss;
                oss << std::put_time(std::localtime(&time_t_val), "%Y-%m-%d %H:%M:%S");
                run["modified"] = oss.str();
                
                result["runs"].push_back(run);
            }
        }
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// get-run-status <run_dir>
// ============================================================================
int cmd_get_run_status(const std::string& run_dir) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["exists"] = fs::exists(p);
    result["status"] = "unknown";
    result["current_phase"] = nullptr;
    result["progress"] = 0;
    result["events"] = json::array();
    
    if (!fs::exists(p)) {
        print_json(result);
        return 0;
    }
    
    // Read events.jsonl if exists
    fs::path events_file = p / "events.jsonl";
    if (fs::exists(events_file)) {
        std::ifstream ifs(events_file);
        std::string line;
        std::string last_phase;
        std::string last_status;
        
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            try {
                json ev = json::parse(line);
                result["events"].push_back(ev);
                
                if (ev.contains("type")) {
                    std::string type = ev["type"].get<std::string>();
                    if (type == "phase_start" && ev.contains("phase")) {
                        last_phase = ev["phase"].get<std::string>();
                        last_status = "running";
                    } else if (type == "phase_end" && ev.contains("status")) {
                        last_status = ev["status"].get<std::string>();
                    } else if (type == "run_end") {
                        if (ev.contains("success") && ev["success"].get<bool>()) {
                            last_status = "completed";
                        } else {
                            last_status = "failed";
                        }
                    }
                }
            } catch (...) {
                // Skip malformed lines
            }
        }
        
        result["current_phase"] = last_phase.empty() ? nullptr : json(last_phase);
        result["status"] = last_status.empty() ? "unknown" : last_status;
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// get-run-logs <run_dir> [--tail N]
// ============================================================================
int cmd_get_run_logs(const std::string& run_dir, int tail) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["log_lines"] = json::array();
    
    fs::path log_file = p / "run.log";
    if (!fs::exists(log_file)) {
        print_json(result);
        return 0;
    }
    
    std::vector<std::string> lines;
    std::ifstream ifs(log_file);
    std::string line;
    while (std::getline(ifs, line)) {
        lines.push_back(line);
    }
    
    if (tail > 0 && static_cast<int>(lines.size()) > tail) {
        lines.erase(lines.begin(), lines.end() - tail);
    }
    
    for (const auto& l : lines) {
        result["log_lines"].push_back(l);
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// list-artifacts <run_dir>
// ============================================================================
int cmd_list_artifacts(const std::string& run_dir) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["artifacts"] = json::array();
    
    if (!fs::exists(p) || !fs::is_directory(p)) {
        print_json(result);
        return 0;
    }
    
    std::regex artifact_regex(R"(.*\.(fit|fits|fts|png|jpg|jpeg|tif|tiff|json|yaml|yml)$)", std::regex::icase);
    
    std::function<void(const fs::path&)> scan_dir = [&](const fs::path& dir) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                if (std::regex_match(name, artifact_regex)) {
                    json art;
                    art["path"] = entry.path().string();
                    art["filename"] = name;
                    art["size_bytes"] = static_cast<int64_t>(fs::file_size(entry.path()));
                    art["relative_path"] = fs::relative(entry.path(), p).string();
                    result["artifacts"].push_back(art);
                }
            } else if (entry.is_directory()) {
                scan_dir(entry.path());
            }
        }
    };
    
    scan_dir(p);
    print_json(result);
    return 0;
}

// ============================================================================
// Main
// ============================================================================
void print_usage() {
    std::cout << "Usage: tile_compile_cli <command> [options]\n"
              << "\nCommands:\n"
              << "  get-schema                      Print JSON schema for config\n"
              << "  load-gui-state [--path P]       Load GUI state from file\n"
              << "  save-gui-state [--path P] [--stdin | JSON]  Save GUI state\n"
              << "  load-config <path>              Load config YAML file\n"
              << "  save-config <path> [--stdin | YAML]  Save config YAML file\n"
              << "  validate-config (--path P | --yaml Y | --stdin)  Validate config\n"
              << "  scan <input_path> [--frames-min N]  Scan input directory for frames\n"
              << "  list-runs <runs_dir>            List pipeline runs\n"
              << "  get-run-status <run_dir>        Get status of a run\n"
              << "  get-run-logs <run_dir> [--tail N]  Get run logs\n"
              << "  list-artifacts <run_dir>        List artifacts in run directory\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    // Helper to find argument value
    auto get_arg = [&](const char* name, const char* short_name = nullptr) -> std::string {
        for (int i = 2; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], name) == 0 || (short_name && std::strcmp(argv[i], short_name) == 0)) {
                return argv[i + 1];
            }
        }
        return "";
    };
    
    auto has_flag = [&](const char* name) -> bool {
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], name) == 0) return true;
        }
        return false;
    };
    
    auto get_positional = [&](int pos) -> std::string {
        int count = 0;
        for (int i = 2; i < argc; ++i) {
            if (argv[i][0] != '-') {
                if (count == pos) return argv[i];
                ++count;
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                ++i; // Skip argument value
            }
        }
        return "";
    };
    
    if (command == "get-schema") {
        return cmd_get_schema();
    }
    
    if (command == "load-gui-state") {
        return cmd_load_gui_state(get_arg("--path"));
    }
    
    if (command == "save-gui-state") {
        return cmd_save_gui_state(get_arg("--path"), get_positional(0), has_flag("--stdin"));
    }
    
    if (command == "load-config") {
        std::string path = get_positional(0);
        if (path.empty()) {
            std::cerr << "load-config requires a path argument\n";
            return 1;
        }
        return cmd_load_config(path);
    }
    
    if (command == "save-config") {
        std::string path = get_positional(0);
        if (path.empty()) {
            std::cerr << "save-config requires a path argument\n";
            return 1;
        }
        return cmd_save_config(path, get_positional(1), has_flag("--stdin"));
    }
    
    if (command == "validate-config") {
        std::string path = get_arg("--path");
        std::string yaml = get_arg("--yaml");
        bool use_stdin = has_flag("--stdin");
        bool strict = has_flag("--strict-exit-codes");
        
        if (path.empty() && yaml.empty() && !use_stdin) {
            std::cerr << "validate-config requires --path, --yaml, or --stdin\n";
            return 1;
        }
        return cmd_validate_config(path, yaml, use_stdin, strict);
    }
    
    if (command == "scan") {
        std::string input_path = get_positional(0);
        if (input_path.empty()) {
            std::cerr << "scan requires an input_path argument\n";
            return 1;
        }
        std::string frames_min_str = get_arg("--frames-min");
        int frames_min = frames_min_str.empty() ? 1 : std::stoi(frames_min_str);
        bool with_checksums = has_flag("--with-checksums");
        return cmd_scan(input_path, frames_min, with_checksums);
    }
    
    if (command == "list-runs") {
        std::string runs_dir = get_positional(0);
        if (runs_dir.empty()) {
            std::cerr << "list-runs requires a runs_dir argument\n";
            return 1;
        }
        return cmd_list_runs(runs_dir);
    }
    
    if (command == "get-run-status") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "get-run-status requires a run_dir argument\n";
            return 1;
        }
        return cmd_get_run_status(run_dir);
    }
    
    if (command == "get-run-logs") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "get-run-logs requires a run_dir argument\n";
            return 1;
        }
        std::string tail_str = get_arg("--tail");
        int tail = tail_str.empty() ? 0 : std::stoi(tail_str);
        return cmd_get_run_logs(run_dir, tail);
    }
    
    if (command == "list-artifacts") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "list-artifacts requires a run_dir argument\n";
            return 1;
        }
        return cmd_list_artifacts(run_dir);
    }
    
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
