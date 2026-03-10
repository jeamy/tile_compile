#include "routes/tools_routes.hpp"
#include "services/download_manager.hpp"
#include "subprocess_manager.hpp"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <cstdlib>
#include <chrono>

namespace fs = std::filesystem;

namespace {

constexpr const char* ASTAP_CLI_URL = "https://sourceforge.net/projects/astap-program/files/linux_installer/astap_command-line_version_Linux_amd64.zip/download";
constexpr const char* ASTAP_SF_BASE = "https://sourceforge.net/projects/astap-program/files/star_databases/";
constexpr int SIRIL_NUM_CHUNKS = 48;
constexpr const char* SIRIL_URL_TEMPLATE_PREFIX = "https://zenodo.org/records/14738271/files/siril_cat1_healpix8_xpsamp_";
constexpr const char* SIRIL_URL_TEMPLATE_SUFFIX = ".dat.bz2?download=1";

const std::map<std::string, std::string> ASTAP_CATALOGS = {
    {"d05", "d05_star_database.zip"},
    {"d20", "d20_star_database.zip"},
    {"d50", "d50_star_database.zip"},
    {"d80", "d80_star_database.deb"},
};

std::string getenv_or(const char* name, const std::string& fallback = "") {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}

fs::path user_home_dir() {
#ifdef _WIN32
    auto home = getenv_or("USERPROFILE");
#else
    auto home = getenv_or("HOME");
#endif
    return home.empty() ? fs::current_path() : fs::path(home);
}

fs::path default_astap_data_dir() {
    return user_home_dir() / ".local" / "share" / "tile_compile" / "astap";
}

fs::path default_siril_catalog_dir() {
    return user_home_dir() / ".local" / "share" / "siril" / "siril_cat1_healpix8_xpsamp";
}

bool is_astap_catalog_installed(const fs::path& catalog_dir, const std::string& catalog_id) {
    if (!fs::exists(catalog_dir)) return false;
    const std::string prefix = catalog_id + "_";
    for (const auto& entry : fs::directory_iterator(catalog_dir)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

std::vector<int> missing_siril_chunks(const fs::path& catalog_dir) {
    std::vector<int> missing;
    for (int i = 0; i < SIRIL_NUM_CHUNKS; ++i) {
        if (!fs::exists(catalog_dir / ("siril_cat1_healpix8_xpsamp_" + std::to_string(i) + ".dat"))) {
            missing.push_back(i);
        }
    }
    return missing;
}

std::string siril_chunk_url(int chunk) {
    return std::string(SIRIL_URL_TEMPLATE_PREFIX) + std::to_string(chunk) + SIRIL_URL_TEMPLATE_SUFFIX;
}

fs::path find_astap_candidate(const fs::path& root) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name == "astap_cli" || name == "astap_cli.exe" || name == "astap") return entry.path();
    }
    return {};
}

bool extract_zip_archive(const fs::path& archive, const fs::path& dest, std::string& error) {
#ifdef _WIN32
    auto res = run_subprocess({"powershell", "-NoProfile", "-Command",
        "Expand-Archive -LiteralPath '" + archive.string() + "' -DestinationPath '" + dest.string() + "' -Force"});
#else
    auto res = run_subprocess({"unzip", "-o", archive.string(), "-d", dest.string()});
#endif
    if (res.exit_code != 0) {
        error = res.stderr_str.empty() ? res.stdout_str : res.stderr_str;
        return false;
    }
    return true;
}

bool extract_deb_archive(const fs::path& archive, const fs::path& dest, std::string& error) {
#ifdef _WIN32
    error = "deb extraction unsupported on Windows";
    return false;
#else
    auto res = run_subprocess({"dpkg-deb", "-x", archive.string(), dest.string()});
    if (res.exit_code != 0) {
        error = res.stderr_str.empty() ? res.stdout_str : res.stderr_str;
        return false;
    }
    return true;
#endif
}

bool decompress_bz2_archive(const fs::path& archive, std::string& error) {
#ifdef _WIN32
    auto res = run_subprocess({"bzip2", "-d", "-f", archive.string()});
#else
    auto res = run_subprocess({"bzip2", "-d", "-f", archive.string()});
#endif
    if (res.exit_code != 0) {
        error = res.stderr_str.empty() ? res.stdout_str : res.stderr_str;
        return false;
    }
    return true;
}

size_t curl_discard_write(char* ptr, size_t size, size_t nmemb, void*) {
    return size * nmemb;
}

} // namespace

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    return json_resp({{"error", {{"message", msg}}}}, status);
}
static crow::response err_resp(const std::string& code,
                               const std::string& msg,
                               int status,
                               const nlohmann::json& details) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

static std::optional<std::string> denied_path(const BackendRuntime& runtime,
                                              const fs::path& access_path,
                                              const fs::path& reported_path = {}) {
    if (access_path.empty() || access_path.is_relative()) return std::nullopt;
    if (runtime.is_path_allowed(access_path)) return std::nullopt;
    fs::path path_for_error = reported_path.empty() ? access_path : reported_path;
    return path_for_error.string();
}

// Shared cancel flags per job (simple global map protected by mutex)
static std::mutex g_cancel_mutex;
static std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> g_cancel_flags;

static std::shared_ptr<std::atomic<bool>> get_or_create_flag(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_cancel_mutex);
    auto it = g_cancel_flags.find(key);
    if (it != g_cancel_flags.end()) return it->second;
    auto flag = std::make_shared<std::atomic<bool>>(false);
    g_cancel_flags[key] = flag;
    return flag;
}
static void cancel_flag(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_cancel_mutex);
    auto it = g_cancel_flags.find(key);
    if (it != g_cancel_flags.end()) it->second->store(true);
}

void register_tools_routes(CrowApp& app,
                             std::shared_ptr<AppState> state) {

    // ---------------------------------------------------------------
    // ASTROMETRY (ASTAP)
    // ---------------------------------------------------------------

    CROW_ROUTE(app, "/api/tools/astrometry/detect").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        std::string astap_cli      = body.is_discarded() ? "" : body.value("astap_cli", "");
        std::string astap_data_dir = body.is_discarded() ? "" : body.value("astap_data_dir", "");
        std::string catalog_dir_str = body.is_discarded() ? "" : body.value("catalog_dir", "");

        if (!astap_cli.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(astap_cli)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }
        if (!astap_data_dir.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(astap_data_dir)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path data_dir = astap_data_dir.empty() ? default_astap_data_dir() : fs::path(astap_data_dir);
        fs::path catalog_dir = catalog_dir_str.empty() ? data_dir : fs::path(catalog_dir_str);
        if (!catalog_dir_str.empty()) {
            if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path binary_path;
        if (!astap_cli.empty()) {
            binary_path = fs::path(astap_cli);
        } else {
            fs::path candidate = data_dir / "astap_cli";
            if (fs::exists(candidate)) binary_path = candidate;
            else binary_path = fs::path("astap_cli");
        }

        auto res = run_subprocess({binary_path.string(), "-h"});
        bool installed = (res.exit_code == 0 || res.exit_code == 1);

        nlohmann::json catalogs = nlohmann::json::object();
        for (const auto& [catalog_id, _] : ASTAP_CATALOGS) catalogs[catalog_id] = is_astap_catalog_installed(catalog_dir, catalog_id);

        return json_resp({
            {"installed",  installed},
            {"binary",     installed ? binary_path.string() : std::string()},
            {"data_dir",   data_dir.string()},
            {"catalog_dir", catalog_dir.string()},
            {"catalogs",   catalogs},
        });
    });

    CROW_ROUTE(app, "/api/tools/astrometry/install-cli").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        fs::path data_dir = body.is_discarded() || body.value("astap_data_dir", "").empty()
            ? default_astap_data_dir()
            : fs::path(body.value("astap_data_dir", ""));
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        auto cancel_flag_ptr = get_or_create_flag("astap_install");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("astrometry_install");
        state->job_store.update_state(job_id, JobState::running, {{"data_dir", data_dir.string()}, {"url", ASTAP_CLI_URL}});
        std::thread([state, job_id, data_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / "astap_cli.zip";
                auto dl = download_file(ASTAP_CLI_URL, archive, *cancel_flag_ptr);
                if (!dl.ok) throw std::runtime_error(dl.error);
                std::string error;
                if (!extract_zip_archive(archive, data_dir, error)) throw std::runtime_error(error);
                fs::remove(archive);
                fs::path candidate = find_astap_candidate(data_dir);
                if (candidate.empty()) throw std::runtime_error("astap_cli executable not found after extraction");
                fs::path target = data_dir / candidate.filename();
                if (candidate != target) fs::copy_file(candidate, target, fs::copy_options::overwrite_existing);
                state->job_store.update_state(job_id, JobState::ok, {{"binary", target.string()}, {"data_dir", data_dir.string()}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"data_dir", data_dir.string()}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/install-cli/retry").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        auto body2 = body;
        fs::path data_dir = body2.is_discarded() || body2.value("astap_data_dir", "").empty()
            ? default_astap_data_dir()
            : fs::path(body2.value("astap_data_dir", ""));
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        auto cancel_flag_ptr = get_or_create_flag("astap_install");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("astrometry_install");
        state->job_store.update_state(job_id, JobState::running, {{"data_dir", data_dir.string()}, {"url", ASTAP_CLI_URL}, {"resume", true}});
        std::thread([state, job_id, data_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / "astap_cli.zip";
                auto dl = download_file(ASTAP_CLI_URL, archive, *cancel_flag_ptr);
                if (!dl.ok) throw std::runtime_error(dl.error);
                std::string error;
                if (!extract_zip_archive(archive, data_dir, error)) throw std::runtime_error(error);
                fs::remove(archive);
                fs::path candidate = find_astap_candidate(data_dir);
                if (candidate.empty()) throw std::runtime_error("astap_cli executable not found after extraction");
                fs::path target = data_dir / candidate.filename();
                if (candidate != target) fs::copy_file(candidate, target, fs::copy_options::overwrite_existing);
                state->job_store.update_state(job_id, JobState::ok, {{"binary", target.string()}, {"data_dir", data_dir.string()}, {"resume", true}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"data_dir", data_dir.string()}, {"resume", true}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/download").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        std::string catalog_id = body.value("catalog_id", std::string("d50"));
        auto it = ASTAP_CATALOGS.find(catalog_id);
        if (it == ASTAP_CATALOGS.end()) return err_resp("BAD_REQUEST", "unknown catalog_id '" + catalog_id + "'", 400, nlohmann::json::object());
        fs::path data_dir = body.value("astap_data_dir", "").empty()
            ? default_astap_data_dir()
            : fs::path(body.value("astap_data_dir", ""));
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        auto cancel_flag_ptr = get_or_create_flag("astap_catalog");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("astrometry_catalog_download");
        state->job_store.update_state(job_id, JobState::running, {{"catalog_id", catalog_id}, {"data_dir", data_dir.string()}});
        std::thread([state, job_id, catalog_id, filename = it->second, data_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / filename;
                auto dl = download_file(std::string(ASTAP_SF_BASE) + filename + "/download", archive, *cancel_flag_ptr);
                if (!dl.ok) throw std::runtime_error(dl.error);
                std::string error;
                if (archive.extension() == ".zip") {
                    if (!extract_zip_archive(archive, data_dir, error)) throw std::runtime_error(error);
                } else if (archive.extension() == ".deb") {
                    fs::path tmp = data_dir / "_deb_extract";
                    fs::create_directories(tmp);
                    if (!extract_deb_archive(archive, tmp, error)) throw std::runtime_error(error);
                    for (const auto& entry : fs::recursive_directory_iterator(tmp)) {
                        if (!entry.is_regular_file()) continue;
                        const auto name = entry.path().filename().string();
                        if (name.rfind(catalog_id + "_", 0) == 0) {
                            fs::copy_file(entry.path(), data_dir / entry.path().filename(), fs::copy_options::overwrite_existing);
                        }
                    }
                    fs::remove_all(tmp);
                } else {
                    throw std::runtime_error("unsupported archive format");
                }
                fs::remove(archive);
                state->job_store.update_state(job_id, JobState::ok, {{"catalog_id", catalog_id}, {"installed", is_astap_catalog_installed(data_dir, catalog_id)}, {"data_dir", data_dir.string()}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"catalog_id", catalog_id}, {"data_dir", data_dir.string()}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/download/retry").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        std::string catalog_id = body.value("catalog_id", std::string("d50"));
        auto it = ASTAP_CATALOGS.find(catalog_id);
        if (it == ASTAP_CATALOGS.end()) return err_resp("BAD_REQUEST", "unknown catalog_id '" + catalog_id + "'", 400, nlohmann::json::object());
        fs::path data_dir = body.value("astap_data_dir", "").empty()
            ? default_astap_data_dir()
            : fs::path(body.value("astap_data_dir", ""));
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        auto cancel_flag_ptr = get_or_create_flag("astap_catalog");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("astrometry_catalog_download");
        state->job_store.update_state(job_id, JobState::running, {{"catalog_id", catalog_id}, {"data_dir", data_dir.string()}, {"resume", true}});
        std::thread([state, job_id, catalog_id, filename = it->second, data_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / filename;
                auto dl = download_file(std::string(ASTAP_SF_BASE) + filename + "/download", archive, *cancel_flag_ptr);
                if (!dl.ok) throw std::runtime_error(dl.error);
                std::string error;
                if (archive.extension() == ".zip") {
                    if (!extract_zip_archive(archive, data_dir, error)) throw std::runtime_error(error);
                } else if (archive.extension() == ".deb") {
                    fs::path tmp = data_dir / "_deb_extract";
                    fs::create_directories(tmp);
                    if (!extract_deb_archive(archive, tmp, error)) throw std::runtime_error(error);
                    for (const auto& entry : fs::recursive_directory_iterator(tmp)) {
                        if (!entry.is_regular_file()) continue;
                        const auto name = entry.path().filename().string();
                        if (name.rfind(catalog_id + "_", 0) == 0) {
                            fs::copy_file(entry.path(), data_dir / entry.path().filename(), fs::copy_options::overwrite_existing);
                        }
                    }
                    fs::remove_all(tmp);
                } else {
                    throw std::runtime_error("unsupported archive format");
                }
                fs::remove(archive);
                state->job_store.update_state(job_id, JobState::ok, {{"catalog_id", catalog_id}, {"installed", is_astap_catalog_installed(data_dir, catalog_id)}, {"data_dir", data_dir.string()}, {"resume", true}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"catalog_id", catalog_id}, {"data_dir", data_dir.string()}, {"resume", true}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/cancel").methods("POST"_method)
    ([state](const crow::request&) {
        cancel_flag("astap_catalog");
        bool cancelled = false;
        auto jobs = state->job_store.list(50);
        for (auto& j : jobs) {
            if (j.type == "astrometry_catalog_download" &&
                (j.state == JobState::running || j.state == JobState::pending)) {
                state->job_store.cancel(j.job_id);
                cancelled = true;
            }
        }
        return json_resp({{"ok", cancelled}});
    });

    CROW_ROUTE(app, "/api/tools/astrometry/solve").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("solve_file"))
            return err_resp("BAD_REQUEST", "solve_file is required", 400, nlohmann::json::object());

        std::string solve_file     = body["solve_file"].get<std::string>();
        std::string astap_cli      = body.value("astap_cli", "");
        std::string astap_data_dir = body.value("astap_data_dir", "");

        if (auto denied = denied_path(state->runtime, fs::path(solve_file)); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        if (!astap_cli.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(astap_cli)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }
        if (!astap_data_dir.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(astap_data_dir)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path astap_bin;
        if (!astap_cli.empty()) astap_bin = fs::path(astap_cli);
        else {
            fs::path default_bin = (astap_data_dir.empty() ? default_astap_data_dir() : fs::path(astap_data_dir)) / "astap_cli";
            astap_bin = fs::exists(default_bin) ? default_bin : fs::path("astap_cli");
        }
        auto probe = run_subprocess({astap_bin.string(), "-h"});
        if (!(probe.exit_code == 0 || probe.exit_code == 1)) {
            return err_resp("BAD_REQUEST", "ASTAP CLI not found; install or provide astap_cli path", 400, nlohmann::json::object());
        }

        std::vector<std::string> args = {astap_bin.string(), "-f", solve_file, "-wcs"};
        if (!astap_data_dir.empty()) { args.push_back("-d"); args.push_back(astap_data_dir); }

        std::string job_id = state->subprocess_manager.launch("astrometry_solve", args,
                                                               state->runtime.project_root.string());
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/save-solved").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return err_resp("Invalid JSON");
        std::string input_path  = body.value("input_path", "");
        std::string output_path = body.value("output_path", "");
        std::string wcs_path    = body.value("wcs_path", "");

        if (input_path.empty() || output_path.empty())
            return err_resp("input_path and output_path required");

        fs::path input = fs::path(input_path);
        fs::path output = fs::path(output_path);
        if (auto denied = denied_path(state->runtime, input); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        if (auto denied = denied_path(state->runtime, output); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        if (!fs::exists(input)) return err_resp("input_path not found", 404);
        fs::create_directories(output.parent_path());
        fs::copy_file(input, output, fs::copy_options::overwrite_existing);
        std::string copied_wcs;
        if (!wcs_path.empty()) {
            fs::path wcs = fs::path(wcs_path);
            if (auto denied = denied_path(state->runtime, wcs); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
            if (fs::exists(wcs)) {
                fs::path out_wcs = output;
                out_wcs.replace_extension(".wcs");
                fs::copy_file(wcs, out_wcs, fs::copy_options::overwrite_existing);
                copied_wcs = out_wcs.string();
            }
        }
        return json_resp({{"output_path", output.string()}, {"wcs_path", copied_wcs.empty() ? nlohmann::json(nullptr) : nlohmann::json(copied_wcs)}});
    });

    // ---------------------------------------------------------------
    // PCC (Siril catalog)
    // ---------------------------------------------------------------

    CROW_ROUTE(app, "/api/tools/pcc/siril/status").methods("GET"_method)
    ([state](const crow::request& req) {
        fs::path catalog_dir = req.url_params.get("catalog_dir") && std::string(req.url_params.get("catalog_dir")).size() > 0
            ? fs::path(req.url_params.get("catalog_dir"))
            : default_siril_catalog_dir();
        if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        auto missing = missing_siril_chunks(catalog_dir);
        return json_resp({
            {"installed",   SIRIL_NUM_CHUNKS - static_cast<int>(missing.size())},
            {"total",       SIRIL_NUM_CHUNKS},
            {"missing",     missing},
            {"catalog_dir", catalog_dir.string()},
        });
    });

    CROW_ROUTE(app, "/api/tools/pcc/siril/download-missing").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        fs::path catalog_dir = body.is_discarded() || body.value("catalog_dir", "").empty()
            ? default_siril_catalog_dir()
            : fs::path(body.value("catalog_dir", ""));
        if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        auto cancel_flag_ptr = get_or_create_flag("pcc_siril");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("pcc_siril_download");
        state->job_store.update_state(job_id, JobState::running, {{"catalog_dir", catalog_dir.string()}});
        std::thread([state, job_id, catalog_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(catalog_dir);
                auto missing = missing_siril_chunks(catalog_dir);
                int completed = 0;
                for (int chunk : missing) {
                    if (cancel_flag_ptr->load()) {
                        state->job_store.update_state(job_id, JobState::cancelled, {{"catalog_dir", catalog_dir.string()}, {"current_chunk", chunk}});
                        return;
                    }
                    fs::path archive = catalog_dir / ("siril_cat1_healpix8_xpsamp_" + std::to_string(chunk) + ".dat.bz2");
                    auto dl = download_file(siril_chunk_url(chunk), archive, *cancel_flag_ptr,
                        [state, job_id, completed, total = static_cast<int>(missing.size())](double ratio) {
                            state->job_store.update_progress(job_id, total > 0 ? ((completed + ratio) / total) * 100.0 : 100.0);
                        });
                    if (!dl.ok) throw std::runtime_error(dl.error);
                    std::string error;
                    if (!decompress_bz2_archive(archive, error)) throw std::runtime_error(error);
                    ++completed;
                    state->job_store.update_progress(job_id, missing.empty() ? 100.0 : (100.0 * completed / missing.size()));
                }
                state->job_store.update_state(job_id, JobState::ok, {{"catalog_dir", catalog_dir.string()}, {"missing_after", missing_siril_chunks(catalog_dir)}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"catalog_dir", catalog_dir.string()}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/pcc/siril/download-missing/retry").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        fs::path catalog_dir = body.value("catalog_dir", "").empty()
            ? default_siril_catalog_dir()
            : fs::path(body.value("catalog_dir", ""));
        if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        auto cancel_flag_ptr = get_or_create_flag("pcc_siril");
        cancel_flag_ptr->store(false);
        std::string job_id = state->job_store.create("pcc_siril_download");
        state->job_store.update_state(job_id, JobState::running, {{"catalog_dir", catalog_dir.string()}, {"resume", true}});
        std::thread([state, job_id, catalog_dir, cancel_flag_ptr]() {
            try {
                fs::create_directories(catalog_dir);
                auto missing = missing_siril_chunks(catalog_dir);
                int completed = 0;
                for (int chunk : missing) {
                    if (cancel_flag_ptr->load()) {
                        state->job_store.update_state(job_id, JobState::cancelled, {{"catalog_dir", catalog_dir.string()}, {"current_chunk", chunk}, {"resume", true}});
                        return;
                    }
                    fs::path archive = catalog_dir / ("siril_cat1_healpix8_xpsamp_" + std::to_string(chunk) + ".dat.bz2");
                    auto dl = download_file(siril_chunk_url(chunk), archive, *cancel_flag_ptr,
                        [state, job_id, completed, total = static_cast<int>(missing.size())](double ratio) {
                            state->job_store.update_progress(job_id, total > 0 ? ((completed + ratio) / total) * 100.0 : 100.0);
                        });
                    if (!dl.ok) throw std::runtime_error(dl.error);
                    std::string error;
                    if (!decompress_bz2_archive(archive, error)) throw std::runtime_error(error);
                    ++completed;
                    state->job_store.update_progress(job_id, missing.empty() ? 100.0 : (100.0 * completed / missing.size()));
                }
                state->job_store.update_state(job_id, JobState::ok, {{"catalog_dir", catalog_dir.string()}, {"missing_after", missing_siril_chunks(catalog_dir)}, {"resume", true}});
            } catch (const std::exception& e) {
                state->job_store.update_state(job_id, cancel_flag_ptr->load() ? JobState::cancelled : JobState::error,
                    {{"catalog_dir", catalog_dir.string()}, {"resume", true}}, e.what());
            }
        }).detach();
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/pcc/siril/cancel").methods("POST"_method)
    ([state](const crow::request&) {
        cancel_flag("pcc_siril");
        bool cancelled = false;
        auto jobs = state->job_store.list(50);
        for (auto& j : jobs) {
            if (j.type == "pcc_siril_download" &&
                (j.state == JobState::running || j.state == JobState::pending)) {
                state->job_store.cancel(j.job_id);
                cancelled = true;
            }
        }
        return json_resp({{"ok", cancelled}});
    });

    CROW_ROUTE(app, "/api/tools/pcc/check-online").methods("POST"_method)
    ([state](const crow::request&) {
        (void)state;
        const std::string url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c=0%200&-c.rd=0.01&-out=RA_ICRS,DE_ICRS,Gmag&-out.max=1";
        CURL* curl = curl_easy_init();
        if (!curl) return json_resp({{"ok", false}, {"latency_ms", 0}, {"error", "curl init failed"}});
        const auto t0 = std::chrono::steady_clock::now();
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_discard_write);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        CURLcode rc = curl_easy_perform(curl);
        long latency_ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
        curl_easy_cleanup(curl);
        if (rc == CURLE_OK) return json_resp({{"ok", true}, {"latency_ms", latency_ms}});
        return json_resp({{"ok", false}, {"latency_ms", latency_ms}, {"error", curl_easy_strerror(rc)}});
    });

    CROW_ROUTE(app, "/api/tools/pcc/run").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("input_rgb") || !body.contains("output_rgb") || !body.contains("wcs_file"))
            return err_resp("BAD_REQUEST", "input_rgb, output_rgb and wcs_file are required", 400, nlohmann::json::object());

        std::string input_rgb   = body["input_rgb"].get<std::string>();
        std::string output_rgb  = body["output_rgb"].get<std::string>();
        std::string wcs_file    = body.value("wcs_file", "");
        std::string source      = body.value("source", "auto");
        std::string catalog_dir = body.value("catalog_dir", "");

        if (source != "auto" && source != "siril" && source != "vizier_gaia" && source != "vizier_apass") {
            return err_resp("BAD_REQUEST", "unsupported pcc source '" + source + "'", 422, nlohmann::json::object());
        }

        if (auto denied = denied_path(state->runtime, fs::path(input_rgb)); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        if (auto denied = denied_path(state->runtime, fs::path(output_rgb)); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }
        if (!wcs_file.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(wcs_file)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }
        if (!catalog_dir.empty()) {
            if (auto denied = denied_path(state->runtime, fs::path(catalog_dir)); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        std::vector<std::string> args = {state->runtime.cli_exe, "pcc-run"};
        args.push_back(input_rgb);
        args.push_back(output_rgb);
        if (!wcs_file.empty())    { args.push_back("--wcs"); args.push_back(wcs_file); }
        if (!source.empty())      { args.push_back("--source"); args.push_back(source); }
        if (!catalog_dir.empty()) { args.push_back("--siril-catalog-dir"); args.push_back(catalog_dir); }

        if (body.contains("mag_limit") && !body["mag_limit"].is_null())
            { args.push_back("--mag-limit"); args.push_back(std::to_string(body["mag_limit"].get<double>())); }
        if (body.contains("min_stars") && !body["min_stars"].is_null())
            { args.push_back("--min-stars"); args.push_back(std::to_string(body["min_stars"].get<int>())); }
        if (body.contains("sigma_clip") && !body["sigma_clip"].is_null())
            { args.push_back("--sigma-clip"); args.push_back(std::to_string(body["sigma_clip"].get<double>())); }

        std::string job_id = state->subprocess_manager.launch("pcc_run", args,
                                                               state->runtime.project_root.string());
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/tools/pcc/save-corrected").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("output_rgb"))
            return err_resp("BAD_REQUEST", "output_rgb is required", 400, nlohmann::json::object());

        std::string output_rgb = body["output_rgb"].get<std::string>();
        if (auto denied = denied_path(state->runtime, fs::path(output_rgb)); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        nlohmann::json channels = nlohmann::json::array();
        if (body.contains("output_channels") && body["output_channels"].is_array()) {
            for (const auto& channel : body["output_channels"]) {
                std::string channel_path = channel.get<std::string>();
                if (auto denied = denied_path(state->runtime, fs::path(channel_path)); denied) {
                    return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
                }
                channels.push_back(channel_path);
            }
        }

        return json_resp({
            {"output_rgb",      output_rgb},
            {"output_channels", channels},
        });
    });
}
