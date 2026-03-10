#include "routes/tools_routes.hpp"
#include "services/download_manager.hpp"
#include "subprocess_manager.hpp"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <algorithm>
#include <cctype>
#include <map>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

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

fs::path expand_user_path(const fs::path& raw) {
    if (raw.empty()) return raw;
    const std::string text = raw.string();
    if (text == "~") return user_home_dir();
    if (text.rfind("~/", 0) == 0 || text.rfind("~\\", 0) == 0) {
        return user_home_dir() / text.substr(2);
    }
    return raw;
}

fs::path path_from_user_input(const std::string& raw, const fs::path& fallback) {
    if (raw.empty()) return fallback;
    return expand_user_path(fs::path(raw));
}

bool is_command_name(const std::string& raw) {
    if (raw.empty()) return false;
    const fs::path path(raw);
    if (path.is_absolute()) return false;
    return raw.find('/') == std::string::npos && raw.find('\\') == std::string::npos;
}

bool is_astap_binary_name(const std::string& name) {
    return name == "astap_cli" || name == "astap_cli.exe" || name == "astap";
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
    const fs::path expanded = expand_user_path(root);
    std::error_code ec;
    if (expanded.empty() || !fs::exists(expanded, ec)) return {};
    if (fs::is_regular_file(expanded, ec) && is_astap_binary_name(expanded.filename().string())) {
        return expanded;
    }
    if (!fs::is_directory(expanded, ec)) return {};
    fs::recursive_directory_iterator end;
    for (fs::recursive_directory_iterator it(expanded, fs::directory_options::skip_permission_denied, ec); it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        const auto name = it->path().filename().string();
        if (is_astap_binary_name(name)) return it->path();
    }
    return {};
}

bool astap_probe_ok(const fs::path& candidate) {
    if (candidate.empty()) return false;
    auto res = run_subprocess({candidate.string(), "-h"});
    return res.exit_code == 0 || res.exit_code == 1;
}

fs::path resolve_astap_binary(const std::string& astap_cli_raw, const fs::path& data_dir) {
    std::vector<fs::path> candidates;
    const auto add_candidate = [&candidates](const fs::path& raw_candidate) {
        const fs::path candidate = expand_user_path(raw_candidate);
        if (candidate.empty()) return;
        if (std::find(candidates.begin(), candidates.end(), candidate) != candidates.end()) return;
        candidates.push_back(candidate);
    };
    const auto add_search_root = [&add_candidate](const fs::path& root) {
        if (root.empty()) return;
        add_candidate(root / "astap_cli");
        add_candidate(root / "astap_cli.exe");
        add_candidate(root / "astap");
        const fs::path nested = find_astap_candidate(root);
        if (!nested.empty()) add_candidate(nested);
    };

    if (!astap_cli_raw.empty()) {
        if (is_command_name(astap_cli_raw)) {
            add_candidate(fs::path(astap_cli_raw));
        } else {
            const fs::path explicit_path = expand_user_path(fs::path(astap_cli_raw));
            std::error_code ec;
            if (fs::is_directory(explicit_path, ec)) add_search_root(explicit_path);
            else add_candidate(explicit_path);
        }
    }

    add_search_root(data_dir);
    add_search_root(data_dir / "bin");
    add_search_root(data_dir / "usr" / "bin");
    add_candidate(fs::path("astap_cli"));
    add_candidate(fs::path("astap"));

    for (const auto& candidate : candidates) {
        if (astap_probe_ok(candidate)) return candidate;
    }
    return {};
}

fs::path guess_wcs_path(const fs::path& fits_path) {
    const std::string name = fits_path.filename().string();
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const std::string& ext : {".fits.fz", ".fit.fz", ".fts.fz", ".fits", ".fit", ".fts"}) {
        if (lower.size() >= ext.size() && lower.compare(lower.size() - ext.size(), ext.size(), ext) == 0) {
            return fits_path.parent_path() / (name.substr(0, name.size() - ext.size()) + ".wcs");
        }
    }
    return fits_path;
}

nlohmann::json parse_astrometry_wcs_summary(const fs::path& wcs_path) {
    std::ifstream in(wcs_path);
    std::map<std::string, double> values;
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string raw = line.substr(eq + 1);
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        const auto slash = raw.find('/');
        if (slash != std::string::npos) raw = raw.substr(0, slash);
        raw.erase(0, raw.find_first_not_of(" \t'\""));
        raw.erase(raw.find_last_not_of(" \t'\"") + 1);
        if (key.empty() || raw.empty()) continue;
        try {
            values[key] = std::stod(raw);
        } catch (...) {}
    }

    const auto get_value = [&values](const std::string& key, double fallback = 0.0) {
        auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    };
    const auto has_value = [&values](const std::string& key) {
        return values.find(key) != values.end();
    };

    const double crval1 = has_value("CRVAL1") ? get_value("CRVAL1") : std::numeric_limits<double>::quiet_NaN();
    const double crval2 = has_value("CRVAL2") ? get_value("CRVAL2") : std::numeric_limits<double>::quiet_NaN();
    const double crpix1 = get_value("CRPIX1");
    const double crpix2 = get_value("CRPIX2");
    const double naxis1_raw = has_value("NAXIS1") ? get_value("NAXIS1") : crpix1 * 2.0;
    const double naxis2_raw = has_value("NAXIS2") ? get_value("NAXIS2") : crpix2 * 2.0;
    const int image_width = naxis1_raw > 0.0 ? static_cast<int>(std::llround(naxis1_raw)) : 0;
    const int image_height = naxis2_raw > 0.0 ? static_cast<int>(std::llround(naxis2_raw)) : 0;

    double pixel_scale_arcsec = 0.0;
    double rotation_deg = 0.0;
    if (has_value("CD1_1") && has_value("CD1_2") && has_value("CD2_1") && has_value("CD2_2")) {
        const double cd11 = get_value("CD1_1");
        const double cd12 = get_value("CD1_2");
        const double cd21 = get_value("CD2_1");
        const double cd22 = get_value("CD2_2");
        const double scale_x = std::hypot(cd11, cd21);
        const double scale_y = std::hypot(cd12, cd22);
        pixel_scale_arcsec = ((scale_x + scale_y) / 2.0) * 3600.0;
        rotation_deg = std::atan2(cd21, cd11) * 180.0 / M_PI;
    } else {
        const double cdelt1 = get_value("CDELT1");
        const double cdelt2 = get_value("CDELT2");
        if (cdelt1 != 0.0 || cdelt2 != 0.0) pixel_scale_arcsec = ((std::abs(cdelt1) + std::abs(cdelt2)) / 2.0) * 3600.0;
        if (has_value("CROTA2")) rotation_deg = get_value("CROTA2");
        else if (has_value("CROTA1")) rotation_deg = get_value("CROTA1");
    }

    const double fov_width_deg = (pixel_scale_arcsec > 0.0 && image_width > 0) ? (pixel_scale_arcsec * image_width / 3600.0) : 0.0;
    const double fov_height_deg = (pixel_scale_arcsec > 0.0 && image_height > 0) ? (pixel_scale_arcsec * image_height / 3600.0) : 0.0;

    nlohmann::json out = {
        {"ra_deg", std::isnan(crval1) ? nlohmann::json(nullptr) : nlohmann::json(crval1)},
        {"dec_deg", std::isnan(crval2) ? nlohmann::json(nullptr) : nlohmann::json(crval2)},
        {"pixel_scale_arcsec", pixel_scale_arcsec},
        {"rotation_deg", rotation_deg},
        {"fov_width_deg", fov_width_deg},
        {"fov_height_deg", fov_height_deg},
        {"image_width", image_width},
        {"image_height", image_height},
    };
    return out;
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
    if (res.exit_code == 0) return true;

    const std::string dpkg_error = res.stderr_str.empty() ? res.stdout_str : res.stderr_str;
    fs::path ar_tmp = dest / "_deb_ar";
    std::error_code ec;
    fs::remove_all(ar_tmp, ec);
    fs::create_directories(ar_tmp, ec);

    auto ar_res = run_subprocess({"ar", "x", archive.string()}, ar_tmp.string());
    if (ar_res.exit_code != 0) {
        std::string ar_error = ar_res.stderr_str.empty() ? ar_res.stdout_str : ar_res.stderr_str;
        error = "dpkg-deb failed";
        if (!dpkg_error.empty()) error += ": " + dpkg_error;
        if (!ar_error.empty()) error += " | ar fallback failed: " + ar_error;
        fs::remove_all(ar_tmp, ec);
        return false;
    }

    fs::path data_archive;
    for (const char* name : {"data.tar.zst", "data.tar.xz", "data.tar.gz", "data.tar.bz2", "data.tar"}) {
        fs::path candidate = ar_tmp / name;
        if (fs::exists(candidate)) {
            data_archive = candidate;
            break;
        }
    }
    if (data_archive.empty()) {
        error = "deb archive missing data payload";
        if (!dpkg_error.empty()) error += " (dpkg-deb: " + dpkg_error + ")";
        fs::remove_all(ar_tmp, ec);
        return false;
    }

    auto tar_res = run_subprocess({"tar", "-xf", data_archive.string(), "-C", dest.string()});
    fs::remove_all(ar_tmp, ec);
    if (tar_res.exit_code != 0) {
        error = tar_res.stderr_str.empty() ? tar_res.stdout_str : tar_res.stderr_str;
        if (!dpkg_error.empty()) error += " | dpkg-deb: " + dpkg_error;
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

std::string payload_text(const nlohmann::json& payload, const std::string& key, const std::string& fallback = "") {
    if (!payload.is_object() || !payload.contains(key) || payload[key].is_null()) return fallback;
    if (payload[key].is_string()) return payload[key].get<std::string>();
    return payload[key].dump();
}

std::optional<double> payload_float(const nlohmann::json& payload, const std::string& key) {
    if (!payload.is_object() || !payload.contains(key) || payload[key].is_null()) return std::nullopt;
    try {
        if (payload[key].is_string()) {
            std::string raw = payload[key].get<std::string>();
            if (raw.empty()) return std::nullopt;
            return std::stod(raw);
        }
        return payload[key].get<double>();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> payload_int(const nlohmann::json& payload, const std::string& key) {
    if (!payload.is_object() || !payload.contains(key) || payload[key].is_null()) return std::nullopt;
    try {
        if (payload[key].is_string()) {
            std::string raw = payload[key].get<std::string>();
            if (raw.empty()) return std::nullopt;
            return std::stoi(raw);
        }
        return payload[key].get<int>();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> payload_bool(const nlohmann::json& payload, const std::string& key) {
    if (!payload.is_object() || !payload.contains(key) || payload[key].is_null()) return std::nullopt;
    try {
        if (payload[key].is_boolean()) return payload[key].get<bool>();
        std::string raw = payload[key].is_string() ? payload[key].get<std::string>() : payload[key].dump();
        std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (raw == "1" || raw == "true" || raw == "yes" || raw == "on") return true;
        if (raw == "0" || raw == "false" || raw == "no" || raw == "off") return false;
    } catch (...) {}
    return std::nullopt;
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
    if (access_path.empty()) return std::nullopt;
    auto resolved = runtime.resolve_input_path(access_path, false);
    if (resolved.status != PathStatus::not_allowed) return std::nullopt;
    fs::path path_for_error = reported_path.empty()
        ? (access_path.is_absolute() ? access_path : resolved.path)
        : reported_path;
    return path_for_error.string();
}

struct JobCancelled : std::runtime_error {
    JobCancelled() : std::runtime_error("cancelled") {}
};

static DownloadOptions download_options_from_payload(const nlohmann::json& payload, long default_timeout_s) {
    DownloadOptions options;
    options.timeout_s = payload.value("timeout_s", default_timeout_s);
    options.retry_count = std::max(0, payload.value("retry_count", 2));
    options.retry_backoff_s = std::max(0.0, payload.value("retry_backoff_sec", 1.5));
    options.resume = payload.value("resume", true);
    return options;
}

static void throw_if_job_cancelled(const std::shared_ptr<AppState>& state, const std::string& job_id) {
    if (state->job_store.is_cancelled(job_id)) throw JobCancelled();
}

static void set_download_progress(const std::shared_ptr<AppState>& state,
                                  const std::string& job_id,
                                  std::uintmax_t received,
                                  std::uintmax_t total) {
    nlohmann::json progress_value = nullptr;
    if (total > 0) {
        progress_value = static_cast<double>(received) / static_cast<double>(total);
    }
    nlohmann::json patch = {
        {"bytes_received", received},
        {"bytes_total", total},
        {"progress", progress_value},
    };
    state->job_store.merge_data(job_id, patch);
    if (total > 0) state->job_store.update_progress(job_id, (static_cast<double>(received) / static_cast<double>(total)) * 100.0);
    throw_if_job_cancelled(state, job_id);
}

static void set_download_state(const std::shared_ptr<AppState>& state,
                               const std::string& job_id,
                               const nlohmann::json& patch) {
    state->job_store.merge_data(job_id, patch);
    throw_if_job_cancelled(state, job_id);
}

static void finish_job_cancelled(const std::shared_ptr<AppState>& state,
                                 const std::string& job_id,
                                 const nlohmann::json& patch = nlohmann::json::object()) {
    nlohmann::json data = patch;
    if (auto snapshot = state->job_store.get(job_id); snapshot && snapshot->data.is_object()) {
        for (auto it = snapshot->data.begin(); it != snapshot->data.end(); ++it) {
            if (!data.contains(it.key())) data[it.key()] = it.value();
        }
    }
    state->job_store.update_state(job_id, JobState::cancelled, data);
}

void register_tools_routes(CrowApp& app,
                             std::shared_ptr<AppState> state) {
    auto cancel_jobs_by_type = [state](const std::string& type,
                                       const std::string& event,
                                       const std::string& source) -> crow::response {
        bool cancelled = false;
        auto jobs = state->job_store.list(200);
        for (const auto& job : jobs) {
            if (job.type != type) continue;
            if (job.state != JobState::running && job.state != JobState::pending) continue;
            if (state->job_store.cancel(job.job_id)) {
                cancelled = true;
                state->ui_event_store.push(event, source, {{"ok", true}}, std::nullopt, job.job_id);
            }
        }
        return json_resp({{"ok", cancelled}});
    };

    auto start_astrometry_install = [state](nlohmann::json body) -> crow::response {
        fs::path data_dir = path_from_user_input(body.value("astap_data_dir", ""), default_astap_data_dir());
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        const DownloadOptions options = download_options_from_payload(body, 1800);
        const bool force_restart = body.value("force_restart", false);
        std::string job_id = state->job_store.create("astrometry_install_cli");
        state->job_store.update_state(job_id, JobState::running, {
            {"data_dir", data_dir.string()},
            {"url", ASTAP_CLI_URL},
            {"stage", "download"},
            {"progress", 0.0},
            {"resume_enabled", options.resume},
            {"retry_count", options.retry_count},
        });

        std::thread([state, job_id, data_dir, options, force_restart]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / "astap_cli.zip";
                if (force_restart) {
                    std::error_code ec;
                    fs::remove(archive, ec);
                }
                auto dl = download_file_with_retry(
                    ASTAP_CLI_URL,
                    archive,
                    options,
                    [state, job_id]() { return state->job_store.is_cancelled(job_id); },
                    [state, job_id](std::uintmax_t received, std::uintmax_t total) {
                        set_download_progress(state, job_id, received, total);
                    },
                    [state, job_id](const nlohmann::json& patch) {
                        nlohmann::json merged = patch;
                        merged["stage"] = "download";
                        set_download_state(state, job_id, merged);
                    });
                if (!dl.ok) {
                    if (dl.error == "cancelled" || state->job_store.is_cancelled(job_id)) throw JobCancelled();
                    throw std::runtime_error(dl.error);
                }

                set_download_state(state, job_id, {{"stage", "extract"}, {"archive", archive.string()}});
                std::string error;
                if (!extract_zip_archive(archive, data_dir, error)) throw std::runtime_error(error);
                std::error_code rm_ec;
                fs::remove(archive, rm_ec);

                fs::path candidate = find_astap_candidate(data_dir);
                if (candidate.empty()) throw std::runtime_error("astap_cli executable not found after extraction");
                fs::path target = data_dir / candidate.filename();
                if (candidate != target) fs::copy_file(candidate, target, fs::copy_options::overwrite_existing);

                state->job_store.merge_data(job_id, {
                    {"binary", target.string()},
                    {"data_dir", data_dir.string()},
                    {"stage", "done"},
                    {"progress", 1.0},
                });
                auto snapshot = state->job_store.get(job_id);
                state->job_store.update_state(job_id, JobState::ok, snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object());
            } catch (const JobCancelled&) {
                finish_job_cancelled(state, job_id);
            } catch (const std::exception& e) {
                auto snapshot = state->job_store.get(job_id);
                nlohmann::json data = snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object();
                data["error"] = e.what();
                if (state->job_store.is_cancelled(job_id)) finish_job_cancelled(state, job_id, data);
                else state->job_store.update_state(job_id, JobState::error, data, e.what());
            }
        }).detach();

        state->ui_event_store.push("tools.astrometry.install_cli", "tools.astrometry_install", {{"data_dir", data_dir.string()}}, std::nullopt, job_id);
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    };

    auto start_astrometry_catalog_download = [state](nlohmann::json body) -> crow::response {
        const std::string catalog_id = body.value("catalog_id", std::string("d50"));
        auto it = ASTAP_CATALOGS.find(catalog_id);
        if (it == ASTAP_CATALOGS.end()) {
            return err_resp("BAD_REQUEST", "unknown catalog_id '" + catalog_id + "'", 400, nlohmann::json::object());
        }

        fs::path data_dir = path_from_user_input(body.value("astap_data_dir", ""), default_astap_data_dir());
        if (auto denied = denied_path(state->runtime, data_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        const DownloadOptions options = download_options_from_payload(body, 1800);
        const bool force_restart = body.value("force_restart", false);
        const std::string url = std::string(ASTAP_SF_BASE) + it->second + "/download";
        std::string job_id = state->job_store.create("astrometry_catalog_download");
        state->job_store.update_state(job_id, JobState::running, {
            {"catalog_id", catalog_id},
            {"url", url},
            {"data_dir", data_dir.string()},
            {"stage", "download"},
            {"progress", 0.0},
            {"resume_enabled", options.resume},
            {"retry_count", options.retry_count},
        });

        std::thread([state, job_id, catalog_id, filename = it->second, data_dir, options, force_restart, url]() {
            try {
                fs::create_directories(data_dir);
                fs::path archive = data_dir / filename;
                if (force_restart) {
                    std::error_code ec;
                    fs::remove(archive, ec);
                }
                auto dl = download_file_with_retry(
                    url,
                    archive,
                    options,
                    [state, job_id]() { return state->job_store.is_cancelled(job_id); },
                    [state, job_id](std::uintmax_t received, std::uintmax_t total) {
                        set_download_progress(state, job_id, received, total);
                    },
                    [state, job_id](const nlohmann::json& patch) {
                        nlohmann::json merged = patch;
                        merged["stage"] = "download";
                        set_download_state(state, job_id, merged);
                    });
                if (!dl.ok) {
                    if (dl.error == "cancelled" || state->job_store.is_cancelled(job_id)) throw JobCancelled();
                    throw std::runtime_error(dl.error);
                }

                set_download_state(state, job_id, {{"stage", "extract"}, {"archive", archive.string()}});
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
                    std::error_code ec;
                    fs::remove_all(tmp, ec);
                } else {
                    throw std::runtime_error("unsupported archive format");
                }
                std::error_code rm_ec;
                fs::remove(archive, rm_ec);

                state->job_store.merge_data(job_id, {
                    {"catalog_id", catalog_id},
                    {"installed", is_astap_catalog_installed(data_dir, catalog_id)},
                    {"data_dir", data_dir.string()},
                    {"stage", "done"},
                    {"progress", 1.0},
                });
                auto snapshot = state->job_store.get(job_id);
                state->job_store.update_state(job_id, JobState::ok, snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object());
            } catch (const JobCancelled&) {
                finish_job_cancelled(state, job_id);
            } catch (const std::exception& e) {
                auto snapshot = state->job_store.get(job_id);
                nlohmann::json data = snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object();
                data["error"] = e.what();
                if (state->job_store.is_cancelled(job_id)) finish_job_cancelled(state, job_id, data);
                else state->job_store.update_state(job_id, JobState::error, data, e.what());
            }
        }).detach();

        state->ui_event_store.push("tools.astrometry.catalog.download", "tools.astrometry_catalog_download", {{"catalog_id", catalog_id}, {"data_dir", data_dir.string()}}, std::nullopt, job_id);
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    };

    auto start_pcc_siril_download = [state](nlohmann::json body) -> crow::response {
        fs::path catalog_dir = path_from_user_input(body.value("catalog_dir", ""), default_siril_catalog_dir());
        if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        const DownloadOptions options = download_options_from_payload(body, 3600);
        const bool force_restart = body.value("force_restart", false);
        nlohmann::json requested_chunks = nlohmann::json::array();
        if (body.contains("chunk_ids") && body["chunk_ids"].is_array()) {
            for (const auto& item : body["chunk_ids"]) {
                try {
                    int idx = item.get<int>();
                    if (idx >= 0 && idx < SIRIL_NUM_CHUNKS) requested_chunks.push_back(idx);
                } catch (...) {}
            }
        }
        const int max_chunks = body.value("max_chunks", 0);
        std::string job_id = state->job_store.create("pcc_siril_download");
        state->job_store.update_state(job_id, JobState::running, {
            {"catalog_dir", catalog_dir.string()},
            {"pending_chunks", requested_chunks},
            {"total_chunks", 0},
            {"progress", 0.0},
            {"resume_enabled", options.resume},
            {"retry_count", options.retry_count},
        });

        std::thread([state, job_id, catalog_dir, options, force_restart, requested_chunks, max_chunks]() {
            try {
                fs::create_directories(catalog_dir);
                auto missing = missing_siril_chunks(catalog_dir);
                if (!requested_chunks.empty()) {
                    std::vector<int> requested;
                    for (const auto& item : requested_chunks) requested.push_back(item.get<int>());
                    missing.erase(std::remove_if(missing.begin(), missing.end(), [&requested](int value) {
                        return std::find(requested.begin(), requested.end(), value) == requested.end();
                    }), missing.end());
                }
                if (max_chunks > 0 && static_cast<int>(missing.size()) > max_chunks) {
                    missing.resize(static_cast<size_t>(max_chunks));
                }

                nlohmann::json pending_chunks = nlohmann::json::array();
                for (int chunk : missing) pending_chunks.push_back(chunk);
                state->job_store.merge_data(job_id, {
                    {"catalog_dir", catalog_dir.string()},
                    {"pending_chunks", pending_chunks},
                    {"total_chunks", static_cast<int>(missing.size())},
                    {"current_index", 0},
                    {"completed_chunks", 0},
                    {"stage", "download"},
                });

                int completed = 0;
                for (size_t index = 0; index < missing.size(); ++index) {
                    throw_if_job_cancelled(state, job_id);
                    const int chunk = missing[index];
                    state->job_store.merge_data(job_id, {
                        {"current_chunk", chunk},
                        {"current_index", static_cast<int>(index)},
                        {"completed_chunks", completed},
                    });

                    fs::path archive = catalog_dir / ("siril_cat1_healpix8_xpsamp_" + std::to_string(chunk) + ".dat.bz2");
                    if (force_restart) {
                        std::error_code ec;
                        fs::remove(archive, ec);
                    }
                    auto dl = download_file_with_retry(
                        siril_chunk_url(chunk),
                        archive,
                        options,
                        [state, job_id]() { return state->job_store.is_cancelled(job_id); },
                        [state, job_id](std::uintmax_t received, std::uintmax_t total) {
                            set_download_progress(state, job_id, received, total);
                        },
                        [state, job_id, chunk](const nlohmann::json& patch) {
                            nlohmann::json merged = patch;
                            merged["stage"] = "download";
                            merged["chunk_id"] = chunk;
                            set_download_state(state, job_id, merged);
                        });
                    if (!dl.ok) {
                        if (dl.error == "cancelled" || state->job_store.is_cancelled(job_id)) throw JobCancelled();
                        throw std::runtime_error(dl.error);
                    }

                    set_download_state(state, job_id, {{"stage", "decompress"}, {"chunk_id", chunk}, {"archive", archive.string()}});
                    std::string error;
                    if (!decompress_bz2_archive(archive, error)) throw std::runtime_error(error);
                    std::error_code rm_ec;
                    fs::remove(archive, rm_ec);

                    ++completed;
                    if (!pending_chunks.empty()) pending_chunks.erase(pending_chunks.begin());
                    state->job_store.merge_data(job_id, {
                        {"completed_chunks", completed},
                        {"pending_chunks", pending_chunks},
                        {"stage", "download"},
                        {"progress", missing.empty() ? 1.0 : static_cast<double>(completed) / static_cast<double>(missing.size())},
                    });
                }

                state->job_store.merge_data(job_id, {
                    {"missing_after", missing_siril_chunks(catalog_dir)},
                    {"pending_chunks", nlohmann::json::array()},
                    {"completed_chunks", completed},
                    {"stage", "done"},
                    {"progress", 1.0},
                });
                auto snapshot = state->job_store.get(job_id);
                state->job_store.update_state(job_id, JobState::ok, snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object());
            } catch (const JobCancelled&) {
                finish_job_cancelled(state, job_id);
            } catch (const std::exception& e) {
                auto snapshot = state->job_store.get(job_id);
                nlohmann::json data = snapshot && snapshot->data.is_object() ? snapshot->data : nlohmann::json::object();
                data["error"] = e.what();
                if (state->job_store.is_cancelled(job_id)) finish_job_cancelled(state, job_id, data);
                else state->job_store.update_state(job_id, JobState::error, data, e.what());
            }
        }).detach();

        state->ui_event_store.push("tools.pcc.siril.download_missing", "tools.pcc_siril_download_missing", {{"catalog_dir", catalog_dir.string()}}, std::nullopt, job_id);
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    };

    // ---------------------------------------------------------------
    // ASTROMETRY (ASTAP)
    // ---------------------------------------------------------------

    CROW_ROUTE(app, "/api/tools/astrometry/detect").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        std::string astap_cli      = body.is_discarded() ? "" : body.value("astap_cli", "");
        std::string astap_data_dir = body.is_discarded() ? "" : body.value("astap_data_dir", "");
        std::string catalog_dir_str = body.is_discarded() ? "" : body.value("catalog_dir", "");

        if (!astap_cli.empty() && !is_command_name(astap_cli)) {
            fs::path cli_path = expand_user_path(fs::path(astap_cli));
            if (auto denied = denied_path(state->runtime, cli_path, cli_path); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }
        if (!astap_data_dir.empty()) {
            fs::path data_dir_path = expand_user_path(fs::path(astap_data_dir));
            if (auto denied = denied_path(state->runtime, data_dir_path, data_dir_path); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path data_dir = path_from_user_input(astap_data_dir, default_astap_data_dir());
        fs::path catalog_dir = catalog_dir_str.empty() ? data_dir : expand_user_path(fs::path(catalog_dir_str));
        if (!catalog_dir_str.empty()) {
            if (auto denied = denied_path(state->runtime, catalog_dir); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path binary_path = resolve_astap_binary(astap_cli, data_dir);
        bool installed = !binary_path.empty();

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
    ([start_astrometry_install](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        return start_astrometry_install(body);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/install-cli/retry").methods("POST"_method)
    ([start_astrometry_install](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        return start_astrometry_install(body);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/download").methods("POST"_method)
    ([start_astrometry_catalog_download](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        return start_astrometry_catalog_download(body);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/download/retry").methods("POST"_method)
    ([start_astrometry_catalog_download](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        return start_astrometry_catalog_download(body);
    });

    CROW_ROUTE(app, "/api/tools/astrometry/catalog/cancel").methods("POST"_method)
    ([cancel_jobs_by_type](const crow::request&) {
        return cancel_jobs_by_type("astrometry_catalog_download", "tools.astrometry.catalog.cancel", "tools.astrometry_catalog_cancel");
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
        if (!astap_cli.empty() && !is_command_name(astap_cli)) {
            fs::path cli_path = expand_user_path(fs::path(astap_cli));
            if (auto denied = denied_path(state->runtime, cli_path, cli_path); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }
        if (!astap_data_dir.empty()) {
            fs::path data_dir_path = expand_user_path(fs::path(astap_data_dir));
            if (auto denied = denied_path(state->runtime, data_dir_path, data_dir_path); denied) {
                return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
            }
        }

        fs::path resolved_astap_data_dir = path_from_user_input(astap_data_dir, default_astap_data_dir());
        fs::path reported_astap_dir = astap_data_dir.empty() ? resolved_astap_data_dir : expand_user_path(fs::path(astap_data_dir));
        if (auto denied = denied_path(state->runtime, resolved_astap_data_dir, reported_astap_dir); denied) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + *denied, 403, {{"path", *denied}});
        }

        fs::path astap_bin = resolve_astap_binary(astap_cli, resolved_astap_data_dir);
        if (astap_bin.empty()) {
            return err_resp("BAD_REQUEST", "ASTAP CLI not found; install or provide astap_cli path", 400, nlohmann::json::object());
        }

        const int search_radius = body.contains("search_radius_deg") && !body["search_radius_deg"].is_null()
            ? body["search_radius_deg"].get<int>()
            : 180;
        const fs::path wcs_path = guess_wcs_path(fs::path(solve_file));

        std::vector<std::string> args = {
            astap_bin.string(),
            "-f", solve_file,
            "-d", resolved_astap_data_dir.string(),
            "-r", std::to_string(search_radius)
        };

        std::string job_id = state->job_store.create("astrometry_solve");
        state->job_store.update_state(job_id, JobState::running, {
            {"command", args},
            {"wcs_path", wcs_path.string()},
            {"solve_file", solve_file},
            {"astap_data_dir", resolved_astap_data_dir.string()}
        });
        std::thread([state, job_id, args, wcs_path]() {
            auto res = run_subprocess(args, state->runtime.project_root.string());
            nlohmann::json data = {
                {"command", args},
                {"stdout", res.stdout_str},
                {"stderr", res.stderr_str},
                {"exit_code", res.exit_code},
                {"wcs_path", wcs_path.string()},
            };
            if (res.exit_code != 0) {
                state->job_store.update_state(job_id, JobState::error, data,
                    res.stderr_str.empty() ? "ASTAP solve failed" : res.stderr_str.substr(0, 256));
                return;
            }
            if (!fs::exists(wcs_path)) {
                state->job_store.update_state(job_id, JobState::error, data,
                    "ASTAP solve completed without producing a WCS file");
                return;
            }
            data["result"] = parse_astrometry_wcs_summary(wcs_path);
            data["result"]["wcs_path"] = wcs_path.string();
            state->job_store.update_state(job_id, JobState::ok, data);
        }).detach();
        state->ui_event_store.push("tools.astrometry.solve", "tools.astrometry_solve", {{"solve_file", solve_file}, {"wcs_path", wcs_path.string()}}, std::nullopt, job_id);
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
        state->ui_event_store.push("tools.astrometry.save_solved", "tools.astrometry_save_solved", {{"output_path", output.string()}, {"wcs_path", copied_wcs.empty() ? nlohmann::json(nullptr) : nlohmann::json(copied_wcs)}});
        return json_resp({{"output_path", output.string()}, {"wcs_path", copied_wcs.empty() ? nlohmann::json(nullptr) : nlohmann::json(copied_wcs)}});
    });

    // ---------------------------------------------------------------
    // PCC (Siril catalog)
    // ---------------------------------------------------------------

    CROW_ROUTE(app, "/api/tools/pcc/siril/status").methods("GET"_method)
    ([state](const crow::request& req) {
        fs::path catalog_dir = req.url_params.get("catalog_dir") && std::string(req.url_params.get("catalog_dir")).size() > 0
            ? expand_user_path(fs::path(req.url_params.get("catalog_dir")))
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
    ([start_pcc_siril_download](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        return start_pcc_siril_download(body);
    });

    CROW_ROUTE(app, "/api/tools/pcc/siril/download-missing/retry").methods("POST"_method)
    ([start_pcc_siril_download](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded()) body = nlohmann::json::object();
        body["resume"] = true;
        return start_pcc_siril_download(body);
    });

    CROW_ROUTE(app, "/api/tools/pcc/siril/cancel").methods("POST"_method)
    ([cancel_jobs_by_type](const crow::request&) {
        return cancel_jobs_by_type("pcc_siril_download", "tools.pcc.siril.cancel", "tools.pcc_siril_cancel");
    });

    CROW_ROUTE(app, "/api/tools/pcc/check-online").methods("POST"_method)
    ([state](const crow::request&) {
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
        nlohmann::json response = rc == CURLE_OK
            ? nlohmann::json{{"ok", true}, {"latency_ms", latency_ms}}
            : nlohmann::json{{"ok", false}, {"latency_ms", latency_ms}, {"error", curl_easy_strerror(rc)}};
        state->ui_event_store.push("tools.pcc.check_online", "tools.pcc_check_online", response);
        return json_resp(response);
    });

    CROW_ROUTE(app, "/api/tools/pcc/run").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("input_rgb") || !body.contains("output_rgb") || !body.contains("wcs_file"))
            return err_resp("BAD_REQUEST", "input_rgb, output_rgb and wcs_file are required", 400, nlohmann::json::object());

        std::string input_rgb   = expand_user_path(fs::path(body["input_rgb"].get<std::string>())).string();
        std::string output_rgb  = expand_user_path(fs::path(body["output_rgb"].get<std::string>())).string();
        std::string wcs_file    = expand_user_path(fs::path(body.value("wcs_file", ""))).string();
        std::string source      = body.value("source", "auto");
        std::string catalog_dir = expand_user_path(fs::path(body.value("catalog_dir", ""))).string();

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
        if (!fs::exists(fs::path(input_rgb))) {
            return err_resp("PATH_NOT_FOUND", "input_rgb not found", 422, {{"path", input_rgb}});
        }
        if (!fs::exists(fs::path(wcs_file))) {
            return err_resp("PATH_NOT_FOUND", "wcs_file not found", 422, {{"path", wcs_file}});
        }
        if (!catalog_dir.empty() && !fs::exists(fs::path(catalog_dir))) {
            return err_resp("PATH_NOT_FOUND", "catalog_dir not found", 422, {{"path", catalog_dir}});
        }

        std::error_code ec;
        fs::create_directories(fs::path(output_rgb).parent_path(), ec);

        std::vector<std::string> args = {state->runtime.cli_exe, "pcc-run"};
        args.push_back(input_rgb);
        args.push_back(output_rgb);
        if (!wcs_file.empty())    { args.push_back("--wcs"); args.push_back(wcs_file); }
        if (!source.empty())      { args.push_back("--source"); args.push_back(source); }
        if (!catalog_dir.empty()) { args.push_back("--siril-catalog-dir"); args.push_back(catalog_dir); }

        if (auto value = payload_float(body, "mag_limit"); value.has_value())
            { args.push_back("--mag-limit"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "mag_bright_limit"); value.has_value())
            { args.push_back("--mag-bright-limit"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_int(body, "min_stars"); value.has_value())
            { args.push_back("--min-stars"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "sigma_clip"); value.has_value())
            { args.push_back("--sigma-clip"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "aperture_radius_px"); value.has_value())
            { args.push_back("--aperture-radius-px"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "annulus_inner_px"); value.has_value())
            { args.push_back("--annulus-inner-px"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "annulus_outer_px"); value.has_value())
            { args.push_back("--annulus-outer-px"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "chroma_strength"); value.has_value())
            { args.push_back("--chroma-strength"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_float(body, "k_max"); value.has_value())
            { args.push_back("--k-max"); args.push_back(std::to_string(*value)); }
        if (auto value = payload_bool(body, "apply_attenuation"); value.has_value())
            { args.push_back("--apply-attenuation"); args.push_back(*value ? "1" : "0"); }

        std::string job_id = state->subprocess_manager.launch("pcc_run", args,
                                                               state->runtime.project_root.string(),
                                                               "",
                                                               {
                                                                   {"payload", body},
                                                                   {"command", args},
                                                                   {"input_rgb", input_rgb},
                                                                   {"output_rgb", output_rgb},
                                                                   {"wcs_file", wcs_file},
                                                                   {"source", source},
                                                                   {"catalog_dir", catalog_dir}
                                                               });
        state->ui_event_store.push("tools.pcc.run", "tools.pcc_run", {{"input_rgb", input_rgb}, {"output_rgb", output_rgb}, {"wcs_file", wcs_file}, {"source", source}, {"catalog_dir", catalog_dir}}, std::nullopt, job_id);
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

        nlohmann::json response = {
            {"output_rgb",      output_rgb},
            {"output_channels", channels},
        };
        state->ui_event_store.push("tools.pcc.save_corrected", "tools.pcc_save_corrected", response);
        return json_resp(response);
    });
}
