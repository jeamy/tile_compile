#include "services/pi/pi_scan_manifest.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <stdexcept>
#include <string>

namespace tile_compile::pi {
namespace fs = std::filesystem;
using nlohmann::json;

namespace {
bool is_fits_path(const fs::path& path) {
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* suffix : {".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz"}) {
        const std::string ext(suffix);
        if (name.size() >= ext.size() && name.compare(name.size() - ext.size(), ext.size(), ext) == 0) return true;
    }
    return false;
}
} // namespace

json build_scan_dataset_manifest(const fs::path& input_dir, const json& scan_metrics) {
    if (!fs::is_directory(input_dir) || !scan_metrics.is_object() ||
        !scan_metrics.contains("frames") || !scan_metrics["frames"].is_array())
        throw std::runtime_error("scan input or measured frame list is unavailable");

    std::set<std::string> measured, actual;
    for (const auto& frame : scan_metrics["frames"]) {
        if (!frame.is_object() || !frame.contains("file_name") || !frame["file_name"].is_string())
            throw std::runtime_error("scan metrics contain a frame without a file name");
        const std::string name = frame["file_name"].get<std::string>();
        if (name.empty() || fs::path(name).filename() != fs::path(name) || !is_fits_path(name) || !measured.insert(name).second)
            throw std::runtime_error("scan metrics contain an invalid or duplicate FITS file name");
    }
    if (measured.empty()) throw std::runtime_error("scan metrics contain no frames");

    json manifest = json::array();
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file() || !is_fits_path(entry.path())) continue;
        const std::string name = entry.path().filename().string();
        actual.insert(name);
        manifest.push_back({{"id", name},
                            {"size", entry.file_size()},
                            {"mtime", entry.last_write_time().time_since_epoch().count()}});
    }
    if (actual != measured) throw std::runtime_error("FITS file set differs from the scan metrics");
    std::sort(manifest.begin(), manifest.end(), [](const json& a, const json& b) { return a["id"] < b["id"]; });
    return manifest;
}
} // namespace tile_compile::pi
