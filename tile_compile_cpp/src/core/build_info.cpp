#include "tile_compile/core/build_info.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile_build_info_generated.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace tile_compile::core {
namespace {

nlohmann::json feature_flags_json(const std::string& encoded) {
    nlohmann::json flags = nlohmann::json::object();
    std::istringstream stream(encoded);
    std::string item;
    while (std::getline(stream, item, '|')) {
        const auto separator = item.find('=');
        if (separator == std::string::npos || separator == 0) continue;
        const std::string key = item.substr(0, separator);
        const std::string value = item.substr(separator + 1);
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (lower == "on" || lower == "true" || lower == "1") {
            flags[key] = true;
        } else if (lower == "off" || lower == "false" || lower == "0") {
            flags[key] = false;
        } else {
            flags[key] = value;
        }
    }
    return flags;
}

} // namespace

std::filesystem::path current_executable_path() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length > 0 && length < buffer.size()) {
        return std::filesystem::path(std::wstring(buffer.data(), length));
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1u, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(buffer.data(), ec);
        return ec ? std::filesystem::path(buffer.data()) : canonical;
    }
#else
    std::error_code ec;
    auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !path.empty()) return path;
#endif
    return {};
}

nlohmann::json binary_provenance_json(const std::filesystem::path& path) {
    nlohmann::json result = {
        {"path", path.empty() ? std::string() : path.string()},
        {"exists", false},
        {"size_bytes", nullptr},
        {"sha256", nullptr}
    };
    if (path.empty()) {
        result["error"] = "executable_path_unavailable";
        return result;
    }

    std::error_code ec;
    const bool exists = std::filesystem::is_regular_file(path, ec);
    result["exists"] = exists && !ec;
    if (!exists || ec) {
        result["error"] = ec ? ec.message() : "not_a_regular_file";
        return result;
    }

    const auto size = std::filesystem::file_size(path, ec);
    if (!ec) result["size_bytes"] = size;
    try {
        result["sha256"] = sha256_file(path);
    } catch (const std::exception& e) {
        result["error"] = e.what();
    }
    return result;
}

nlohmann::json build_info_json(bool include_runtime_binary) {
    namespace generated = generated_build_info;
    nlohmann::json result = {
        {"schema_version", 1},
        {"component", generated::component},
        {"project_version", generated::project_version},
        {"build_id", generated::build_id},
        {"source", {
            {"git_sha", generated::git_sha},
            {"git_describe", generated::git_describe},
            {"git_dirty", generated::git_dirty},
            {"dirty_tree_digest", generated::dirty_tree_digest},
            {"source_tree_digest", generated::source_tree_digest},
            {"source_file_count", generated::source_file_count}
        }},
        {"toolchain", {
            {"build_type", generated::build_type},
            {"compiler_id", generated::compiler_id},
            {"compiler_version", generated::compiler_version},
            {"compiler_path", generated::compiler_path},
            {"compiler_abi", generated::compiler_abi},
            {"os", generated::os_name},
            {"os_version", generated::os_version},
            {"architecture", generated::architecture}
        }},
        {"dependencies", {
            {"opencv", generated::opencv_version},
            {"cuda", generated::cuda_version}
        }},
        {"feature_flags", feature_flags_json(generated::feature_flags)},
        {"ci", {
            {"github_repository", generated::github_repository},
            {"github_run_id", generated::github_run_id},
            {"github_run_attempt", generated::github_run_attempt},
            {"github_run_url", generated::github_run_url}
        }},
        {"reproducibility", {
            {"source_date_epoch", generated::source_date_epoch},
            {"build_timestamp_utc", generated::build_timestamp_utc}
        }}
    };
    if (include_runtime_binary) {
        result["binary"] = binary_provenance_json(current_executable_path());
    }
    return result;
}

std::string build_info_text() {
    namespace generated = generated_build_info;
    std::ostringstream out;
    out << generated::component << ' ' << generated::project_version
        << " build=" << generated::build_id
        << " git=" << generated::git_describe;
    if (generated::git_dirty) out << " dirty";
    out << " " << generated::build_type
        << " " << generated::compiler_abi
        << " " << generated::os_name << '/' << generated::architecture;
    return out.str();
}

} // namespace tile_compile::core
