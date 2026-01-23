#include "tile_compile/runner/utils.hpp"

#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace tile_compile::runner {

std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern) {
    std::vector<fs::path> paths;
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        return paths;
    }
    
    // Convert glob pattern to regex
    std::string regex_pattern = pattern;
    std::replace(regex_pattern.begin(), regex_pattern.end(), '.', '\\');
    std::replace(regex_pattern.begin(), regex_pattern.end(), '*', '.');
    regex_pattern = ".*" + regex_pattern;
    
    std::regex pattern_regex(regex_pattern, std::regex::icase);
    
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, pattern_regex)) {
                paths.push_back(entry.path());
            }
        }
    }
    
    std::sort(paths.begin(), paths.end());
    return paths;
}

void copy_config(const fs::path& config_path, const fs::path& out_path) {
    fs::create_directories(out_path.parent_path());
    fs::copy_file(config_path, out_path, fs::copy_options::overwrite_existing);
}

void safe_symlink_or_copy(const fs::path& src, const fs::path& dst) {
    fs::create_directories(dst.parent_path());
    
    if (fs::exists(dst) || fs::is_symlink(dst)) {
        return;
    }
    
    try {
        fs::create_symlink(fs::absolute(src), dst);
    } catch (...) {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }
}

std::string sha256_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }
    
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    
    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        SHA256_Update(&ctx, buffer, file.gcount());
    }
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);
    
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::string sha256_bytes(const std::vector<uint8_t>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), hash);
    
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    
    return buffer;
}

std::string json_dumps_canonical(const nlohmann::json& obj) {
    return obj.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
}

} // namespace tile_compile::runner
