#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::web {

struct HmePreviewResult {
    bool ok{false};
    int status{500};
    std::string error;
    std::vector<unsigned char> png;
    nlohmann::json diagnostics = nlohmann::json::object();
};

HmePreviewResult create_hme_preview(const std::filesystem::path& run_dir,
                                    const nlohmann::json& params);

} // namespace tile_compile::web
