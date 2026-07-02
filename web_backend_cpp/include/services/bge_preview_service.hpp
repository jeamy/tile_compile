#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::web {

struct BgePreviewResult {
    bool ok{false};
    int status{500};
    std::string error;
    std::vector<unsigned char> png;
    nlohmann::json diagnostics = nlohmann::json::object();
};

BgePreviewResult create_bge_preview(const std::filesystem::path& run_dir,
                                    const nlohmann::json& params,
                                    const nlohmann::json& polygons,
                                    const std::string& view);

} // namespace tile_compile::web
