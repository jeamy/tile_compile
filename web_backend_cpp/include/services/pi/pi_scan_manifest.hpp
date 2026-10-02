#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

namespace tile_compile::pi {

// Snapshot the files measured by scan-metrics and fail if the current FITS file set differs.
// This is metadata identity (name, size, mtime), not proof of unchanged file contents.
nlohmann::json build_scan_dataset_manifest(const std::filesystem::path& input_dir,
                                           const nlohmann::json& scan_metrics);

} // namespace tile_compile::pi
