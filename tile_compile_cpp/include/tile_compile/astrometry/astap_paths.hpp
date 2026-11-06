#pragma once

#include <filesystem>
#include <string>

namespace tile_compile::astrometry {

/// ASTAP data directory and binary as used by the solver phases.
struct AstapPaths {
  std::string data_dir;
  std::filesystem::path binary;
};

/// Per-user default ASTAP data directory
/// (`%LOCALAPPDATA%\tile_compile\astap` or `~/.local/share/tile_compile/astap`).
/// Empty when the environment provides no base directory.
std::string default_astap_data_dir();

/// Resolve an ASTAP CLI binary path across platforms. Tries the configured
/// binary, the data directory, PATH, and (Windows) common install locations.
/// Returns an empty path when nothing is found.
std::filesystem::path resolve_astap_binary_path(const std::string &astap_bin_cfg,
                                                const std::string &astap_data_dir);

/// Resolve the ASTAP data directory and binary. An empty or non-existent
/// `data_dir_cfg` falls back to `default_astap_data_dir()`; a non-existent
/// `binary_cfg` falls through to the data directory and PATH lookup. When the
/// resolved binary lies outside the data directory, its parent directory is
/// used as data directory.
AstapPaths resolve_astap_paths(const std::string &binary_cfg,
                               const std::string &data_dir_cfg);

} // namespace tile_compile::astrometry
