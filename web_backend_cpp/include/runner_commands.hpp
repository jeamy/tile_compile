#pragma once

#include <algorithm>
#include <string>
#include <vector>

/// @brief Subcommands of tile_compile_runner that start or continue a run.
/// @details Must match the subcommands in tile_compile_cpp/apps/runner_main.cpp.
/// The backend uses this to recognise a live runner process for a run.
inline bool is_runner_run_subcommand(const std::string& arg) {
    return arg == "reconstruct" || arg == "resume-reconstruction" || arg == "preprocess";
}

/// @brief True if argv (after the executable) names a run subcommand.
inline bool argv_has_runner_run_subcommand(const std::vector<std::string>& argv) {
    if (argv.size() < 2) return false;
    return std::any_of(argv.begin() + 1, argv.end(), is_runner_run_subcommand);
}
