#pragma once

#include <string>

/// Run the separate raw-preprocessing pipeline.
int preprocess_command(const std::string& config_path,
                       const std::string& runs_dir,
                       const std::string& project_root,
                       const std::string& run_id_override,
                       bool config_from_stdin);
