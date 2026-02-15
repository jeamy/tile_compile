#pragma once

#include <string>

int run_pipeline_command(const std::string &config_path,
                         const std::string &input_dir,
                         const std::string &runs_dir,
                         const std::string &project_root,
                         const std::string &run_id_override,
                         bool dry_run,
                         int max_frames,
                         int max_tiles,
                         bool config_from_stdin);
