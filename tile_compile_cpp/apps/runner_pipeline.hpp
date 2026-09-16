#pragma once

#include <string>

/// Run a new reconstruction pipeline invocation from CLI arguments.
///
/// The command wrapper resolves configuration/input/run directories, applies
/// optional command-line overrides, supports config-from-stdin workflows, and
/// delegates to the production runner. The return value is suitable as the
/// process exit code for `tile_compile_runner reconstruct`.
int run_pipeline_command(const std::string &config_path,
                         const std::string &input_dir,
                         const std::string &runs_dir,
                         const std::string &project_root,
                         const std::string &run_id_override,
                         bool dry_run,
                         int max_frames,
                         bool config_from_stdin);
