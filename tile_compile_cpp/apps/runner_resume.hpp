#pragma once

#include <string>

/// Resume an existing run from the requested pipeline phase.
///
/// The command inspects the run directory, preserves/configures revision
/// metadata as needed, and replays the minimum required downstream phases from
/// `from_phase`. The integer result is the CLI/process exit code.
int resume_command(const std::string &run_dir_path,
                   const std::string &from_phase);
