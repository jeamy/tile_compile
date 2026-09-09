#pragma once

#include "runner_shared.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"
#include <functional>

namespace tile_compile::runner {
// Restore photometry once on the selected per-channel working-space files.
// Raw and reconstructed_* remain immutable inputs. Outputs retain full canvas.
void write_forward_downstream_inputs(const fs::path &run_dir,
    const registration::RegistrationSamplingPlan &sampling,
    const config::ReconstructionDrizzleConfig &drizzle);
// Shared RGB postprocessing. Forward runs supply newly committed linear
// inputs and never reuse a WCS or Classic tile metrics from a previous image.
int run_rgb_downstream(const fs::path &run_dir, const std::string &run_id,
    const config::Config &cfg, std::string phase_upper, std::ostream &log,
    const std::function<bool(const std::string &)> &abort_if_runtime_limit_exceeded,
    bool forward = false);
}
