#pragma once

#include "runner_shared.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"
#include <functional>
#include <ostream>

namespace tile_compile::runner {
// Restore photometry once on the selected per-channel working-space files.
// Raw and reconstructed_* remain immutable inputs. Outputs retain full canvas.
//
// P0.2/P0.3 (redundant_data_reload_analysis, "schwarze Artefakte"): distinguishes
// "never geometrically covered" from "clipped away after full coverage" instead
// of conflating both into canvas_mask, and warns (does not fail the run --
// see the comment above the warning call) when a nontrivial fraction of
// geometrically covered pixels were still zeroed by the cross-channel
// support AND. `emitter`/`run_id`/`log` are optional so existing callers
// (tests) that only need the FITS/JSON outputs keep working unchanged.
void write_forward_downstream_inputs(const fs::path &run_dir,
    const registration::RegistrationSamplingPlan &sampling,
    const config::ReconstructionDrizzleConfig &drizzle,
    core::EventEmitter *emitter = nullptr, const std::string &run_id = {},
    std::ostream *log = nullptr);
// Shared RGB postprocessing. Forward runs supply newly committed linear
// inputs and never reuse a WCS or Classic tile metrics from a previous image.
int run_rgb_downstream(const fs::path &run_dir, const std::string &run_id,
    const config::Config &cfg, std::string phase_upper, std::ostream &log,
    const std::function<bool(const std::string &)> &abort_if_runtime_limit_exceeded,
    bool forward = false);
}
