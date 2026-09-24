#pragma once
#include "runner_shared.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/registration/registration_sampling_plan.hpp"
namespace tile_compile::runner {
// Fresh: seal cache, geometry, common overlap, quality, paired profile store.
// Resume: validate checkpoint/predecessors before any phase or artifact writes.
bool run_forward_drizzle_stages(const std::string &run_id, const config::Config &cfg,
    const fs::path &run_dir, const registration::RegistrationSamplingPlan &sampling,
    RunnerFrameCache *fresh_cache, core::EventEmitter &emitter, std::ostream &log,
    const std::string &resume_from = "");
}
int resume_forward_drizzle_command(const std::string &run_dir,
                                  const std::string &from_phase);
