#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace tile_compile::pi {

// The single copy of the resume section scope (mirrors allowed_reconstruction_resume_sections and the downstream resume
// gate of the runner). Phases are the names accepted by `resume-reconstruction --from-phase`.
// nullptr: the phase has no section scope (unknown phase).
const std::set<std::string>* resume_allowed_sections(const std::string& phase_upper);

// Resume phases from the latest (cheapest) to the earliest.
const std::vector<std::string>& resume_phases_latest_first();

// The LATEST phase a resume may start at so that a change confined to config section `top_level_section` is still legal
// (every later phase rejects the change), i.e. the smallest allowed-section set that contains it. std::nullopt: no resume
// phase permits the section (for example `reconstruction`), so the change needs a new full run. This is a scope
// statement only: it does not say that the section is consumed at that phase, and it checks neither caches nor
// provenance (the resume dry run stays the authority for feasibility).
std::optional<std::string> min_resume_phase_for_section(const std::string& top_level_section);

// Same for a dotted config path (`reconstruction.drizzle.pixfrac` -> section `reconstruction`).
std::optional<std::string> min_resume_phase_for_path(const std::string& dotted_path);

} // namespace tile_compile::pi
