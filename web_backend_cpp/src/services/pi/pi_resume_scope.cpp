#include "services/pi/pi_resume_scope.hpp"

#include <map>

namespace tile_compile::pi {
namespace {

// GLOBAL_QUALITY/FORWARD_DRIZZLE: the reused reconstruction predecessors are validated against their own checkpoint
// hashes, so a change confined to these sections cannot desync cache from config. `reconstruction` is deliberately
// excluded (quality.pyramid is only consumed while SOURCE_QUALITY_MAPS is rebuilt, which this path skips).
const std::set<std::string> kReconstructionResumeBase = {
    "output", "data", "linearity", "calibration", "normalization", "registration", "dithering", "chroma_denoise",
    "luma_denoise", "astrometry", "pcc", "hypermetric_stretch", "bge", "stacking", "runtime_limits"};

const std::map<std::string, std::set<std::string>>& table() {
    static const std::map<std::string, std::set<std::string>> t = {
        {"HYPERMETRIC_STRETCH", {"hypermetric_stretch", "runtime_limits"}},
        {"PCC", {"pcc", "chroma_denoise", "hypermetric_stretch", "runtime_limits"}},
        {"BGE", {"bge", "pcc", "chroma_denoise", "luma_denoise", "hypermetric_stretch", "runtime_limits"}},
        {"ASTROMETRY", {"astrometry", "bge", "pcc", "chroma_denoise", "luma_denoise", "hypermetric_stretch", "runtime_limits"}},
        // FORWARD_DRIZZLE does not recompute GLOBAL_QUALITY, so a global_metrics edit is only legal from GLOBAL_QUALITY.
        {"GLOBAL_QUALITY", [] { auto s = kReconstructionResumeBase; s.insert("global_metrics"); return s; }()},
        {"FORWARD_DRIZZLE", kReconstructionResumeBase},
    };
    return t;
}

} // namespace

const std::set<std::string>* resume_allowed_sections(const std::string& phase_upper) {
    const auto it = table().find(phase_upper);
    return it == table().end() ? nullptr : &it->second;
}

const std::vector<std::string>& resume_phases_latest_first() {
    static const std::vector<std::string> order = {"HYPERMETRIC_STRETCH", "PCC", "BGE", "ASTROMETRY", "FORWARD_DRIZZLE", "GLOBAL_QUALITY"};
    return order;
}

std::optional<std::string> min_resume_phase_for_section(const std::string& section) {
    for (const auto& phase : resume_phases_latest_first())
        if (resume_allowed_sections(phase)->count(section)) return phase;
    return std::nullopt;
}

std::optional<std::string> min_resume_phase_for_path(const std::string& dotted_path) {
    return min_resume_phase_for_section(dotted_path.substr(0, dotted_path.find('.')));
}

} // namespace tile_compile::pi
