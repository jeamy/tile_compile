#pragma once

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

inline constexpr const char* kActionPlanSchemaVersion = "pi.action-plan.v1";

nlohmann::json build_scan_analysis_action_plan(const nlohmann::json& analysis,
                                               const nlohmann::json& updates);

} // namespace tile_compile::pi
