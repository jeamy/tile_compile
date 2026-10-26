#pragma once
// Record builders for the LLM and config-apply write points (docs/PI/DECISIONTRACE/pi_decision_trace_plan_de.md §2.2, §2.4).
// Builders only shape the record; callers append it through PiDecisionRecordStore and treat failures as best effort.

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace tile_compile::pi {

class PiDecisionRecordStore;

// One llm_proposal per served analysis: actor llm, basis llm_hypothesis, recommendations as alternatives.
// Context is "analysis:<analysis_id>" so the later config_apply can find its parent.
nlohmann::json build_llm_proposal_record(const nlohmann::json& analysis, const std::string& analysis_id);

struct ConfigApplyEvent {
    std::string context_id;          // "analysis:<id>" for scan-AI applies, empty for preview applies
    std::string action_plan_id;      // empty for scan-AI applies
    std::string preview_id;          // empty for legacy applies
    std::string revision_id;
    std::string config_sha256;
    std::string idempotency_key;     // empty disables deduplication
    std::string parent_decision_id;  // llm_proposal the apply accepted from, if any
    std::vector<std::string> paths;
};

nlohmann::json build_config_apply_record(const ConfigApplyEvent& event);

// Id of the most recent llm_proposal for an analysis, or empty when none was recorded.
std::string find_llm_proposal_id(const PiDecisionRecordStore& store, const std::string& analysis_id);

} // namespace tile_compile::pi
