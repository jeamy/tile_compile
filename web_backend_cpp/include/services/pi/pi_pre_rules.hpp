#pragma once

#include "services/pi/pi_decision_policy.hpp"

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

// Result of building the candidate set the provider may choose from. Deterministic and
// order-independent: candidates are evaluated independently and reported sorted by id.
//   applicable: [{candidate_id, candidate_version, group, experimental, requires_review, updates,
//                 evidence_refs, rationale}]   (always contains keep_current and insufficient_evidence)
//   excluded:   [{candidate_id, reasons[]}]    (why a candidate is NOT offered; sorted, unique)
struct PreRunCandidates {
    nlohmann::json applicable = nlohmann::json::array();
    nlohmann::json excluded = nlohmann::json::array();
    int catalog_version = 0;
    std::string policy_version;
    // ids the provider may name, for pi.decisions.request.v1.allowed_candidates
    std::vector<std::string> allowed_ids() const;
};

// Pure. Uses validate_decision_candidate() for every candidate, so what is OFFERED and what would
// later be ACCEPTED come from one code path (a candidate offered here cannot be rejected later for
// a reason that was already knowable).
PreRunCandidates build_pre_run_candidates(const nlohmann::json& state,
                                          const nlohmann::json& current_config,
                                          const DecisionPolicy& policy,
                                          const DecisionCatalog& catalog,
                                          const ConfigValidator& validate_config);

} // namespace tile_compile::pi
