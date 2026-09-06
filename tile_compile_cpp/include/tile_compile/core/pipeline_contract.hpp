#pragma once

// Pipeline contract version --- milestone M0 of the CFA-forward-drizzle plan
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 17.1, 18.1, 23/M0).
//
// Every run records a `pipeline_contract_version` in run_provenance.json and in
// the run_start event. The value identifies which reconstruction contract wrote
// the run so that resume (runner_resume.cpp, section 18.1) can fail-closed on a
// run it cannot legitimately continue.
//
//   0  --- legacy / cutover-in-progress. The active pipeline is still the
//          PREWARP-AQMH (or Classic) path. A run stamped 0 is NOT resumable with
//          the single-method runner; a full new run from the unchanged sources
//          is the only supported continuation.
//   1  --- CFA-forward-drizzle single-method contract. Reserved; only stamped
//          once the new pipeline is the one that actually produced the run
//          (milestone M10). See kPipelineContractVersionSingleMethod.
//
// A run with NO pipeline_contract_version field is treated exactly like 0.

#include <string>

namespace tile_compile::core {

// The target single-method contract version. Stamped on real runs only from the
// M10 cutover onward.
inline constexpr int kPipelineContractVersionSingleMethod = 1;

// The value the runner currently stamps. While the active pipeline is still the
// legacy PREWARP-AQMH path this is 0 ("cutover in progress"). It becomes
// kPipelineContractVersionSingleMethod at M10 when the method branch is removed
// and the CFA-forward-drizzle path is the only one that can produce a run.
inline constexpr int kPipelineContractVersionActive = 0;

// True when `version` identifies a run that the single-method runner may
// directly resume. Anything below the single-method contract (including a
// missing field, represented by the caller as a negative value) is legacy.
inline constexpr bool pipeline_contract_is_single_method(int version) {
  return version == kPipelineContractVersionSingleMethod;
}

// Stable human-readable label for the contract a run was written under.
inline std::string pipeline_contract_label(int version) {
  if (version == kPipelineContractVersionSingleMethod)
    return "cfa_forward_drizzle_multiband";
  return "legacy_prewarp_aqmh_cutover_in_progress";
}

// Milestone M0-M2 (plan section 23/M0): the production runner must refuse every
// run mutation while the single-method pipeline is under construction. Returns
// the reason string when the caller should abort, or an empty string when the
// run may proceed. The frozen legacy-reference build
// (-DTILE_COMPILE_LEGACY_REFERENCE) always returns "" so regression / bisection
// runs stay possible.
inline std::string pipeline_unavailable_reason() {
#ifdef TILE_COMPILE_LEGACY_REFERENCE
  return {};
#else
  if (pipeline_contract_is_single_method(kPipelineContractVersionActive)) {
    return {};  // M10+: the single-method pipeline is the real one.
  }
  return "PIPELINE_UNAVAILABLE_DURING_CUTOVER --- the production runner is "
         "locked while the CFA-forward-drizzle single-method pipeline is under "
         "construction (plan milestones M0-M2). Use the tile_compile_legacy_"
         "reference binary for reproducible regression / bisection runs.";
#endif
}

}  // namespace tile_compile::core
