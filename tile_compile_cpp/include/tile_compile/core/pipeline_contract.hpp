#pragma once

// Pipeline contract version --- CFA-forward-drizzle single-method contract
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 17.1, 18.1).
//
// Every run records a `pipeline_contract_version` in run_provenance.json and
// in the run_start event. The value identifies which reconstruction contract
// wrote the run so that resume can fail-closed on a run it cannot
// legitimately continue.
//
//   0  --- legacy runs produced before the single-method cutover. A run
//          stamped 0 is NOT resumable; a full new run from the unchanged
//          sources is the only supported continuation.
//   1  --- CFA-forward-drizzle single-method contract. Stamped by the
//          `reconstruct` pipeline on every run it produces.
//
// A run with NO pipeline_contract_version field is treated exactly like 0.

#include <string>

namespace tile_compile::core {

// The single-method contract version stamped on every new run.
inline constexpr int kPipelineContractVersionSingleMethod = 1;

// The value the runner stamps. The CFA-forward-drizzle path is the only one
// that can produce a run, so this is always the single-method contract.
inline constexpr int kPipelineContractVersionActive =
    kPipelineContractVersionSingleMethod;

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
  return "legacy_prewarp";
}

}  // namespace tile_compile::core
