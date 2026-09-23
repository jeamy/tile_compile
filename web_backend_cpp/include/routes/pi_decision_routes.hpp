#pragma once
#include "crow_app.hpp"
#include "../app_state.hpp"
#include <memory>

namespace tile_compile::routes {

/// Jev pre-run advice (docs/PI/pi_jev_implementierungsplan_de.md, M4). All decision logic lives in
/// services/pi/pi_decision_service; these routes only translate HTTP <-> AdviceRequest.
///   POST /api/scan/decisions                 -> 202 {proposal_id}; starts an advice, never a run
///   GET  /api/scan/decisions/<id>            -> persisted status/result (survives restarts)
///   POST /api/scan/decisions/<id>/apply      -> patched config DRAFT (yaml) after CAS + revalidation;
///                                               never saves a file and never starts a run
///   GET  /api/pi/decisions/status            -> sidecar mode/model/key presence (no key value)
void register_pi_decision_routes(CrowApp& app, std::shared_ptr<AppState> state);

} // namespace tile_compile::routes
