#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct AppState;

namespace tile_compile::pi {

// Live-editor analogue of build_scan_feature_vector() (pi_feature_vector.hpp) — same
// pi.feature-vector.v1 schema, domain "live_edit". Kept in this file rather than
// pi_feature_vector.{hpp,cpp} because it needs OpenCV (cv::Mat pixel statistics); the scan
// feature vector is a pure JSON projection and should not force an OpenCV dependency onto every
// file that includes it (docs/PI/pi_local_learning_plan_de.md, Abschnitt 4.1).
//
// Computed from original_fits (the image state the session STARTED from), not current_fits: the
// point is predicting a good operation for an image someone is about to edit, not describing the
// edited result.
nlohmann::json build_live_edit_feature_vector(const cv::Mat& fits);

// Schritt 4 (docs/PI/pi_local_learning_plan_de.md, Abschnitt 0.4/7). Called once, at Live-Editor
// session close, AFTER the session's durable copy (operation_history) has already been persisted —
// never on every undo/redo/adjust, unlike the general-purpose try_persist_live_session() this sits
// alongside. Builds one "live_edit_operation" memory candidate per op-type that survived to the
// final undo_stack (label = terminal value, per Abschnitt 0.4 — not every adjust_step increment,
// even though repeated adjust clicks each push their own stack entry) with outcome.retained=true,
// and one per op-type that appears in edit_history but was fully undone before close, with
// outcome.retained=false. No same-frames delta concept applies here (unlike scan) — "retained
// until session close" is itself the observed outcome, known immediately, not awaited from a
// future run. Best-effort: swallows its own errors, never breaks session close.
void record_live_edit_session_outcome(const std::shared_ptr<AppState>& state,
                                      const std::string& run_id,
                                      const cv::Mat& original_fits,
                                      const std::vector<nlohmann::json>& final_undo_stack,
                                      const nlohmann::json& edit_history);

} // namespace tile_compile::pi
