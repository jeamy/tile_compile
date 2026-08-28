#pragma once

#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

struct AppState;

namespace tile_compile::pi {

// docs/PI/pi_local_learning_plan_de.md, Abschnitt 4.3 — Schritt 3 implements this convention for
// the first time. project_root/pi_models, override via TILE_COMPILE_PI_MODELS_DIR, same pattern as
// pi_storage_paths.cpp's runs_dir/.pi_memory.
std::filesystem::path default_pi_models_dir(const std::shared_ptr<AppState>& state);
std::filesystem::path pi_models_dir(const std::shared_ptr<AppState>& state);

// SHA-256 of a file's raw bytes, hex-encoded; empty string if the file cannot be read. Used to pin
// a scan-domain model to the config_schema it was trained against (see predict_param_nn()) — public
// so tests and the offline training script's cross-check can compute the same value independently.
std::string compute_file_sha256(const std::filesystem::path& path);

// Layout per (domain, target path) (Abschnitt 4.3): pi_models_dir()/<domain>/<target_path>/v<N>/
// {metadata.json, reference_points.jsonl} — domain is "scan" (Schritt 3) or "live_edit" (Schritt 5),
// matching the two example subtrees in Abschnitt 4.3 (scan/bge.method/... vs.
// live_edit/brightness.midtones/...). No training pipeline exists yet (Schritt 6) — these
// directories are normally absent, and that is the expected bootstrap state (Abschnitt 4.3
// "Bootstrap-Realismus"), not an error. predict_param_nn() must degrade cleanly when they are.
struct ParamShadowPrediction {
    bool available = false;
    nlohmann::json predicted_value = nullptr;
    double confidence = 0.0;
    int n_reference_points = 0;
    int k_used = 0;
    std::string model_version;
    std::string reason;  // set when available == false
};

// Weighted nearest-neighbor prediction (Abschnitt 4.2: the explicitly named low-data fallback —
// "funktioniert schon ab ~20 Beispielen, kein Training nötig, nur Distanzsuche"). Reads the highest
// version directory under pi_models_dir()/<domain>/<target_path>/ that has both metadata.json and
// reference_points.jsonl; returns available=false with a reason otherwise. Pure read, no side
// effect — safe to call from a shadow-logging hot path.
//
// Regression vs. classification is decided per call from the reference data itself, not a
// parameter: if every one of the k nearest points' "value" is a JSON number, the prediction is an
// inverse-distance-weighted mean (continuous targets — e.g. live_edit's brightness.midtones);
// otherwise it falls back to the Schritt-3 weighted majority vote (discrete targets — e.g. scan's
// bge.method). A reference set must not mix numbers and non-numbers for the same target_path.
ParamShadowPrediction predict_param_nn(const std::shared_ptr<AppState>& state,
                                       const std::string& domain,
                                       const std::string& target_path,
                                       const nlohmann::json& feature_vector);

// Schritt 3 (docs/PI/pi_local_learning_plan_de.md, Abschnitt 7): for the PoC target paths
// (bge.method, normalization.mode), builds the scan feature vector, runs predict_param_nn(), and
// logs {feature_vector, model prediction, LLM's actual validated_updates value, agreement} to
// pi_models_dir()/scan/<path>/shadow_predictions.jsonl — purely for later comparison. Never
// influences validated_updates/action_plan; best-effort, swallows its own errors so a shadow
// logging failure can never break scan analysis itself.
void log_scan_param_shadow_predictions(const std::shared_ptr<AppState>& state,
                                       const nlohmann::json& scan_metrics,
                                       const nlohmann::json& scan_result,
                                       const nlohmann::json& validated_updates);

// Schritt 5 (docs/PI/pi_local_learning_plan_de.md, Abschnitt 7) — the parameter-regression half
// only (see the design note in that section for why the free-text "op-intent classifier" half is
// explicitly NOT implemented here). Called once per applied, chat-sourced live-edit operation, with
// the feature vector of the image BEFORE the operation (what should drive a prediction of good
// params) and the actual params the LLM chose. For each numeric field in "actual_params", predicts
// via predict_param_nn(domain="live_edit", target_path="<op_type>.<field>") and logs
// {feature_vector, model prediction, actual value, agreement-by-tolerance} to
// pi_models_dir()/live_edit/<op_type>.<field>/shadow_predictions.jsonl. Never influences the
// applied operation; best-effort, swallows its own errors.
void log_live_edit_param_shadow_predictions(const std::shared_ptr<AppState>& state,
                                            const std::string& op_type,
                                            const nlohmann::json& feature_vector_before,
                                            const nlohmann::json& actual_params);

} // namespace tile_compile::pi
