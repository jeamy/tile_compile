#pragma once
#include "../job_store.hpp"
#include <optional>
#include <nlohmann/json.hpp>

/// @brief Returns the newest input-scan job known to the job store.
std::optional<Job> latest_scan_job(const InMemoryJobStore& store);
/// @brief Converts the latest scan job into the compact UI summary shape.
nlohmann::json summarize_scan_job(const std::optional<Job>& job,
                                  const std::string& fallback_input_path = "");
/// @brief Returns quality-related scan summary data for dashboard endpoints.
nlohmann::json scan_quality(const InMemoryJobStore& store);
/// @brief Returns scan guardrail warnings and limits for dashboard endpoints.
nlohmann::json scan_guardrails(const InMemoryJobStore& store);
