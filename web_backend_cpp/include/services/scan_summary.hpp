#pragma once
#include "../job_store.hpp"
#include <optional>
#include <nlohmann/json.hpp>

std::optional<Job> latest_scan_job(const InMemoryJobStore& store);
nlohmann::json summarize_scan_job(const std::optional<Job>& job,
                                  const std::string& fallback_input_path = "");
nlohmann::json scan_quality(const InMemoryJobStore& store);
nlohmann::json scan_guardrails(const InMemoryJobStore& store);
