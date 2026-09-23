#pragma once
#include "app_state.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

// Scan-metrics result lookup shared by the scan routes and the Jev advice route: a finished scan-metrics
// job in the job store, or the on-disk cache written when such a job completed. Moved out of
// scan_routes.cpp unchanged so both callers use one definition of the cache key.
std::string scan_metrics_cache_key(const std::string& input_path, const std::string& object_name, int frame_count);
nlohmann::json find_disk_cached_scan_metrics(const std::shared_ptr<AppState>& state, const std::string& cache_key);
nlohmann::json find_cached_scan_metrics(const std::shared_ptr<AppState>& state, const std::string& input_path,
                                        const std::string& object_name, int frame_count);
