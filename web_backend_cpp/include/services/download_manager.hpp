#pragma once
#include <filesystem>
#include <string>
#include <functional>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

/// @brief User-facing network download policy.
/// @details Configures timeout, retry count, backoff, resume behavior, and user agent for
/// tool/catalog downloads started by backend routes.
struct DownloadOptions {
    long timeout_s{120};
    int retry_count{2};
    double retry_backoff_s{1.5};
    bool resume{true};
    std::string user_agent{"TileCompileGUI2/1.0"};
};

/// @brief Final outcome and transfer metadata for a download attempt sequence.
struct DownloadResult {
    bool ok{false};
    fs::path dest_path;
    std::string error;
    int attempts{0};
    std::uintmax_t bytes_written{0};
    std::uintmax_t bytes_total{0};
    long status_code{0};
    bool resumed{false};
};

/// @brief Callback checked periodically to stop a download cooperatively.
using DownloadShouldCancel = std::function<bool()>;
/// @brief Callback receiving transferred and total byte counts.
using DownloadProgressCallback = std::function<void(std::uintmax_t, std::uintmax_t)>;
/// @brief Callback receiving structured state changes for job data overlays.
using DownloadStateCallback = std::function<void(const nlohmann::json&)>;

/// @brief Downloads a URL to disk with retry, resume, progress, and cancellation support.
DownloadResult download_file_with_retry(const std::string& url,
                                        const fs::path& dest,
                                        const DownloadOptions& options,
                                        DownloadShouldCancel should_cancel = {},
                                        DownloadProgressCallback on_progress = {},
                                        DownloadStateCallback on_state = {});
