#pragma once
#include <filesystem>
#include <string>
#include <functional>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct DownloadOptions {
    long timeout_s{120};
    int retry_count{2};
    double retry_backoff_s{1.5};
    bool resume{true};
    std::string user_agent{"TileCompileGUI2/1.0"};
};

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

using DownloadShouldCancel = std::function<bool()>;
using DownloadProgressCallback = std::function<void(std::uintmax_t, std::uintmax_t)>;
using DownloadStateCallback = std::function<void(const nlohmann::json&)>;

DownloadResult download_file_with_retry(const std::string& url,
                                        const fs::path& dest,
                                        const DownloadOptions& options,
                                        DownloadShouldCancel should_cancel = {},
                                        DownloadProgressCallback on_progress = {},
                                        DownloadStateCallback on_state = {});
