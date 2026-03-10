#pragma once
#include <filesystem>
#include <string>
#include <atomic>
#include <functional>

namespace fs = std::filesystem;

struct DownloadResult {
    bool ok{false};
    fs::path dest_path;
    std::string error;
};

DownloadResult download_file(const std::string& url,
                              const fs::path& dest,
                              std::atomic<bool>& cancelled,
                              std::function<void(double)> on_progress = {});
