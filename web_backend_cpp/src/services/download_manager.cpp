#include "services/download_manager.hpp"
#include <curl/curl.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <sstream>
#include <string_view>
#include <thread>

namespace {

struct DownloadContext {
    fs::path dest;
    DownloadShouldCancel should_cancel;
    DownloadProgressCallback on_progress;
    DownloadStateCallback on_state;
    DownloadOptions options;
    int attempt{1};
    std::uintmax_t requested_existing{0};
    std::uintmax_t effective_existing{0};
    std::uintmax_t bytes_total{0};
    long status_code{0};
    bool resumed{false};
    bool headers_processed{false};
    bool stream_ready{false};
    std::ofstream out;
};

std::string trim_ascii(std::string_view raw) {
    size_t start = 0;
    while (start < raw.size() && std::isspace(static_cast<unsigned char>(raw[start]))) ++start;
    size_t end = raw.size();
    while (end > start && std::isspace(static_cast<unsigned char>(raw[end - 1]))) --end;
    return std::string(raw.substr(start, end - start));
}

std::uintmax_t parse_total_from_content_range(const std::string& value) {
    const auto slash = value.rfind('/');
    if (slash == std::string::npos) return 0;
    const std::string total = trim_ascii(std::string_view(value).substr(slash + 1));
    if (total.empty() || total == "*") return 0;
    try {
        return static_cast<std::uintmax_t>(std::stoull(total));
    } catch (...) {
        return 0;
    }
}

bool ensure_output_stream(DownloadContext& ctx) {
    if (ctx.stream_ready) return ctx.out.good();
    const bool append = ctx.requested_existing > 0 && ctx.status_code == 206;
    ctx.resumed = append;
    ctx.effective_existing = append ? ctx.requested_existing : 0;
    std::ios::openmode mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
    ctx.out.open(ctx.dest, mode);
    ctx.stream_ready = true;
    if (ctx.on_state) {
        ctx.on_state({
            {"attempt", ctx.attempt},
            {"status_code", ctx.status_code},
            {"resumed", ctx.resumed},
            {"existing_bytes", ctx.effective_existing},
            {"bytes_total", ctx.bytes_total},
        });
    }
    return ctx.out.good();
}

size_t write_to_file(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* ctx = static_cast<DownloadContext*>(userdata);
    if (!ctx) return 0;
    if (!ensure_output_stream(*ctx)) return 0;
    ctx->out.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(size * nmemb));
    return ctx->out.good() ? size * nmemb : 0;
}

size_t header_cb(char* buffer, size_t size, size_t nitems, void* userdata) {
    auto* ctx = static_cast<DownloadContext*>(userdata);
    if (!ctx) return 0;
    const std::string line(buffer, size * nitems);
    const std::string trimmed = trim_ascii(line);
    if (trimmed.rfind("HTTP/", 0) == 0) {
        std::istringstream iss(trimmed);
        std::string http_version;
        long status = 0;
        iss >> http_version >> status;
        if (status > 0) ctx->status_code = status;
        ctx->headers_processed = true;
    } else {
        const auto colon = trimmed.find(':');
        if (colon != std::string::npos) {
            std::string key = trimmed.substr(0, colon);
            std::string value = trim_ascii(std::string_view(trimmed).substr(colon + 1));
            std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (key == "content-range") {
                std::uintmax_t total = parse_total_from_content_range(value);
                if (total > 0) ctx->bytes_total = total;
            } else if (key == "content-length" && ctx->bytes_total == 0) {
                try {
                    std::uintmax_t content_length = static_cast<std::uintmax_t>(std::stoull(value));
                    ctx->bytes_total = (ctx->requested_existing > 0 && ctx->status_code == 206)
                        ? (ctx->requested_existing + content_length)
                        : content_length;
                } catch (...) {}
            }
        }
    }
    return size * nitems;
}

int progress_cb(void* userdata, curl_off_t dltotal, curl_off_t dlnow,
                curl_off_t, curl_off_t) {
    auto* ctx = static_cast<DownloadContext*>(userdata);
    if (!ctx) return 0;
    if (ctx->should_cancel && ctx->should_cancel()) return 1;
    if (ctx->on_progress) {
        std::uintmax_t total = ctx->bytes_total;
        if (total == 0 && dltotal > 0) {
            total = (ctx->requested_existing > 0 && ctx->status_code == 206)
                ? (ctx->requested_existing + static_cast<std::uintmax_t>(dltotal))
                : static_cast<std::uintmax_t>(dltotal);
            ctx->bytes_total = total;
        }
        std::uintmax_t received = ctx->effective_existing + (dlnow > 0 ? static_cast<std::uintmax_t>(dlnow) : 0U);
        ctx->on_progress(received, total);
    }
    return 0;
}

DownloadResult download_once(const std::string& url,
                             const fs::path& dest,
                             const DownloadOptions& options,
                             int attempt,
                             DownloadShouldCancel should_cancel,
                             DownloadProgressCallback on_progress,
                             DownloadStateCallback on_state) {
    DownloadResult result;
    result.dest_path = dest;
    result.attempts = attempt;

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error = "curl_easy_init failed";
        return result;
    }

    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);
    DownloadContext ctx{
        dest,
        std::move(should_cancel),
        std::move(on_progress),
        std::move(on_state),
        options,
        attempt,
        options.resume && fs::exists(dest) ? fs::file_size(dest, ec) : 0U,
    };
    if (ec) ctx.requested_existing = 0;

    if (ctx.on_state) {
        ctx.on_state({
            {"attempt", ctx.attempt},
            {"retry_count", options.retry_count},
            {"resume_enabled", options.resume},
            {"existing_bytes", ctx.requested_existing},
        });
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, options.timeout_s);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, options.user_agent.c_str());
    if (ctx.requested_existing > 0 && options.resume) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(ctx.requested_existing));
    }

    CURLcode rc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &ctx.status_code);
    if (!ctx.stream_ready) ensure_output_stream(ctx);
    if (ctx.out.is_open()) ctx.out.close();

    result.status_code = ctx.status_code;
    result.resumed = ctx.resumed;
    result.bytes_total = ctx.bytes_total;
    if (fs::exists(dest)) result.bytes_written = fs::file_size(dest, ec);
    if (ec) result.bytes_written = 0;

    if (rc == CURLE_ABORTED_BY_CALLBACK) {
        result.error = (ctx.should_cancel && ctx.should_cancel()) ? "cancelled" : "download aborted";
    } else if (rc != CURLE_OK) {
        result.error = curl_easy_strerror(rc);
    } else if (ctx.status_code >= 400) {
        result.error = "HTTP " + std::to_string(ctx.status_code);
    } else {
        result.ok = true;
    }

    if (ctx.on_state) {
        ctx.on_state({
            {"attempt", ctx.attempt},
            {"status_code", ctx.status_code},
            {"resumed", ctx.resumed},
            {"existing_bytes", ctx.effective_existing},
            {"bytes_total", ctx.bytes_total},
            {"bytes_received", result.bytes_written},
        });
    }

    curl_easy_cleanup(curl);
    return result;
}

} // namespace

DownloadResult download_file_with_retry(const std::string& url,
                                        const fs::path& dest,
                                        const DownloadOptions& options,
                                        DownloadShouldCancel should_cancel,
                                        DownloadProgressCallback on_progress,
                                        DownloadStateCallback on_state) {
    const int attempts_total = std::max(1, options.retry_count + 1);
    DownloadResult last;
    for (int attempt = 1; attempt <= attempts_total; ++attempt) {
        last = download_once(url, dest, options, attempt, should_cancel, on_progress, on_state);
        if (last.ok || last.error == "cancelled") return last;

        const bool retrying = attempt < attempts_total;
        if (on_state) {
            on_state({
                {"attempt", attempt},
                {"retrying", retrying},
                {"error", last.error},
            });
        }
        if (!retrying) break;

        const double sleep_s = std::max(0.0, options.retry_backoff_s) * static_cast<double>(attempt);
        if (sleep_s > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
        }
    }
    return last;
}
