#include "services/download_manager.hpp"
#include <curl/curl.h>
#include <fstream>
#include <atomic>
#include <stdexcept>

static size_t write_to_file(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* f = static_cast<std::ofstream*>(userdata);
    f->write(static_cast<const char*>(ptr), (std::streamsize)(size * nmemb));
    return size * nmemb;
}

struct ProgressState {
    std::atomic<bool>* cancelled{nullptr};
    std::function<void(double)> on_progress;
};

static int progress_cb(void* userdata, curl_off_t dltotal, curl_off_t dlnow,
                       curl_off_t, curl_off_t) {
    ProgressState* state = static_cast<ProgressState*>(userdata);
    if (state && state->on_progress) {
        double ratio = 0.0;
        if (dltotal > 0) ratio = static_cast<double>(dlnow) / static_cast<double>(dltotal);
        state->on_progress(ratio);
    }
    return (state && state->cancelled && state->cancelled->load()) ? 1 : 0;
}

DownloadResult download_file(const std::string& url,
                              const fs::path& dest,
                              std::atomic<bool>& cancelled,
                              std::function<void(double)> on_progress) {
    DownloadResult res;
    CURL* curl = curl_easy_init();
    if (!curl) { res.error = "curl_easy_init failed"; return res; }
    ProgressState progress_state{&cancelled, on_progress};

    fs::create_directories(dest.parent_path());
    std::ofstream out(dest, std::ios::binary);
    if (!out) { res.error = "Cannot open dest: " + dest.string(); curl_easy_cleanup(curl); return res; }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_state);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode rc = curl_easy_perform(curl);
    out.close();

    if (rc == CURLE_ABORTED_BY_CALLBACK) {
        fs::remove(dest);
        res.error = "cancelled";
    } else if (rc != CURLE_OK) {
        fs::remove(dest);
        res.error = curl_easy_strerror(rc);
    } else {
        res.ok = true;
        res.dest_path = dest;
    }
    curl_easy_cleanup(curl);
    return res;
}
