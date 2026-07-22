#include "services/hme_preview_service.hpp"

#include "tile_compile/image/hypermetric_stretch.hpp"

#include <fitsio.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;
using tile_compile::Matrix2Df;

namespace tile_compile::web {
namespace {

struct Proxy {
    Matrix2Df r, g, b;
    std::vector<uint8_t> statistics_mask;
    std::vector<uint8_t> output_mask;
    std::string source;
    std::string signature;
};

std::mutex cache_mutex;
std::unordered_map<std::string, Proxy> cache;
constexpr int kMaxProxyEdge = 1600;
constexpr size_t kMaxCacheEntries = 4;

std::string fits_error(int status) {
    char text[FLEN_STATUS]{};
    fits_get_errstatus(status, text);
    return text;
}

std::vector<float> read_fits_plane(const fs::path& path, long plane,
                                   long& width, long& height, long& planes) {
    fitsfile* file = nullptr;
    int status = 0;
    if (fits_open_file(&file, path.string().c_str(), READONLY, &status))
        throw std::runtime_error("Cannot open FITS " + path.string() + ": " + fits_error(status));
    int naxis = 0;
    long axes[3]{1, 1, 1};
    if (fits_get_img_dim(file, &naxis, &status) ||
        fits_get_img_size(file, 3, axes, &status) || naxis < 2) {
        fits_close_file(file, &status);
        throw std::runtime_error("Invalid FITS image " + path.string());
    }
    width = axes[0]; height = axes[1]; planes = naxis >= 3 ? axes[2] : 1;
    if (plane < 1 || plane > planes) {
        fits_close_file(file, &status);
        throw std::runtime_error("Missing RGB plane in " + path.string());
    }
    std::vector<float> pixels(static_cast<size_t>(width * height));
    long first[3]{1, 1, plane};
    int any_null = 0;
    if (fits_read_pix(file, TFLOAT, first, width * height, nullptr,
                      pixels.data(), &any_null, &status)) {
        const auto message = fits_error(status);
        fits_close_file(file, &status);
        throw std::runtime_error("Cannot read FITS " + path.string() + ": " + message);
    }
    fits_close_file(file, &status);
    return pixels;
}

Matrix2Df to_matrix(const std::vector<float>& values, long width, long height) {
    Matrix2Df result(static_cast<int>(height), static_cast<int>(width));
    for (long y = 0; y < height; ++y)
        for (long x = 0; x < width; ++x)
            result(static_cast<int>(y), static_cast<int>(x)) =
                values[static_cast<size_t>(y * width + x)];
    return result;
}

std::string file_signature(const std::vector<fs::path>& paths) {
    std::string out;
    for (const auto& path : paths) {
        out += path.string() + ":" + std::to_string(fs::file_size(path)) + ":" +
               std::to_string(static_cast<long long>(
                   fs::last_write_time(path).time_since_epoch().count())) + ";";
    }
    return out;
}

Matrix2Df resize_matrix(const Matrix2Df& input, int rows, int cols) {
    cv::Mat source(input.rows(), input.cols(), CV_32F);
    for (int y = 0; y < input.rows(); ++y)
        for (int x = 0; x < input.cols(); ++x) source.at<float>(y, x) = input(y, x);
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
    Matrix2Df result(rows, cols);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) result(y, x) = resized.at<float>(y, x);
    return result;
}

std::vector<uint8_t> resize_mask(const std::vector<uint8_t>& input, int input_rows,
                                 int input_cols, int rows, int cols) {
    cv::Mat source(input_rows, input_cols, CV_8U);
    for (int y = 0; y < input_rows; ++y)
        for (int x = 0; x < input_cols; ++x)
            source.at<uint8_t>(y, x) = input[static_cast<size_t>(y) * input_cols + x];
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
    std::vector<uint8_t> result(static_cast<size_t>(rows) * cols);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            result[static_cast<size_t>(y) * cols + x] = resized.at<uint8_t>(y, x);
    return result;
}

std::vector<uint8_t> read_mask(const fs::path& path, long width, long height) {
    long mask_width = 0, mask_height = 0, planes = 0;
    const auto values = read_fits_plane(path, 1, mask_width, mask_height, planes);
    if (mask_width != width || mask_height != height)
        throw std::runtime_error("Mask dimensions do not match PCC RGB: " + path.string());
    std::vector<uint8_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = values[i] > 0.0f ? 1u : 0u;
    return result;
}

Proxy load_proxy(const fs::path& run_dir) {
    const fs::path outputs = run_dir / "outputs";
    const std::vector<fs::path> channels = {
        outputs / "pcc_R.fit", outputs / "pcc_G.fit", outputs / "pcc_B.fit"};
    const fs::path cube = outputs / "stacked_rgb_pcc.fits";
    const fs::path statistics_mask_path = outputs / "common_overlap_mask.fits";
    const fs::path output_mask_path = outputs / "canvas_mask.fits";
    std::vector<fs::path> source_paths;
    Proxy proxy;
    long w = 0, h = 0, p = 0;

    if (fs::exists(cube)) {
        auto rv = read_fits_plane(cube, 1, w, h, p);
        if (p < 3) throw std::runtime_error("stacked_rgb_pcc.fits has fewer than three planes");
        long w2, h2, p2;
        proxy.r = to_matrix(rv, w, h);
        auto gv = read_fits_plane(cube, 2, w2, h2, p2);
        proxy.g = to_matrix(gv, w2, h2);
        auto bv = read_fits_plane(cube, 3, w2, h2, p2);
        proxy.b = to_matrix(bv, w2, h2);
        proxy.source = "stacked_rgb_pcc.fits";
        source_paths = {cube};
    } else if (std::all_of(channels.begin(), channels.end(), [](const fs::path& pth) { return fs::exists(pth); })) {
        long gw, gh, gp, bw, bh, bp;
        auto rv = read_fits_plane(channels[0], 1, w, h, p);
        proxy.r = to_matrix(rv, w, h);
        auto gv = read_fits_plane(channels[1], 1, gw, gh, gp);
        auto bv = read_fits_plane(channels[2], 1, bw, bh, bp);
        if (gw != w || gh != h || bw != w || bh != h)
            throw std::runtime_error("PCC channel dimensions do not match");
        proxy.g = to_matrix(gv, w, h); proxy.b = to_matrix(bv, w, h);
        proxy.source = "pcc_R.fit + pcc_G.fit + pcc_B.fit";
        source_paths = channels;
    } else {
        throw std::runtime_error("No PCC RGB artifact found");
    }

    if (fs::exists(statistics_mask_path)) {
        proxy.statistics_mask = read_mask(statistics_mask_path, w, h);
        source_paths.push_back(statistics_mask_path);
    }
    if (fs::exists(output_mask_path)) {
        proxy.output_mask = read_mask(output_mask_path, w, h);
        source_paths.push_back(output_mask_path);
    }
    const size_t pixels = static_cast<size_t>(w) * h;
    if (proxy.statistics_mask.empty()) proxy.statistics_mask.assign(pixels, 1u);
    if (proxy.output_mask.empty()) proxy.output_mask.assign(pixels, 1u);

    const int max_edge = std::max(proxy.r.rows(), proxy.r.cols());
    if (max_edge > kMaxProxyEdge) {
        const int input_rows = proxy.r.rows();
        const int input_cols = proxy.r.cols();
        const double scale = static_cast<double>(kMaxProxyEdge) / max_edge;
        const int rows = std::max(1, static_cast<int>(std::lround(input_rows * scale)));
        const int cols = std::max(1, static_cast<int>(std::lround(input_cols * scale)));
        proxy.r = resize_matrix(proxy.r, rows, cols);
        proxy.g = resize_matrix(proxy.g, rows, cols);
        proxy.b = resize_matrix(proxy.b, rows, cols);
        proxy.statistics_mask = resize_mask(proxy.statistics_mask, input_rows, input_cols,
                                            rows, cols);
        proxy.output_mask = resize_mask(proxy.output_mask, input_rows, input_cols,
                                        rows, cols);
    }
    proxy.signature = file_signature(source_paths);
    return proxy;
}

Proxy get_proxy(const fs::path& run_dir) {
    const std::string key = fs::weakly_canonical(run_dir).string();
    const fs::path outputs = run_dir / "outputs";
    const std::vector<fs::path> channels = {outputs / "pcc_R.fit", outputs / "pcc_G.fit", outputs / "pcc_B.fit"};
    std::vector<fs::path> sources;
    if (fs::exists(outputs / "stacked_rgb_pcc.fits"))
        sources = {outputs / "stacked_rgb_pcc.fits"};
    else if (std::all_of(channels.begin(), channels.end(), [](const fs::path& pth) { return fs::exists(pth); }))
        sources = channels;
    else
        throw std::runtime_error("No PCC RGB artifact found");
    for (const auto& mask : {outputs / "common_overlap_mask.fits",
                             outputs / "canvas_mask.fits"}) {
        if (fs::exists(mask)) sources.push_back(mask);
    }
    const std::string signature = file_signature(sources);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end() && it->second.signature == signature) return it->second;
    }
    Proxy loaded = load_proxy(run_dir);
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache.size() >= kMaxCacheEntries && cache.find(key) == cache.end()) cache.erase(cache.begin());
    cache[key] = loaded;
    return loaded;
}

template <typename T>
T value(const nlohmann::json& p, const char* key, T fallback) {
    return p.contains(key) ? p.at(key).get<T>() : fallback;
}

bool finite_range(float v, float lo, float hi) {
    return std::isfinite(v) && v >= lo && v <= hi;
}

} // namespace

HmePreviewResult create_hme_preview(const fs::path& run_dir, const nlohmann::json& params) {
    HmePreviewResult result;
    try {
        if (!params.is_object()) { result.status = 400; result.error = "params must be an object"; return result; }
        image::HyperMetricStretchConfig cfg;
        cfg.mode = value<std::string>(params, "mode", cfg.mode);
        cfg.sensor_profile = value<std::string>(params, "sensor_profile", cfg.sensor_profile);
        cfg.fallback_profile = value<std::string>(params, "fallback_profile", cfg.fallback_profile);
        cfg.adaptive_anchor = value<bool>(params, "adaptive_anchor", cfg.adaptive_anchor);
        cfg.target_bg = value<float>(params, "target_bg", cfg.target_bg);
        cfg.protect_b = value<float>(params, "protect_b", cfg.protect_b);
        cfg.convergence_power = value<float>(params, "convergence_power", cfg.convergence_power);
        cfg.log_d_mode = value<std::string>(params, "log_d_mode", cfg.log_d_mode);
        cfg.fixed_log_d = value<float>(params, "fixed_log_d", cfg.fixed_log_d);
        cfg.color_strategy = value<std::string>(params, "color_strategy", cfg.color_strategy);
        cfg.fixed_color_strategy = value<float>(params, "fixed_color_strategy", cfg.fixed_color_strategy);
        cfg.color_grip = value<float>(params, "color_grip", cfg.color_grip);
        cfg.shadow_convergence = value<float>(params, "shadow_convergence", cfg.shadow_convergence);
        cfg.linear_expansion = value<float>(params, "linear_expansion", cfg.linear_expansion);
        std::vector<std::string> invalid;
        auto invalid_enum = [&invalid](const char* name, const std::string& actual,
                                       const char* expected) {
            invalid.emplace_back(std::string(name) + "='" + actual + "' (expected " + expected + ")");
        };
        auto invalid_number = [&invalid](const char* name, float actual,
                                         float low, float high) {
            if (finite_range(actual, low, high)) return;
            std::ostringstream text;
            text << name << '=' << actual << " (expected " << low << ".." << high << ')';
            invalid.push_back(text.str());
        };
        if (cfg.mode != "ready_to_use" && cfg.mode != "scientific")
            invalid_enum("mode", cfg.mode, "ready_to_use or scientific");
        if (cfg.log_d_mode != "auto" && cfg.log_d_mode != "fixed")
            invalid_enum("log_d_mode", cfg.log_d_mode, "auto or fixed");
        if (cfg.color_strategy != "auto" && cfg.color_strategy != "fixed")
            invalid_enum("color_strategy", cfg.color_strategy, "auto or fixed");
        invalid_number("target_bg", cfg.target_bg, .05f, .5f);
        invalid_number("protect_b", cfg.protect_b, .1f, 15.f);
        invalid_number("convergence_power", cfg.convergence_power, 1.f, 10.f);
        invalid_number("fixed_log_d", cfg.fixed_log_d, 0.f, 7.f);
        invalid_number("fixed_color_strategy", cfg.fixed_color_strategy, -1.f, 1.f);
        invalid_number("color_grip", cfg.color_grip, 0.f, 1.f);
        invalid_number("shadow_convergence", cfg.shadow_convergence, 0.f, 3.f);
        invalid_number("linear_expansion", cfg.linear_expansion, 0.f, 1.f);
        if (!invalid.empty()) {
            std::ostringstream message;
            message << "Invalid HMS parameter value" << (invalid.size() > 1 ? "s: " : ": ");
            for (size_t i = 0; i < invalid.size(); ++i) {
                if (i) message << "; ";
                message << invalid[i];
            }
            result.status = 400;
            result.error = message.str();
            return result;
        }

        Proxy proxy = get_proxy(run_dir);
        Matrix2Df r = proxy.r, g = proxy.g, b = proxy.b;
        if (r.rows() != g.rows() || r.rows() != b.rows() || r.cols() != g.cols() || r.cols() != b.cols()) {
            throw std::runtime_error("Proxy RGB dimensions mismatch: R=" + std::to_string(r.rows()) + "x" + std::to_string(r.cols()) +
                " G=" + std::to_string(g.rows()) + "x" + std::to_string(g.cols()) +
                " B=" + std::to_string(b.rows()) + "x" + std::to_string(b.cols()));
        }
        auto diag = image::run_hypermetric_stretch_rgb(
            r, g, b, cfg, &proxy.statistics_mask, r.rows(), r.cols(),
            &proxy.output_mask);
        if (!diag.success) { result.status = 400; result.error = diag.error_message; return result; }

        cv::Mat rgb(r.rows(), r.cols(), CV_8UC3);
        for (int y = 0; y < r.rows(); ++y) for (int x = 0; x < r.cols(); ++x) {
            auto& px = rgb.at<cv::Vec3b>(y, x);
            px[2] = cv::saturate_cast<unsigned char>(std::clamp(r(y,x), 0.f, 1.f) * 255.f);
            px[1] = cv::saturate_cast<unsigned char>(std::clamp(g(y,x), 0.f, 1.f) * 255.f);
            px[0] = cv::saturate_cast<unsigned char>(std::clamp(b(y,x), 0.f, 1.f) * 255.f);
        }
        if (!cv::imencode(".png", rgb, result.png)) throw std::runtime_error("PNG encoding failed");
        result.diagnostics = {{"success", true}, {"status", diag.status}, {"source", proxy.source},
            {"profile", diag.profile}, {"profile_source", diag.profile_source}, {"anchor", diag.anchor},
            {"log_d", diag.log_d}, {"star_pressure", diag.star_pressure},
            {"black_clip_percent", diag.black_clip_percent}, {"white_clip_percent", diag.white_clip_percent}};
        result.ok = true; result.status = 200;
    } catch (const std::exception& e) {
        result.status = std::string(e.what()).find("No PCC") != std::string::npos ? 404 : 400;
        result.error = e.what();
    }
    return result;
}

} // namespace tile_compile::web
