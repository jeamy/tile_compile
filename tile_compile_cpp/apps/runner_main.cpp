#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/registration/registration.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef HAVE_CLI11
#include <CLI/CLI.hpp>
#endif

namespace fs = std::filesystem;

namespace {

using tile_compile::Matrix2Df;
using tile_compile::ColorMode;

class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* a, std::streambuf* b) : a_(a), b_(b) {}

protected:
    int overflow(int c) override {
        if (c == EOF) return EOF;
        const int ra = a_ ? a_->sputc(static_cast<char>(c)) : c;
        const int rb = b_ ? b_->sputc(static_cast<char>(c)) : c;
        return (ra == EOF || rb == EOF) ? EOF : c;
    }

    int sync() override {
        int ra = a_ ? a_->pubsync() : 0;
        int rb = b_ ? b_->pubsync() : 0;
        return (ra == 0 && rb == 0) ? 0 : -1;
    }

private:
    std::streambuf* a_;
    std::streambuf* b_;
};

struct NormalizationScales {
    bool is_osc = false;
    float scale_mono = 1.0f;
    float scale_r = 1.0f;
    float scale_g = 1.0f;
    float scale_b = 1.0f;
};

float median_of(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];
    if ((n % 2) == 1) {
        return hi;
    }
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

float robust_sigma_mad(std::vector<float>& pixels) {
    if (pixels.empty()) return 0.0f;
    float med = median_of(pixels);
    for (float& x : pixels) x = std::fabs(x - med);
    float mad = median_of(pixels);
    return 1.4826f * mad;
}

struct LinearityThresholds {
    float skewness_max = 1.2f;
    float kurtosis_max = 1.2f;
    float variance_max = 0.5f;
    float energy_ratio_min = 0.95f;
    float gradient_consistency_max = 0.5f;
};

struct LinearityFrameResult {
    bool is_linear = false;
    float score = 0.0f;
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    float variance_coeff = 0.0f;
    float energy_ratio = 0.0f;
    float gradient_consistency = 0.0f;
    bool moment_ok = false;
    bool spectral_ok = false;
    bool spatial_ok = false;
};

static LinearityThresholds linearity_thresholds_for(const std::string& strictness) {
    if (strictness == "moderate") {
        return {1.2f, 1.2f, 0.7f, 0.9f, 0.7f};
    }
    if (strictness == "permissive") {
        return {1.5f, 1.5f, 1.0f, 0.8f, 1.0f};
    }
    return {1.2f, 1.2f, 0.5f, 0.95f, 0.5f};
}

static std::vector<size_t> sample_indices(size_t count, int max_frames) {
    std::vector<size_t> out;
    if (count == 0 || max_frames <= 0) return out;
    size_t n = std::min(count, static_cast<size_t>(max_frames));
    if (n == 1) {
        out.push_back(0);
        return out;
    }
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n - 1);
        size_t idx = static_cast<size_t>(std::round(t * static_cast<float>(count - 1)));
        if (out.empty() || out.back() != idx) {
            out.push_back(idx);
        }
    }
    return out;
}

static float percentile_from_sorted(const std::vector<float>& sorted, float pct) {
    if (sorted.empty()) return 0.0f;
    float clamped = std::min(std::max(pct, 0.0f), 100.0f);
    float pos = (clamped / 100.0f) * static_cast<float>(sorted.size() - 1);
    size_t idx = static_cast<size_t>(std::round(pos));
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
}

static LinearityFrameResult validate_linearity_frame(const Matrix2Df& img, const std::string& strictness) {
    LinearityFrameResult out;
    if (img.size() <= 0) return out;

    cv::Mat cv_img(img.rows(), img.cols(), CV_32F, const_cast<float*>(img.data()));
    cv::Mat small = cv_img;

    const int max_dim = 256;
    if (cv_img.rows > max_dim || cv_img.cols > max_dim) {
        float scale = static_cast<float>(max_dim) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));
        cv::resize(cv_img, small, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    std::vector<float> values;
    values.reserve(static_cast<size_t>(small.rows) * static_cast<size_t>(small.cols));
    for (int y = 0; y < small.rows; ++y) {
        const float* row = small.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x) {
            float v = row[x];
            if (std::isfinite(v)) values.push_back(v);
        }
    }

    if (values.empty()) {
        return out;
    }

    double mean = 0.0;
    double m2 = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        double x = static_cast<double>(values[i]);
        double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }
    double var = (values.size() > 1) ? (m2 / static_cast<double>(values.size() - 1)) : 0.0;
    double stddev = std::sqrt(std::max(0.0, var));

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    float p1 = percentile_from_sorted(sorted, 1.0f);
    float p5 = percentile_from_sorted(sorted, 5.0f);
    float p50 = percentile_from_sorted(sorted, 50.0f);
    float p95 = percentile_from_sorted(sorted, 95.0f);
    float p99 = percentile_from_sorted(sorted, 99.0f);

    float denom_skew = (p50 - p1) + 1.0e-12f;
    float denom_kurt = (p50 - p5) + 1.0e-12f;
    out.skewness = (p99 - p50) / denom_skew;
    out.kurtosis = (p95 - p50) / denom_kurt;
    out.variance_coeff = static_cast<float>(stddev / (std::fabs(mean) + 1.0e-12));

    cv::Mat gx, gy, mag;
    cv::Sobel(small, gx, CV_32F, 1, 0, 3);
    cv::Sobel(small, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    double mean_grad = cv::mean(mag)[0];
    double mean_frame = cv::mean(small)[0];
    out.gradient_consistency = static_cast<float>(2.0 * (mean_grad / (std::fabs(mean_frame) + 1.0e-12)));

    out.energy_ratio = 0.0f;
    if (small.rows >= 8 && small.cols >= 8) {
        cv::Mat dft;
        cv::dft(small, dft, cv::DFT_COMPLEX_OUTPUT);
        std::vector<cv::Mat> planes;
        cv::split(dft, planes);
        cv::Mat mag2 = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
        double total_energy = cv::sum(mag2)[0];
        int r = std::max(1, std::min(mag2.rows, mag2.cols) / 8);
        double low_energy = 0.0;
        low_energy += cv::sum(mag2(cv::Rect(0, 0, r, r)))[0];
        low_energy += cv::sum(mag2(cv::Rect(0, mag2.rows - r, r, r)))[0];
        low_energy += cv::sum(mag2(cv::Rect(mag2.cols - r, 0, r, r)))[0];
        low_energy += cv::sum(mag2(cv::Rect(mag2.cols - r, mag2.rows - r, r, r)))[0];
        if (total_energy > 0.0) {
            out.energy_ratio = static_cast<float>(low_energy / total_energy);
        }
    }

    LinearityThresholds th = linearity_thresholds_for(strictness);
    out.moment_ok = (std::fabs(out.skewness) < th.skewness_max) &&
                    (std::fabs(out.kurtosis) < th.kurtosis_max) &&
                    (out.variance_coeff < th.variance_max);
    out.spectral_ok = (out.energy_ratio >= th.energy_ratio_min);
    out.spatial_ok = (out.gradient_consistency < th.gradient_consistency_max);

    out.score = (static_cast<float>(out.moment_ok) + static_cast<float>(out.spectral_ok) +
                 static_cast<float>(out.spatial_ok)) / 3.0f;
    out.is_linear = out.moment_ok && out.spectral_ok && out.spatial_ok;
    return out;
}

float estimate_fwhm_from_patch(const cv::Mat& patch) {
    if (patch.empty()) return 0.0f;
    std::vector<float> v;
    v.reserve(static_cast<size_t>(patch.rows) * static_cast<size_t>(patch.cols));
    for (int y = 0; y < patch.rows; ++y) {
        const float* row = patch.ptr<float>(y);
        for (int x = 0; x < patch.cols; ++x) {
            v.push_back(row[x]);
        }
    }
    if (v.empty()) return 0.0f;
    float bg = median_of(v);
    float sigma = robust_sigma_mad(v);

    double maxv = 0.0;
    cv::minMaxLoc(patch, nullptr, &maxv);
    if (!(maxv > 0.0)) return 0.0f;
    if (maxv <= static_cast<double>(bg) + 3.0 * static_cast<double>(sigma)) return 0.0f;

    cv::Mat p = patch - bg;
    cv::threshold(p, p, 0.0, 0.0, cv::THRESH_TOZERO);
    double peak = 0.0;
    cv::minMaxLoc(p, nullptr, &peak);
    if (!(peak > 0.0)) return 0.0f;
    double half = 0.5 * peak;

    int cnt = 0;
    for (int y = 0; y < p.rows; ++y) {
        const float* row = p.ptr<float>(y);
        for (int x = 0; x < p.cols; ++x) {
            if (static_cast<double>(row[x]) >= half) {
                ++cnt;
            }
        }
    }
    if (cnt <= 0) return 0.0f;
    return static_cast<float>(std::sqrt(static_cast<double>(cnt) / 3.14159265358979323846));
}

float stddev_of(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0f;
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x);
    const double mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) {
        const double d = static_cast<double>(x) - mean;
        var += d * d;
    }
    var /= static_cast<double>(v.size());
    if (var <= 0.0) return 0.0f;
    return static_cast<float>(std::sqrt(var));
}

float estimate_background_sigma_clip(std::vector<float> pixels) {
    if (pixels.empty()) return 0.0f;
    for (int iter = 0; iter < 5; ++iter) {
        float mu = median_of(pixels);
        float sigma = stddev_of(pixels);
        if (!(sigma > 0.0f)) break;

        std::vector<float> clipped;
        clipped.reserve(pixels.size());
        const float thr = 3.0f * sigma;
        for (float x : pixels) {
            if (std::fabs(x - mu) < thr) {
                clipped.push_back(x);
            }
        }
        if (clipped.size() == pixels.size() || clipped.empty()) break;
        pixels.swap(clipped);
    }
    return median_of(pixels);
}

cv::Mat1b build_background_mask_sigma_clip(const cv::Mat& frame_norm, float k_sigma, int dilate_radius) {
    const int h = frame_norm.rows;
    const int w = frame_norm.cols;
    cv::Mat1b obj = cv::Mat1b::zeros(h, w);

    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(h) * static_cast<size_t>(w));
    for (int y = 0; y < h; ++y) {
        const float* row = frame_norm.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            vals.push_back(row[x]);
        }
    }

    float mu = median_of(vals);
    float sigma = robust_sigma_mad(vals);
    if (!(sigma > 0.0f)) {
        return cv::Mat1b(h, w, uint8_t(1));
    }

    const float thr = k_sigma * sigma;
    for (int y = 0; y < h; ++y) {
        const float* row = frame_norm.ptr<float>(y);
        uint8_t* mrow = obj.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            mrow[x] = (std::fabs(row[x] - mu) > thr) ? uint8_t(1) : uint8_t(0);
        }
    }

    cv::Mat1b obj_d;
    const int r = std::max(0, dilate_radius);
    if (r > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1));
        cv::dilate(obj, obj_d, kernel);
    } else {
        obj_d = obj;
    }

    cv::Mat1b bg = cv::Mat1b::zeros(h, w);
    for (int y = 0; y < h; ++y) {
        const uint8_t* orow = obj_d.ptr<uint8_t>(y);
        uint8_t* brow = bg.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            brow[x] = (orow[x] == 0) ? uint8_t(1) : uint8_t(0);
        }
    }
    return bg;
}

void bayer_offsets(const std::string& bayer_pattern,
                   int& r_row, int& r_col,
                   int& b_row, int& b_col) {
    std::string bp = bayer_pattern;
    std::transform(bp.begin(), bp.end(), bp.begin(), ::toupper);
    r_row = 1; r_col = 0;
    b_row = 0; b_col = 1;
    if (bp == "RGGB") {
        r_row = 0; r_col = 0;
        b_row = 1; b_col = 1;
    } else if (bp == "BGGR") {
        r_row = 1; r_col = 1;
        b_row = 0; b_col = 0;
    } else if (bp == "GRBG") {
        r_row = 0; r_col = 1;
        b_row = 1; b_col = 0;
    }
}

void apply_normalization_inplace(Matrix2Df& img,
                                 const NormalizationScales& s,
                                 ColorMode mode,
                                 const std::string& bayer_pattern,
                                 int origin_x,
                                 int origin_y) {
    if (img.size() <= 0) return;
    if (mode != ColorMode::OSC) {
        img *= s.scale_mono;
        return;
    }

    int r_row, r_col, b_row, b_col;
    bayer_offsets(bayer_pattern, r_row, r_col, b_row, b_col);
    for (int y = 0; y < img.rows(); ++y) {
        const int gy = origin_y + y;
        for (int x = 0; x < img.cols(); ++x) {
            const int gx = origin_x + x;
            const int py = gy & 1;
            const int px = gx & 1;
            if (py == r_row && px == r_col) {
                img(y, x) *= s.scale_r;
            } else if (py == b_row && px == b_col) {
                img(y, x) *= s.scale_b;
            } else {
                img(y, x) *= s.scale_g;
            }
        }
    }
}

}

void print_usage() {
    std::cout << "Usage: tile_compile_runner <command> [options]\n\n"
              << "Commands:\n"
              << "  run      Run the pipeline\n"
              << "  resume   Resume a run from a specific phase\n"
              << "\nOptions:\n"
              << "  --config <path>       Path to config.yaml\n"
              << "  --input-dir <path>    Input directory with FITS frames\n"
              << "  --runs-dir <path>     Directory for run outputs\n"
              << "  --project-root <path> Project root directory\n"
              << "  --max-frames <n>      Limit number of frames (0 = no limit)\n"
              << "  --max-tiles <n>       Limit number of tiles in Phase 5/6 (0 = no limit)\n"
              << "  --dry-run             Dry run (no actual processing)\n"
              << std::endl;
}

int run_command(const std::string& config_path, const std::string& input_dir,
                const std::string& runs_dir, const std::string& project_root,
                bool dry_run, int max_frames, int max_tiles, bool config_from_stdin) {
    using namespace tile_compile;
    
    fs::path cfg_path(config_path);
    fs::path in_dir(input_dir);
    fs::path runs(runs_dir);

    const bool use_stdin_config = config_from_stdin || (config_path == "-");
    fs::path proj_root;
    
    if (!fs::exists(in_dir)) {
        std::cerr << "Error: Input directory not found: " << input_dir << std::endl;
        return 1;
    }

    config::Config cfg;
    std::string cfg_text;
    if (use_stdin_config) {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        cfg_text = ss.str();
        if (cfg_text.empty()) {
            std::cerr << "Error: --stdin provided but no config YAML received" << std::endl;
            return 1;
        }
        YAML::Node node = YAML::Load(cfg_text);
        cfg = config::Config::from_yaml(node);
        cfg.validate();
        proj_root = project_root.empty() ? fs::current_path() : fs::path(project_root);
    } else {
        if (!fs::exists(cfg_path)) {
            std::cerr << "Error: Config file not found: " << config_path << std::endl;
            return 1;
        }
        cfg = config::Config::load(cfg_path);
        cfg.validate();
        proj_root = project_root.empty() ? core::resolve_project_root(cfg_path) : fs::path(project_root);
    }
    
    auto frames = core::discover_frames(in_dir, "*.fit*");
    std::sort(frames.begin(), frames.end());
    if (max_frames > 0 && frames.size() > static_cast<size_t>(max_frames)) {
        frames.resize(static_cast<size_t>(max_frames));
    }
    if (frames.empty()) {
        std::cerr << "Error: No FITS frames found in " << input_dir << std::endl;
        return 1;
    }
    
    std::string run_id = core::get_run_id();
    fs::path run_dir = runs / run_id;
    fs::create_directories(run_dir / "logs");
    fs::create_directories(run_dir / "outputs");
    fs::create_directories(run_dir / "artifacts");

    if (use_stdin_config) {
        std::ofstream out(run_dir / "config.yaml", std::ios::out);
        out << cfg_text;
    } else {
        core::copy_config(cfg_path, run_dir / "config.yaml");
    }
    
    std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl");
    TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
    std::ostream log_file(&tee_buf);

    const bool debug_tile_reg = cfg.v4.debug_tile_registration;
    std::ofstream debug_log_file;
    std::mutex debug_log_mutex;
    if (debug_tile_reg) {
        debug_log_file.open(run_dir / "logs" / "debug.log", std::ios::out);
    }
    
    core::EventEmitter emitter;
    emitter.run_start(run_id, {
        {"config_path", config_path},
        {"input_dir", input_dir},
        {"run_dir", run_dir.string()},
        {"frames_discovered", frames.size()},
        {"dry_run", dry_run}
    }, log_file);
    
    std::cout << "Run ID: " << run_id << std::endl;
    std::cout << "Frames: " << frames.size() << std::endl;
    std::cout << "Output: " << run_dir.string() << std::endl;
    if (max_tiles > 0) {
        std::cout << "Max tiles (Phase 5/6): " << max_tiles << std::endl;
    }

    if (dry_run) {
        emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "skipped", {{"reason", "dry_run"}, {"input_dir", input_dir}}, log_file);

        std::cout << "Dry run - no processing" << std::endl;
        emitter.run_end(run_id, true, "ok", log_file);
        return 0;
    }

    // Phase 0: SCAN_INPUT (Methodik v4)
    emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);

    int width = 0;
    int height = 0;
    int naxis = 0;
    ColorMode detected_mode = ColorMode::MONO;
    BayerPattern detected_bayer = BayerPattern::UNKNOWN;
    Matrix2Df first_frame;
    io::FitsHeader first_header;

    try {
        std::tie(width, height, naxis) = io::get_fits_dimensions(frames.front());
        auto first = io::read_fits_float(frames.front());
        first_frame = std::move(first.first);
        first_header = std::move(first.second);

        detected_mode = io::detect_color_mode(first_header, naxis);
        detected_bayer = io::detect_bayer_pattern(first_header);
    } catch (const std::exception& e) {
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error", {{"error", e.what()}, {"input_dir", input_dir}}, log_file);
        emitter.run_end(run_id, false, "error", log_file);
        std::cerr << "Error during SCAN_INPUT: " << e.what() << std::endl;
        return 1;
    }

    std::string detected_mode_str = color_mode_to_string(detected_mode);
    std::string detected_bayer_str = bayer_pattern_to_string(detected_bayer);

    if (!cfg.data.color_mode.empty() && cfg.data.color_mode != detected_mode_str) {
        emitter.warning(run_id,
                        "Detected color mode '" + detected_mode_str + "' differs from config.data.color_mode '" +
                            cfg.data.color_mode + "'",
                        log_file);
    }
    if (!cfg.data.bayer_pattern.empty() && detected_mode == ColorMode::OSC &&
        cfg.data.bayer_pattern != detected_bayer_str && detected_bayer != BayerPattern::UNKNOWN) {
        emitter.warning(run_id,
                        "Detected bayer pattern '" + detected_bayer_str +
                            "' differs from config.data.bayer_pattern '" + cfg.data.bayer_pattern + "'",
                        log_file);
    }

    core::json linearity_info;
    if (cfg.linearity.enabled) {
        auto indices = sample_indices(frames.size(), cfg.linearity.max_frames);
        int failed = 0;
        float score_sum = 0.0f;
        std::vector<std::string> failed_names;
        for (size_t idx : indices) {
            Matrix2Df frame_img;
            if (idx == 0) {
                frame_img = first_frame;
            } else {
                frame_img = io::read_fits_float(frames[idx]).first;
            }
            LinearityFrameResult res = validate_linearity_frame(frame_img, cfg.linearity.strictness);
            score_sum += res.is_linear ? 1.0f : 0.0f;
            if (!res.is_linear) {
                failed++;
                if (failed_names.size() < 5) {
                    failed_names.push_back(frames[idx].filename().string());
                }
            }
        }

        float overall_linearity = indices.empty() ? 0.0f : (score_sum / static_cast<float>(indices.size()));
        linearity_info["enabled"] = true;
        linearity_info["sampled_frames"] = static_cast<int>(indices.size());
        linearity_info["overall_linearity"] = overall_linearity;
        linearity_info["min_overall_linearity"] = cfg.linearity.min_overall_linearity;
        linearity_info["failed_frames"] = failed;
        if (!failed_names.empty()) {
            linearity_info["failed_frame_names"] = failed_names;
        }

        if (overall_linearity + 1.0e-6f < cfg.linearity.min_overall_linearity) {
            core::json err;
            err["error"] = "linearity_check_failed";
            err["linearity"] = linearity_info;
            emitter.phase_end(run_id, Phase::SCAN_INPUT, "error", err, log_file);
            emitter.run_end(run_id, false, "error", log_file);
            return 1;
        }
        if (failed > 0) {
            emitter.warning(run_id,
                            "Linearity check: " + std::to_string(failed) +
                                " sampled frames flagged non-linear (overall_linearity=" +
                                std::to_string(overall_linearity) + ")",
                            log_file);
        }
    }

    core::json scan_extra = {
        {"input_dir", input_dir},
        {"frames_scanned", frames.size()},
        {"image_width", width},
        {"image_height", height},
        {"color_mode", detected_mode_str},
        {"bayer_pattern", detected_bayer_str},
    };
    if (!linearity_info.is_null()) {
        scan_extra["linearity"] = linearity_info;
    }

    emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", scan_extra, log_file);

    // Phase 1: CHANNEL_SPLIT (metadata-only; actual split happens later)
    emitter.phase_start(run_id, Phase::CHANNEL_SPLIT, "CHANNEL_SPLIT", log_file);

    core::json extra;
    if (detected_mode == ColorMode::OSC) {
        extra["mode"] = "OSC";
        extra["channels"] = core::json::array({"R", "G", "B"});
        extra["bayer_pattern"] = detected_bayer_str;
        extra["note"] = "deferred_to_tile_processing";
    } else {
        extra["mode"] = "MONO";
        extra["channels"] = core::json::array({"L"});
    }
    emitter.phase_end(run_id, Phase::CHANNEL_SPLIT, "ok", extra, log_file);

    // Phase 2: NORMALIZATION (Methodik v4 §3)
    emitter.phase_start(run_id, Phase::NORMALIZATION, "NORMALIZATION", log_file);

    if (!cfg.normalization.enabled) {
        emitter.phase_end(run_id, Phase::NORMALIZATION, "error",
                          {{"error", "NORMALIZATION: disabled but required for v4"}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return 1;
    }

    const float eps_b = 1.0e-6f;
    std::vector<NormalizationScales> norm_scales(frames.size());
    std::vector<float> B_mono(frames.size(), 0.0f);
    std::vector<float> B_r(frames.size(), 0.0f);
    std::vector<float> B_g(frames.size(), 0.0f);
    std::vector<float> B_b(frames.size(), 0.0f);

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& path = frames[i];
        try {
            auto frame_pair = io::read_fits_float(path);
            const Matrix2Df& img = frame_pair.first;

            NormalizationScales s;
            {
                std::vector<float> all;
                all.reserve(static_cast<size_t>(img.size()));
                for (Eigen::Index k = 0; k < img.size(); ++k) {
                    all.push_back(img.data()[k]);
                }
                const float b0 = median_of(all);
                const float coarse_scale = (b0 > eps_b) ? (1.0f / b0) : 1.0f;
                Matrix2Df coarse_norm = img * coarse_scale;
                cv::Mat coarse_cv(coarse_norm.rows(), coarse_norm.cols(), CV_32F, coarse_norm.data());
                const cv::Mat1b bg_mask = build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);

                if (detected_mode == ColorMode::OSC) {
                    s.is_osc = true;
                    int r_row, r_col, b_row, b_col;
                    bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);

                    std::vector<float> pr_bg;
                    std::vector<float> pg_bg;
                    std::vector<float> pb_bg;
                    pr_bg.reserve(static_cast<size_t>(img.size()) / 4);
                    pg_bg.reserve(static_cast<size_t>(img.size()) / 2);
                    pb_bg.reserve(static_cast<size_t>(img.size()) / 4);

                    for (int y = 0; y < img.rows(); ++y) {
                        const uint8_t* mrow = bg_mask.ptr<uint8_t>(y);
                        const int py = y & 1;
                        for (int x = 0; x < img.cols(); ++x) {
                            if (mrow[x] == 0) continue;
                            const int px = x & 1;
                            const float v = img(y, x);
                            if (py == r_row && px == r_col) {
                                pr_bg.push_back(v);
                            } else if (py == b_row && px == b_col) {
                                pb_bg.push_back(v);
                            } else {
                                pg_bg.push_back(v);
                            }
                        }
                    }

                    float br = pr_bg.empty() ? 0.0f : median_of(pr_bg);
                    float bg = pg_bg.empty() ? 0.0f : median_of(pg_bg);
                    float bb = pb_bg.empty() ? 0.0f : median_of(pb_bg);

                    if (!(br > eps_b)) {
                        std::vector<float> pr;
                        pr.reserve(static_cast<size_t>(img.size()) / 4);
                        for (int y = 0; y < img.rows(); ++y) {
                            const int py = y & 1;
                            for (int x = 0; x < img.cols(); ++x) {
                                const int px = x & 1;
                                if (py == r_row && px == r_col) pr.push_back(img(y, x));
                            }
                        }
                        br = estimate_background_sigma_clip(std::move(pr));
                    }
                    if (!(bg > eps_b)) {
                        std::vector<float> pg;
                        pg.reserve(static_cast<size_t>(img.size()) / 2);
                        for (int y = 0; y < img.rows(); ++y) {
                            const int py = y & 1;
                            for (int x = 0; x < img.cols(); ++x) {
                                const int px = x & 1;
                                if (!((py == r_row && px == r_col) || (py == b_row && px == b_col))) pg.push_back(img(y, x));
                            }
                        }
                        bg = estimate_background_sigma_clip(std::move(pg));
                    }
                    if (!(bb > eps_b)) {
                        std::vector<float> pb;
                        pb.reserve(static_cast<size_t>(img.size()) / 4);
                        for (int y = 0; y < img.rows(); ++y) {
                            const int py = y & 1;
                            for (int x = 0; x < img.cols(); ++x) {
                                const int px = x & 1;
                                if (py == b_row && px == b_col) pb.push_back(img(y, x));
                            }
                        }
                        bb = estimate_background_sigma_clip(std::move(pb));
                    }

                    if (!(br > eps_b) || !(bg > eps_b) || !(bb > eps_b)) {
                        emitter.phase_end(run_id, Phase::NORMALIZATION, "error",
                                          {{"error", "NORMALIZATION: invalid background estimate"}},
                                          log_file);
                        emitter.run_end(run_id, false, "error", log_file);
                        return 1;
                    }

                    s.scale_r = 1.0f / br;
                    s.scale_g = 1.0f / bg;
                    s.scale_b = 1.0f / bb;
                    B_r[i] = br;
                    B_g[i] = bg;
                    B_b[i] = bb;
                } else {
                    std::vector<float> p_bg;
                    p_bg.reserve(static_cast<size_t>(img.size()));
                    for (int y = 0; y < img.rows(); ++y) {
                        const uint8_t* mrow = bg_mask.ptr<uint8_t>(y);
                        for (int x = 0; x < img.cols(); ++x) {
                            if (mrow[x] != 0) p_bg.push_back(img(y, x));
                        }
                    }
                    float b = p_bg.empty() ? 0.0f : median_of(p_bg);
                    if (!(b > eps_b)) {
                        std::vector<float> p;
                        p.reserve(static_cast<size_t>(img.size()));
                        for (Eigen::Index k = 0; k < img.size(); ++k) {
                            p.push_back(img.data()[k]);
                        }
                        b = estimate_background_sigma_clip(std::move(p));
                    }
                    if (!(b > eps_b)) {
                        emitter.phase_end(run_id, Phase::NORMALIZATION, "error",
                                          {{"error", "NORMALIZATION: invalid background estimate"}},
                                          log_file);
                        emitter.run_end(run_id, false, "error", log_file);
                        return 1;
                    }
                    s.scale_mono = 1.0f / b;
                    B_mono[i] = b;
                }
            }
            norm_scales[i] = s;
        } catch (const std::exception& e) {
            emitter.phase_end(run_id, Phase::NORMALIZATION, "error", { {"error", e.what()} }, log_file);
            emitter.run_end(run_id, false, "error", log_file);
            std::cerr << "Error during NORMALIZATION: " << e.what() << std::endl;
            return 1;
        }

        const float progress = frames.empty() ? 1.0f : static_cast<float>(i + 1) / static_cast<float>(frames.size());
        emitter.phase_progress(run_id, Phase::NORMALIZATION, progress,
                               "normalize " + std::to_string(i + 1) + "/" + std::to_string(frames.size()),
                               log_file);
    }

    {
        core::json artifact;
        artifact["mode"] = (detected_mode == ColorMode::OSC) ? "OSC" : "MONO";
        artifact["bayer_pattern"] = detected_bayer_str;
        artifact["B_mono"] = core::json::array();
        artifact["B_r"] = core::json::array();
        artifact["B_g"] = core::json::array();
        artifact["B_b"] = core::json::array();
        for (size_t i = 0; i < frames.size(); ++i) {
            artifact["B_mono"].push_back(B_mono[i]);
            artifact["B_r"].push_back(B_r[i]);
            artifact["B_g"].push_back(B_g[i]);
            artifact["B_b"].push_back(B_b[i]);
        }
        core::write_text(run_dir / "artifacts" / "normalization.json", artifact.dump(2));
    }

    auto median_finite_positive = [&](const std::vector<float>& v, float fallback) -> float {
        std::vector<float> p;
        p.reserve(v.size());
        for (float x : v) {
            if (std::isfinite(x) && x > 0.0f) p.push_back(x);
        }
        if (p.empty()) return fallback;
        return median_of(p);
    };

    const float output_pedestal = 32768.0f;
    const float output_bg_mono = median_finite_positive(B_mono, 1.0f);
    const float output_bg_r = median_finite_positive(B_r, 1.0f);
    const float output_bg_g = median_finite_positive(B_g, 1.0f);
    const float output_bg_b = median_finite_positive(B_b, 1.0f);

    auto apply_output_scaling_inplace = [&](Matrix2Df& img, int origin_x, int origin_y) {
        if (img.size() <= 0) return;
        if (detected_mode != ColorMode::OSC) {
            img *= output_bg_mono;
            img.array() += output_pedestal;
            return;
        }

        int r_row, r_col, b_row, b_col;
        bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);
        for (int y = 0; y < img.rows(); ++y) {
            const int gy = origin_y + y;
            for (int x = 0; x < img.cols(); ++x) {
                const int gx = origin_x + x;
                const int py = gy & 1;
                const int px = gx & 1;
                if (py == r_row && px == r_col) {
                    img(y, x) = img(y, x) * output_bg_r + output_pedestal;
                } else if (py == b_row && px == b_col) {
                    img(y, x) = img(y, x) * output_bg_b + output_pedestal;
                } else {
                    img(y, x) = img(y, x) * output_bg_g + output_pedestal;
                }
            }
        }
    };

    emitter.phase_end(run_id, Phase::NORMALIZATION, "ok",
                      {
                          {"num_frames", static_cast<int>(frames.size())},
                      },
                      log_file);

    // Phase 3: GLOBAL_METRICS
    emitter.phase_start(run_id, Phase::GLOBAL_METRICS, "GLOBAL_METRICS", log_file);

    std::vector<FrameMetrics> frame_metrics;
    frame_metrics.resize(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& path = frames[i];
        try {
            auto frame_pair = io::read_fits_float(path);
            Matrix2Df img = frame_pair.first;
            if (img.size() <= 0) {
                emitter.warning(run_id,
                                "GLOBAL_METRICS: empty frame for " + path.filename().string(),
                                log_file);
                FrameMetrics m;
                m.background = 0.0f;
                m.noise = 0.0f;
                m.gradient_energy = 0.0f;
                m.quality_score = 1.0f;
                frame_metrics[i] = m;
            } else {
                apply_normalization_inplace(img, norm_scales[i], detected_mode, detected_bayer_str, 0, 0);
                frame_metrics[i] = metrics::calculate_frame_metrics(img);
            }
        } catch (const std::exception& e) {
            emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "error", {{"error", e.what()}}, log_file);
            emitter.run_end(run_id, false, "error", log_file);
            std::cerr << "Error during GLOBAL_METRICS: " << e.what() << std::endl;
            return 1;
        }

        const float progress = frames.empty() ? 1.0f : static_cast<float>(i + 1) / static_cast<float>(frames.size());
        emitter.phase_progress(run_id, Phase::GLOBAL_METRICS, progress,
                               "metrics " + std::to_string(i + 1) + "/" + std::to_string(frames.size()),
                               log_file);
    }

    VectorXf global_weights = metrics::calculate_global_weights(
        frame_metrics,
        cfg.global_metrics.weights.background,
        cfg.global_metrics.weights.noise,
        cfg.global_metrics.weights.gradient,
        cfg.global_metrics.clamp[0],
        cfg.global_metrics.clamp[1]);

    {
        core::json artifact;
        artifact["metrics"] = core::json::array();
        for (size_t i = 0; i < frame_metrics.size(); ++i) {
            core::json m;
            m["background"] = frame_metrics[i].background;
            m["noise"] = frame_metrics[i].noise;
            m["gradient_energy"] = frame_metrics[i].gradient_energy;
            m["quality_score"] = frame_metrics[i].quality_score;
            m["global_weight"] = (i < static_cast<size_t>(global_weights.size())) ? global_weights[static_cast<int>(i)] : 0.0f;
            artifact["metrics"].push_back(m);
        }

        artifact["weights"] = {
            {"background", cfg.global_metrics.weights.background},
            {"noise", cfg.global_metrics.weights.noise},
            {"gradient", cfg.global_metrics.weights.gradient}
        };
        artifact["clamp"] = {cfg.global_metrics.clamp[0], cfg.global_metrics.clamp[1]};
        artifact["adaptive_weights"] = cfg.global_metrics.adaptive_weights;
        core::write_text(run_dir / "artifacts" / "global_metrics.json", artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "ok",
                      {
                          {"num_frames", static_cast<int>(frame_metrics.size())},
                      },
                      log_file);

    // Phase 4: TILE_GRID (with adaptive optimization)
    emitter.phase_start(run_id, Phase::TILE_GRID, "TILE_GRID", log_file);

    float seeing_fwhm_med = 3.0f;
    {
        const int patch_r = 10;
        const int patch_sz = 2 * patch_r + 1;
        const size_t n_probe = std::min<size_t>(5, frames.size());
        std::vector<float> fwhms;
        for (size_t pi = 0; pi < n_probe; ++pi) {
            size_t fi = (n_probe <= 1) ? 0 : (pi * (frames.size() - 1)) / (n_probe - 1);
            const int roi_w = std::min(width, 1024);
            const int roi_h = std::min(height, 1024);
            const int roi_x0 = std::max(0, (width - roi_w) / 2);
            const int roi_y0 = std::max(0, (height - roi_h) / 2);

            Matrix2Df img = io::read_fits_region_float(frames[fi], roi_x0, roi_y0, roi_w, roi_h);
            apply_normalization_inplace(img, norm_scales[fi], detected_mode, detected_bayer_str, roi_x0, roi_y0);
            cv::Mat img_cv(img.rows(), img.cols(), CV_32F, const_cast<float*>(img.data()));
            cv::Mat blur;
            cv::blur(img_cv, blur, cv::Size(31, 31), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
            cv::Mat resid = img_cv - blur;

            std::vector<cv::Point2f> corners;
            try {
                cv::goodFeaturesToTrack(resid, corners, 400, 0.01, 6);
            } catch (...) {
                corners.clear();
            }

            for (const auto& p : corners) {
                int cx = static_cast<int>(std::round(p.x));
                int cy = static_cast<int>(std::round(p.y));
                int x0 = cx - patch_r;
                int y0 = cy - patch_r;
                if (x0 < 0 || y0 < 0 || (x0 + patch_sz) > img_cv.cols || (y0 + patch_sz) > img_cv.rows) continue;
                cv::Mat patch = img_cv(cv::Rect(x0, y0, patch_sz, patch_sz));
                float f = estimate_fwhm_from_patch(patch);
                if (f > 0.0f && std::isfinite(f)) {
                    fwhms.push_back(f);
                }
            }
        }

        if (fwhms.size() >= 25) {
            seeing_fwhm_med = median_of(fwhms);
        }
    }

    int seeing_tile_size = 0;
    float seeing_overlap_fraction = 0.25f;
    {
        float t0 = 32.0f * seeing_fwhm_med;
        int tmin = 64;
        int tmax = std::max(1, std::min(width, height) / 6);
        float tc = std::min(std::max(t0, static_cast<float>(tmin)), static_cast<float>(tmax));
        seeing_tile_size = static_cast<int>(std::floor(tc));
        if (seeing_tile_size < tmin) seeing_tile_size = tmin;
    }

    bool adaptive_enabled = cfg.v4.adaptive_tiles.enabled;
    bool use_warp_probe = adaptive_enabled && cfg.v4.adaptive_tiles.use_warp_probe;
    bool use_hierarchical = adaptive_enabled && cfg.v4.adaptive_tiles.use_hierarchical;

    cfg.tile.overlap_fraction = seeing_overlap_fraction;
    if (adaptive_enabled) {
        cfg.v4.adaptive_tiles.initial_tile_size = seeing_tile_size;
    }

    tile_compile::pipeline::WarpGradientField grad_field;
    bool has_grad_field = false;

    if (use_warp_probe && frames.size() >= 3) {
        int probe_window = cfg.v4.adaptive_tiles.probe_window;
        int num_probe_frames = cfg.v4.adaptive_tiles.num_probe_frames;

        if (probe_window <= width && probe_window <= height) {
            try {
                grad_field = tile_compile::pipeline::compute_warp_gradient_field(
                    frames,
                    width,
                    height,
                    probe_window,
                    num_probe_frames,
                    nullptr,
                    [&](float p) {
                        emitter.phase_progress(run_id, Phase::TILE_GRID, p, "warp_probe", log_file);
                    });
                has_grad_field = grad_field.grid.size() > 0;
            } catch (const std::exception& e) {
                emitter.warning(run_id,
                                std::string("warp_probe failed: ") + e.what() + "; continuing without gradient field",
                                log_file);
                has_grad_field = false;
                grad_field = tile_compile::pipeline::WarpGradientField();
            } catch (...) {
                emitter.warning(run_id,
                                "warp_probe failed: unknown error; continuing without gradient field",
                                log_file);
                has_grad_field = false;
                grad_field = tile_compile::pipeline::WarpGradientField();
            }
        } else {
            emitter.warning(run_id, "warp_probe skipped: probe_window larger than image", log_file);
        }
    }

    std::vector<Tile> tiles;
    int uniform_tile_size = 0;

    if (adaptive_enabled) {
        if (use_hierarchical) {
            tiles = tile_compile::pipeline::build_hierarchical_tile_grid(width, height, cfg, has_grad_field ? &grad_field : nullptr);
        } else {
            tiles = tile_compile::pipeline::build_adaptive_tile_grid(width, height, cfg, has_grad_field ? &grad_field : nullptr);
        }
    } else {
        uniform_tile_size = seeing_tile_size;

        tiles = tile_compile::pipeline::build_initial_tile_grid(width, height, uniform_tile_size, cfg.tile.overlap_fraction);
    }

    {
        core::json artifact;
        artifact["image_width"] = width;
        artifact["image_height"] = height;
        artifact["num_tiles"] = static_cast<int>(tiles.size());
        artifact["adaptive_enabled"] = adaptive_enabled;
        artifact["use_warp_probe"] = use_warp_probe;
        artifact["use_hierarchical"] = use_hierarchical;
        artifact["overlap_fraction"] = cfg.tile.overlap_fraction;
        artifact["seeing_fwhm_median"] = seeing_fwhm_med;
        artifact["seeing_tile_size"] = seeing_tile_size;
        artifact["seeing_overlap_px"] = static_cast<int>(std::floor(cfg.tile.overlap_fraction * static_cast<float>(seeing_tile_size)));
        if (!adaptive_enabled) {
            artifact["uniform_tile_size"] = uniform_tile_size;
        }

        artifact["tiles"] = core::json::array();
        for (const auto& t : tiles) {
            artifact["tiles"].push_back({
                {"x", t.x},
                {"y", t.y},
                {"width", t.width},
                {"height", t.height},
            });
        }

        core::write_text(run_dir / "artifacts" / "tile_grid.json", artifact.dump(2));
    }

    if (has_grad_field) {
        core::json g;
        g["probe_window"] = grad_field.probe_window;
        g["step"] = grad_field.step;
        g["grid_h"] = grad_field.grid_h;
        g["grid_w"] = grad_field.grid_w;
        g["probe_indices"] = core::json::array();
        for (int idx : grad_field.probe_indices) g["probe_indices"].push_back(idx);
        g["min"] = grad_field.min_val;
        g["max"] = grad_field.max_val;
        g["mean"] = grad_field.mean_val;
        g["grid"] = core::json::array();
        for (int r = 0; r < grad_field.grid.rows(); ++r) {
            core::json row = core::json::array();
            for (int c = 0; c < grad_field.grid.cols(); ++c) {
                row.push_back(grad_field.grid(r, c));
            }
            g["grid"].push_back(row);
        }
        core::write_text(run_dir / "artifacts" / "warp_gradient_field.json", g.dump(2));
    }

    emitter.phase_end(run_id, Phase::TILE_GRID, "ok",
                      {
                          {"num_tiles", static_cast<int>(tiles.size())},
                          {"adaptive", adaptive_enabled},
                          {"gradient_field", has_grad_field},
                      },
                      log_file);

    // Helpers for Phase 5/6
    auto load_frame_normalized = [&](size_t frame_index) -> std::pair<Matrix2Df, io::FitsHeader> {
        auto frame_pair = io::read_fits_float(frames[frame_index]);
        Matrix2Df img = frame_pair.first;
        apply_normalization_inplace(img, norm_scales[frame_index], detected_mode, detected_bayer_str, 0, 0);
        return {img, frame_pair.second};
    };

    auto extract_tile = [&](const Matrix2Df& img, const Tile& t) -> Matrix2Df {
        int cols = static_cast<int>(img.cols());
        int rows = static_cast<int>(img.rows());
        int x0 = std::max(0, t.x);
        int y0 = std::max(0, t.y);
        int x1 = std::min(cols, t.x + t.width);
        int y1 = std::min(rows, t.y + t.height);
        int tw = std::max(0, x1 - x0);
        int th = std::max(0, y1 - y0);
        if (tw <= 0 || th <= 0) return Matrix2Df();
        return img.block(y0, x0, th, tw);
    };

    auto make_hann_1d = [&](int n) -> std::vector<float> {
        std::vector<float> w;
        if (n <= 0) return w;
        w.resize(static_cast<size_t>(n));
        if (n == 1) {
            w[0] = 1.0f;
            return w;
        }
        const float pi = 3.14159265358979323846f;
        for (int i = 0; i < n; ++i) {
            float x = static_cast<float>(i) / static_cast<float>(n - 1);
            w[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(2.0f * pi * x));
        }
        return w;
    };

    std::vector<Tile> tiles_phase56 = tiles;
    if (max_tiles > 0 && tiles_phase56.size() > static_cast<size_t>(max_tiles)) {
        tiles_phase56.resize(static_cast<size_t>(max_tiles));
    }

    // Phase 5: LOCAL_METRICS (compute tile metrics per frame)
    emitter.phase_start(run_id, Phase::LOCAL_METRICS, "LOCAL_METRICS", log_file);

    std::vector<std::vector<TileMetrics>> local_metrics;
    local_metrics.resize(frames.size());
    std::vector<std::vector<float>> local_weights;
    local_weights.resize(frames.size());

    const std::string phase5_io_mode = cfg.v4.phase6_io.mode;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        local_metrics[fi].reserve(tiles_phase56.size());
        local_weights[fi].reserve(tiles_phase56.size());

        Matrix2Df img;
        if (phase5_io_mode != "roi") {
            auto pair = load_frame_normalized(fi);
            img = std::move(pair.first);
        }

        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
            const Tile& t = tiles_phase56[ti];
            Matrix2Df tile_img;

            if (phase5_io_mode == "roi") {
                const int rx0 = std::max(0, t.x);
                const int ry0 = std::max(0, t.y);
                tile_img = io::read_fits_region_float(frames[fi], rx0, ry0, t.width, t.height);
                apply_normalization_inplace(tile_img, norm_scales[fi], detected_mode, detected_bayer_str, rx0, ry0);
            } else {
                tile_img = extract_tile(img, t);
            }

            if (tile_img.size() <= 0) {
                TileMetrics z;
                z.fwhm = 0.0f;
                z.roundness = 0.0f;
                z.contrast = 0.0f;
                z.sharpness = 0.0f;
                z.background = 0.0f;
                z.noise = 0.0f;
                z.gradient_energy = 0.0f;
                z.star_count = 0;
                z.type = TileType::STRUCTURE;
                z.quality_score = 0.0f;
                local_metrics[fi].push_back(z);
                local_weights[fi].push_back(1.0f);
                continue;
            }

            TileMetrics tm = metrics::calculate_tile_metrics(tile_img);
            local_metrics[fi].push_back(tm);
            local_weights[fi].push_back(1.0f);
        }

        const float p = frames.empty() ? 1.0f : static_cast<float>(fi + 1) / static_cast<float>(frames.size());
        emitter.phase_progress(run_id, Phase::LOCAL_METRICS, p,
                               "local_metrics " + std::to_string(fi + 1) + "/" + std::to_string(frames.size()),
                               log_file);
    }

    {
        auto robust_tilde = [&](const std::vector<float>& v, std::vector<float>& out) {
            out.assign(v.size(), 0.0f);
            if (v.empty()) return;
            std::vector<float> tmp = v;
            float med = median_of(tmp);
            for (float& x : tmp) x = std::fabs(x - med);
            float mad = median_of(tmp);
            float sigma = 1.4826f * mad;
            if (!(sigma > 0.0f)) {
                std::fill(out.begin(), out.end(), 0.0f);
                return;
            }
            for (size_t i = 0; i < v.size(); ++i) {
                out[i] = (v[i] - med) / sigma;
            }
        };

        auto clip3 = [&](float x) -> float {
            return std::min(std::max(x, cfg.local_metrics.clamp[0]), cfg.local_metrics.clamp[1]);
        };

        const int star_thr = cfg.tile.star_min_count;
        const float eps = 1.0e-12f;

        const size_t n_frames = local_metrics.size();
        const size_t n_tiles = tiles_phase56.size();
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            std::vector<float> fwhm_log;
            std::vector<float> roundness;
            std::vector<float> contrast;
            std::vector<float> bg;
            std::vector<float> noise;
            std::vector<float> energy;
            std::vector<float> star_counts;

            fwhm_log.reserve(n_frames);
            roundness.reserve(n_frames);
            contrast.reserve(n_frames);
            bg.reserve(n_frames);
            noise.reserve(n_frames);
            energy.reserve(n_frames);
            star_counts.reserve(n_frames);

            for (size_t fi = 0; fi < n_frames; ++fi) {
                const TileMetrics& tm = local_metrics[fi][ti];
                fwhm_log.push_back(std::log(std::max(tm.fwhm, 1.0e-6f)));
                roundness.push_back(tm.roundness);
                contrast.push_back(tm.contrast);
                bg.push_back(tm.background);
                noise.push_back(tm.noise);
                energy.push_back(tm.gradient_energy);
                star_counts.push_back(static_cast<float>(tm.star_count));
            }

            std::vector<float> sc_tmp = star_counts;
            float sc_med = median_of(sc_tmp);
            const TileType tile_type = (sc_med >= static_cast<float>(star_thr)) ? TileType::STAR : TileType::STRUCTURE;

            std::vector<float> fwhm_t, r_t, c_t, b_t, s_t, e_t;
            robust_tilde(fwhm_log, fwhm_t);
            robust_tilde(roundness, r_t);
            robust_tilde(contrast, c_t);
            robust_tilde(bg, b_t);
            robust_tilde(noise, s_t);
            robust_tilde(energy, e_t);

            for (size_t fi = 0; fi < n_frames; ++fi) {
                TileMetrics& tm = local_metrics[fi][ti];
                tm.type = tile_type;

                float q = 0.0f;
                if (tile_type == TileType::STAR) {
                    q = cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_t[fi]) +
                        cfg.local_metrics.star_mode.weights.roundness * (r_t[fi]) +
                        cfg.local_metrics.star_mode.weights.contrast * (c_t[fi]);
                } else {
                    float denom = s_t[fi];
                    float ratio = (std::fabs(denom) > eps) ? (e_t[fi] / denom) : 0.0f;
                    q = cfg.local_metrics.structure_mode.metric_weight * ratio +
                        cfg.local_metrics.structure_mode.background_weight * (-b_t[fi]);
                }

                q = clip3(q);
                tm.quality_score = q;
                local_weights[fi][ti] = std::exp(q);
            }
        }

        core::json artifact;
        artifact["num_frames"] = static_cast<int>(frames.size());
        artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
        artifact["tile_metrics"] = core::json::array();

        for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
            core::json fm = core::json::array();
            for (size_t ti = 0; ti < local_metrics[fi].size(); ++ti) {
                const auto& m = local_metrics[fi][ti];
                fm.push_back({
                    {"fwhm", m.fwhm},
                    {"roundness", m.roundness},
                    {"contrast", m.contrast},
                    {"sharpness", m.sharpness},
                    {"background", m.background},
                    {"noise", m.noise},
                    {"gradient_energy", m.gradient_energy},
                    {"star_count", m.star_count},
                    {"tile_type", (m.type == TileType::STAR) ? "STAR" : "STRUCTURE"},
                    {"quality_score", m.quality_score},
                    {"local_weight", local_weights[fi][ti]},
                });
            }
            artifact["tile_metrics"].push_back(fm);
        }

        core::write_text(run_dir / "artifacts" / "local_metrics.json", artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::LOCAL_METRICS, "ok",
                      {
                          {"num_frames", static_cast<int>(frames.size())},
                          {"num_tiles", static_cast<int>(tiles_phase56.size())},
                      },
                      log_file);

    const bool reduced_mode = (static_cast<int>(frames.size()) < cfg.assumptions.frames_reduced_threshold);
    const bool skip_clustering_in_reduced = (reduced_mode && cfg.assumptions.reduced_mode_skip_clustering);

    // Phase 6: TILE_RECONSTRUCTION_TLR (Methodik v4 compliant)
    // Features: iterative refinement, temporal smoothing, variance window
    emitter.phase_start(run_id, Phase::TILE_RECONSTRUCTION_TLR, "TILE_RECONSTRUCTION_TLR", log_file);

    const int passes_total = std::max(1, cfg.v4.iterations);
    const int total_tiles_planned = static_cast<int>(tiles_phase56.size()) * passes_total;
    std::atomic<int> cumulative_tiles_processed{0};

    // Helper: compute warp variance (Methodik v4 §9, §10)
    auto compute_warp_variance = [](const std::vector<std::pair<float, float>>& translations) -> float {
        if (translations.size() < 2) return 0.0f;
        float sum_x = 0.0f, sum_y = 0.0f;
        for (const auto& [dx, dy] : translations) {
            sum_x += dx;
            sum_y += dy;
        }
        float mean_x = sum_x / static_cast<float>(translations.size());
        float mean_y = sum_y / static_cast<float>(translations.size());
        float var_x = 0.0f, var_y = 0.0f;
        for (const auto& [dx, dy] : translations) {
            var_x += (dx - mean_x) * (dx - mean_x);
            var_y += (dy - mean_y) * (dy - mean_y);
        }
        var_x /= static_cast<float>(translations.size());
        var_y /= static_cast<float>(translations.size());
        return var_x + var_y;
    };

    // Helper: variance window weight ψ(var) (Methodik v4 §9)
    auto variance_window_weight = [](float warp_variance, float sigma) -> float {
        float w = std::exp(-warp_variance / (2.0f * sigma * sigma));
        return std::max(w, 1.0e-3f);
    };

    // Helper: temporal smoothing of warps (Methodik v4 §5.3) - median filter
    auto smooth_warps_translation = [](std::vector<std::pair<float, float>>& warps, int window) {
        if (static_cast<int>(warps.size()) < window) return;
        int half = window / 2;
        std::vector<std::pair<float, float>> smoothed(warps.size());
        for (size_t i = 0; i < warps.size(); ++i) {
            std::vector<float> xs, ys;
            for (int j = std::max(0, static_cast<int>(i) - half);
                 j < std::min(static_cast<int>(warps.size()), static_cast<int>(i) + half + 1); ++j) {
                xs.push_back(warps[static_cast<size_t>(j)].first);
                ys.push_back(warps[static_cast<size_t>(j)].second);
            }
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            smoothed[i].first = xs[xs.size() / 2];
            smoothed[i].second = ys[ys.size() / 2];
        }
        warps = smoothed;
    };

    // Get first frame for dimensions
    auto [first_img, first_hdr] = load_frame_normalized(0);
    Matrix2Df recon = Matrix2Df::Zero(first_img.rows(), first_img.cols());
    Matrix2Df weight_sum = Matrix2Df::Zero(first_img.rows(), first_img.cols());

    bool run_validation_failed = false;

    const int prev_cv_threads_recon = cv::getNumThreads();
    cv::setNumThreads(1);

    // Config parameters
    int min_valid_frames = cfg.registration.local_tiles.min_valid_frames;
    float ecc_cc_min = cfg.registration.local_tiles.ecc_cc_min;
    int iterations = cfg.v4.iterations;
    const int temporal_smoothing_window = cfg.registration.local_tiles.temporal_smoothing_window;
    const float max_warp_delta_px = cfg.registration.local_tiles.max_warp_delta_px;
    const float variance_sigma = cfg.registration.local_tiles.variance_window_sigma;

    const std::string phase6_io_mode = cfg.v4.phase6_io.mode;
    const int phase6_lru_capacity = cfg.v4.phase6_io.lru_capacity;

    // Parallel processing configuration
    int parallel_tiles = cfg.v4.parallel_tiles;
    int cpu_cores = std::thread::hardware_concurrency();
    if (cpu_cores == 0) cpu_cores = 1;
    if (parallel_tiles > cpu_cores) {
        std::cout << "[WARNING] parallel_tiles (" << parallel_tiles << ") exceeds CPU cores (" 
                  << cpu_cores << "), capping to " << cpu_cores << std::endl;
        parallel_tiles = cpu_cores;
    }
    if (parallel_tiles < 1) parallel_tiles = 1;

    std::cout << "[Phase 6] Using " << parallel_tiles << " parallel workers for " 
              << tiles_phase56.size() << " tiles" << std::endl;

    std::vector<int> tile_valid_counts(tiles_phase56.size(), 0);
    std::vector<float> tile_warp_variances(tiles_phase56.size(), 0.0f);
    std::vector<float> tile_mean_correlations(tiles_phase56.size(), 0.0f);

    // Thread-safe structures for parallel processing
    std::mutex recon_mutex;
    std::atomic<size_t> tiles_completed{0};
    std::atomic<size_t> tiles_failed{0};

    class FrameLruCache {
    public:
        explicit FrameLruCache(size_t capacity) : capacity_(capacity) {}

        const Matrix2Df& get_or_load(size_t key, const std::function<Matrix2Df()>& loader) {
            if (capacity_ == 0) {
                scratch_ = loader();
                return scratch_;
            }

            auto it = map_.find(key);
            if (it != map_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second);
                return it->second->value;
            }

            Matrix2Df value = loader();
            lru_.push_front(Node{key, std::move(value)});
            map_[key] = lru_.begin();

            if (lru_.size() > capacity_) {
                auto last = std::prev(lru_.end());
                map_.erase(last->key);
                lru_.pop_back();
            }

            return lru_.begin()->value;
        }

    private:
        struct Node {
            size_t key;
            Matrix2Df value;
        };

        size_t capacity_ = 0;
        std::list<Node> lru_;
        std::unordered_map<size_t, std::list<Node>::iterator> map_;
        Matrix2Df scratch_;
    };

    // Worker function for parallel tile processing
    auto process_tile = [&](size_t ti, FrameLruCache& frame_cache) {
        const Tile& t = tiles_phase56[ti];

        auto dbg = [&](const std::string& s) {
            if (!debug_tile_reg || !debug_log_file.is_open()) return;
            std::lock_guard<std::mutex> lock(debug_log_mutex);
            debug_log_file << s << "\n";
        };

        dbg("tile=" + std::to_string(ti) +
            " x=" + std::to_string(t.x) +
            " y=" + std::to_string(t.y) +
            " w=" + std::to_string(t.width) +
            " h=" + std::to_string(t.height));

        auto load_tile_normalized = [&](size_t fi) -> Matrix2Df {
            const int rx0 = std::max(0, t.x);
            const int ry0 = std::max(0, t.y);
            if (phase6_io_mode == "roi") {
                Matrix2Df tile_img = io::read_fits_region_float(frames[fi], rx0, ry0, t.width, t.height);
                apply_normalization_inplace(tile_img, norm_scales[fi], detected_mode, detected_bayer_str, rx0, ry0);
                return tile_img;
            }
            if (phase6_io_mode == "lru") {
                const Matrix2Df& full = frame_cache.get_or_load(fi, [&]() {
                    auto frame_pair = io::read_fits_float(frames[fi]);
                    Matrix2Df img = frame_pair.first;
                    apply_normalization_inplace(img, norm_scales[fi], detected_mode, detected_bayer_str, 0, 0);
                    return img;
                });
                return extract_tile(full, t);
            }

            auto [img, _hdr] = load_frame_normalized(fi);
            return extract_tile(img, t);
        };

        // Initial reference: median of all tiles (simplified: use first frame)
        Matrix2Df ref_tile;
        {
            std::vector<Matrix2Df> tile_samples;
            for (size_t fi = 0; fi < std::min(frames.size(), size_t(5)); ++fi) {
                Matrix2Df sample = load_tile_normalized(fi);
                if (sample.size() > 0) tile_samples.push_back(sample);
            }
            if (tile_samples.empty()) {
                tiles_failed++;
                return;
            }
            // Use first sample as initial reference
            ref_tile = tile_samples[0];
        }

        if (ref_tile.size() <= 0) {
            tiles_failed++;
            return;
        }

        std::vector<float> hann_x = make_hann_1d(ref_tile.cols());
        std::vector<float> hann_y = make_hann_1d(ref_tile.rows());

        // Iterative refinement (Methodik v4 §5.2)
        std::vector<std::pair<float, float>> final_translations;
        std::vector<float> final_correlations;
        std::vector<Matrix2Df> final_warped_tiles;
        std::vector<float> final_weights;

        for (int iter = 0; iter < iterations; ++iter) {
            // For OSC: use green proxy for registration (Methodik v4)
            Matrix2Df ref_reg = (detected_mode == ColorMode::OSC) 
                ? image::cfa_green_proxy_downsample2x2(ref_tile, detected_bayer_str)
                : ref_tile;
            Matrix2Df ref_ecc = registration::prepare_ecc_image(ref_reg);

            std::vector<std::pair<float, float>> translations;
            std::vector<float> correlations;
            std::vector<Matrix2Df> warped_tiles;
            std::vector<float> weights;
            std::vector<size_t> frame_indices;

            int ecc_failures = 0;
            int tile_size_mismatches = 0;
            for (size_t fi = 0; fi < frames.size(); ++fi) {
                Matrix2Df mov_tile = load_tile_normalized(fi);
                if (mov_tile.size() <= 0 || mov_tile.rows() != ref_tile.rows() || mov_tile.cols() != ref_tile.cols()) {
                    tile_size_mismatches++;
                    dbg("tile=" + std::to_string(ti) + " iter=" + std::to_string(iter) +
                        " fi=" + std::to_string(fi) + " reject=size_mismatch");
                    continue;
                }

                // For OSC: use green proxy for registration (Methodik v4)
                Matrix2Df mov_reg = (detected_mode == ColorMode::OSC)
                    ? image::cfa_green_proxy_downsample2x2(mov_tile, detected_bayer_str)
                    : mov_tile;
                Matrix2Df mov_ecc = registration::prepare_ecc_image(mov_reg);

                auto [dx, dy] = registration::phasecorr_translation(mov_ecc, ref_ecc);
                float max_shift = 0.5f * static_cast<float>(std::min(mov_reg.rows(), mov_reg.cols()));
                dx = std::max(-max_shift, std::min(max_shift, dx));
                dy = std::max(-max_shift, std::min(max_shift, dy));

                WarpMatrix init = registration::identity_warp();
                init(0, 2) = dx;
                init(1, 2) = dy;

                RegistrationResult rr = registration::ecc_warp(
                    mov_ecc, ref_ecc, false, init, 50, 1.0e-4f);

                // Use phase correlation as fallback if ECC fails (like Python reference)
                float cc = 0.0f;
                float final_dx = dx;
                float final_dy = dy;

                if (rr.success) {
                    cc = rr.correlation;
                    final_dx = rr.warp(0, 2);
                    final_dy = rr.warp(1, 2);
                } else {
                    ecc_failures++;
                    // Continue with phase correlation result (cc=0.0)
                }

                dbg("tile=" + std::to_string(ti) +
                    " iter=" + std::to_string(iter) +
                    " fi=" + std::to_string(fi) +
                    " init_dx=" + std::to_string(dx) +
                    " init_dy=" + std::to_string(dy) +
                    " ecc_success=" + std::string(rr.success ? "1" : "0") +
                    " cc=" + std::to_string(cc) +
                    " final_dx=" + std::to_string(final_dx) +
                    " final_dy=" + std::to_string(final_dy));

                translations.push_back({final_dx, final_dy});
                correlations.push_back(cc);
                frame_indices.push_back(fi);

                // Use final warp (ECC result or phase correlation fallback)
                WarpMatrix final_warp = registration::identity_warp();
                final_warp(0, 2) = final_dx;
                final_warp(1, 2) = final_dy;

                // For OSC: use CFA-aware warping to preserve Bayer pattern (Methodik v4)
                Matrix2Df mov_warped = (detected_mode == ColorMode::OSC)
                    ? image::warp_cfa_mosaic_via_subplanes(mov_tile, final_warp, mov_tile.rows(), mov_tile.cols())
                    : registration::apply_warp(mov_tile, final_warp);
                warped_tiles.push_back(mov_warped);

                float L_ft = 1.0f;
                if (fi < local_weights.size() && ti < local_weights[fi].size()) {
                    L_ft = local_weights[fi][ti];
                }
                float G_f = (fi < static_cast<size_t>(global_weights.size()))
                                ? global_weights[static_cast<int>(fi)]
                                : 1.0f;
                weights.push_back(G_f * L_ft);
            }

            if (translations.empty()) {
                if (iter == 0 && ti == 0) {
                    std::cerr << "Warning: Tile " << ti << " iter " << iter << ": All registrations failed. "
                              << "ECC failures: " << ecc_failures << ", size mismatches: " << tile_size_mismatches 
                              << ", ref_tile size: " << ref_tile.rows() << "x" << ref_tile.cols() << std::endl;
                }
                break;
            }

            // Temporal smoothing of warps (Methodik v4 §5.3)
            if (temporal_smoothing_window > 1) {
                smooth_warps_translation(translations, temporal_smoothing_window);
            }

            // Warp consistency check (Methodik v4 - prevents double stars)
            if (max_warp_delta_px > 0.0f && translations.size() > 1) {
                std::vector<float> dxs, dys;
                for (const auto& [dx, dy] : translations) {
                    dxs.push_back(dx);
                    dys.push_back(dy);
                }
                std::sort(dxs.begin(), dxs.end());
                std::sort(dys.begin(), dys.end());
                float median_dx = dxs[dxs.size() / 2];
                float median_dy = dys[dys.size() / 2];

                std::vector<bool> valid_mask(translations.size(), true);
                for (size_t i = 0; i < translations.size(); ++i) {
                    float delta = std::sqrt(
                        (translations[i].first - median_dx) * (translations[i].first - median_dx) +
                        (translations[i].second - median_dy) * (translations[i].second - median_dy));
                    if (delta > max_warp_delta_px) {
                        valid_mask[i] = false;
                        if (i < frame_indices.size()) {
                            dbg("tile=" + std::to_string(ti) +
                                " iter=" + std::to_string(iter) +
                                " fi=" + std::to_string(frame_indices[i]) +
                                " reject=warp_delta" +
                                " delta=" + std::to_string(delta) +
                                " max=" + std::to_string(max_warp_delta_px) +
                                " median_dx=" + std::to_string(median_dx) +
                                " median_dy=" + std::to_string(median_dy));
                        }
                    }
                }

                // Filter out invalid frames
                std::vector<std::pair<float, float>> filtered_translations;
                std::vector<float> filtered_correlations;
                std::vector<Matrix2Df> filtered_warped;
                std::vector<float> filtered_weights;
                for (size_t i = 0; i < valid_mask.size(); ++i) {
                    if (valid_mask[i]) {
                        filtered_translations.push_back(translations[i]);
                        filtered_correlations.push_back(correlations[i]);
                        filtered_warped.push_back(warped_tiles[i]);
                        filtered_weights.push_back(weights[i]);
                    }
                }
                translations = filtered_translations;
                correlations = filtered_correlations;
                warped_tiles = filtered_warped;
                weights = filtered_weights;
                {
                    dbg("tile=" + std::to_string(ti) +
                        " iter=" + std::to_string(iter) +
                        " warp_consistency_kept=" + std::to_string(translations.size()));
                }
            }

            // Filter by ECC correlation threshold (but keep phase correlation fallbacks with cc=0)
            // Note: Python reference keeps all registrations, even with low cc
            {
                std::vector<std::pair<float, float>> filtered_translations;
                std::vector<float> filtered_correlations;
                std::vector<Matrix2Df> filtered_warped;
                std::vector<float> filtered_weights;
                for (size_t i = 0; i < correlations.size(); ++i) {
                    // Keep if: (1) ECC succeeded with good correlation, OR (2) phase correlation fallback
                    bool keep = (correlations[i] >= ecc_cc_min) || (correlations[i] == 0.0f);
                    if (keep) {
                        filtered_translations.push_back(translations[i]);
                        filtered_correlations.push_back(correlations[i]);
                        filtered_warped.push_back(warped_tiles[i]);
                        filtered_weights.push_back(weights[i]);
                    } else {
                        if (i < frame_indices.size()) {
                            dbg("tile=" + std::to_string(ti) +
                                " iter=" + std::to_string(iter) +
                                " fi=" + std::to_string(frame_indices[i]) +
                                " reject=ecc_cc" +
                                " cc=" + std::to_string(correlations[i]) +
                                " min=" + std::to_string(ecc_cc_min));
                        }
                    }
                }
                translations = filtered_translations;
                correlations = filtered_correlations;
                warped_tiles = filtered_warped;
                weights = filtered_weights;
            }

            if (static_cast<int>(warped_tiles.size()) < min_valid_frames) break;

            // Weighted reconstruction for next iteration reference
            float wsum = 0.0f;
            for (float w : weights) wsum += w;

            Matrix2Df new_ref = Matrix2Df::Zero(ref_tile.rows(), ref_tile.cols());
            if (wsum > 1.0e-12f) {
                for (size_t i = 0; i < warped_tiles.size(); ++i) {
                    new_ref += warped_tiles[i] * (weights[i] / wsum);
                }
            } else {
                for (size_t i = 0; i < warped_tiles.size(); ++i) {
                    new_ref += warped_tiles[i] / static_cast<float>(warped_tiles.size());
                }
            }
            ref_tile = new_ref;

            // Store for final iteration
            final_translations = translations;
            final_correlations = correlations;
            final_warped_tiles = warped_tiles;
            final_weights = weights;

            // Per-iteration progress (Python legacy style)
            const int global_done = cumulative_tiles_processed.fetch_add(1) + 1;
            emitter.phase_progress_counts(
                run_id,
                Phase::TILE_RECONSTRUCTION_TLR,
                global_done,
                total_tiles_planned,
                "Tile " + std::to_string(static_cast<int>(ti) + 1) + " von " + std::to_string(static_cast<int>(tiles_phase56.size())),
                "Iteration " + std::to_string(iter + 1) + " von " + std::to_string(passes_total),
                log_file);
        }

        int valid = static_cast<int>(final_warped_tiles.size());
        tile_valid_counts[ti] = valid;

        if (valid < std::max(1, min_valid_frames)) {
            tiles_failed++;
            return;
        }

        // Compute warp variance for this tile
        float warp_var = compute_warp_variance(final_translations);
        tile_warp_variances[ti] = warp_var;

        // Compute mean correlation
        float mean_cc = 0.0f;
        for (float cc : final_correlations) mean_cc += cc;
        mean_cc /= static_cast<float>(final_correlations.size());
        tile_mean_correlations[ti] = mean_cc;

        // Final weighted sum
        float wsum = 0.0f;
        for (float w : final_weights) wsum += w;

        Matrix2Df tile_rec = Matrix2Df::Zero(ref_tile.rows(), ref_tile.cols());
        if (wsum > 1.0e-12f) {
            for (size_t i = 0; i < final_warped_tiles.size(); ++i) {
                tile_rec += final_warped_tiles[i] * (final_weights[i] / wsum);
            }
        } else {
            for (size_t i = 0; i < final_warped_tiles.size(); ++i) {
                tile_rec += final_warped_tiles[i] / static_cast<float>(final_warped_tiles.size());
            }
        }

        // Variance window weight ψ(var) (Methodik v4 §9)
        float psi = variance_window_weight(warp_var, variance_sigma);

        dbg("tile=" + std::to_string(ti) +
            " valid_frames=" + std::to_string(valid) +
            " warp_var=" + std::to_string(warp_var) +
            " mean_cc=" + std::to_string(mean_cc) +
            " psi=" + std::to_string(psi));

        // Overlap-add into full image with variance-weighted Hanning window (thread-safe)
        {
            std::lock_guard<std::mutex> lock(recon_mutex);
            int x0 = std::max(0, t.x);
            int y0 = std::max(0, t.y);
            for (int yy = 0; yy < tile_rec.rows(); ++yy) {
                for (int xx = 0; xx < tile_rec.cols(); ++xx) {
                    int iy = y0 + yy;
                    int ix = x0 + xx;
                    if (iy < 0 || iy >= recon.rows() || ix < 0 || ix >= recon.cols()) continue;

                    float win = hann_y[static_cast<size_t>(yy)] * hann_x[static_cast<size_t>(xx)] * psi;
                    recon(iy, ix) += tile_rec(yy, xx) * win;
                    weight_sum(iy, ix) += win;
                }
            }
        }
        
        // Update per-tile completion stats (legacy progress is per-iteration)
        tiles_completed++;
    };
    
    // Execute tiles in parallel or serial based on parallel_tiles setting
    if (parallel_tiles > 1) {
        std::cout << "  Processing " << tiles_phase56.size() << " tiles with " << parallel_tiles << " workers..." << std::endl;
        
        std::vector<std::thread> workers;
        std::atomic<size_t> next_tile{0};
        
        for (int w = 0; w < parallel_tiles; ++w) {
            workers.emplace_back([&]() {
                FrameLruCache frame_cache((phase6_io_mode == "lru") ? static_cast<size_t>(std::max(0, phase6_lru_capacity)) : 0);
                while (true) {
                    size_t ti = next_tile.fetch_add(1);
                    if (ti >= tiles_phase56.size()) break;
                    process_tile(ti, frame_cache);
                }
            });
        }
        
        for (auto& worker : workers) {
            worker.join();
        }
        
        std::cout << "  Completed " << tiles_completed.load() << " tiles (" 
                  << tiles_failed.load() << " failed)" << std::endl;
    } else {
        std::cout << "  Processing " << tiles_phase56.size() << " tiles serially..." << std::endl;
        FrameLruCache frame_cache((phase6_io_mode == "lru") ? static_cast<size_t>(std::max(0, phase6_lru_capacity)) : 0);
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
            process_tile(ti, frame_cache);
        }
    }

    cv::setNumThreads(prev_cv_threads_recon);

    if (debug_tile_reg && debug_log_file.is_open()) {
        std::lock_guard<std::mutex> lock(debug_log_mutex);
        debug_log_file.flush();
    }

    // Normalize reconstruction
    for (int i = 0; i < recon.size(); ++i) {
        float ws = weight_sum.data()[i];
        if (ws > 1.0e-12f) {
            recon.data()[i] /= ws;
        } else {
            recon.data()[i] = first_img.data()[i];
        }
    }

    // Write reconstruction output (stacked.fits is the main output per Python reference)

    {
        core::json artifact;
        artifact["num_frames"] = static_cast<int>(frames.size());
        artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
        artifact["iterations"] = iterations;
        artifact["beta"] = cfg.v4.beta;
        artifact["temporal_smoothing_window"] = temporal_smoothing_window;
        artifact["max_warp_delta_px"] = max_warp_delta_px;
        artifact["variance_sigma"] = variance_sigma;
        artifact["min_valid_frames"] = min_valid_frames;
        artifact["ecc_cc_min"] = ecc_cc_min;
        artifact["tile_valid_counts"] = core::json::array();
        artifact["tile_warp_variances"] = core::json::array();
        artifact["tile_mean_correlations"] = core::json::array();
        for (size_t i = 0; i < tiles_phase56.size(); ++i) {
            artifact["tile_valid_counts"].push_back(tile_valid_counts[i]);
            artifact["tile_warp_variances"].push_back(tile_warp_variances[i]);
            artifact["tile_mean_correlations"].push_back(tile_mean_correlations[i]);
        }
        core::write_text(run_dir / "artifacts" / "tile_reconstruction_tlr.json", artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::TILE_RECONSTRUCTION_TLR, "ok",
                      {
                          {"output", (run_dir / "outputs" / "reconstructed_L.fit").string()},
                          {"valid_tiles", std::count_if(tile_valid_counts.begin(), tile_valid_counts.end(),
                                                        [&](int c) { return c >= min_valid_frames; })},
                      },
                      log_file);

    // Phase 7: STATE_CLUSTERING (Methodik v4 §10)
    emitter.phase_start(run_id, Phase::STATE_CLUSTERING, "STATE_CLUSTERING", log_file);

    // Reduced Mode: optionally skip clustering/synthetic frames when N is small
    bool use_synthetic_frames = true;
    std::vector<int> cluster_labels(static_cast<size_t>(frames.size()), 0);
    int n_clusters = 1;
    if (skip_clustering_in_reduced) {
        use_synthetic_frames = false;
        emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "skipped",
                          {{"reason", "reduced_mode"},
                           {"frame_count", static_cast<int>(frames.size())},
                           {"frames_reduced_threshold", cfg.assumptions.frames_reduced_threshold}},
                          log_file);
    }

    if (!skip_clustering_in_reduced) {
        // Build state vectors for clustering: [G_f, mean_local_quality, var_local_quality, background, noise]
        const int n_frames_cluster = static_cast<int>(frames.size());
        std::vector<std::vector<float>> state_vectors(static_cast<size_t>(n_frames_cluster));

        std::vector<float> G_for_cluster(static_cast<size_t>(n_frames_cluster), 1.0f);

        for (size_t fi = 0; fi < frames.size(); ++fi) {
            float G_f = (fi < static_cast<size_t>(global_weights.size())) ? global_weights[static_cast<int>(fi)] : 1.0f;
            float bg = (fi < frame_metrics.size()) ? frame_metrics[fi].background : 0.0f;
            float noise = (fi < frame_metrics.size()) ? frame_metrics[fi].noise : 0.0f;

            // Compute mean/var of local tile quality for this frame
            float mean_local = 0.0f, var_local = 0.0f;
            if (fi < local_metrics.size() && !local_metrics[fi].empty()) {
                for (const auto& tm : local_metrics[fi]) {
                    mean_local += tm.quality_score;
                }
                mean_local /= static_cast<float>(local_metrics[fi].size());
                for (const auto& tm : local_metrics[fi]) {
                    float diff = tm.quality_score - mean_local;
                    var_local += diff * diff;
                }
                var_local /= static_cast<float>(local_metrics[fi].size());
            }

            state_vectors[fi] = {G_f, mean_local, var_local, bg, noise};
            G_for_cluster[fi] = G_f;
        }

        std::vector<std::vector<float>> X = state_vectors;
        if (n_frames_cluster > 0) {
            const size_t D = 5;
            std::vector<float> means(D, 0.0f);
            std::vector<float> stds(D, 0.0f);

            for (size_t d = 0; d < D; ++d) {
                double sum = 0.0;
                for (size_t i = 0; i < X.size(); ++i) sum += static_cast<double>(X[i][d]);
                means[d] = static_cast<float>(sum / static_cast<double>(X.size()));
                double var = 0.0;
                for (size_t i = 0; i < X.size(); ++i) {
                    double diff = static_cast<double>(X[i][d]) - static_cast<double>(means[d]);
                    var += diff * diff;
                }
                var /= std::max<double>(1.0, static_cast<double>(X.size()));
                stds[d] = static_cast<float>(std::sqrt(std::max(0.0, var)));
            }

            const float eps = 1.0e-12f;
            for (size_t i = 0; i < X.size(); ++i) {
                for (size_t d = 0; d < D; ++d) {
                    float sd = stds[d];
                    X[i][d] = (sd > eps) ? ((X[i][d] - means[d]) / sd) : 0.0f;
                }
            }
        }

        // Determine cluster count: K = clip(floor(N/10), K_min, K_max)
        int k_min = cfg.clustering.cluster_count_range[0];
        int k_max = cfg.clustering.cluster_count_range[1];
        int k_default = std::max(k_min, std::min(k_max, n_frames_cluster / 10));

        // Simple k-means clustering
        n_clusters = std::min(k_default, n_frames_cluster);

    std::string clustering_method = "kmeans";

    if (n_clusters > 1 && n_frames_cluster > 1) {
        // Initialize cluster centers (k-means++ style: just pick evenly spaced frames)
        std::vector<std::vector<float>> centers(static_cast<size_t>(n_clusters));
        for (int c = 0; c < n_clusters; ++c) {
            int idx = (c * n_frames_cluster) / n_clusters;
            centers[static_cast<size_t>(c)] = X[static_cast<size_t>(idx)];
        }

        // K-means iterations
        for (int iter = 0; iter < 20; ++iter) {
            // Assign labels
            for (size_t fi = 0; fi < X.size(); ++fi) {
                float best_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = 0.0f;
                    for (size_t d = 0; d < X[fi].size(); ++d) {
                        float diff = X[fi][d] - centers[static_cast<size_t>(c)][d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                cluster_labels[fi] = best_c;
            }

            // Update centers
            std::vector<std::vector<float>> new_centers(static_cast<size_t>(n_clusters),
                                                        std::vector<float>(5, 0.0f));
            std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
            for (size_t fi = 0; fi < X.size(); ++fi) {
                int c = cluster_labels[fi];
                for (size_t d = 0; d < X[fi].size(); ++d) {
                    new_centers[static_cast<size_t>(c)][d] += X[fi][d];
                }
                counts[static_cast<size_t>(c)]++;
            }
            for (int c = 0; c < n_clusters; ++c) {
                if (counts[static_cast<size_t>(c)] > 0) {
                    for (size_t d = 0; d < 5; ++d) {
                        new_centers[static_cast<size_t>(c)][d] /= static_cast<float>(counts[static_cast<size_t>(c)]);
                    }
                }
            }
            centers = new_centers;
        }
    }

    {
        std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
        for (int lbl : cluster_labels) {
            if (lbl >= 0 && lbl < n_clusters) counts[static_cast<size_t>(lbl)]++;
        }

        bool degenerate = false;
        for (int c = 0; c < n_clusters; ++c) {
            if (counts[static_cast<size_t>(c)] <= 0) {
                degenerate = true;
                break;
            }
        }

        if (degenerate && n_clusters > 1) {
            clustering_method = "quantile";
            std::vector<std::pair<float, int>> order;
            order.reserve(G_for_cluster.size());
            for (size_t i = 0; i < G_for_cluster.size(); ++i) {
                order.push_back({G_for_cluster[i], static_cast<int>(i)});
            }
            std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t r = 0; r < order.size(); ++r) {
                int label = static_cast<int>((r * static_cast<size_t>(n_clusters)) / std::max<size_t>(1, order.size()));
                if (label >= n_clusters) label = n_clusters - 1;
                cluster_labels[static_cast<size_t>(order[r].second)] = label;
            }
        }
    }

    {
        core::json artifact;
        artifact["n_clusters"] = n_clusters;
        artifact["k_min"] = k_min;
        artifact["k_max"] = k_max;
        artifact["method"] = clustering_method;
        artifact["cluster_labels"] = core::json::array();
        for (int lbl : cluster_labels) artifact["cluster_labels"].push_back(lbl);
        artifact["cluster_sizes"] = core::json::array();
        for (int c = 0; c < n_clusters; ++c) {
            int count = static_cast<int>(std::count(cluster_labels.begin(), cluster_labels.end(), c));
            artifact["cluster_sizes"].push_back(count);
        }
        core::write_text(run_dir / "artifacts" / "state_clustering.json", artifact.dump(2));
    }

        emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "ok",
                          {{"n_clusters", n_clusters}}, log_file);
    }

    // Phase 8: SYNTHETIC_FRAMES (Methodik v4 §11)
    emitter.phase_start(run_id, Phase::SYNTHETIC_FRAMES, "SYNTHETIC_FRAMES", log_file);

    std::vector<Matrix2Df> synthetic_frames;
    int synth_min = cfg.synthetic.frames_min;
    int synth_max = cfg.synthetic.frames_max;

    if (!use_synthetic_frames) {
        emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped",
                          {{"reason", "reduced_mode"},
                           {"frame_count", static_cast<int>(frames.size())},
                           {"frames_reduced_threshold", cfg.assumptions.frames_reduced_threshold}},
                          log_file);
    } else {
        for (int c = 0; c < n_clusters; ++c) {
            Matrix2Df sum = Matrix2Df::Zero(first_img.rows(), first_img.cols());
            double wsum = 0.0;
            int count = 0;

            for (size_t fi = 0; fi < frames.size(); ++fi) {
                if (cluster_labels[fi] != c) continue;
                if (fi >= static_cast<size_t>(global_weights.size())) continue;

                float w = global_weights[static_cast<int>(fi)];
                if (!(w > 0.0f) || !std::isfinite(w)) continue;

                auto [img, _hdr] = load_frame_normalized(fi);
                if (img.size() <= 0) continue;

                sum += img * w;
                wsum += static_cast<double>(w);
                count++;
            }

            if (count >= synth_min && wsum > 0.0) {
                synthetic_frames.push_back(sum / static_cast<float>(wsum));
            }

            if (static_cast<int>(synthetic_frames.size()) >= synth_max) break;
        }

        if (synthetic_frames.empty()) {
            // If there are not enough frames to satisfy frames_min, treat as a valid skip.
            if (static_cast<int>(frames.size()) < synth_min) {
                use_synthetic_frames = false;
                emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped",
                                  {{"reason", "insufficient_frames"},
                                   {"frame_count", static_cast<int>(frames.size())},
                                   {"frames_min", synth_min}},
                                  log_file);
            } else {
                emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "error",
                                  {{"error", "SYNTHETIC_FRAMES: no synthetic frames"}},
                                  log_file);
                emitter.run_end(run_id, false, "error", log_file);
                return 1;
            }
        }
    }

    if (use_synthetic_frames) {
        // Save synthetic frames
        for (size_t si = 0; si < synthetic_frames.size(); ++si) {
            std::string fname = "synthetic_" + std::to_string(si) + ".fit";
            Matrix2Df out = synthetic_frames[si];
            apply_output_scaling_inplace(out, 0, 0);
            io::write_fits_float(run_dir / "outputs" / fname, out, first_hdr);
        }

        {
            core::json artifact;
            artifact["num_synthetic"] = static_cast<int>(synthetic_frames.size());
            artifact["frames_min"] = synth_min;
            artifact["frames_max"] = synth_max;
            core::write_text(run_dir / "artifacts" / "synthetic_frames.json", artifact.dump(2));
        }

        emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "ok",
                          {{"num_synthetic", static_cast<int>(synthetic_frames.size())}}, log_file);
    }

    // Phase 9: STACKING (final overlap-add already done in Phase 6)
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);

    if (use_synthetic_frames) {
        recon = Matrix2Df::Zero(synthetic_frames[0].rows(), synthetic_frames[0].cols());
        for (const auto& sf : synthetic_frames) {
            recon += sf;
        }
        recon /= static_cast<float>(synthetic_frames.size());
    }

    Matrix2Df recon_out = recon;
    apply_output_scaling_inplace(recon_out, 0, 0);
    io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon_out, first_hdr);
    io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit", recon_out, first_hdr);

    emitter.phase_end(run_id, Phase::STACKING, "ok",
                      {{"note", use_synthetic_frames ? "overlap_add_done_in_phase6" : "reduced_mode_reuse_phase6"}},
                      log_file);

    {
        bool validation_ok = true;
        core::json v;

        float output_fwhm_med = 0.0f;
        {
            const int patch_r = 10;
            const int patch_sz = 2 * patch_r + 1;
            cv::Mat img_cv(recon.rows(), recon.cols(), CV_32F, const_cast<float*>(recon.data()));
            cv::Mat blur;
            cv::blur(img_cv, blur, cv::Size(31, 31), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
            cv::Mat resid = img_cv - blur;
            std::vector<cv::Point2f> corners;
            try {
                cv::goodFeaturesToTrack(resid, corners, 400, 0.01, 6);
            } catch (...) {
                corners.clear();
            }
            std::vector<float> fwhms;
            for (const auto& p : corners) {
                int cx = static_cast<int>(std::round(p.x));
                int cy = static_cast<int>(std::round(p.y));
                int x0 = cx - patch_r;
                int y0 = cy - patch_r;
                if (x0 < 0 || y0 < 0 || (x0 + patch_sz) > img_cv.cols || (y0 + patch_sz) > img_cv.rows) continue;
                cv::Mat patch = img_cv(cv::Rect(x0, y0, patch_sz, patch_sz));
                float f = estimate_fwhm_from_patch(patch);
                if (f > 0.0f && std::isfinite(f)) fwhms.push_back(f);
            }
            if (fwhms.size() >= 25) {
                output_fwhm_med = median_of(fwhms);
            }
        }

        float fwhm_improvement_percent = 0.0f;
        if (seeing_fwhm_med > 1.0e-6f && output_fwhm_med > 0.0f) {
            fwhm_improvement_percent = (seeing_fwhm_med - output_fwhm_med) / seeing_fwhm_med * 100.0f;
        }
        v["seeing_fwhm_median"] = seeing_fwhm_med;
        v["output_fwhm_median"] = output_fwhm_med;
        v["fwhm_improvement_percent"] = fwhm_improvement_percent;
        if (fwhm_improvement_percent < cfg.validation.min_fwhm_improvement_percent) {
            validation_ok = false;
            v["fwhm_improvement_ok"] = false;
        } else {
            v["fwhm_improvement_ok"] = true;
        }

        float tile_weight_variance = 0.0f;
        {
            std::vector<float> tile_means;
            tile_means.reserve(tiles_phase56.size());
            for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
                double sum = 0.0;
                int cnt = 0;
                for (size_t fi = 0; fi < frames.size(); ++fi) {
                    float G_f = (fi < static_cast<size_t>(global_weights.size())) ? global_weights[static_cast<int>(fi)] : 1.0f;
                    float L_ft = (fi < local_weights.size() && ti < local_weights[fi].size()) ? local_weights[fi][ti] : 1.0f;
                    sum += static_cast<double>(G_f * L_ft);
                    cnt++;
                }
                tile_means.push_back(cnt > 0 ? static_cast<float>(sum / static_cast<double>(cnt)) : 0.0f);
            }
            double mean = 0.0;
            for (float x : tile_means) mean += static_cast<double>(x);
            mean /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
            double var = 0.0;
            for (float x : tile_means) {
                double d = static_cast<double>(x) - mean;
                var += d * d;
            }
            var /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
            tile_weight_variance = static_cast<float>(var / (mean * mean + 1.0e-12));
        }
        v["tile_weight_variance"] = tile_weight_variance;
        if (tile_weight_variance < cfg.validation.min_tile_weight_variance) {
            validation_ok = false;
            v["tile_weight_variance_ok"] = false;
        } else {
            v["tile_weight_variance_ok"] = true;
        }

        bool tile_pattern_ok = true;
        if (cfg.validation.require_no_tile_pattern) {
            cv::Mat img_cv(recon.rows(), recon.cols(), CV_32F, const_cast<float*>(recon.data()));
            cv::Mat gx, gy;
            cv::Sobel(img_cv, gx, CV_32F, 1, 0, 3);
            cv::Sobel(img_cv, gy, CV_32F, 0, 1, 3);
            cv::Mat mag;
            cv::magnitude(gx, gy, mag);

            std::vector<int> xb;
            std::vector<int> yb;
            xb.reserve(tiles.size());
            yb.reserve(tiles.size());
            for (const auto& t : tiles) {
                if (t.x > 0) xb.push_back(t.x);
                if (t.y > 0) yb.push_back(t.y);
            }
            std::sort(xb.begin(), xb.end());
            xb.erase(std::unique(xb.begin(), xb.end()), xb.end());
            std::sort(yb.begin(), yb.end());
            yb.erase(std::unique(yb.begin(), yb.end()), yb.end());

            auto line_mean_x = [&](int x) -> float {
                if (x < 0 || x >= mag.cols) return 0.0f;
                double sum = 0.0;
                for (int y = 0; y < mag.rows; ++y) sum += static_cast<double>(mag.at<float>(y, x));
                return static_cast<float>(sum / static_cast<double>(mag.rows));
            };
            auto line_mean_y = [&](int y) -> float {
                if (y < 0 || y >= mag.rows) return 0.0f;
                double sum = 0.0;
                for (int x = 0; x < mag.cols; ++x) sum += static_cast<double>(mag.at<float>(y, x));
                return static_cast<float>(sum / static_cast<double>(mag.cols));
            };

            float worst_ratio = 1.0f;
            for (int x : xb) {
                float b = line_mean_x(x);
                float n = 0.5f * (line_mean_x(x - 2) + line_mean_x(x + 2));
                float r = b / (n + 1.0e-12f);
                if (r > worst_ratio) worst_ratio = r;
            }
            for (int y : yb) {
                float b = line_mean_y(y);
                float n = 0.5f * (line_mean_y(y - 2) + line_mean_y(y + 2));
                float r = b / (n + 1.0e-12f);
                if (r > worst_ratio) worst_ratio = r;
            }
            v["tile_pattern_ratio"] = worst_ratio;
            tile_pattern_ok = (worst_ratio < 1.5f);
            v["tile_pattern_ok"] = tile_pattern_ok;
            if (!tile_pattern_ok) validation_ok = false;
        }

        core::write_text(run_dir / "artifacts" / "validation.json", v.dump(2));

        // Do not abort here: we still want to run DEBAYER so GUI gets outputs.
        // We will mark the run as validation_failed at the end.
        if (!validation_ok) {
            run_validation_failed = true;
        }
    }

    // Phase 10: DEBAYER (for OSC data)
    emitter.phase_start(run_id, Phase::DEBAYER, "DEBAYER", log_file);

    if (detected_mode == ColorMode::OSC) {
        // Simple bilinear debayer
        int h = static_cast<int>(recon.rows());
        int w = static_cast<int>(recon.cols());

        Matrix2Df R = Matrix2Df::Zero(h, w);
        Matrix2Df G = Matrix2Df::Zero(h, w);
        Matrix2Df B = Matrix2Df::Zero(h, w);

        // Determine Bayer pattern offsets (default GBRG)
        int r_row = 1, r_col = 0;  // R at odd rows, even cols
        int b_row = 0, b_col = 1;  // B at even rows, odd cols

        if (detected_bayer == BayerPattern::RGGB) {
            r_row = 0; r_col = 0;
            b_row = 1; b_col = 1;
        } else if (detected_bayer == BayerPattern::BGGR) {
            r_row = 1; r_col = 1;
            b_row = 0; b_col = 0;
        } else if (detected_bayer == BayerPattern::GRBG) {
            r_row = 0; r_col = 1;
            b_row = 1; b_col = 0;
        }
        // GBRG is default

        // Simple nearest-neighbor debayer
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int y2 = y & ~1;
                int x2 = x & ~1;

                float r_val = recon(std::min(y2 + r_row, h - 1), std::min(x2 + r_col, w - 1));
                float b_val = recon(std::min(y2 + b_row, h - 1), std::min(x2 + b_col, w - 1));

                // Green is at two positions per 2x2 block
                float g_val;
                if ((y + x) % 2 == 0) {
                    // This pixel is green
                    g_val = recon(y, x);
                } else {
                    // Average neighboring greens
                    int gy1 = (y % 2 == r_row) ? y : y;
                    int gx1 = (x % 2 == r_col) ? x + 1 : x - 1;
                    int gy2 = (y % 2 == r_row) ? y + 1 : y - 1;
                    int gx2 = (x % 2 == r_col) ? x : x;
                    gx1 = std::max(0, std::min(w - 1, gx1));
                    gx2 = std::max(0, std::min(w - 1, gx2));
                    gy1 = std::max(0, std::min(h - 1, gy1));
                    gy2 = std::max(0, std::min(h - 1, gy2));
                    g_val = (recon(gy1, gx1) + recon(gy2, gx2)) * 0.5f;
                }

                R(y, x) = r_val;
                G(y, x) = g_val;
                B(y, x) = b_val;
            }
        }

        // Save individual RGB channels
        Matrix2Df R_out = R;
        Matrix2Df G_out = G;
        Matrix2Df B_out = B;
        R_out *= output_bg_r;
        G_out *= output_bg_g;
        B_out *= output_bg_b;
        R_out.array() += output_pedestal;
        G_out.array() += output_pedestal;
        B_out.array() += output_pedestal;
        io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", R_out, first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", G_out, first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", B_out, first_hdr);

        // Save stacked_rgb.fits as 3-plane RGB cube (NAXIS3=3)
        io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb.fits", R_out, G_out, B_out, first_hdr);

        emitter.phase_end(run_id, Phase::DEBAYER, "ok",
                          {{"mode", "OSC"}, {"bayer_pattern", bayer_pattern_to_string(detected_bayer)},
                           {"output_rgb", (run_dir / "outputs" / "stacked_rgb.fits").string()}}, log_file);
    } else {
        emitter.phase_end(run_id, Phase::DEBAYER, "ok", {{"mode", "MONO"}}, log_file);
    }

    // Phase 11: DONE
    emitter.phase_start(run_id, Phase::DONE, "DONE", log_file);
    emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_file);

    if (run_validation_failed) {
        emitter.run_end(run_id, false, "validation_failed", log_file);
        
        std::cout << "Pipeline completed with validation_failed" << std::endl;
        return 1;
    }

    emitter.run_end(run_id, true, "ok", log_file);
    
    std::cout << "Pipeline completed successfully" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
#ifdef HAVE_CLI11
    CLI::App app{"Tile-Compile Runner (C++)"};
    
    std::string config_path, input_dir, runs_dir, project_root;
    bool dry_run = false;
    int max_frames = 0;
    int max_tiles = 0;
    bool config_from_stdin = false;
    
    auto run_cmd = app.add_subcommand("run", "Run the pipeline");
    run_cmd->add_option("--config", config_path, "Path to config.yaml")->required();
    run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
    run_cmd->add_option("--runs-dir", runs_dir, "Runs directory")->required();
    run_cmd->add_option("--project-root", project_root, "Project root");
    run_cmd->add_option("--max-frames", max_frames, "Limit number of frames (0 = no limit)");
    run_cmd->add_option("--max-tiles", max_tiles, "Limit number of tiles in Phase 5/6 (0 = no limit)");
    run_cmd->add_flag("--dry-run", dry_run, "Dry run");
    run_cmd->add_flag("--stdin", config_from_stdin, "Read config YAML from stdin (use with --config -)");
    
    CLI11_PARSE(app, argc, argv);
    
    if (run_cmd->parsed()) {
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run, max_frames, max_tiles, config_from_stdin);
    }
    
    print_usage();
    return 1;
#else
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    std::string config_path, input_dir, runs_dir, project_root;
    bool dry_run = false;
    int max_frames = 0;
    int max_tiles = 0;
    bool config_from_stdin = false;
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--runs-dir" && i + 1 < argc) runs_dir = argv[++i];
        else if (arg == "--project-root" && i + 1 < argc) project_root = argv[++i];
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
        else if (arg == "--max-tiles" && i + 1 < argc) max_tiles = std::stoi(argv[++i]);
        else if (arg == "--dry-run") dry_run = true;
        else if (arg == "--stdin") config_from_stdin = true;
    }
    
    if (command == "run") {
        if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
            std::cerr << "Error: --config, --input-dir, and --runs-dir are required" << std::endl;
            return 1;
        }
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run, max_frames, max_tiles, config_from_stdin);
    }
    
    print_usage();
    return 1;
#endif
}
