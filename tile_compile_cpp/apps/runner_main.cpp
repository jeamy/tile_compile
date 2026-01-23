#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/registration/registration.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef HAVE_CLI11
#include <CLI/CLI.hpp>
#endif

namespace fs = std::filesystem;

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
                bool dry_run, int max_frames, int max_tiles) {
    using namespace tile_compile;
    
    fs::path cfg_path(config_path);
    fs::path in_dir(input_dir);
    fs::path runs(runs_dir);
    fs::path proj_root = project_root.empty() ? 
        core::resolve_project_root(cfg_path) : fs::path(project_root);
    
    if (!fs::exists(cfg_path)) {
        std::cerr << "Error: Config file not found: " << config_path << std::endl;
        return 1;
    }
    
    if (!fs::exists(in_dir)) {
        std::cerr << "Error: Input directory not found: " << input_dir << std::endl;
        return 1;
    }
    
    config::Config cfg = config::Config::load(cfg_path);
    cfg.validate();
    
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
    
    core::copy_config(cfg_path, run_dir / "config.yaml");
    
    std::ofstream log_file(run_dir / "logs" / "run_events.jsonl");
    
    core::EventEmitter emitter;
    emitter.run_start(run_id, {
        {"config_path", config_path},
        {"input_dir", input_dir},
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
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "skipped", {{"reason", "dry_run"}}, log_file);

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

    try {
        std::tie(width, height, naxis) = io::get_fits_dimensions(frames.front());
        auto first = io::read_fits_float(frames.front());
        const auto& header = first.second;

        detected_mode = io::detect_color_mode(header, naxis);
        detected_bayer = io::detect_bayer_pattern(header);
    } catch (const std::exception& e) {
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error", {{"error", e.what()}}, log_file);
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

    emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok",
                      {
                          {"frames_scanned", frames.size()},
                          {"image_width", width},
                          {"image_height", height},
                          {"color_mode", detected_mode_str},
                          {"bayer_pattern", detected_bayer_str},
                      },
                      log_file);

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

    // Phase 2: NORMALIZATION (compute normalization factors; apply later during tile loading)
    emitter.phase_start(run_id, Phase::NORMALIZATION, "NORMALIZATION", log_file);

    std::vector<float> frame_medians;
    frame_medians.reserve(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& path = frames[i];

        float med = 0.0f;
        try {
            auto frame_pair = io::read_fits_float(path);
            med = core::compute_median(frame_pair.first);
        } catch (const std::exception& e) {
            emitter.warning(run_id, "Failed to load frame for normalization: " + path.filename().string() + ": " + e.what(), log_file);
            med = 0.0f;
        }

        frame_medians.push_back(med);

        const float progress = frames.empty() ? 1.0f : static_cast<float>(i + 1) / static_cast<float>(frames.size());
        emitter.phase_progress(run_id, Phase::NORMALIZATION, progress,
                               "median " + std::to_string(i + 1) + "/" + std::to_string(frames.size()),
                               log_file);
    }

    float target_median = 0.0f;
    {
        std::vector<float> positive;
        positive.reserve(frame_medians.size());
        for (float m : frame_medians) {
            if (m > 0.0f) positive.push_back(m);
        }
        if (!positive.empty()) {
            std::sort(positive.begin(), positive.end());
            size_t n = positive.size();
            target_median = (n % 2 == 0) ? (positive[n/2 - 1] + positive[n/2]) / 2.0f : positive[n/2];
        }
    }

    {
        core::json artifact;
        artifact["frame_medians"] = core::json::array();
        for (float m : frame_medians) artifact["frame_medians"].push_back(m);
        artifact["target_median"] = target_median;
        artifact["note"] = "normalization_applied_during_tile_loading";
        core::write_text(run_dir / "artifacts" / "normalization.json", artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::NORMALIZATION, "ok",
                      {
                          {"target_median", target_median},
                          {"note", "normalization_applied_during_tile_loading"},
                      },
                      log_file);

    // Phase 3: GLOBAL_METRICS
    emitter.phase_start(run_id, Phase::GLOBAL_METRICS, "GLOBAL_METRICS", log_file);

    std::vector<FrameMetrics> frame_metrics;
    frame_metrics.reserve(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& path = frames[i];
        try {
            int roi = std::min(1024, std::min(width, height));
            int x0 = std::max(0, (width - roi) / 2);
            int y0 = std::max(0, (height - roi) / 2);
            Matrix2Df roi_img = io::read_fits_region_float(path, x0, y0, roi, roi);
            if (roi_img.size() <= 0) {
                emitter.warning(run_id,
                                "GLOBAL_METRICS: empty ROI for " + path.filename().string(),
                                log_file);
                continue;
            }
            frame_metrics.push_back(metrics::calculate_frame_metrics(roi_img));
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

    bool adaptive_enabled = cfg.v4.adaptive_tiles.enabled;
    bool use_warp_probe = adaptive_enabled && cfg.v4.adaptive_tiles.use_warp_probe;
    bool use_hierarchical = adaptive_enabled && cfg.v4.adaptive_tiles.use_hierarchical;

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
        // Uniform tile size based on Methodik v3/v4 style: T0 = size_factor * FWHM, upper bound min(W,H)/max_divisor.
        const float assumed_fwhm = 3.0f;
        float t0 = static_cast<float>(cfg.tile.size_factor) * assumed_fwhm;
        int tmin = cfg.tile.min_size;
        int tmax = std::max(1, std::min(width, height) / std::max(1, cfg.tile.max_divisor));
        float tc = std::min(std::max(t0, static_cast<float>(tmin)), static_cast<float>(tmax));
        uniform_tile_size = static_cast<int>(std::floor(tc));

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
    auto load_frame_normalized = [&](const fs::path& path) -> std::pair<Matrix2Df, io::FitsHeader> {
        auto frame_pair = io::read_fits_float(path);
        Matrix2Df img = frame_pair.first;
        if (target_median > 0.0f) {
            float med = core::compute_median(img);
            if (med > 0.0f) {
                float scale = target_median / med;
                img *= scale;
            }
        }
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

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        auto [img, _hdr] = load_frame_normalized(frames[fi]);
        local_metrics[fi].reserve(tiles_phase56.size());

        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
            Matrix2Df tile_img = extract_tile(img, tiles_phase56[ti]);
            if (tile_img.size() <= 0) {
                local_metrics[fi].push_back(TileMetrics{0, 0, 0, 0, 0});
                continue;
            }
            local_metrics[fi].push_back(metrics::calculate_tile_metrics(tile_img));
        }

        const float p = frames.empty() ? 1.0f : static_cast<float>(fi + 1) / static_cast<float>(frames.size());
        emitter.phase_progress(run_id, Phase::LOCAL_METRICS, p,
                               "local_metrics " + std::to_string(fi + 1) + "/" + std::to_string(frames.size()),
                               log_file);
    }

    {
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
                    {"quality_score", m.quality_score},
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

    // Phase 6: TILE_RECONSTRUCTION_TLR (Methodik v4 compliant)
    // Features: iterative refinement, temporal smoothing, R_{f,t} weighting, variance window
    emitter.phase_start(run_id, Phase::TILE_RECONSTRUCTION_TLR, "TILE_RECONSTRUCTION_TLR", log_file);

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
    auto [first_img, first_hdr] = load_frame_normalized(frames[0]);
    Matrix2Df recon = Matrix2Df::Zero(first_img.rows(), first_img.cols());
    Matrix2Df weight_sum = Matrix2Df::Zero(first_img.rows(), first_img.cols());

    const int prev_cv_threads_recon = cv::getNumThreads();
    cv::setNumThreads(1);

    // Config parameters
    int min_valid_frames = cfg.registration.local_tiles.min_valid_frames;
    float ecc_cc_min = cfg.registration.local_tiles.ecc_cc_min;
    int iterations = cfg.v4.iterations;
    float beta = cfg.v4.beta;
    int temporal_smoothing_window = cfg.registration.local_tiles.temporal_smoothing_window;
    float max_warp_delta_px = cfg.registration.local_tiles.max_warp_delta_px;
    float variance_sigma = cfg.registration.local_tiles.variance_window_sigma;

    std::vector<int> tile_valid_counts(tiles_phase56.size(), 0);
    std::vector<float> tile_warp_variances(tiles_phase56.size(), 0.0f);
    std::vector<float> tile_mean_correlations(tiles_phase56.size(), 0.0f);

    for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        const Tile& t = tiles_phase56[ti];

        // Initial reference: median of all tiles (simplified: use first frame)
        Matrix2Df ref_tile;
        {
            std::vector<Matrix2Df> tile_samples;
            for (size_t fi = 0; fi < std::min(frames.size(), size_t(5)); ++fi) {
                auto [img, _hdr] = load_frame_normalized(frames[fi]);
                Matrix2Df sample = extract_tile(img, t);
                if (sample.size() > 0) tile_samples.push_back(sample);
            }
            if (tile_samples.empty()) continue;
            // Use first sample as initial reference
            ref_tile = tile_samples[0];
        }

        if (ref_tile.size() <= 0) continue;

        std::vector<float> hann_x = make_hann_1d(ref_tile.cols());
        std::vector<float> hann_y = make_hann_1d(ref_tile.rows());

        // Iterative refinement (Methodik v4 §5.2)
        std::vector<std::pair<float, float>> final_translations;
        std::vector<float> final_correlations;
        std::vector<Matrix2Df> final_warped_tiles;
        std::vector<float> final_weights;

        for (int iter = 0; iter < iterations; ++iter) {
            Matrix2Df ref_ecc = registration::prepare_ecc_image(ref_tile);

            std::vector<std::pair<float, float>> translations;
            std::vector<float> correlations;
            std::vector<Matrix2Df> warped_tiles;
            std::vector<float> weights;
            std::vector<size_t> frame_indices;

            int ecc_failures = 0;
            int tile_size_mismatches = 0;
            for (size_t fi = 0; fi < frames.size(); ++fi) {
                auto [img, _hdr] = load_frame_normalized(frames[fi]);
                Matrix2Df mov_tile = extract_tile(img, t);
                if (mov_tile.size() <= 0 || mov_tile.rows() != ref_tile.rows() || mov_tile.cols() != ref_tile.cols()) {
                    tile_size_mismatches++;
                    continue;
                }

                Matrix2Df mov_ecc = registration::prepare_ecc_image(mov_tile);

                auto [dx, dy] = registration::phasecorr_translation(mov_ecc, ref_ecc);
                float max_shift = 0.5f * static_cast<float>(std::min(mov_tile.rows(), mov_tile.cols()));
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

                translations.push_back({final_dx, final_dy});
                correlations.push_back(cc);
                frame_indices.push_back(fi);

                // Use final warp (ECC result or phase correlation fallback)
                WarpMatrix final_warp = registration::identity_warp();
                final_warp(0, 2) = final_dx;
                final_warp(1, 2) = final_dy;
                Matrix2Df mov_warped = registration::apply_warp(mov_tile, final_warp);
                warped_tiles.push_back(mov_warped);

                // W_{f,t} = G_f · R_{f,t} where R_{f,t} = exp(β·(cc-1))
                float G_f = (fi < static_cast<size_t>(global_weights.size()))
                                ? global_weights[static_cast<int>(fi)]
                                : 1.0f;
                float R_ft = std::exp(beta * (cc - 1.0f));
                weights.push_back(G_f * R_ft);
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
            if (wsum <= 0.0f) break;

            Matrix2Df new_ref = Matrix2Df::Zero(ref_tile.rows(), ref_tile.cols());
            for (size_t i = 0; i < warped_tiles.size(); ++i) {
                new_ref += warped_tiles[i] * (weights[i] / wsum);
            }
            ref_tile = new_ref;

            // Store for final iteration
            final_translations = translations;
            final_correlations = correlations;
            final_warped_tiles = warped_tiles;
            final_weights = weights;
        }

        int valid = static_cast<int>(final_warped_tiles.size());
        tile_valid_counts[ti] = valid;

        if (valid < std::max(1, min_valid_frames)) continue;

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
        if (wsum <= 0.0f) continue;

        Matrix2Df tile_rec = Matrix2Df::Zero(ref_tile.rows(), ref_tile.cols());
        for (size_t i = 0; i < final_warped_tiles.size(); ++i) {
            tile_rec += final_warped_tiles[i] * (final_weights[i] / wsum);
        }

        // Variance window weight ψ(var) (Methodik v4 §9)
        float psi = variance_window_weight(warp_var, variance_sigma);

        // Overlap-add into full image with variance-weighted Hanning window
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

        const float p = tiles_phase56.empty() ? 1.0f : static_cast<float>(ti + 1) / static_cast<float>(tiles_phase56.size());
        emitter.phase_progress(run_id, Phase::TILE_RECONSTRUCTION_TLR, p,
                               "tile_recon " + std::to_string(ti + 1) + "/" + std::to_string(tiles_phase56.size()),
                               log_file);
    }

    cv::setNumThreads(prev_cv_threads_recon);

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
    io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon, first_hdr);
    io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit", recon, first_hdr);

    {
        core::json artifact;
        artifact["num_frames"] = static_cast<int>(frames.size());
        artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
        artifact["iterations"] = iterations;
        artifact["beta"] = beta;
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

    // Build state vectors for clustering: [G_f, mean_local_quality, var_local_quality, background, noise]
    int n_frames_cluster = static_cast<int>(frames.size());
    std::vector<std::vector<float>> state_vectors(static_cast<size_t>(n_frames_cluster));

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
    }

    // Determine cluster count: K = clip(floor(N/10), K_min, K_max)
    int k_min = cfg.clustering.cluster_count_range[0];
    int k_max = cfg.clustering.cluster_count_range[1];
    int k_default = std::max(k_min, std::min(k_max, n_frames_cluster / 10));

    // Simple k-means clustering
    std::vector<int> cluster_labels(static_cast<size_t>(n_frames_cluster), 0);
    int n_clusters = std::min(k_default, n_frames_cluster);

    if (n_clusters > 1 && n_frames_cluster > 1) {
        // Initialize cluster centers (k-means++ style: just pick evenly spaced frames)
        std::vector<std::vector<float>> centers(static_cast<size_t>(n_clusters));
        for (int c = 0; c < n_clusters; ++c) {
            int idx = (c * n_frames_cluster) / n_clusters;
            centers[static_cast<size_t>(c)] = state_vectors[static_cast<size_t>(idx)];
        }

        // K-means iterations
        for (int iter = 0; iter < 20; ++iter) {
            // Assign labels
            for (size_t fi = 0; fi < state_vectors.size(); ++fi) {
                float best_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = 0.0f;
                    for (size_t d = 0; d < state_vectors[fi].size(); ++d) {
                        float diff = state_vectors[fi][d] - centers[static_cast<size_t>(c)][d];
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
            for (size_t fi = 0; fi < state_vectors.size(); ++fi) {
                int c = cluster_labels[fi];
                for (size_t d = 0; d < state_vectors[fi].size(); ++d) {
                    new_centers[static_cast<size_t>(c)][d] += state_vectors[fi][d];
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
        core::json artifact;
        artifact["n_clusters"] = n_clusters;
        artifact["k_min"] = k_min;
        artifact["k_max"] = k_max;
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

    // Phase 8: SYNTHETIC_FRAMES (Methodik v4 §11)
    emitter.phase_start(run_id, Phase::SYNTHETIC_FRAMES, "SYNTHETIC_FRAMES", log_file);

    std::vector<Matrix2Df> synthetic_frames;
    int synth_min = cfg.synthetic.frames_min;
    int synth_max = cfg.synthetic.frames_max;

    for (int c = 0; c < n_clusters; ++c) {
        // Count frames in this cluster
        int cluster_count = static_cast<int>(std::count(cluster_labels.begin(), cluster_labels.end(), c));
        if (cluster_count < synth_min) continue;

        // Weighted average of frames in cluster
        Matrix2Df synth = Matrix2Df::Zero(first_img.rows(), first_img.cols());
        float wsum = 0.0f;

        for (size_t fi = 0; fi < frames.size(); ++fi) {
            if (cluster_labels[fi] != c) continue;

            auto [img, _hdr] = load_frame_normalized(frames[fi]);
            float G_f = (fi < static_cast<size_t>(global_weights.size())) ? global_weights[static_cast<int>(fi)] : 1.0f;
            synth += img * G_f;
            wsum += G_f;
        }

        if (wsum > 0.0f) {
            synth /= wsum;
            synthetic_frames.push_back(synth);
        }

        if (static_cast<int>(synthetic_frames.size()) >= synth_max) break;
    }

    // Save synthetic frames
    for (size_t si = 0; si < synthetic_frames.size(); ++si) {
        std::string fname = "synthetic_" + std::to_string(si) + ".fit";
        io::write_fits_float(run_dir / "outputs" / fname, synthetic_frames[si], first_hdr);
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

    // Phase 9: STACKING (final overlap-add already done in Phase 6)
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);
    emitter.phase_end(run_id, Phase::STACKING, "ok",
                      {{"note", "overlap_add_done_in_phase6"}}, log_file);

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
        io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", R, first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", G, first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", B, first_hdr);

        // Save stacked_rgb.fits as 3-plane RGB cube (NAXIS3=3)
        io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb.fits", R, G, B, first_hdr);

        emitter.phase_end(run_id, Phase::DEBAYER, "ok",
                          {{"mode", "OSC"}, {"bayer_pattern", bayer_pattern_to_string(detected_bayer)},
                           {"output_rgb", (run_dir / "outputs" / "stacked_rgb.fits").string()}}, log_file);
    } else {
        emitter.phase_end(run_id, Phase::DEBAYER, "ok", {{"mode", "MONO"}}, log_file);
    }

    // Phase 11: DONE
    emitter.phase_start(run_id, Phase::DONE, "DONE", log_file);
    emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_file);

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
    
    auto run_cmd = app.add_subcommand("run", "Run the pipeline");
    run_cmd->add_option("--config", config_path, "Path to config.yaml")->required();
    run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
    run_cmd->add_option("--runs-dir", runs_dir, "Runs directory")->required();
    run_cmd->add_option("--project-root", project_root, "Project root");
    run_cmd->add_option("--max-frames", max_frames, "Limit number of frames (0 = no limit)");
    run_cmd->add_option("--max-tiles", max_tiles, "Limit number of tiles in Phase 5/6 (0 = no limit)");
    run_cmd->add_flag("--dry-run", dry_run, "Dry run");
    
    CLI11_PARSE(app, argc, argv);
    
    if (run_cmd->parsed()) {
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run, max_frames, max_tiles);
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
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--runs-dir" && i + 1 < argc) runs_dir = argv[++i];
        else if (arg == "--project-root" && i + 1 < argc) project_root = argv[++i];
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
        else if (arg == "--max-tiles" && i + 1 < argc) max_tiles = std::stoi(argv[++i]);
        else if (arg == "--dry-run") dry_run = true;
    }
    
    if (command == "run") {
        if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
            std::cerr << "Error: --config, --input-dir, and --runs-dir are required" << std::endl;
            return 1;
        }
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run, max_frames, max_tiles);
    }
    
    print_usage();
    return 1;
#endif
}
