#include "runner_phase_preprocess_pipeline.hpp"
#include "runner_phase_metrics.hpp"
#include "runner_shared.hpp"

#include "tile_compile/core/events.hpp"
#include "tile_compile/core/utils.hpp"
#include <nlohmann/json.hpp>
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace {
inline std::string pname(preprocessing::Phase p) {
    return preprocessing::phase_to_string(p);
}
}

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace metrics = tile_compile::metrics;
namespace registration = tile_compile::registration;

namespace {

/// Convert preprocessing input_mode / detected NAXIS to ColorMode.
ColorMode resolve_color_mode(const preprocessing::Config& cfg,
                              const io::FitsHeader& hdr, int naxis) {
    const ColorMode detected = io::detect_color_mode(hdr, naxis);
    if (cfg.input_mode == "cfa_osc") return ColorMode::OSC;
    if (cfg.input_mode == "mono")    return ColorMode::MONO;
    // "auto": use header detection
    return detected;
}

/// Resolve Bayer pattern from config or header.
std::string resolve_bayer_pattern(const preprocessing::Config& cfg,
                                   const io::FitsHeader& hdr,
                                   ColorMode mode) {
    if (mode != ColorMode::OSC) return "";
    if (!cfg.bayer_pattern.empty() && cfg.bayer_pattern != "auto") {
        return cfg.bayer_pattern;
    }
    if (const auto bp = hdr.get_string("BAYERPAT")) return *bp;
    if (const auto bp = hdr.get_string("COLORTYP")) return *bp;
    return "RGGB"; // safe default
}

/// Select reference frame index. Strategy: "best_quality" picks the frame with
/// the highest quality_score; fall back to temporal center for ties / zero.
int select_reference_frame(const preprocessing::Config& cfg,
                            const std::vector<FrameMetrics>& fm,
                            std::string& strategy_out) {
    if (fm.empty()) { strategy_out = "temporal_center"; return 0; }

    if (cfg.registration_reference == "best_quality") {
        strategy_out = "best_quality";
        int best = 0;
        float best_q = fm[0].quality_score;
        for (int i = 1; i < static_cast<int>(fm.size()); ++i) {
            if (fm[i].quality_score > best_q) {
                best_q = fm[i].quality_score;
                best = i;
            }
        }
        return best;
    }

    // Fallback: temporal center
    strategy_out = "temporal_center";
    return static_cast<int>(fm.size() / 2);
}

/// Build a lightweight registration proxy from a normalized frame.
Matrix2Df make_proxy(const Matrix2Df& img, ColorMode mode,
                     const std::string& bayer) {
    return build_registration_proxy(img, mode, bayer);
}

// ---------------------------------------------------------------------------
// Calibration helpers (Bias / Dark / Flat)
// ---------------------------------------------------------------------------

constexpr float kFlatFloor = 1.0e-6f;

bool load_average_master(const std::vector<fs::path>& paths,
                         int expected_rows, int expected_cols,
                         Matrix2Df& out, std::string& err) {
    if (paths.empty()) { err = "no calibration frames"; return false; }
    Matrix2Df accum;
    size_t loaded = 0;
    for (const auto& p : paths) {
        Matrix2Df img;
        try { img = io::read_fits_pixels_float(p); }
        catch (const std::exception& e) {
            err = "cannot read '" + p.string() + "': " + e.what();
            return false;
        }
        if (img.rows() != expected_rows || img.cols() != expected_cols) {
            err = "dimension mismatch in '" + p.string() + "'";
            return false;
        }
        if (loaded == 0) accum = img; else accum += img;
        ++loaded;
    }
    out = accum / static_cast<float>(loaded);
    return true;
}

std::vector<std::string> split_patterns(const std::string& raw) {
    std::vector<std::string> patterns;
    std::string cur;
    for (char c : raw) {
        if (c == ';') {
            if (!cur.empty()) patterns.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) patterns.push_back(cur);
    if (patterns.empty()) patterns.push_back("*");
    return patterns;
}

std::vector<fs::path> discover_calib_frames(const fs::path& dir,
                                            const std::string& pattern) {
    std::vector<fs::path> frames;
    for (const auto& pat : split_patterns(pattern)) {
        auto part = core::discover_frames(dir, pat);
        frames.insert(frames.end(), part.begin(), part.end());
    }
    frames.erase(std::remove_if(frames.begin(), frames.end(),
                                [](const fs::path& p){ return !io::is_fits_image_path(p); }),
                 frames.end());
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    return frames;
}

Matrix2Df load_master_file(const fs::path& path,
                           int expected_rows,
                           int expected_cols,
                           std::string& err) {
    Matrix2Df master;
    try {
        master = io::read_fits_pixels_float(path);
    } catch (const std::exception& e) {
        err = "cannot read master '" + path.string() + "': " + e.what();
        return {};
    }
    if (master.rows() != expected_rows || master.cols() != expected_cols) {
        err = "dimension mismatch in master '" + path.string() + "'";
        return {};
    }
    return master;
}

struct PrepCalibMasters {
    Matrix2Df bias;
    Matrix2Df dark;
    Matrix2Df flat;
    std::vector<fs::path> dark_frames;
    bool have_bias = false;
    bool have_dark = false;
    bool have_flat = false;
    bool dark_auto_select = false;
};

std::optional<double> header_exposure(const io::FitsHeader& hdr) {
    if (auto v = hdr.get_double("EXPTIME")) return *v;
    if (auto v = hdr.get_double("EXPOSURE")) return *v;
    return std::nullopt;
}

std::optional<double> header_temperature(const io::FitsHeader& hdr) {
    if (auto v = hdr.get_double("CCD-TEMP")) return *v;
    if (auto v = hdr.get_double("SENSOR-T")) return *v;
    if (auto v = hdr.get_double("TEMP")) return *v;
    return std::nullopt;
}

bool build_calib_masters(const preprocessing::Config& cfg,
                         const fs::path& proj_root,
                         int rows, int cols,
                         PrepCalibMasters& out,
                         nlohmann::json& artifact,
                         std::string& err) {
    namespace json = nlohmann;
    auto resolve = [&](const std::string& raw) -> fs::path {
        if (raw.empty()) return {};
        fs::path p(raw);
        return p.is_absolute() ? p : proj_root / p;
    };

    const bool use_bias = cfg.calibration.use_bias || !cfg.bias_dir.empty() ||
                          !cfg.calibration.bias_master.empty();
    const bool use_dark = cfg.calibration.use_dark || !cfg.darks_dir.empty() ||
                          !cfg.calibration.dark_master.empty();
    const bool use_flat = cfg.calibration.use_flat || !cfg.flats_dir.empty() ||
                          !cfg.calibration.flat_master.empty();

    if (use_bias) {
        if (cfg.calibration.bias_use_master || !cfg.calibration.bias_master.empty()) {
            out.bias = load_master_file(resolve(cfg.calibration.bias_master), rows, cols, err);
            if (!err.empty()) return false;
            artifact["bias"] = {{"source", resolve(cfg.calibration.bias_master).string()},
                                {"master", true}};
        } else {
            const fs::path dir = resolve(cfg.bias_dir);
            auto frames = discover_calib_frames(dir, cfg.calibration.pattern);
            if (frames.empty()) { err = "bias_dir empty: " + dir.string(); return false; }
            if (!load_average_master(frames, rows, cols, out.bias, err)) return false;
            artifact["bias"] = {{
                {"source", dir.string()}, {"frames", static_cast<int>(frames.size())},
                {"master", false}
            }};
        }
        out.have_bias = true;
    }
    if (use_dark) {
        if (cfg.calibration.dark_use_master || !cfg.calibration.dark_master.empty()) {
            out.dark = load_master_file(resolve(cfg.calibration.dark_master), rows, cols, err);
            if (!err.empty()) return false;
            artifact["dark"] = {{"source", resolve(cfg.calibration.dark_master).string()},
                                {"master", true},
                                {"bias_corrected", false}};
        } else {
            const fs::path dir = resolve(cfg.darks_dir);
            auto frames = discover_calib_frames(dir, cfg.calibration.pattern);
            if (frames.empty()) { err = "darks_dir empty: " + dir.string(); return false; }
            if (!load_average_master(frames, rows, cols, out.dark, err)) return false;
            artifact["dark"] = {{
                {"source", dir.string()}, {"frames", static_cast<int>(frames.size())},
                {"master", false}, {"bias_corrected", out.have_bias}
            }};
            if (cfg.calibration.dark_auto_select) {
                out.dark_frames = frames;
                out.dark_auto_select = true;
                artifact["dark"]["auto_select"] = true;
            }
        }
        if (out.have_bias) out.dark -= out.bias;
        out.have_dark = true;
        artifact["dark"]["bias_corrected"] = out.have_bias;
    }
    if (use_flat) {
        if (cfg.calibration.flat_use_master || !cfg.calibration.flat_master.empty()) {
            out.flat = load_master_file(resolve(cfg.calibration.flat_master), rows, cols, err);
            if (!err.empty()) return false;
            artifact["flat"] = {{"source", resolve(cfg.calibration.flat_master).string()},
                                {"master", true}};
        } else {
            const fs::path dir = resolve(cfg.flats_dir);
            auto frames = discover_calib_frames(dir, cfg.calibration.pattern);
            if (frames.empty()) { err = "flats_dir empty: " + dir.string(); return false; }
            if (!load_average_master(frames, rows, cols, out.flat, err)) return false;
            artifact["flat"] = {{
                {"source", dir.string()}, {"frames", static_cast<int>(frames.size())},
                {"master", false}
            }};
        }
        // normalize flat to median
        std::vector<float> samples;
        samples.reserve(static_cast<size_t>(out.flat.size()));
        for (Eigen::Index i = 0; i < out.flat.size(); ++i) {
            const float v = out.flat.data()[i];
            if (std::isfinite(v) && v > kFlatFloor) samples.push_back(v);
        }
        if (samples.empty()) { err = "flat master has no valid samples"; return false; }
        std::sort(samples.begin(), samples.end());
        const float median = samples[samples.size() / 2];
        if (!std::isfinite(median) || median <= kFlatFloor) {
            err = "flat median invalid"; return false;
        }
        out.flat.array() /= median;
        out.have_flat = true;
        artifact["flat"]["normalization_median"] = median;
    }
    if (out.have_flat &&
        (!cfg.darkflats_dir.empty() || !cfg.calibration.darkflat_master.empty())) {
        // Dark-flats: subtract from flat master if both exist
        if (cfg.calibration.darkflat_use_master || !cfg.calibration.darkflat_master.empty()) {
            Matrix2Df darkflat_master =
                load_master_file(resolve(cfg.calibration.darkflat_master), rows, cols, err);
            if (!err.empty()) return false;
            const float mean = out.flat.mean();
            if (std::isfinite(mean) && std::fabs(mean) > kFlatFloor) {
                out.flat -= darkflat_master / mean;
                artifact["darkflat"] = {{"source", resolve(cfg.calibration.darkflat_master).string()},
                                        {"master", true}};
            }
        } else {
            const fs::path dir = resolve(cfg.darkflats_dir);
            auto frames = discover_calib_frames(dir, cfg.calibration.pattern);
            if (!frames.empty()) {
                Matrix2Df darkflat_master;
                if (load_average_master(frames, rows, cols, darkflat_master, err)) {
                    const float mean = out.flat.mean();
                    if (std::isfinite(mean) && std::fabs(mean) > kFlatFloor) {
                        out.flat -= darkflat_master / mean;
                    }
                    artifact["darkflat"] = {{
                        {"source", dir.string()}, {"frames", static_cast<int>(frames.size())},
                        {"master", false}
                    }};
                }
            }
        }
    }
    return true;
}

bool select_dark_for_light(const preprocessing::Config& cfg,
                           const PrepCalibMasters& cal,
                           const io::FitsHeader& light_hdr,
                           int rows,
                           int cols,
                           Matrix2Df& dark_out,
                           std::string& detail) {
    if (!cal.dark_auto_select || cal.dark_frames.empty()) {
        dark_out = cal.dark;
        return cal.have_dark;
    }
    const auto light_exp = header_exposure(light_hdr);
    const auto light_temp = header_temperature(light_hdr);
    if (!light_exp) {
        detail = "light exposure missing; using global dark master";
        dark_out = cal.dark;
        return cal.have_dark;
    }
    const double tol_frac =
        static_cast<double>(cfg.calibration.dark_match_exposure_tolerance_percent) / 100.0;
    std::vector<fs::path> selected;
    for (const auto& p : cal.dark_frames) {
        io::FitsHeader hdr;
        try { hdr = io::read_fits_header(p); } catch (...) { continue; }
        const auto dark_exp = header_exposure(hdr);
        if (!dark_exp) continue;
        const double exp_tol = std::max(1.0e-9, std::fabs(*light_exp) * tol_frac);
        if (std::fabs(*dark_exp - *light_exp) > exp_tol) continue;
        if (cfg.calibration.dark_match_use_temp) {
            const auto dark_temp = header_temperature(hdr);
            if (!light_temp || !dark_temp) continue;
            if (std::fabs(*dark_temp - *light_temp) >
                cfg.calibration.dark_match_temp_tolerance_c) {
                continue;
            }
        }
        selected.push_back(p);
    }
    if (selected.empty()) {
        detail = "no exposure/temp matched darks; using global dark master";
        dark_out = cal.dark;
        return cal.have_dark;
    }
    std::string err;
    if (!load_average_master(selected, rows, cols, dark_out, err)) {
        detail = err + "; using global dark master";
        dark_out = cal.dark;
        return cal.have_dark;
    }
    if (cal.have_bias) dark_out -= cal.bias;
    detail = "matched dark frames: " + std::to_string(selected.size());
    return true;
}

void apply_calib_to_frame(Matrix2Df& img,
                          const PrepCalibMasters& cal,
                          const Matrix2Df* selected_dark) {
    if (cal.have_bias) img -= cal.bias;
    if (selected_dark != nullptr) img -= *selected_dark;
    else if (cal.have_dark) img -= cal.dark;
    if (cal.have_flat) {
        for (Eigen::Index px = 0; px < img.size(); ++px) {
            const float d = cal.flat.data()[px];
            if (std::isfinite(d) && d > kFlatFloor)
                img.data()[px] /= d;
        }
    }
}

} // namespace

bool run_preprocess_pipeline(
    const std::string& run_id,
    const preprocessing::Config& cfg,
    const fs::path& run_dir,
    const fs::path& proj_root,
    core::EventEmitter& emitter,
    std::ostream& log_file,
    PreprocessPipelineContext& out)
{
    // -----------------------------------------------------------------------
    // Phase: INPUT_FRAMES – scan lights_dir, validate, read first header
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::INPUT_SCAN), log_file);

    const fs::path lights_dir =
        cfg.lights_dir.empty() ? fs::path{}
                               : (fs::path(cfg.lights_dir).is_absolute()
                                      ? fs::path(cfg.lights_dir)
                                      : proj_root / cfg.lights_dir);

    if (lights_dir.empty() || !fs::exists(lights_dir)) {
        emitter.phase_end(run_id, pname(preprocessing::Phase::INPUT_SCAN), "error",
                          {{"error", "lights_dir not found: " + lights_dir.string()}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return false;
    }

    auto frames = core::discover_frames(lights_dir, "*");
    frames.erase(std::remove_if(frames.begin(), frames.end(),
                                [](const fs::path& p) { return !io::is_fits_image_path(p); }),
                 frames.end());
    if (frames.empty()) {
        emitter.phase_end(run_id, pname(preprocessing::Phase::INPUT_SCAN), "error",
                          {{"error", "no supported FITS frames found in lights_dir"}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return false;
    }

    // Read first header for dimension + color mode detection
    int image_width = 0, image_height = 0;
    io::FitsHeader first_header;
    try {
        auto [w, h, nax] = io::get_fits_dimensions(frames.front());
        image_width  = w;
        image_height = h;
        first_header = io::read_fits_header(frames.front());
        (void)nax;
    } catch (const std::exception& e) {
        emitter.phase_end(run_id, pname(preprocessing::Phase::INPUT_SCAN), "error",
                          {{"error", std::string("failed to read first frame header: ") + e.what()}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return false;
    }

    int naxis_for_color = 2;
    try {
        auto [w2, h2, nax2] = io::get_fits_dimensions(frames.front());
        naxis_for_color = nax2; (void)w2; (void)h2;
    } catch (...) {}

    out.color_mode    = resolve_color_mode(cfg, first_header, naxis_for_color);
    out.bayer_pattern = resolve_bayer_pattern(cfg, first_header, out.color_mode);

    const std::string color_mode_str =
        (out.color_mode == ColorMode::OSC)  ? "OSC"  :
        (out.color_mode == ColorMode::RGB)  ? "RGB"  : "MONO";

    emitter.phase_end(run_id, pname(preprocessing::Phase::INPUT_SCAN), "ok",
                      {
                          {"frames_found",  static_cast<int>(frames.size())},
                          {"image_width",   image_width},
                          {"image_height",  image_height},
                          {"color_mode",    color_mode_str},
                          {"bayer_pattern", out.bayer_pattern},
                          {"lights_dir",    lights_dir.string()},
                      }, log_file);

    // -----------------------------------------------------------------------
    // Phase: CALIBRATION – optional bias/dark/flat; skip when unconfigured
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::CALIBRATION), log_file);

    const bool want_calibration = cfg.calibration.use_bias
                                 || cfg.calibration.use_dark
                                 || cfg.calibration.use_flat
                                 || !cfg.bias_dir.empty()
                                 || !cfg.darks_dir.empty()
                                 || !cfg.flats_dir.empty()
                                 || !cfg.darkflats_dir.empty()
                                 || !cfg.calibration.bias_master.empty()
                                 || !cfg.calibration.dark_master.empty()
                                 || !cfg.calibration.flat_master.empty()
                                 || !cfg.calibration.darkflat_master.empty();
    if (!want_calibration) {
        out.effective_frames     = frames;
        out.calibration_applied  = false;
        emitter.phase_end(run_id, pname(preprocessing::Phase::CALIBRATION), "skipped",
                          {{"reason", "no calibration dirs configured"}},
                          log_file);
    } else {
        // Build master frames (bias, dark, flat)
        PrepCalibMasters cal_masters;
        nlohmann::json cal_artifact;
        std::string cal_err;
        if (!build_calib_masters(cfg, proj_root, image_height, image_width,
                                 cal_masters, cal_artifact, cal_err)) {
            emitter.phase_end(run_id, pname(preprocessing::Phase::CALIBRATION), "error",
                              {{"error", cal_err}}, log_file);
            emitter.run_end(run_id, false, "error", log_file);
            return false;
        }

        // Apply calibration to each frame and write calibrated copy
        const fs::path calib_dir = run_dir / "outputs" / "calibrated";
        fs::create_directories(calib_dir);
        out.effective_frames.clear();
        out.effective_frames.reserve(frames.size());
        bool any_failed = false;
        for (size_t i = 0; i < frames.size(); ++i) {
            try {
                auto [img, hdr] = io::read_fits_float(frames[i]);
                if (img.rows() != image_height || img.cols() != image_width) {
                    emitter.warning(run_id,
                        "CALIBRATION: skipping frame " + frames[i].filename().string() +
                        " (dimension mismatch)", log_file);
                    out.effective_frames.push_back(frames[i]);
                    continue;
                }
                Matrix2Df selected_dark;
                Matrix2Df* selected_dark_ptr = nullptr;
                std::string dark_detail;
                if (select_dark_for_light(cfg, cal_masters, hdr, image_height, image_width,
                                          selected_dark, dark_detail)) {
                    selected_dark_ptr = &selected_dark;
                }
                apply_calib_to_frame(img, cal_masters, selected_dark_ptr);
                hdr.set("CALIBRAT", true);
                hdr.set("BIASCORR", cal_masters.have_bias);
                hdr.set("DARKCORR", selected_dark_ptr != nullptr || cal_masters.have_dark);
                hdr.set("FLATCORR", cal_masters.have_flat);
                if (!dark_detail.empty()) hdr.set("DARKSEL", dark_detail);
                std::ostringstream fname;
                fname << "cal_" << std::setfill('0') << std::setw(5) << (i + 1) << ".fit";
                const fs::path dst = calib_dir / fname.str();
                io::write_fits_float(dst, img, hdr);
                out.effective_frames.push_back(dst);
            } catch (const std::exception& e) {
                emitter.warning(run_id,
                    "CALIBRATION: failed for frame " + frames[i].filename().string() +
                    ": " + e.what(), log_file);
                out.effective_frames.push_back(frames[i]); // use uncalibrated fallback
                any_failed = true;
            }
        }
        out.calibration_applied = true;
        cal_artifact["frames_calibrated"] = static_cast<int>(frames.size());
        cal_artifact["calibrated_dir"] = calib_dir.string();
        if (any_failed) cal_artifact["some_frames_failed"] = true;
        emitter.phase_end(run_id, pname(preprocessing::Phase::CALIBRATION), "ok",
                          cal_artifact, log_file);
    }

    // -----------------------------------------------------------------------
    // Phase: CFA_CHANNEL_PREP – normalization via existing pipeline
    // Mono: single channel path, no Bayer assumptions.
    // OSC:  CFA-aware normalization, channel deferred to stacking.
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::CFA_CHANNEL_PREP), log_file);

    // Build a thin config::Config shim with the normalization defaults
    // so we can call run_phase_channel_split_normalization_global_metrics.
    config::Config shim_cfg;
    shim_cfg.normalization.enabled      = true;
    shim_cfg.normalization.mode         = cfg.stacking.normalization; // e.g. "addscale"
    shim_cfg.normalization.per_channel  = (out.color_mode == tile_compile::ColorMode::OSC);
    shim_cfg.global_metrics.weights.background = 0.25f;
    shim_cfg.global_metrics.weights.noise      = 0.50f;
    shim_cfg.global_metrics.weights.gradient   = 0.25f;
    shim_cfg.global_metrics.clamp[0]           = -3.0f;
    shim_cfg.global_metrics.clamp[1]           =  3.0f;
    shim_cfg.global_metrics.adaptive_weights   = false;
    shim_cfg.global_metrics.weight_exponent_scale = 1.0f;
    shim_cfg.runtime_limits.parallel_workers   = 0; // auto

    PhaseMetricsContext metrics_ctx;
    if (!run_phase_channel_split_normalization_global_metrics(
            run_id, shim_cfg, out.effective_frames, run_dir,
            out.color_mode, out.bayer_pattern,
            emitter, log_file, metrics_ctx)) {
        // error already emitted inside
        return false;
    }

    out.norm_scales    = std::move(metrics_ctx.norm_scales);
    out.frame_metrics  = std::move(metrics_ctx.frame_metrics);
    out.global_weights = std::move(metrics_ctx.global_weights);

    emitter.phase_end(run_id, pname(preprocessing::Phase::CFA_CHANNEL_PREP), "ok",
                      {
                          {"color_mode",    color_mode_str},
                          {"bayer_pattern", out.bayer_pattern},
                          {"frames",        static_cast<int>(out.effective_frames.size())},
                          {"normalization", cfg.stacking.normalization},
                      }, log_file);

    // -----------------------------------------------------------------------
    // Phase: REFERENCE_SELECTION – choose reference frame
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::REFERENCE_SELECTION), log_file);

    out.reference_frame_index = select_reference_frame(
        cfg, out.frame_metrics, out.reference_selection_strategy);

    emitter.phase_end(run_id, pname(preprocessing::Phase::REFERENCE_SELECTION), "ok",
                      {
                          {"strategy",        out.reference_selection_strategy},
                          {"reference_index", out.reference_frame_index},
                          {"reference_path",
                           out.effective_frames.empty() ? std::string{}
                             : out.effective_frames[static_cast<size_t>(out.reference_frame_index)].filename().string()},
                          {"quality_score",
                           out.frame_metrics.empty() ? 0.0f
                             : out.frame_metrics[static_cast<size_t>(out.reference_frame_index)].quality_score},
                      }, log_file);

    // -----------------------------------------------------------------------
    // Phase: REGISTRATION – reuse global registration
    // Tile-specific phases (TILE_GRID, TILE_RECONSTRUCTION, SYNTHETIC_FRAMES,
    // STATE_CLUSTERING) are explicitly NOT started here.
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::REGISTRATION), log_file);

    const size_t n_frames = out.effective_frames.size();
    out.frame_warps.assign(n_frames, registration::identity_warp());
    out.frame_cc.assign(n_frames, 0.0f);
    out.frame_has_data.assign(n_frames, 0);

    // Build registration proxy for reference frame
    auto load_proxy = [&](size_t fi) -> Matrix2Df {
        try {
            Matrix2Df img;
            if (metrics_ctx.frame_cache &&
                metrics_ctx.frame_cache->try_load_normalized(fi, img)) {
                return make_proxy(img, out.color_mode, out.bayer_pattern);
            }
            img = io::read_fits_pixels_float(out.effective_frames[fi]);
            if (img.size() > 0) {
                image::apply_normalization_inplace(
                    img, out.norm_scales[fi],
                    out.color_mode, out.bayer_pattern, 0, 0);
            }
            return make_proxy(img, out.color_mode, out.bayer_pattern);
        } catch (...) {
            return Matrix2Df{};
        }
    };

    Matrix2Df ref_proxy = load_proxy(static_cast<size_t>(out.reference_frame_index));
    out.frame_cc[static_cast<size_t>(out.reference_frame_index)]        = 1.0f;
    out.frame_has_data[static_cast<size_t>(out.reference_frame_index)]  = 1;

    // Simple registration config derived from preprocessing config
    config::RegistrationConfig reg_cfg;
    reg_cfg.engine             = "triangle_star_matching";
    reg_cfg.transform_model    = "affine";
    reg_cfg.allow_rotation     = true;
    reg_cfg.auto_engine        = false;

    std::atomic<size_t> reg_done{0};
    std::mutex reg_mutex;

    const int reg_workers = std::max(1, static_cast<int>(
        std::thread::hardware_concurrency() / 2));

    auto register_frame = [&](size_t fi) {
        if (static_cast<int>(fi) == out.reference_frame_index) return;
        Matrix2Df proxy = load_proxy(fi);
        if (proxy.size() == 0 || ref_proxy.size() == 0 ||
            proxy.rows() != ref_proxy.rows() ||
            proxy.cols() != ref_proxy.cols()) {
            const size_t done = reg_done.fetch_add(1) + 1;
            const float pct = static_cast<float>(done) /
                              static_cast<float>(n_frames);
            std::lock_guard<std::mutex> lk(reg_mutex);
            emitter.phase_progress(run_id, pname(preprocessing::Phase::REGISTRATION),
                                   pct, "frame " + std::to_string(fi) +
                                   " skipped (proxy unavailable)", log_file);
            return;
        }
        try {
            const auto sfr = registration::register_single_frame(
                proxy, ref_proxy, reg_cfg);
            {
                std::lock_guard<std::mutex> lk(reg_mutex);
                if (sfr.reg.success) {
                    // Scale warp from proxy resolution to full resolution
                    const float scale_x = static_cast<float>(image_width)  /
                                          static_cast<float>(proxy.cols());
                    const float scale_y = static_cast<float>(image_height) /
                                          static_cast<float>(proxy.rows());
                    WarpMatrix w = sfr.reg.warp;
                    w(0, 2) *= scale_x;
                    w(1, 2) *= scale_y;
                    out.frame_warps[fi]    = w;
                    out.frame_cc[fi]       = sfr.ncc_warped;
                    out.frame_has_data[fi] = 1;
                } else {
                    out.frame_cc[fi] = 0.0f;
                    emitter.warning(run_id,
                        "REGISTRATION: frame " + out.effective_frames[fi].filename().string() +
                        " registration failed", log_file);
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(reg_mutex);
            emitter.warning(run_id,
                "REGISTRATION: frame " + std::to_string(fi) + " threw: " + e.what(),
                log_file);
        }
        const size_t done = reg_done.fetch_add(1) + 1;
        const float pct = static_cast<float>(done) / static_cast<float>(n_frames);
        std::lock_guard<std::mutex> lk(reg_mutex);
        emitter.phase_progress(run_id, pname(preprocessing::Phase::REGISTRATION),
                               pct, "registered " + std::to_string(done) +
                               "/" + std::to_string(n_frames), log_file);
    };

    if (ref_proxy.size() > 0 && n_frames > 1) {
        std::atomic<size_t> next_fi{0};
        auto worker = [&]() {
            while (true) {
                const size_t fi = next_fi.fetch_add(1);
                if (fi >= n_frames) break;
                register_frame(fi);
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(reg_workers));
        for (int w = 0; w < reg_workers; ++w) workers.emplace_back(worker);
        for (auto& t : workers) if (t.joinable()) t.join();
    } else if (n_frames == 1) {
        out.frame_has_data[0] = 1;
    }

    int n_registered = 0;
    for (size_t fi = 0; fi < n_frames; ++fi) {
        if (out.frame_has_data[fi]) ++n_registered;
    }

    // Write registration artifact
    {
        core::json reg_art;
        reg_art["reference_index"]  = out.reference_frame_index;
        reg_art["n_frames"]         = static_cast<int>(n_frames);
        reg_art["n_registered"]     = n_registered;
        reg_art["engine"]           = reg_cfg.engine;
        reg_art["transform_model"]  = reg_cfg.transform_model;
        reg_art["frames"]           = core::json::array();
        for (size_t fi = 0; fi < n_frames; ++fi) {
            reg_art["frames"].push_back({
                {"index",     static_cast<int>(fi)},
                {"filename",  out.effective_frames[fi].filename().string()},
                {"cc",        out.frame_cc[fi]},
                {"has_data",  static_cast<bool>(out.frame_has_data[fi])},
            });
        }
        fs::create_directories(run_dir / "artifacts" / "preprocess");
        core::write_text(run_dir / "artifacts" / "preprocess" / "preprocessing_registration.json",
                         reg_art.dump(2));
    }

    if (n_registered == 0) {
        emitter.phase_end(run_id, pname(preprocessing::Phase::REGISTRATION), "error",
                          {{"error", "no frames could be registered"}}, log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return false;
    }

    // Prewarp accepted frames onto common canvas
    out.canvas_width  = image_width;
    out.canvas_height = image_height;
    out.prewarped_frames = DiskCacheFrameStore(
        run_dir / ".prewarped_cache", n_frames, image_height, image_width);

    std::atomic<size_t> prewarp_next{0};
    std::atomic<size_t> prewarp_done{0};
    std::mutex prewarp_mutex;
    auto prewarp_frame = [&](size_t fi) {
        if (!out.frame_has_data[fi]) return;
        try {
            Matrix2Df img;
            if (metrics_ctx.frame_cache &&
                metrics_ctx.frame_cache->try_load_normalized(fi, img)) {
                // already normalized
            } else {
                img = io::read_fits_pixels_float(out.effective_frames[fi]);
                if (img.size() > 0) {
                    image::apply_normalization_inplace(
                        img, out.norm_scales[fi],
                        out.color_mode, out.bayer_pattern, 0, 0);
                }
            }
            if (img.size() > 0) {
                if (cfg.stacking.per_frame_cosmetic_correction) {
                    if (out.color_mode == tile_compile::ColorMode::OSC) {
                        img = image::cosmetic_correction_cfa(
                            img, cfg.stacking.per_frame_cosmetic_correction_sigma,
                            true, 0, 0);
                    } else {
                        img = image::cosmetic_correction(
                            img, cfg.stacking.per_frame_cosmetic_correction_sigma,
                            true);
                    }
                }
                const WarpMatrix& W = out.frame_warps[fi];
                const bool is_identity =
                    (W - registration::identity_warp()).cwiseAbs().maxCoeff() < 1e-5f;
                if (is_identity) {
                    out.prewarped_frames.store(fi, img);
                } else {
                    Matrix2Df warped = image::apply_global_warp(
                        img, W, out.color_mode, image_height, image_width);
                    out.prewarped_frames.store(fi, warped);
                }
                out.frame_has_data[fi] = 1;
            }
        } catch (const std::exception& e) {
            out.frame_has_data[fi] = 0;
            std::lock_guard<std::mutex> lk(prewarp_mutex);
            emitter.warning(run_id,
                "REGISTRATION: prewarp failed for frame " +
                out.effective_frames[fi].filename().string() + ": " + e.what(),
                log_file);
        }
        const size_t done = prewarp_done.fetch_add(1) + 1;
        if (done % std::max<size_t>(1, n_registered / 20) == 0 || done == static_cast<size_t>(n_registered)) {
            std::lock_guard<std::mutex> lk(prewarp_mutex);
            emitter.phase_progress(run_id, pname(preprocessing::Phase::REGISTRATION),
                                   0.75f + 0.25f * static_cast<float>(done) /
                                   static_cast<float>(std::max(1, n_registered)),
                                   "prewarped " + std::to_string(done) + "/" +
                                   std::to_string(n_registered), log_file);
        }
    };
    auto prewarp_worker = [&]() {
        while (true) {
            const size_t fi = prewarp_next.fetch_add(1);
            if (fi >= n_frames) break;
            prewarp_frame(fi);
        }
    };
    std::vector<std::thread> prewarp_workers;
    prewarp_workers.reserve(static_cast<size_t>(reg_workers));
    for (int w = 0; w < reg_workers; ++w) prewarp_workers.emplace_back(prewarp_worker);
    for (auto& t : prewarp_workers) if (t.joinable()) t.join();

    emitter.phase_end(run_id, pname(preprocessing::Phase::REGISTRATION), "ok",
                      {
                          {"n_frames",     static_cast<int>(n_frames)},
                          {"n_registered", n_registered},
                          {"reference_index", out.reference_frame_index},
                          {"engine",       reg_cfg.engine},
                      }, log_file);

    return true;
}

} // namespace tile_compile::runner
