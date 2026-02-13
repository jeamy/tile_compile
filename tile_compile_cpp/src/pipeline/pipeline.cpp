#include "tile_compile/core/types.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/registration/registration.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tile_compile::pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

// NOTE(deprecated): This pipeline is a legacy/minimal runner kept for
// compatibility. The normative Methodik-v3.2 implementation lives in
// apps/runner_main.cpp. New algorithmic changes must be applied there first.


static tile_compile::Matrix2Df downsample2x2_mean_local(const tile_compile::Matrix2Df& in) {
    const int h = in.rows();
    const int w = in.cols();
    const int h2 = h - (h % 2);
    const int w2 = w - (w % 2);
    const int out_h = std::max(1, h2 / 2);
    const int out_w = std::max(1, w2 / 2);
    tile_compile::Matrix2Df out(out_h, out_w);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const int sy = y * 2;
            const int sx = x * 2;
            const float a = in(sy, sx);
            const float b = in(sy, sx + 1);
            const float c = in(sy + 1, sx);
            const float d = in(sy + 1, sx + 1);
            out(y, x) = 0.25f * (a + b + c + d);
        }
    }
    return out;
}

static json warp_to_json(const tile_compile::WarpMatrix& w) {
    json j;
    j["a00"] = w(0,0);
    j["a01"] = w(0,1);
    j["tx"]  = w(0,2);
    j["a10"] = w(1,0);
    j["a11"] = w(1,1);
    j["ty"]  = w(1,2);
    return j;
}

static json frame_metrics_to_json(const tile_compile::FrameMetrics& m) {
    json j;
    j["background"] = m.background;
    j["noise"] = m.noise;
    j["gradient_energy"] = m.gradient_energy;
    j["quality_score"] = m.quality_score;
    return j;
}

class PipelineRunner {
public:
    bool run(const std::string& run_id,
             const fs::path& run_dir,
             const fs::path& project_root,
             const config::Config& cfg,
             const std::vector<fs::path>& input_frames,
             std::ostream& log_stream,
             std::atomic<bool>* stop_flag = nullptr) {

        log_stream << "[deprecated] src/pipeline/pipeline.cpp is legacy; "
                      "use apps/runner_main.cpp for normative v3.2 runs\n";

        (void)project_root;
        core::EventEmitter emitter;

        // --- SCAN_INPUT ---
        emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_stream);
        json scan_out;
        scan_out["n_frames"] = static_cast<int>(input_frames.size());
        scan_out["frames"] = json::array();
        for (const auto& p : input_frames) {
            scan_out["frames"].push_back(p.string());
        }
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", scan_out, log_stream);

        if (stop_flag && stop_flag->load()) {
            emitter.phase_end(run_id, Phase::DONE, "aborted", {}, log_stream);
            return false;
        }

        // --- REGISTRATION ---
        emitter.phase_start(run_id, Phase::REGISTRATION, "REGISTRATION", log_stream);

        try {
            fs::create_directories(run_dir / "artifacts");
            fs::create_directories(run_dir / "registered");

            // Load frames
            std::vector<Matrix2Df> frames;
            frames.reserve(input_frames.size());

            io::FitsHeader ref_header;
            ColorMode mode = ColorMode::MONO;
            BayerPattern bayer = BayerPattern::GBRG;

            for (size_t i = 0; i < input_frames.size(); ++i) {
                if (stop_flag && stop_flag->load()) {
                    throw std::runtime_error("stop requested");
                }
                auto [img, hdr] = io::read_fits_float(input_frames[i]);
                if (i == 0) {
                    ref_header = hdr;
                    mode = io::detect_color_mode(hdr, 2);
                    bayer = io::detect_bayer_pattern(hdr);
                }
                frames.push_back(std::move(img));
            }

            // Compute lightweight global metrics for reference selection (proxy frames).
            std::vector<FrameMetrics> frame_metrics;
            frame_metrics.reserve(frames.size());
            for (size_t i = 0; i < frames.size(); ++i) {
                tile_compile::Matrix2Df proxy = (mode == ColorMode::OSC)
                    ? tile_compile::image::cfa_green_proxy_downsample2x2(frames[i], tile_compile::bayer_pattern_to_string(bayer))
                    : downsample2x2_mean_local(frames[i]);
                frame_metrics.push_back(tile_compile::metrics::calculate_frame_metrics(proxy));
            }
            VectorXf global_weights = tile_compile::metrics::calculate_global_weights(
                frame_metrics,
                cfg.global_metrics.weights.background,
                cfg.global_metrics.weights.noise,
                cfg.global_metrics.weights.gradient,
                cfg.global_metrics.clamp[0],
                cfg.global_metrics.clamp[1]);

            // Write artifacts/global_metrics.json so later phases can reuse global metrics and weights.
            {
                json gm;
                gm["n_frames"] = static_cast<int>(frames.size());
                gm["color_mode"] = (mode == ColorMode::OSC) ? "OSC" : "MONO";
                gm["bayer_pattern"] = tile_compile::bayer_pattern_to_string(bayer);
                gm["weights"] = {
                    {"background", cfg.global_metrics.weights.background},
                    {"noise", cfg.global_metrics.weights.noise},
                    {"gradient", cfg.global_metrics.weights.gradient}
                };
                gm["clamp"] = {cfg.global_metrics.clamp[0], cfg.global_metrics.clamp[1]};
                gm["frames"] = json::array();
                for (size_t i = 0; i < frame_metrics.size(); ++i) {
                    json jf;
                    jf["index"] = static_cast<int>(i);
                    jf["input_path"] = input_frames[i].string();
                    jf["metrics"] = frame_metrics_to_json(frame_metrics[i]);
                    jf["global_weight"] = (i < static_cast<size_t>(global_weights.size())) ? global_weights(static_cast<int>(i)) : 0.0f;
                    gm["frames"].push_back(jf);
                }

                std::ofstream ofs(run_dir / "artifacts" / "global_metrics.json");
                ofs << gm.dump(2);
            }

            auto reg = tile_compile::registration::register_frames_to_reference(
                frames, mode, bayer, cfg.registration, &frame_metrics, &global_weights);

            // Apply warps and write registered frames
            for (size_t i = 0; i < frames.size(); ++i) {
                if (stop_flag && stop_flag->load()) {
                    throw std::runtime_error("stop requested");
                }

                Matrix2Df out_img;
                const auto& w = reg.warps_fullres[i];

                if (mode == ColorMode::OSC) {
                    out_img = tile_compile::image::warp_cfa_mosaic_via_subplanes(frames[i], w);
                } else {
                    out_img = tile_compile::registration::apply_warp(frames[i], w);
                }

                std::ostringstream name;
                name << "frame_" << std::setw(4) << std::setfill('0') << i << ".fits";
                io::write_fits_float(run_dir / "registered" / name.str(), out_img, ref_header);
            }

            // Write artifacts/global_registration.json
            json j;
            j["engine"] = reg.engine_used;
            j["ref_idx"] = reg.ref_idx;
            j["ref_selection_method"] = reg.ref_selection_method;
            j["ref_selection_value"] = reg.ref_selection_value;
            j["downsample_scale"] = reg.downsample_scale;
            j["allow_rotation"] = cfg.registration.allow_rotation;
            j["frames"] = json::array();

            for (size_t i = 0; i < reg.warps_fullres.size(); ++i) {
                json jf;
                jf["index"] = static_cast<int>(i);
                jf["input_path"] = input_frames[i].string();
                jf["success"] = reg.success[i];
                jf["score"] = reg.scores[i];
                jf["error"] = reg.errors[i];
                jf["warp"] = warp_to_json(reg.warps_fullres[i]);
                j["frames"].push_back(jf);
            }

            {
                std::ofstream ofs(run_dir / "artifacts" / "global_registration.json");
                ofs << j.dump(2);
            }

            emitter.phase_end(run_id, Phase::REGISTRATION, "ok", j, log_stream);

        } catch (const std::exception& e) {
            json err;
            err["error"] = e.what();
            emitter.phase_end(run_id, Phase::REGISTRATION, "error", err, log_stream);
            if (cfg.pipeline.abort_on_fail) {
                emitter.phase_start(run_id, Phase::DONE, "DONE", log_stream);
                emitter.phase_end(run_id, Phase::DONE, "error", err, log_stream);
                return false;
            }
        }

        emitter.phase_start(run_id, Phase::DONE, "DONE", log_stream);
        emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_stream);
        return true;
    }
};

} // namespace tile_compile::pipeline
