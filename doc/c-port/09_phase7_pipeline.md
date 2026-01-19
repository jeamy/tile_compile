# Phase 7: Pipeline-Integration

## Ziel

Integration aller Komponenten in die Hauptpipeline und CLI.

**Geschätzte Dauer**: 2-3 Wochen

---

## 7.1 Pipeline-Orchestrierung (pipeline/phases_impl.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"
#include <atomic>
#include <functional>
#include <ostream>
#include <vector>

namespace tile_compile::pipeline {

// Phase-Definitionen
enum class Phase {
    CALIBRATION = 0,
    REGISTRATION = 1,
    CHANNEL_SPLIT = 2,
    NORMALIZATION = 3,
    GLOBAL_METRICS = 4,
    TILE_GRID = 5,
    LOCAL_METRICS = 6,
    QUALITY_INDICES = 7,
    CLUSTERING = 8,
    SYNTHETIC_FRAMES = 9,
    RECONSTRUCTION = 10,
    OUTPUT = 11
};

constexpr int PHASE_COUNT = 12;

std::string phase_name(Phase phase);

// Pipeline-Kontext
struct PipelineContext {
    std::string run_id;
    fs::path run_dir;
    fs::path project_root;
    config::Config cfg;
    std::vector<fs::path> input_frames;
    
    // Laufzeit-Daten
    std::vector<fs::path> calibrated_frames;
    std::vector<fs::path> registered_frames;
    std::map<std::string, std::vector<Matrix2Df>> channels;
    std::map<std::string, metrics::ChannelMetrics> channel_metrics;
    std::map<std::string, clustering::ClusteringResult> clustering_results;
    std::map<std::string, std::vector<Matrix2Df>> synthetic_frames;
    std::map<std::string, Matrix2Df> reconstructed_channels;
    
    // Metadaten
    std::string color_mode;  // "OSC", "RGB", "MONO"
    BayerPattern bayer_pattern = BayerPattern::UNKNOWN;
    bool is_cfa = false;
    
    // Steuerung
    std::atomic<bool>* stop_flag = nullptr;
    std::ostream* log_stream = nullptr;
    bool dry_run = false;
};

// Phase-Ergebnis
struct PhaseResult {
    bool success;
    std::string status;  // "ok", "skipped", "error"
    std::string message;
    double elapsed_seconds;
    nlohmann::json extra_data;
};

// Phase-Handler-Typ
using PhaseHandler = std::function<PhaseResult(PipelineContext&)>;

// Pipeline-Runner
class PipelineRunner {
public:
    PipelineRunner();
    
    // Hauptausführung
    bool run(
        PipelineContext& ctx,
        std::optional<Phase> resume_from = std::nullopt
    );
    
    // Einzelne Phase ausführen
    PhaseResult run_phase(Phase phase, PipelineContext& ctx);
    
    // Stop anfordern
    void request_stop();
    bool stop_requested() const;

private:
    std::atomic<bool> stop_flag_{false};
    core::EventEmitter emitter_;
    
    // Phase-Handler
    std::map<Phase, PhaseHandler> handlers_;
    
    void register_handlers();
    
    // Phase-Implementierungen
    PhaseResult phase_calibration(PipelineContext& ctx);
    PhaseResult phase_registration(PipelineContext& ctx);
    PhaseResult phase_channel_split(PipelineContext& ctx);
    PhaseResult phase_normalization(PipelineContext& ctx);
    PhaseResult phase_global_metrics(PipelineContext& ctx);
    PhaseResult phase_tile_grid(PipelineContext& ctx);
    PhaseResult phase_local_metrics(PipelineContext& ctx);
    PhaseResult phase_quality_indices(PipelineContext& ctx);
    PhaseResult phase_clustering(PipelineContext& ctx);
    PhaseResult phase_synthetic_frames(PipelineContext& ctx);
    PhaseResult phase_reconstruction(PipelineContext& ctx);
    PhaseResult phase_output(PipelineContext& ctx);
};

} // namespace tile_compile::pipeline
```

### Implementierung (Auszug)

```cpp
#include "tile_compile/pipeline/phases_impl.hpp"
#include "tile_compile/io/fits_utils.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/registration/opencv_registration.hpp"
#include "tile_compile/calibration/calibration.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/clustering/clustering.hpp"
#include "tile_compile/synthetic/synthetic.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include <chrono>

namespace tile_compile::pipeline {

std::string phase_name(Phase phase) {
    static const std::map<Phase, std::string> names = {
        {Phase::CALIBRATION, "CALIBRATION"},
        {Phase::REGISTRATION, "REGISTRATION"},
        {Phase::CHANNEL_SPLIT, "CHANNEL_SPLIT"},
        {Phase::NORMALIZATION, "NORMALIZATION"},
        {Phase::GLOBAL_METRICS, "GLOBAL_METRICS"},
        {Phase::TILE_GRID, "TILE_GRID"},
        {Phase::LOCAL_METRICS, "LOCAL_METRICS"},
        {Phase::QUALITY_INDICES, "QUALITY_INDICES"},
        {Phase::CLUSTERING, "CLUSTERING"},
        {Phase::SYNTHETIC_FRAMES, "SYNTHETIC_FRAMES"},
        {Phase::RECONSTRUCTION, "RECONSTRUCTION"},
        {Phase::OUTPUT, "OUTPUT"}
    };
    auto it = names.find(phase);
    return it != names.end() ? it->second : "UNKNOWN";
}

PipelineRunner::PipelineRunner() {
    register_handlers();
}

void PipelineRunner::register_handlers() {
    handlers_[Phase::CALIBRATION] = [this](auto& ctx) { return phase_calibration(ctx); };
    handlers_[Phase::REGISTRATION] = [this](auto& ctx) { return phase_registration(ctx); };
    handlers_[Phase::CHANNEL_SPLIT] = [this](auto& ctx) { return phase_channel_split(ctx); };
    handlers_[Phase::NORMALIZATION] = [this](auto& ctx) { return phase_normalization(ctx); };
    handlers_[Phase::GLOBAL_METRICS] = [this](auto& ctx) { return phase_global_metrics(ctx); };
    handlers_[Phase::TILE_GRID] = [this](auto& ctx) { return phase_tile_grid(ctx); };
    handlers_[Phase::LOCAL_METRICS] = [this](auto& ctx) { return phase_local_metrics(ctx); };
    handlers_[Phase::QUALITY_INDICES] = [this](auto& ctx) { return phase_quality_indices(ctx); };
    handlers_[Phase::CLUSTERING] = [this](auto& ctx) { return phase_clustering(ctx); };
    handlers_[Phase::SYNTHETIC_FRAMES] = [this](auto& ctx) { return phase_synthetic_frames(ctx); };
    handlers_[Phase::RECONSTRUCTION] = [this](auto& ctx) { return phase_reconstruction(ctx); };
    handlers_[Phase::OUTPUT] = [this](auto& ctx) { return phase_output(ctx); };
}

bool PipelineRunner::run(
    PipelineContext& ctx,
    std::optional<Phase> resume_from
) {
    int start_phase = resume_from ? static_cast<int>(*resume_from) : 0;
    
    for (int i = start_phase; i < PHASE_COUNT; ++i) {
        if (stop_requested()) {
            if (ctx.log_stream) {
                emitter_.phase_end(ctx.run_id, i, "stopped", *ctx.log_stream);
            }
            return false;
        }
        
        Phase phase = static_cast<Phase>(i);
        auto result = run_phase(phase, ctx);
        
        if (!result.success && result.status != "skipped") {
            return false;
        }
    }
    
    return true;
}

PhaseResult PipelineRunner::run_phase(Phase phase, PipelineContext& ctx) {
    int phase_num = static_cast<int>(phase);
    std::string name = phase_name(phase);
    
    if (ctx.log_stream) {
        emitter_.phase_start(ctx.run_id, phase_num, name, *ctx.log_stream);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    PhaseResult result;
    try {
        auto handler = handlers_.find(phase);
        if (handler != handlers_.end()) {
            result = handler->second(ctx);
        } else {
            result.success = false;
            result.status = "error";
            result.message = "No handler for phase " + name;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.status = "error";
        result.message = e.what();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    
    if (ctx.log_stream) {
        nlohmann::json extra = result.extra_data;
        extra["elapsed_seconds"] = result.elapsed_seconds;
        extra["message"] = result.message;
        emitter_.phase_end(ctx.run_id, phase_num, result.status, *ctx.log_stream, extra);
    }
    
    return result;
}

// Phase 0: Kalibrierung
PhaseResult PipelineRunner::phase_calibration(PipelineContext& ctx) {
    PhaseResult result;
    result.success = true;
    result.status = "ok";
    
    if (!ctx.cfg.calibration.enabled) {
        result.status = "skipped";
        result.message = "Calibration disabled";
        ctx.calibrated_frames = ctx.input_frames;
        return result;
    }
    
    // Master-Frames erstellen
    std::optional<calibration::MasterFrame> bias_master;
    std::optional<calibration::MasterFrame> dark_master;
    std::optional<calibration::MasterFrame> flat_master;
    
    if (ctx.cfg.calibration.bias_dir) {
        auto bias_files = core::discover_frames(*ctx.cfg.calibration.bias_dir, "*.fit*");
        bias_master = calibration::build_master_mean(bias_files);
    }
    
    if (ctx.cfg.calibration.dark_dir) {
        auto dark_files = core::discover_frames(*ctx.cfg.calibration.dark_dir, "*.fit*");
        dark_master = calibration::build_master_mean(dark_files);
        dark_master = calibration::bias_correct_dark(dark_master, bias_master);
    }
    
    if (ctx.cfg.calibration.flat_dir) {
        auto flat_files = core::discover_frames(*ctx.cfg.calibration.flat_dir, "*.fit*");
        flat_master = calibration::build_master_mean(flat_files);
        flat_master = calibration::prepare_flat(flat_master, bias_master, dark_master);
    }
    
    // Frames kalibrieren
    fs::path cal_dir = ctx.run_dir / "outputs" / "calibrated";
    fs::create_directories(cal_dir);
    
    for (const auto& frame_path : ctx.input_frames) {
        auto [data, header] = io::read_fits_float(frame_path);
        
        Matrix2Df calibrated = calibration::apply_calibration(
            data,
            bias_master ? &bias_master->data : nullptr,
            dark_master ? &dark_master->data : nullptr,
            flat_master ? &flat_master->data : nullptr
        );
        
        fs::path out_path = cal_dir / frame_path.filename();
        io::write_fits_float(out_path, calibrated, header);
        ctx.calibrated_frames.push_back(out_path);
    }
    
    result.message = "Calibrated " + std::to_string(ctx.calibrated_frames.size()) + " frames";
    return result;
}

// Phase 1: Registrierung
PhaseResult PipelineRunner::phase_registration(PipelineContext& ctx) {
    PhaseResult result;
    result.success = true;
    result.status = "ok";
    
    const auto& frames = ctx.calibrated_frames.empty() 
                         ? ctx.input_frames 
                         : ctx.calibrated_frames;
    
    if (frames.empty()) {
        result.success = false;
        result.status = "error";
        result.message = "No frames to register";
        return result;
    }
    
    // Referenz-Frame (erster Frame)
    auto [ref_data, ref_header] = io::read_fits_float(frames[0]);
    
    // CFA-Erkennung
    ctx.is_cfa = io::fits_is_cfa(ref_header);
    if (ctx.is_cfa) {
        ctx.bayer_pattern = io::fits_get_bayer_pattern(ref_header);
        ctx.color_mode = "OSC";
    } else if (ref_data.rows() > 0 && ref_data.cols() > 0) {
        // TODO: RGB vs MONO Erkennung
        ctx.color_mode = "MONO";
    }
    
    // Referenz-Bild für Registrierung vorbereiten
    Matrix2Df ref_for_reg;
    if (ctx.is_cfa) {
        ref_for_reg = image::cfa_downsample_sum2x2(ref_data);
    } else {
        ref_for_reg = ref_data;
    }
    Matrix2Df ref01 = registration::prepare_ecc_image(ref_for_reg);
    
    // Output-Verzeichnis
    fs::path reg_dir = ctx.run_dir / "outputs" / ctx.cfg.registration.output_dir;
    fs::create_directories(reg_dir);
    
    // Alle Frames registrieren
    for (size_t i = 0; i < frames.size(); ++i) {
        auto [data, header] = io::read_fits_float(frames[i]);
        
        Matrix2Df registered;
        
        if (i == 0) {
            // Referenz-Frame unverändert
            registered = data;
        } else {
            // Registrierung
            Matrix2Df moving_for_reg;
            if (ctx.is_cfa) {
                moving_for_reg = image::cfa_downsample_sum2x2(data);
            } else {
                moving_for_reg = data;
            }
            Matrix2Df moving01 = registration::prepare_ecc_image(moving_for_reg);
            
            // Initiale Warp finden
            AffineWarp init_warp = registration::best_translation_init(
                moving01, ref01,
                {ctx.cfg.registration.allow_rotation,
                 ctx.cfg.registration.rotation_range_deg,
                 ctx.cfg.registration.rotation_steps}
            );
            
            // ECC-Registrierung
            auto ecc_result = registration::ecc_warp(
                moving01, ref01,
                ctx.cfg.registration.allow_rotation,
                init_warp
            );
            
            if (!ecc_result.success) {
                // Fallback auf initiale Warp
            }
            
            // Warp anwenden
            if (ctx.is_cfa) {
                registered = image::warp_cfa_mosaic_via_subplanes(
                    data, ecc_result.warp
                );
            } else {
                registered = registration::apply_warp(data, ecc_result.warp);
            }
        }
        
        // Speichern
        fs::path out_path = reg_dir / frames[i].filename();
        io::write_fits_float(out_path, registered, header);
        ctx.registered_frames.push_back(out_path);
        
        // Progress
        if (ctx.log_stream) {
            float progress = static_cast<float>(i + 1) / frames.size();
            emitter_.phase_progress(ctx.run_id, 1, progress,
                                    "Registered " + std::to_string(i + 1) + "/" + 
                                    std::to_string(frames.size()),
                                    *ctx.log_stream);
        }
    }
    
    result.message = "Registered " + std::to_string(ctx.registered_frames.size()) + " frames";
    return result;
}

// Weitere Phasen analog implementieren...

} // namespace tile_compile::pipeline
```

---

## 7.2 CLI-Hauptprogramm (apps/tile_compile_runner.cpp)

```cpp
#include <CLI/CLI.hpp>
#include <iostream>
#include <fstream>

#include "tile_compile/core/utils.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/pipeline/phases_impl.hpp"

using namespace tile_compile;

int run_command(
    const std::string& config_path,
    const std::string& input_dir,
    const std::string& pattern,
    const std::string& runs_dir,
    const std::string& project_root_str,
    bool dry_run
) {
    // Pfade auflösen
    fs::path config = fs::absolute(config_path);
    fs::path input = fs::absolute(input_dir);
    
    fs::path project_root;
    if (!project_root_str.empty()) {
        project_root = fs::absolute(project_root_str);
    } else {
        project_root = core::resolve_project_root(config);
    }
    
    fs::path runs = project_root / runs_dir;
    
    // Konfiguration laden
    config::Config cfg;
    try {
        cfg = config::Config::load(config);
        cfg.validate();
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return 1;
    }
    
    // Frames entdecken
    auto frames = core::discover_frames(input, pattern);
    if (frames.empty()) {
        std::cerr << "No frames found in " << input << " matching " << pattern << std::endl;
        return 1;
    }
    
    // Run-ID und Verzeichnis
    std::string run_id = core::get_run_id();
    fs::path run_dir = runs / run_id;
    fs::create_directories(run_dir);
    
    // Config kopieren
    fs::copy_file(config, run_dir / "config.yaml");
    
// Log-Datei
    fs::path log_dir = run_dir / "logs";
    fs::create_directories(log_dir);
    std::ofstream log_file(log_dir / "run_events.jsonl");
    
    // Pipeline-Kontext
    pipeline::PipelineContext ctx;
    ctx.run_id = run_id;
    ctx.run_dir = run_dir;
    ctx.project_root = project_root;
    ctx.cfg = cfg;
    ctx.input_frames = frames;
    ctx.log_stream = &log_file;
    ctx.dry_run = dry_run;
    
    // Run-Start Event
    core::EventEmitter emitter;
    emitter.run_start(run_id, {
        {"config_path", config.string()},
        {"input_dir", input.string()},
        {"frames_discovered", frames.size()},
        {"dry_run", dry_run}
    }, log_file);
    
    // Pipeline ausführen
    pipeline::PipelineRunner runner;
    bool success = runner.run(ctx);
    
    // Run-End Event
    emitter.run_end(run_id, success, success ? "ok" : "error", log_file);
    
    return success ? 0 : 1;
}

int resume_command(
    const std::string& run_dir_str,
    int from_phase,
    const std::string& project_root_str
) {
    fs::path run_dir = fs::absolute(run_dir_str);
    
    if (!fs::exists(run_dir) || !fs::is_directory(run_dir)) {
        std::cerr << "Run directory not found: " << run_dir << std::endl;
        return 1;
    }
    
    // Config laden
    fs::path config_path = run_dir / "config.yaml";
    if (!fs::exists(config_path)) {
        std::cerr << "Config not found in run directory" << std::endl;
        return 1;
    }
    
    config::Config cfg = config::Config::load(config_path);
    
    // Run-ID aus Verzeichnisname
    std::string run_id = run_dir.filename().string();
    
    // Project root
    fs::path project_root;
    if (!project_root_str.empty()) {
        project_root = fs::absolute(project_root_str);
    } else {
        project_root = run_dir.parent_path().parent_path();
    }
    
    // Registrierte Frames laden
    fs::path reg_dir = run_dir / "outputs" / cfg.registration.output_dir;
    auto frames = core::discover_frames(reg_dir, "*.fit*");
    
    if (frames.empty()) {
        std::cerr << "No registered frames found" << std::endl;
        return 1;
    }
    
    // Log-Datei (append)
    std::ofstream log_file(run_dir / "logs" / "run_events.jsonl", std::ios::app);
    
    // Pipeline-Kontext
    pipeline::PipelineContext ctx;
    ctx.run_id = run_id;
    ctx.run_dir = run_dir;
    ctx.project_root = project_root;
    ctx.cfg = cfg;
    ctx.registered_frames = frames;
    ctx.log_stream = &log_file;
    
    // Pipeline fortsetzen
    pipeline::PipelineRunner runner;
    bool success = runner.run(ctx, static_cast<pipeline::Phase>(from_phase));
    
    return success ? 0 : 1;
}

int main(int argc, char** argv) {
    CLI::App app{"Tile-Compile Runner (C++)"};
    
    // run subcommand
    auto run_cmd = app.add_subcommand("run", "Run the pipeline");
    std::string config_path, input_dir, pattern = "*.fit*", runs_dir = "runs", project_root;
    bool dry_run = false;
    
    run_cmd->add_option("--config", config_path, "Path to config.yaml")->required();
    run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
    run_cmd->add_option("--pattern", pattern, "Input file pattern");
    run_cmd->add_option("--runs-dir", runs_dir, "Runs output directory");
    run_cmd->add_option("--project-root", project_root, "Project root directory");
    run_cmd->add_flag("--dry-run", dry_run, "Dry run mode");
    
    // resume subcommand
    auto resume_cmd = app.add_subcommand("resume", "Resume a run");
    std::string run_dir_str;
    int from_phase;
    std::string resume_project_root;
    
    resume_cmd->add_option("--run-dir", run_dir_str, "Run directory")->required();
    resume_cmd->add_option("--from-phase", from_phase, "Phase to resume from")->required();
    resume_cmd->add_option("--project-root", resume_project_root, "Project root");
    
    CLI11_PARSE(app, argc, argv);
    
    if (run_cmd->parsed()) {
        return run_command(config_path, input_dir, pattern, runs_dir, project_root, dry_run);
    }
    
    if (resume_cmd->parsed()) {
        return resume_command(run_dir_str, from_phase, resume_project_root);
    }
    
    std::cout << app.help() << std::endl;
    return 1;
}
```

---

## 7.3 GUI-Integration

Die Python-GUI kann das C++ Backend auf zwei Arten aufrufen:

### Option A: Subprocess (empfohlen)

```python
# gui/main.py - Anpassung

import subprocess
import json

def run_cpp_backend(config_path, input_dir, runs_dir):
    """Run C++ backend via subprocess."""
    cmd = [
        "./tile_compile_runner",  # oder vollständiger Pfad
        "run",
        "--config", str(config_path),
        "--input-dir", str(input_dir),
        "--runs-dir", str(runs_dir)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Events aus Log-Datei lesen
    # ...
    
    return process.wait() == 0
```

### Option B: Shared Library (optional)

```cpp
// tile_compile_cpp/src/bindings/c_api.cpp

extern "C" {

int tile_compile_run(
    const char* config_path,
    const char* input_dir,
    const char* runs_dir,
    void (*progress_callback)(int phase, float progress, const char* message)
) {
    // ... Implementierung ...
}

void tile_compile_stop() {
    // Stop-Flag setzen
}

}
```

```python
# Python-Binding mit ctypes
import ctypes

lib = ctypes.CDLL("./libtile_compile.so")

def run_backend(config_path, input_dir, runs_dir, progress_callback):
    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_float, ctypes.c_char_p)
    
    return lib.tile_compile_run(
        config_path.encode(),
        input_dir.encode(),
        runs_dir.encode(),
        CALLBACK(progress_callback)
    )
```

---

## Checkliste Phase 7

- [ ] phases_impl.hpp Header erstellt
- [ ] `PipelineContext` Struktur definiert
- [ ] `PipelineRunner` Klasse implementiert
- [ ] Alle 12 Phase-Handler implementiert
- [ ] tile_compile_runner.cpp CLI erstellt
- [ ] `run` Subcommand implementiert
- [ ] `resume` Subcommand implementiert
- [ ] Event-Logging funktioniert
- [ ] Progress-Reporting funktioniert
- [ ] Stop-Mechanismus funktioniert
- [ ] GUI-Integration dokumentiert
- [ ] End-to-End-Test durchgeführt
