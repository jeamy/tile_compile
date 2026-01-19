#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <iostream>
#include <fstream>
#include <string>

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
              << "  --dry-run             Dry run (no actual processing)\n"
              << std::endl;
}

int run_command(const std::string& config_path, const std::string& input_dir,
                const std::string& runs_dir, const std::string& project_root,
                bool dry_run) {
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
    
    if (dry_run) {
        std::cout << "Dry run - no processing" << std::endl;
        emitter.run_end(run_id, true, "ok", log_file);
        return 0;
    }
    
    emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", {{"frames", frames.size()}}, log_file);
    
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
    
    auto run_cmd = app.add_subcommand("run", "Run the pipeline");
    run_cmd->add_option("--config", config_path, "Path to config.yaml")->required();
    run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
    run_cmd->add_option("--runs-dir", runs_dir, "Runs directory")->required();
    run_cmd->add_option("--project-root", project_root, "Project root");
    run_cmd->add_flag("--dry-run", dry_run, "Dry run");
    
    CLI11_PARSE(app, argc, argv);
    
    if (run_cmd->parsed()) {
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run);
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
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--runs-dir" && i + 1 < argc) runs_dir = argv[++i];
        else if (arg == "--project-root" && i + 1 < argc) project_root = argv[++i];
        else if (arg == "--dry-run") dry_run = true;
    }
    
    if (command == "run") {
        if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
            std::cerr << "Error: --config, --input-dir, and --runs-dir are required" << std::endl;
            return 1;
        }
        return run_command(config_path, input_dir, runs_dir, project_root, dry_run);
    }
    
    print_usage();
    return 1;
#endif
}
