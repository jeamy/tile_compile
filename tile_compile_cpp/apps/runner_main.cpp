#include "runner_pipeline.hpp"
#include "runner_resume.hpp"

#include <QCoreApplication>
#include <iostream>
#include <string>

#ifdef HAVE_CLI11
#include <CLI/CLI.hpp>
#endif

void print_usage() {
  std::cout << "Usage: tile_compile_runner <command> [options]\n\n"
            << "Commands:\n"
            << "  run      Run the pipeline\n"
            << "  resume   Resume a run from a specific phase\n"
            << "\nOptions:\n"
            << "  --config <path>       Path to config.yaml (run)\n"
            << "  --input-dir <path>    Input directory with FITS frames (run)\n"
            << "  --runs-dir <path>     Directory for run outputs (run)\n"
            << "  --project-root <path> Project root directory (run)\n"
            << "  --run-dir <path>      Existing run directory (resume)\n"
            << "  --from-phase <name>   Phase to resume from (resume; default: PCC)\n"
            << "  --max-frames <n>      Limit number of frames (0 = no limit)\n"
            << "  --max-tiles <n>       Limit number of tiles in Phase 5/6 (0 "
               "= no limit)\n"
            << "  --dry-run             Dry run (no actual processing)\n"
            << std::endl;
}

int run_command(const std::string &config_path, const std::string &input_dir,
                const std::string &runs_dir, const std::string &project_root,
                const std::string &run_id_override,
                bool dry_run, int max_frames, int max_tiles,
                bool config_from_stdin) {
  return run_pipeline_command(config_path, input_dir, runs_dir, project_root,
                              run_id_override,
                              dry_run, max_frames, max_tiles,
                              config_from_stdin);
}

int main(int argc, char *argv[]) {
  QCoreApplication qapp(argc, argv);  // needed for Qt6::Network event loop

#ifdef HAVE_CLI11
  CLI::App app{"Tile-Compile Runner (C++)"};

  std::string config_path, input_dir, runs_dir, project_root;
  std::string run_id_override;
  std::string resume_run_dir;
  std::string resume_from_phase = "PCC";
  bool dry_run = false;
  int max_frames = 0;
  int max_tiles = 0;
  bool config_from_stdin = false;

  auto run_cmd = app.add_subcommand("run", "Run the pipeline");
  run_cmd->add_option("--config", config_path, "Path to config.yaml")
      ->required();
  run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
  run_cmd->add_option("--runs-dir", runs_dir, "Runs directory")->required();
  run_cmd->add_option("--project-root", project_root, "Project root");
  run_cmd->add_option("--run-id", run_id_override,
                      "Optional run-id override (group related runs)");
  run_cmd->add_option("--max-frames", max_frames,
                      "Limit number of frames (0 = no limit)");
  run_cmd->add_option("--max-tiles", max_tiles,
                      "Limit number of tiles in Phase 5/6 (0 = no limit)");
  run_cmd->add_flag("--dry-run", dry_run, "Dry run");
  run_cmd->add_flag("--stdin", config_from_stdin,
                    "Read config YAML from stdin (use with --config -)");

  auto resume_cmd = app.add_subcommand("resume", "Resume an existing run (ASTROMETRY/BGE/PCC)");
  resume_cmd->add_option("--run-dir", resume_run_dir, "Existing run directory")
      ->required();
  resume_cmd->add_option("--from-phase", resume_from_phase,
                         "Phase to resume from: ASTROMETRY|BGE|PCC")
      ->default_val("PCC");

  CLI11_PARSE(app, argc, argv);

  if (run_cmd->parsed()) {
    return run_command(config_path, input_dir, runs_dir, project_root,
                       run_id_override, dry_run,
                       max_frames, max_tiles, config_from_stdin);
  }

  if (resume_cmd->parsed()) {
    return resume_command(resume_run_dir, resume_from_phase);
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
  std::string run_id_override;
  std::string resume_run_dir;
  std::string resume_from_phase = "PCC";
  bool dry_run = false;
  int max_frames = 0;
  int max_tiles = 0;
  bool config_from_stdin = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      config_path = argv[++i];
    else if (arg == "--input-dir" && i + 1 < argc)
      input_dir = argv[++i];
    else if (arg == "--runs-dir" && i + 1 < argc)
      runs_dir = argv[++i];
    else if (arg == "--project-root" && i + 1 < argc)
      project_root = argv[++i];
    else if (arg == "--run-id" && i + 1 < argc)
      run_id_override = argv[++i];
    else if (arg == "--run-dir" && i + 1 < argc)
      resume_run_dir = argv[++i];
    else if (arg == "--from-phase" && i + 1 < argc)
      resume_from_phase = argv[++i];
    else if (arg == "--max-frames" && i + 1 < argc)
      max_frames = std::stoi(argv[++i]);
    else if (arg == "--max-tiles" && i + 1 < argc)
      max_tiles = std::stoi(argv[++i]);
    else if (arg == "--dry-run")
      dry_run = true;
    else if (arg == "--stdin")
      config_from_stdin = true;
  }

  if (command == "run") {
    if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
      std::cerr << "Error: --config, --input-dir, and --runs-dir are required"
                << std::endl;
      return 1;
    }
    return run_command(config_path, input_dir, runs_dir, project_root,
                       run_id_override, dry_run,
                       max_frames, max_tiles, config_from_stdin);
  }

  if (command == "resume") {
    if (resume_run_dir.empty()) {
      std::cerr << "Error: resume requires --run-dir <path>" << std::endl;
      return 1;
    }
    return resume_command(resume_run_dir, resume_from_phase);
  }

  print_usage();
  return 1;
#endif
}
