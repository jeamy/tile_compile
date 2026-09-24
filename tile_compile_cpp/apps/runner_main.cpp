#include "runner_forward_drizzle.hpp"
#include "runner_pipeline.hpp"
#include "runner_preprocess.hpp"
#include "tile_compile/core/build_info.hpp"

#include <iostream>
#include <string>

#ifdef HAVE_CLI11
#include <CLI/CLI.hpp>
#endif

/// @brief Implements print usage.
/// @details Part of the tile_compile_runner executable entry point and command dispatcher; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void print_usage() {
  std::cout << "Usage: tile_compile_runner <command> [options]\n\n"
            << "Commands:\n"
            << "  reconstruct Run the CFA forward-drizzle + multiband pipeline to the final reconstruction image\n"
            << "  resume-reconstruction Resume from GLOBAL_QUALITY, FORWARD_DRIZZLE,\n"
            << "                        ASTROMETRY, BGE, PCC or HYPERMETRIC_STRETCH\n"
            << "  preprocess Run the separate raw-preprocessing pipeline\n"
            << "\nOptions:\n"
            << "  --config <path>       Path to config.yaml (reconstruct)\n"
            << "  --input-dir <path>    Input directory with FITS frames (reconstruct)\n"
            << "  --runs-dir <path>     Directory for run outputs (reconstruct)\n"
            << "  --project-root <path> Project root directory (reconstruct)\n"
            << "  --run-dir <path>      Existing run directory (resume-reconstruction)\n"
            << "  --from-phase <name>   Phase to resume from (resume-reconstruction; supports\n"
            << "                        GLOBAL_QUALITY|FORWARD_DRIZZLE|ASTROMETRY|BGE|PCC|HYPERMETRIC_STRETCH)\n"
            << "  --max-frames <n>      Limit number of frames (0 = no limit)\n"
            << "  --dry-run             Dry run (no actual processing)\n"
            << "  --version            Print build/version information\n"
            << "  --json               Use JSON with --version\n"
            << std::endl;
}

/// @brief Implements main.
/// @details Part of the tile_compile_runner executable entry point and command dispatcher; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int main(int argc, char *argv[]) {
  bool version_requested = false;
  bool json_requested = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    version_requested = version_requested || arg == "--version";
    json_requested = json_requested || arg == "--json";
  }
  if (version_requested) {
    if (json_requested) {
      std::cout << tile_compile::core::build_info_json(true).dump(2)
                << std::endl;
    } else {
      std::cout << tile_compile::core::build_info_text() << std::endl;
    }
    return 0;
  }
#ifdef HAVE_CLI11
  CLI::App app{"Tile-Compile Runner (C++)"};

  std::string config_path, input_dir, runs_dir, project_root;
  std::string run_id_override;
  std::string resume_run_dir;
  std::string resume_from_phase = "PCC";
  std::string preprocess_config_path;
  std::string preprocess_runs_dir;
  std::string preprocess_project_root;
  std::string preprocess_run_id;
  bool dry_run = false;
  int max_frames = 0;
  bool config_from_stdin = false;
  bool preprocess_config_from_stdin = false;

  auto reconstruct_cmd = app.add_subcommand("reconstruct", "Run the CFA forward-drizzle + multiband pipeline to the final reconstruction image");
  reconstruct_cmd->add_option("--config", config_path)->required();
  reconstruct_cmd->add_option("--input-dir", input_dir)->required();
  reconstruct_cmd->add_option("--runs-dir", runs_dir)->required();
  reconstruct_cmd->add_option("--project-root", project_root);
  reconstruct_cmd->add_option("--run-id", run_id_override);
  reconstruct_cmd->add_option("--max-frames", max_frames);
  reconstruct_cmd->add_flag("--dry-run", dry_run);
  reconstruct_cmd->add_flag("--stdin", config_from_stdin);
  std::string reconstruction_resume_phase = "GLOBAL_QUALITY";
  auto reconstruction_resume = app.add_subcommand("resume-reconstruction", "Resume checked M1-M3 predecessors");
  reconstruction_resume->add_option("--run-dir", resume_run_dir)->required();
  reconstruction_resume->add_option("--from-phase", reconstruction_resume_phase);

  auto preprocess_cmd = app.add_subcommand("preprocess", "Run the separate raw-preprocessing pipeline");
  preprocess_cmd->add_option("--config", preprocess_config_path, "Path to preprocessing JSON config")
      ->required();
  preprocess_cmd->add_option("--runs-dir", preprocess_runs_dir, "Runs directory")->required();
  preprocess_cmd->add_option("--project-root", preprocess_project_root, "Project root");
  preprocess_cmd->add_option("--run-id", preprocess_run_id, "Optional run-id override");
  preprocess_cmd->add_flag("--stdin", preprocess_config_from_stdin,
                           "Read preprocessing JSON from stdin (use with --config -)");

  CLI11_PARSE(app, argc, argv);

  if (reconstruct_cmd->parsed())
    return run_pipeline_command(config_path, input_dir, runs_dir, project_root,
        run_id_override, dry_run, max_frames, config_from_stdin);
  if (reconstruction_resume->parsed())
    return resume_forward_drizzle_command(resume_run_dir, reconstruction_resume_phase);

  if (preprocess_cmd->parsed()) {
    return preprocess_command(preprocess_config_path, preprocess_runs_dir,
                              preprocess_project_root, preprocess_run_id,
                              preprocess_config_from_stdin);
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
  std::string resume_from_phase = "GLOBAL_QUALITY";
  std::string preprocess_config_path;
  std::string preprocess_runs_dir;
  std::string preprocess_project_root;
  std::string preprocess_run_id;
  bool dry_run = false;
  int max_frames = 0;
  bool config_from_stdin = false;
  bool preprocess_config_from_stdin = false;

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
    else if (arg == "--dry-run")
      dry_run = true;
    else if (arg == "--stdin")
      config_from_stdin = true;
  }

  if (command == "reconstruct") {
    if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
      std::cerr << "Error: --config, --input-dir, and --runs-dir are required"
                << std::endl;
      return 1;
    }
    return run_pipeline_command(config_path, input_dir, runs_dir, project_root,
                       run_id_override, dry_run,
                       max_frames, config_from_stdin);
  }

  if (command == "resume-reconstruction") {
    if (resume_run_dir.empty()) { std::cerr << "--run-dir required\n"; return 1; }
    return resume_forward_drizzle_command(resume_run_dir, resume_from_phase);
  }

  if (command == "preprocess") {
    preprocess_config_path = config_path;
    preprocess_runs_dir = runs_dir;
    preprocess_project_root = project_root;
    preprocess_run_id = run_id_override;
    preprocess_config_from_stdin = config_from_stdin;
    if (preprocess_config_path.empty() || preprocess_runs_dir.empty()) {
      std::cerr << "Error: preprocess requires --config and --runs-dir"
                << std::endl;
      return 1;
    }
    return preprocess_command(preprocess_config_path, preprocess_runs_dir,
                              preprocess_project_root, preprocess_run_id,
                              preprocess_config_from_stdin);
  }

  print_usage();
  return 1;
#endif
}
