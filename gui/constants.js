export const GUI_CONSTANTS_JSON = `{
  "CLI": {
    "backend_bin": "tile-compile-backend",
    "backend_fallback": "tile_compile_backend_cli.py",
    "sub": {
      "LOAD_GUI_STATE": "load-gui-state",
      "SAVE_GUI_STATE": "save-gui-state",
      "GET_SCHEMA": "get-schema",
      "LOAD_CONFIG": "load-config",
      "SAVE_CONFIG": "save-config",
      "VALIDATE_CONFIG": "validate-config",
      "VALIDATE_SIRIL_SCRIPTS": "validate-siril-scripts",
      "SCAN": "scan",
      "LIST_RUNS": "list-runs",
      "GET_RUN_STATUS": "get-run-status",
      "GET_RUN_LOGS": "get-run-logs",
      "LIST_ARTIFACTS": "list-artifacts"
    }
  },
  "RUNNER": {
    "python": "",
    "script": "tile_compile_runner.py",
    "run_subcommand": "run"
  }
}`;
