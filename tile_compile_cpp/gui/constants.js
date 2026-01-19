export const GUI_CONSTANTS_JSON = `{
  "CLI": {
    "backend_bin": "../build/tile_compile_cli",
    "backend_fallback": null,
    "sub": {
      "LOAD_GUI_STATE": "load-gui-state",
      "SAVE_GUI_STATE": "save-gui-state",
      "GET_SCHEMA": "get-schema",
      "LOAD_CONFIG": "load-config",
      "SAVE_CONFIG": "save-config",
      "VALIDATE_CONFIG": "validate-config",
      "SCAN": "scan",
      "LIST_RUNS": "list-runs",
      "GET_RUN_STATUS": "get-run-status",
      "GET_RUN_LOGS": "get-run-logs",
      "LIST_ARTIFACTS": "list-artifacts",
      "RESUME_RUN": "resume-run"
    }
  },
  "RUNNER": {
    "executable": "../build/tile_compile_runner",
    "run_subcommand": "run"
  },
  "BACKEND": "cpp"
}`;
