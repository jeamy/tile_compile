export const IPC = {
  START_RUN: "start_run",
  STOP_RUN: "stop_run",
  ABORT_RUN: "abort_run",
  SCAN_INPUT: "scan_input",
  GET_SCHEMA: "get_schema",
  LOAD_CONFIG: "load_config",
  SAVE_CONFIG: "save_config",
  VALIDATE_CONFIG: "validate_config",
  LIST_RUNS: "list_runs",
  GET_RUN_STATUS: "get_run_status",
  GET_RUN_LOGS: "get_run_logs",
  LIST_ARTIFACTS: "list_artifacts",
};

export const EVENTS = {
  RUNNER_LINE: "runner_line",
  RUNNER_STDERR: "runner_stderr",
  RUNNER_STARTED: "runner_started",
  RUNNER_EXIT: "runner_exit",
};
