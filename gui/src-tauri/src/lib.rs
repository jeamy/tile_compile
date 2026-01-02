// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use serde::Serialize;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{Emitter, State};

fn resolve_project_root(working_dir: &str) -> Result<PathBuf, String> {
    let wd = Path::new(working_dir);
    let candidates = [wd.to_path_buf(), wd.join("..")];
    for c in candidates {
        if c.join("tile_compile_backend_cli.py").exists() || c.join("tile_compile_runner.py").exists() {
            return Ok(c);
        }
    }
    Err(format!(
        "could not locate project root from working_dir={working_dir} (expected tile_compile_backend_cli.py)"
    ))
}

fn resolve_script(project_root: &Path, filename: &str) -> Result<PathBuf, String> {
    let p = project_root.join(filename);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("missing script {filename} under project root {}", project_root.display()))
    }
}

struct RunnerState {
    child: Arc<Mutex<Option<Child>>>,
}

#[derive(Serialize, Clone)]
struct RunnerLineEvent {
    line: String,
}

#[derive(Serialize, Clone)]
struct StartRunResult {
    pid: u32,
}

#[derive(Serialize, Clone)]
struct StopRunResult {
    stopped: bool,
}

#[derive(Serialize, Clone)]
struct AbortRunResult {
    requested: bool,
    forced: bool,
}

#[derive(Serialize, Clone)]
struct RunnerExitEvent {
    exit_code: Option<i32>,
}

#[tauri::command]
fn start_run(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    working_dir: String,
    config_path: String,
    input_dir: String,
    runs_dir: String,
    pattern: String,
    dry_run: bool,
    color_mode_confirmed: Option<String>,
) -> Result<StartRunResult, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "runner state poisoned".to_string())?;
    if guard.is_some() {
        return Err("run already in progress".to_string());
    }

    let project_root = resolve_project_root(&working_dir)?;
    let runner_script = resolve_script(&project_root, "tile_compile_runner.py")?;

    let mut cmd = Command::new("python3");
    cmd.current_dir(&project_root)
        .arg(runner_script)
        .arg("run")
        .arg("--config")
        .arg(config_path)
        .arg("--input-dir")
        .arg(input_dir)
        .arg("--runs-dir")
        .arg(runs_dir)
        .arg("--pattern")
        .arg(pattern)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(cm) = color_mode_confirmed {
        cmd.arg("--color-mode-confirmed").arg(cm);
    }

    if dry_run {
        cmd.arg("--dry-run");
    }

    let mut child = cmd.spawn().map_err(|e| format!("failed to start runner: {e}"))?;
    let pid = child.id();

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture runner stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture runner stderr".to_string())?;

    *guard = Some(child);

    let app_exit = app.clone();
    let state_exit = Arc::clone(&state.child);
    std::thread::spawn(move || loop {
        let exit_code: Option<Option<i32>> = {
            let mut g = match state_exit.lock() {
                Ok(v) => v,
                Err(_) => return,
            };

            let Some(child) = g.as_mut() else {
                return;
            };

            match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code();
                    *g = None;
                    Some(code)
                }
                Ok(None) => None,
                Err(_) => {
                    *g = None;
                    Some(None)
                }
            }
        };

        if let Some(exit_code) = exit_code {
            let _ = app_exit.emit("runner_exit", RunnerExitEvent { exit_code });
            return;
        }

        std::thread::sleep(Duration::from_millis(250));
    });

    let app_out = app.clone();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).ok().filter(|n| *n > 0).is_some() {
            let trimmed = line.trim_end_matches(['\n', '\r']);
            let _ = app_out.emit("runner_line", RunnerLineEvent {
                line: trimmed.to_string(),
            });
            line.clear();
        }
    });

    let app_err = app.clone();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).ok().filter(|n| *n > 0).is_some() {
            let trimmed = line.trim_end_matches(['\n', '\r']);
            let _ = app_err.emit("runner_stderr", RunnerLineEvent {
                line: trimmed.to_string(),
            });
            line.clear();
        }
    });

    let app_done = app.clone();
    std::thread::spawn(move || {
        let _ = app_done.emit("runner_started", StartRunResult { pid });
    });

    Ok(StartRunResult { pid })
}

#[tauri::command]
fn stop_run(state: State<'_, RunnerState>) -> Result<StopRunResult, String> {
    let mut guard = state.child.lock().map_err(|_| "runner state poisoned".to_string())?;
    let Some(mut child) = guard.take() else {
        return Ok(StopRunResult { stopped: false });
    };

    child.kill().map_err(|e| format!("failed to stop runner: {e}"))?;
    let _ = child.wait();
    Ok(StopRunResult { stopped: true })
}

#[tauri::command]
fn abort_run(state: State<'_, RunnerState>) -> Result<AbortRunResult, String> {
    // Graceful abort: SIGINT (so Python runner sets _STOP and emits run_stop_requested)
    // Fallback: kill if it doesn't exit quickly.
    let pid_opt: Option<u32> = {
        let guard = state
            .child
            .lock()
            .map_err(|_| "runner state poisoned".to_string())?;
        guard.as_ref().map(|c| c.id())
    };

    let Some(pid) = pid_opt else {
        return Ok(AbortRunResult {
            requested: false,
            forced: false,
        });
    };

    #[cfg(unix)]
    unsafe {
        // Best effort; ignore error, fallback will handle.
        let _ = libc::kill(pid as i32, libc::SIGINT);
    }

    // Wait a short grace period.
    let grace_ms: u64 = 2500;
    let step_ms: u64 = 100;
    let mut waited: u64 = 0;

    loop {
        let done: bool = {
            let mut guard = state
                .child
                .lock()
                .map_err(|_| "runner state poisoned".to_string())?;

            match guard.as_mut() {
                None => true,
                Some(child) => match child.try_wait() {
                    Ok(Some(_)) => {
                        *guard = None;
                        true
                    }
                    Ok(None) => false,
                    Err(_) => {
                        *guard = None;
                        true
                    }
                },
            }
        };

        if done {
            return Ok(AbortRunResult {
                requested: true,
                forced: false,
            });
        }

        if waited >= grace_ms {
            break;
        }

        std::thread::sleep(Duration::from_millis(step_ms));
        waited += step_ms;
    }

    // Fallback hard kill
    let forced = {
        let mut guard = state
            .child
            .lock()
            .map_err(|_| "runner state poisoned".to_string())?;

        match guard.take() {
            None => false,
            Some(mut child) => {
                let _ = child.kill();
                let _ = child.wait();
                true
            }
        }
    };

    Ok(AbortRunResult {
        requested: true,
        forced,
    })
}

#[tauri::command]
fn scan_input(working_dir: String, input_path: String, frames_min: u32) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("scan")
        .arg(input_path)
        .arg("--frames-min")
        .arg(frames_min.to_string())
        .output()
        .map_err(|e| format!("failed to run scan: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let msg = if !stderr.trim().is_empty() {
            stderr
        } else if !stdout.trim().is_empty() {
            stdout
        } else {
            format!("scan failed with exit code {:?}", output.status.code())
        };
        return Err(msg);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse scan output as JSON: {e}"))
}

#[tauri::command]
fn get_schema(working_dir: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("get-schema")
        .output()
        .map_err(|e| format!("failed to run get-schema: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse schema output as JSON: {e}"))
}

#[tauri::command]
fn load_config(working_dir: String, path: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("load-config")
        .arg(path)
        .output()
        .map_err(|e| format!("failed to run load-config: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse load-config output as JSON: {e}"))
}

#[tauri::command]
fn save_config(working_dir: String, path: String, yaml_text: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let mut child = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("save-config")
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run save-config: {e}"))?;

    {
        use std::io::Write;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "failed to open stdin for save-config".to_string())?;
        stdin
            .write_all(yaml_text.as_bytes())
            .map_err(|e| format!("failed to write yaml to save-config stdin: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for save-config: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse save-config output as JSON: {e}"))
}

#[tauri::command]
fn validate_config(
    working_dir: String,
    yaml_text: String,
    schema_path: Option<String>,
) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let mut cmd = Command::new("python3");
    cmd.current_dir(&project_root)
        .arg(cli_script)
        .arg("validate-config")
        .arg("--yaml")
        .arg("x")
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(p) = schema_path {
        cmd.arg("--schema").arg(p);
    }

    let mut child = cmd.spawn().map_err(|e| format!("failed to run validate-config: {e}"))?;
    {
        use std::io::Write;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "failed to open stdin for validate-config".to_string())?;
        stdin
            .write_all(yaml_text.as_bytes())
            .map_err(|e| format!("failed to write yaml to validate-config stdin: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for validate-config: {e}"))?;

    // validate-config returns exit code 1 when invalid -> still parse JSON stdout
    if output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse validate-config output as JSON: {e}"))
}

#[tauri::command]
fn list_runs(working_dir: String, runs_dir: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("list-runs")
        .arg(runs_dir)
        .output()
        .map_err(|e| format!("failed to run list-runs: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse list-runs output as JSON: {e}"))
}

#[tauri::command]
fn get_run_logs(working_dir: String, run_dir: String, tail: Option<u32>) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let mut cmd = Command::new("python3");
    cmd.current_dir(&project_root)
        .arg(cli_script)
        .arg("get-run-logs")
        .arg(run_dir);

    if let Some(t) = tail {
        cmd.arg("--tail").arg(t.to_string());
    }

    let output = cmd.output().map_err(|e| format!("failed to run get-run-logs: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse get-run-logs output as JSON: {e}"))
}

#[tauri::command]
fn list_artifacts(working_dir: String, run_dir: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("list-artifacts")
        .arg(run_dir)
        .output()
        .map_err(|e| format!("failed to run list-artifacts: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse list-artifacts output as JSON: {e}"))
}

#[tauri::command]
fn get_run_status(working_dir: String, run_dir: String) -> Result<Value, String> {
    let project_root = resolve_project_root(&working_dir)?;
    let cli_script = resolve_script(&project_root, "tile_compile_backend_cli.py")?;
    let output = Command::new("python3")
        .current_dir(&project_root)
        .arg(cli_script)
        .arg("get-run-status")
        .arg(run_dir)
        .output()
        .map_err(|e| format!("failed to run get-run-status: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(stderr);
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|e| format!("failed to parse get-run-status output as JSON: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(RunnerState {
            child: Arc::new(Mutex::new(None)),
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_run,
            stop_run,
            abort_run,
            scan_input,
            get_schema,
            load_config,
            save_config,
            validate_config,
            list_runs,
            get_run_status,
            get_run_logs,
            list_artifacts
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
