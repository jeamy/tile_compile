// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{Emitter, State};

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
) -> Result<StartRunResult, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "runner state poisoned".to_string())?;
    if guard.is_some() {
        return Err("run already in progress".to_string());
    }

    let mut cmd = Command::new("python3");
    cmd.current_dir(&working_dir)
        .arg("tile_compile_runner.py")
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(RunnerState {
            child: Arc::new(Mutex::new(None)),
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![start_run, stop_run])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
