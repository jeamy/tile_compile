# GUI-Integration

## Übersicht

Die Python-GUI (`gui/`) bleibt unverändert und ruft das C++ Backend als externes Executable auf. Dies ist die einfachste und robusteste Integrationsmethode.

---

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    PyQt6 GUI (Python)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  main.py                                             │   │
│  │  - Konfiguration bearbeiten                          │   │
│  │  - Run starten/stoppen                               │   │
│  │  - Progress anzeigen                                 │   │
│  │  - Ergebnisse visualisieren                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  subprocess.Popen("tile_compile_runner run ...")     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              C++ Backend (tile_compile_runner)              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Pipeline-Ausführung                                 │   │
│  │  - FITS I/O                                          │   │
│  │  - Registrierung                                     │   │
│  │  - Metriken                                          │   │
│  │  - Rekonstruktion                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  run_events.jsonl (Event-Log)                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              GUI liest Events und aktualisiert UI           │
└─────────────────────────────────────────────────────────────┘
```

---

## Änderungen in der GUI

### 1. Backend-Auswahl in constants.js

```javascript
// gui/constants.js

// Backend-Konfiguration
export const BACKEND_CONFIG = {
    // "python" oder "cpp"
    backend: "cpp",
    
    // Pfad zum C++ Executable (relativ zum Projekt-Root)
    cpp_executable: "tile_compile_cpp/build/tile_compile_runner",
    
    // Fallback auf Python wenn C++ nicht verfügbar
    fallback_to_python: true
};

// Bestehende API-Endpoints bleiben unverändert
export const API_ENDPOINTS = {
    // ...
};
```

### 2. Backend-Runner-Klasse

```python
# gui/backend_runner.py

import subprocess
import threading
import json
import os
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class BackendConfig:
    backend: str = "cpp"  # "python" oder "cpp"
    cpp_executable: str = "tile_compile_cpp/build/tile_compile_runner"
    fallback_to_python: bool = True

class BackendRunner:
    """Unified interface for running Python or C++ backend."""
    
    def __init__(self, config: BackendConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.process: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self._event_thread: Optional[threading.Thread] = None
    
    def run(
        self,
        config_path: Path,
        input_dir: Path,
        runs_dir: Path,
        on_event: Optional[Callable[[dict], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """Start pipeline execution."""
        self.stop_requested = False
        
        # Backend auswählen
        if self.config.backend == "cpp":
            success = self._run_cpp(config_path, input_dir, runs_dir, on_event)
            if not success and self.config.fallback_to_python:
                print("C++ backend failed, falling back to Python")
                success = self._run_python(config_path, input_dir, runs_dir, on_event)
        else:
            success = self._run_python(config_path, input_dir, runs_dir, on_event)
        
        if on_complete:
            on_complete(success)
        
        return success
    
    def _run_cpp(
        self,
        config_path: Path,
        input_dir: Path,
        runs_dir: Path,
        on_event: Optional[Callable[[dict], None]]
    ) -> bool:
        """Run C++ backend."""
        exe_path = self.project_root / self.config.cpp_executable
        
        if not exe_path.exists():
            print(f"C++ executable not found: {exe_path}")
            return False
        
        cmd = [
            str(exe_path),
            "run",
            "--config", str(config_path),
            "--input-dir", str(input_dir),
            "--runs-dir", str(runs_dir),
            "--project-root", str(self.project_root)
        ]
        
        return self._run_subprocess(cmd, runs_dir, on_event)
    
    def _run_python(
        self,
        config_path: Path,
        input_dir: Path,
        runs_dir: Path,
        on_event: Optional[Callable[[dict], None]]
    ) -> bool:
        """Run Python backend."""
        runner_path = self.project_root / "tile_compile_runner.py"
        
        cmd = [
            "python3",
            str(runner_path),
            "run",
            "--config", str(config_path),
            "--input-dir", str(input_dir),
            "--runs-dir", str(runs_dir),
            "--project-root", str(self.project_root)
        ]
        
        return self._run_subprocess(cmd, runs_dir, on_event)
    
    def _run_subprocess(
        self,
        cmd: list,
        runs_dir: Path,
        on_event: Optional[Callable[[dict], None]]
    ) -> bool:
        """Execute subprocess and monitor events."""
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Event-Monitoring in separatem Thread
            if on_event:
                self._event_thread = threading.Thread(
                    target=self._monitor_events,
                    args=(runs_dir, on_event)
                )
                self._event_thread.start()
            
            # Auf Prozess-Ende warten
            returncode = self.process.wait()
            
            if self._event_thread:
                self._event_thread.join(timeout=5.0)
            
            return returncode == 0
            
        except Exception as e:
            print(f"Error running backend: {e}")
            return False
    
    def _monitor_events(
        self,
        runs_dir: Path,
        on_event: Callable[[dict], None]
    ):
        """Monitor event log file for new events."""
        import time
        
        # Warten bis Run-Verzeichnis existiert
        timeout = 30
        start = time.time()
        while time.time() - start < timeout:
            run_dirs = sorted(runs_dir.glob("*"))
            if run_dirs:
                break
            time.sleep(0.1)
        
        if not run_dirs:
            return
        
        # Neuestes Run-Verzeichnis
        run_dir = run_dirs[-1]
        log_path = run_dir / "logs" / "run_events.jsonl"
        
        # Warten bis Log-Datei existiert
        while not log_path.exists() and not self.stop_requested:
            time.sleep(0.1)
        
        if not log_path.exists():
            return
        
        # Events lesen
        last_pos = 0
        while not self.stop_requested:
            try:
                with open(log_path, "r") as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                on_event(event)
                            except json.JSONDecodeError:
                                pass
                    last_pos = f.tell()
            except Exception:
                pass
            
            # Prüfen ob Prozess noch läuft
            if self.process and self.process.poll() is not None:
                # Letzte Events lesen
                time.sleep(0.5)
                try:
                    with open(log_path, "r") as f:
                        f.seek(last_pos)
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    event = json.loads(line)
                                    on_event(event)
                                except json.JSONDecodeError:
                                    pass
                except Exception:
                    pass
                break
            
            time.sleep(0.1)
    
    def stop(self):
        """Request stop of running pipeline."""
        self.stop_requested = True
        
        if self.process and self.process.poll() is None:
            # SIGTERM senden
            self.process.terminate()
            
            # Warten auf Beendigung
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # SIGKILL als letzter Ausweg
                self.process.kill()
```

### 3. Integration in main.py

```python
# gui/main.py - Relevante Änderungen

from backend_runner import BackendRunner, BackendConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ...
        
        # Backend-Runner initialisieren
        self.backend_config = BackendConfig(
            backend="cpp",  # oder aus Settings laden
            cpp_executable="tile_compile_cpp/build/tile_compile_runner",
            fallback_to_python=True
        )
        self.backend_runner = BackendRunner(
            self.backend_config,
            self.project_root
        )
    
    def start_run(self):
        """Start pipeline execution."""
        # Konfiguration speichern
        config_path = self.save_current_config()
        
        # Run starten
        self.run_thread = threading.Thread(
            target=self._run_pipeline,
            args=(config_path,)
        )
        self.run_thread.start()
    
    def _run_pipeline(self, config_path: Path):
        """Execute pipeline in background thread."""
        success = self.backend_runner.run(
            config_path=config_path,
            input_dir=self.input_dir,
            runs_dir=self.runs_dir,
            on_event=self._handle_event,
            on_complete=self._handle_complete
        )
    
    def _handle_event(self, event: dict):
        """Handle pipeline event."""
        event_type = event.get("type", "")
        
        if event_type == "phase_start":
            phase = event.get("phase", 0)
            name = event.get("name", "")
            self.update_status(f"Phase {phase}: {name}")
        
        elif event_type == "phase_progress":
            progress = event.get("progress", 0.0)
            message = event.get("message", "")
            self.update_progress(progress, message)
        
        elif event_type == "phase_end":
            status = event.get("status", "")
            if status == "error":
                self.show_error(event.get("message", "Unknown error"))
        
        elif event_type == "run_end":
            success = event.get("success", False)
            if success:
                self.show_success("Pipeline completed successfully")
            else:
                self.show_error("Pipeline failed")
    
    def _handle_complete(self, success: bool):
        """Handle pipeline completion."""
        self.enable_ui()
        if success:
            self.load_results()
    
    def stop_run(self):
        """Stop running pipeline."""
        self.backend_runner.stop()
```

---

## Event-Format

Das Event-Format ist identisch zwischen Python und C++ Backend:

```json
{"type": "run_start", "run_id": "20240115_143022_abc12345", "ts": "2024-01-15T14:30:22Z", ...}
{"type": "phase_start", "run_id": "...", "phase": 0, "name": "CALIBRATION", "ts": "..."}
{"type": "phase_progress", "run_id": "...", "phase": 0, "progress": 0.5, "message": "...", "ts": "..."}
{"type": "phase_end", "run_id": "...", "phase": 0, "status": "ok", "ts": "...", ...}
...
{"type": "run_end", "run_id": "...", "success": true, "status": "ok", "ts": "..."}
```

---

## Build-Integration

### CMake-Target für GUI-Integration

```cmake
# tile_compile_cpp/CMakeLists.txt

# Shared Library für optionale direkte Integration
add_library(tile_compile_shared SHARED
    src/bindings/c_api.cpp
)
target_link_libraries(tile_compile_shared PRIVATE tile_compile_lib)

# Install-Target
install(TARGETS tile_compile_runner tile_compile_shared
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
```

### Build-Script

```bash
#!/bin/bash
# build_cpp_backend.sh

set -e

cd tile_compile_cpp

# Build-Verzeichnis erstellen
mkdir -p build
cd build

# CMake konfigurieren
cmake .. -DCMAKE_BUILD_TYPE=Release

# Bauen
cmake --build . -j$(nproc)

echo "C++ backend built successfully"
echo "Executable: $(pwd)/tile_compile_runner"
```

---

## Zusammenfassung

1. **GUI bleibt in Python** - keine Änderungen an der GUI-Logik nötig
2. **C++ Backend als Subprocess** - einfachste Integration
3. **Event-basierte Kommunikation** - über JSONL-Log-Datei
4. **Fallback auf Python** - wenn C++ nicht verfügbar
5. **Identisches Output-Format** - volle Kompatibilität
