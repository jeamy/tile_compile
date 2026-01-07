import json
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Dict

import yaml

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


METHODIK_V3_PHASES = [
    (0, "SCAN_INPUT", "Eingabe-Validierung"),
    (1, "REGISTRATION", "Frame-Registrierung"),
    (2, "CHANNEL_SPLIT", "Kanal-Trennung (R/G/B)"),
    (3, "NORMALIZATION", "Globale lineare Normalisierung"),
    (4, "GLOBAL_METRICS", "Globale Frame-Metriken (B, σ, E)"),
    (5, "TILE_GRID", "Seeing-adaptive Tile-Geometrie"),
    (6, "LOCAL_METRICS", "Lokale Tile-Metriken"),
    (7, "TILE_RECONSTRUCTION", "Tile-weise Rekonstruktion"),
    (8, "STATE_CLUSTERING", "Zustandsbasiertes Clustering"),
    (9, "SYNTHETIC_FRAMES", "Synthetische Qualitätsframes"),
    (10, "STACKING", "Finales lineares Stacking"),
    (11, "DONE", "Abschluss"),
]

ASSUMPTIONS_DEFAULTS = {
    "frames_min": 50,
    "frames_optimal": 800,
    "frames_reduced_threshold": 200,
    "exposure_time_tolerance_percent": 5.0,
    "registration_residual_warn_px": 0.5,
    "registration_residual_max_px": 1.0,
    "elongation_warn": 0.3,
    "elongation_max": 0.4,
    "tracking_error_max_px": 1.0,
    "reduced_mode_skip_clustering": True,
    "reduced_mode_cluster_range": [5, 10],
}


def _resolve_project_root(start: Path) -> Path:
    p = start
    if p.is_file():
        p = p.parent
    p = p.resolve()
    while True:
        if (p / "tile_compile_runner.py").exists() and (p / "tile_compile_backend_cli.py").exists():
            return p
        if p.parent == p:
            return start.resolve()
        p = p.parent


def _read_gui_constants(project_root: Path) -> dict:
    js_path = project_root / "gui" / "constants.js"
    raw = js_path.read_text(encoding="utf-8")
    key = "GUI_CONSTANTS_JSON"
    i = raw.find(key)
    if i < 0:
        raise RuntimeError("GUI_CONSTANTS_JSON not found in gui/constants.js")
    start = raw.find("`", i)
    end = raw.find("`", start + 1)
    if start < 0 or end < 0:
        raise RuntimeError("failed to parse gui/constants.js")
    payload = raw[start + 1 : end]
    return json.loads(payload)


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


class BackendClient:
    def __init__(self, project_root: Path, constants: dict):
        self.project_root = project_root
        self.constants = constants

    def _backend_cmd(self) -> list[str]:
        cli = self.constants.get("CLI", {})
        backend_bin = str(cli.get("backend_bin") or "").strip()
        backend_fallback = str(cli.get("backend_fallback") or "tile_compile_backend_cli.py").strip()

        if backend_bin and _which(backend_bin):
            return [backend_bin]
        script = str((self.project_root / backend_fallback).resolve())

        def can_import_backend_deps(py: str) -> bool:
            if not py or not _which(py):
                return False
            try:
                cp = subprocess.run(
                    [py, "-c", "import yaml; import tile_compile_backend"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                return cp.returncode == 0
            except Exception:
                return False

        candidates = [sys.executable, "python3", "python"]
        for py in candidates:
            if can_import_backend_deps(py):
                return [py, script]

        return [sys.executable, script]

    def run_json(self, cwd: Path, args: list[str], stdin_text: str | None = None, timeout_s: float = 30.0) -> Any:
        cmd = self._backend_cmd() + args
        cmd_str = " ".join(cmd)
        print(f"Running backend command: {cmd_str} in {cwd}")
        
        # Ensure the working directory exists
        if not cwd.exists():
            print(f"Working directory does not exist: {cwd}")
            raise RuntimeError(f"Working directory does not exist: {cwd}")
            
        try:
            # Log the exact command being executed
            print(f"Executing: {' '.join(cmd)} in {cwd}")
            if stdin_text:
                print(f"With stdin: {stdin_text[:100]}{'...' if len(stdin_text) > 100 else ''}")
                
            cp = subprocess.run(
                cmd,
                cwd=str(cwd),
                input=stdin_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
            
            # Log the command output
            print(f"Command returned with code {cp.returncode}")
            if cp.stdout:
                print(f"stdout: {cp.stdout[:200]}{'...' if len(cp.stdout) > 200 else ''}")
            if cp.stderr:
                print(f"stderr: {cp.stderr[:200]}{'...' if len(cp.stderr) > 200 else ''}")
                
            if cp.returncode != 0:
                err_msg = cp.stderr.strip() or cp.stdout.strip() or "backend command failed"
                print(f"Command failed with code {cp.returncode}: {err_msg}")
                raise RuntimeError(err_msg)
                
            try:
                result = json.loads(cp.stdout)
                print(f"Command succeeded with JSON result")
                return result
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {cp.stdout[:200]}...")
                print(f"JSON error: {str(e)}")
                raise RuntimeError(f"Failed to parse backend JSON output: {e}\nOutput: {cp.stdout[:100]}...")
            except Exception as e:
                print(f"Unexpected error parsing JSON: {str(e)}")
                raise RuntimeError(f"Error processing backend result: {e}")
                
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout_s}s: {cmd_str}")
            raise RuntimeError(f"Backend command timed out after {timeout_s}s")
        except FileNotFoundError as e:
            print(f"Command not found: {cmd[0]}")
            raise RuntimeError(f"Backend command not found: {cmd[0]}")
        except Exception as e:
            print(f"Command execution error: {e}")
            raise RuntimeError(f"Backend command execution error: {e}")


class RunnerProcess:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, cmd: list[str], cwd: Path) -> subprocess.Popen[str]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("runner already running")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return self._proc

    def stop(self) -> None:
        with self._lock:
            p = self._proc
        if p is None or p.poll() is not None:
            return
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            try:
                p.terminate()
            except Exception:
                return


class PhaseProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase_labels: Dict[int, QLabel] = {}
        self._phase_status_labels: Dict[int, QLabel] = {}
        self._reduced_mode = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.reduced_mode_label = QLabel("")
        self.reduced_mode_label.setObjectName("ReducedModeWarning")
        self.reduced_mode_label.setVisible(False)
        layout.addWidget(self.reduced_mode_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(METHODIK_V3_PHASES))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (phase_id, phase_name, phase_desc) in enumerate(METHODIK_V3_PHASES):
            name_label = QLabel(f"{phase_id}. {phase_name}")
            name_label.setToolTip(phase_desc)
            status_label = QLabel("pending")
            status_label.setObjectName("PhasePending")
            status_label.setMinimumWidth(120)
            status_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(name_label, i, 0)
            grid.addWidget(status_label, i, 1)
            self._phase_labels[phase_id] = name_label
            self._phase_status_labels[phase_id] = status_label

        layout.addLayout(grid)
        layout.addStretch(1)

    def set_reduced_mode(self, enabled: bool, frame_count: int = 0):
        self._reduced_mode = enabled
        if enabled:
            self.reduced_mode_label.setText(
                f"⚠ Reduced Mode aktiv ({frame_count} Frames < 200)\n"
                "STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen."
            )
            self.reduced_mode_label.setVisible(True)
        else:
            self.reduced_mode_label.setVisible(False)

    def update_phase(self, phase_name: str, status: str, progress_current: int = 0, progress_total: int = 0):
        phase_id = None
        for pid, pname, _ in METHODIK_V3_PHASES:
            if pname == phase_name:
                phase_id = pid
                break
        if phase_id is None or phase_id not in self._phase_status_labels:
            return
        label = self._phase_status_labels[phase_id]
        
        # Don't overwrite completed status (ok/error/skipped) with running
        current_text = label.text()
        if current_text in ("ok", "error", "skipped") and status.lower() == "running":
            return
        
        status_lower = status.lower()
        if status_lower == "running":
            if progress_total > 0 and progress_current > 0:
                percent = int(100 * progress_current / progress_total)
                label.setText(f"{progress_current}/{progress_total} ({percent}%)")
            else:
                label.setText("running")
            label.setObjectName("PhaseRunning")
        elif status_lower in ("ok", "success"):
            label.setText("ok")
            label.setObjectName("PhaseOk")
        elif status_lower == "error":
            label.setText("error")
            label.setObjectName("PhaseError")
        elif status_lower == "skipped":
            label.setText("skipped")
            label.setObjectName("PhaseSkipped")
        else:
            label.setText(status)
            label.setObjectName("PhasePending")
        label.setStyle(label.style())
        completed = sum(1 for lbl in self._phase_status_labels.values()
                       if lbl.text() in ("ok", "skipped", "error"))
        self.progress_bar.setValue(completed)

    def reset(self):
        for phase_id, label in self._phase_status_labels.items():
            label.setText("pending")
            label.setObjectName("PhasePending")
            label.setStyle(label.style())
        self.progress_bar.setValue(0)
        self.reduced_mode_label.setVisible(False)


class AssumptionsWidget(QWidget):
    assumptions_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hard_box = QGroupBox("Hard Assumptions (Verletzung → Abbruch)")
        hard_layout = QVBoxLayout(hard_box)
        hard_items = [
            "Lineare Daten (kein Stretch, keine nicht-linearen Operatoren)",
            "Keine Frame-Selektion (Pixel-Level Artefakt-Rejection erlaubt)",
            "Kanal-getrennte Verarbeitung (kein Channel Coupling)",
            "Strikt lineare Pipeline (keine Feedback-Loops)",
        ]
        for item in hard_items:
            lbl = QLabel(f"• {item}")
            lbl.setObjectName("HardAssumption")
            lbl.setWordWrap(True)
            hard_layout.addWidget(lbl)
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("• Einheitliche Belichtungszeit (Toleranz: ±"))
        self.exposure_tolerance = QDoubleSpinBox()
        self.exposure_tolerance.setRange(0.1, 20.0)
        self.exposure_tolerance.setValue(ASSUMPTIONS_DEFAULTS["exposure_time_tolerance_percent"])
        self.exposure_tolerance.setSuffix(" %)")
        self.exposure_tolerance.setFixedWidth(100)
        self.exposure_tolerance.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        exp_row.addWidget(self.exposure_tolerance)
        exp_row.addStretch(1)
        hard_layout.addLayout(exp_row)
        layout.addWidget(hard_box)

        soft_box = QGroupBox("Soft Assumptions (mit Toleranzen)")
        soft_layout = QFormLayout(soft_box)
        soft_layout.setSpacing(8)
        frame_row = QHBoxLayout()
        self.frames_min = QSpinBox()
        self.frames_min.setRange(1, 10000)
        self.frames_min.setValue(ASSUMPTIONS_DEFAULTS["frames_min"])
        self.frames_min.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        self.frames_reduced = QSpinBox()
        self.frames_reduced.setRange(1, 10000)
        self.frames_reduced.setValue(ASSUMPTIONS_DEFAULTS["frames_reduced_threshold"])
        self.frames_reduced.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        self.frames_optimal = QSpinBox()
        self.frames_optimal.setRange(1, 100000)
        self.frames_optimal.setValue(ASSUMPTIONS_DEFAULTS["frames_optimal"])
        self.frames_optimal.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        frame_row.addWidget(QLabel("min:"))
        frame_row.addWidget(self.frames_min)
        frame_row.addWidget(QLabel("reduced:"))
        frame_row.addWidget(self.frames_reduced)
        frame_row.addWidget(QLabel("optimal:"))
        frame_row.addWidget(self.frames_optimal)
        frame_row.addStretch(1)
        soft_layout.addRow("Frame-Anzahl", frame_row)
        reg_row = QHBoxLayout()
        self.reg_warn = QDoubleSpinBox()
        self.reg_warn.setRange(0.01, 10.0)
        self.reg_warn.setValue(ASSUMPTIONS_DEFAULTS["registration_residual_warn_px"])
        self.reg_warn.setSuffix(" px")
        self.reg_warn.setDecimals(2)
        self.reg_warn.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        self.reg_max = QDoubleSpinBox()
        self.reg_max.setRange(0.01, 10.0)
        self.reg_max.setValue(ASSUMPTIONS_DEFAULTS["registration_residual_max_px"])
        self.reg_max.setSuffix(" px")
        self.reg_max.setDecimals(2)
        self.reg_max.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        reg_row.addWidget(QLabel("warn:"))
        reg_row.addWidget(self.reg_warn)
        reg_row.addWidget(QLabel("max:"))
        reg_row.addWidget(self.reg_max)
        reg_row.addStretch(1)
        soft_layout.addRow("Registrierungs-Residual", reg_row)
        elong_row = QHBoxLayout()
        self.elong_warn = QDoubleSpinBox()
        self.elong_warn.setRange(0.01, 1.0)
        self.elong_warn.setValue(ASSUMPTIONS_DEFAULTS["elongation_warn"])
        self.elong_warn.setDecimals(2)
        self.elong_warn.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        self.elong_max = QDoubleSpinBox()
        self.elong_max.setRange(0.01, 1.0)
        self.elong_max.setValue(ASSUMPTIONS_DEFAULTS["elongation_max"])
        self.elong_max.setDecimals(2)
        self.elong_max.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        elong_row.addWidget(QLabel("warn:"))
        elong_row.addWidget(self.elong_warn)
        elong_row.addWidget(QLabel("max:"))
        elong_row.addWidget(self.elong_max)
        elong_row.addStretch(1)
        soft_layout.addRow("Stern-Elongation", elong_row)
        layout.addWidget(soft_box)

        implicit_box = QGroupBox("Implicit Assumptions (jetzt explizit)")
        implicit_layout = QFormLayout(implicit_box)
        implicit_layout.setSpacing(8)
        
        tracking_row = QHBoxLayout()
        tracking_row.addWidget(QLabel("• Tracking-Fehler max:"))
        self.tracking_error_max = QDoubleSpinBox()
        self.tracking_error_max.setRange(0.1, 10.0)
        self.tracking_error_max.setValue(ASSUMPTIONS_DEFAULTS["tracking_error_max_px"])
        self.tracking_error_max.setSuffix(" px")
        self.tracking_error_max.setDecimals(2)
        self.tracking_error_max.setFixedWidth(120)
        self.tracking_error_max.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        tracking_row.addWidget(self.tracking_error_max)
        tracking_row.addStretch(1)
        implicit_layout.addRow("", tracking_row)
        
        implicit_items = [
            "Stabile optische Konfiguration (Fokus, Feldkrümmung)",
            "Kein systematischer Drift während der Session",
        ]
        for item in implicit_items:
            lbl = QLabel(f"• {item}")
            lbl.setObjectName("ImplicitAssumption")
            lbl.setWordWrap(True)
            implicit_layout.addRow("", lbl)
        layout.addWidget(implicit_box)

        reduced_box = QGroupBox("Reduced Mode (50–199 Frames)")
        reduced_layout = QFormLayout(reduced_box)
        reduced_layout.setSpacing(8)
        self.skip_clustering = QCheckBox("STATE_CLUSTERING und SYNTHETIC_FRAMES überspringen")
        self.skip_clustering.setChecked(ASSUMPTIONS_DEFAULTS["reduced_mode_skip_clustering"])
        self.skip_clustering.stateChanged.connect(lambda *_: self.assumptions_changed.emit())
        reduced_layout.addRow("", self.skip_clustering)
        cluster_row = QHBoxLayout()
        self.cluster_min = QSpinBox()
        self.cluster_min.setRange(1, 100)
        self.cluster_min.setValue(ASSUMPTIONS_DEFAULTS["reduced_mode_cluster_range"][0])
        self.cluster_min.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        self.cluster_max = QSpinBox()
        self.cluster_max.setRange(1, 100)
        self.cluster_max.setValue(ASSUMPTIONS_DEFAULTS["reduced_mode_cluster_range"][1])
        self.cluster_max.valueChanged.connect(lambda *_: self.assumptions_changed.emit())
        cluster_row.addWidget(QLabel("min:"))
        cluster_row.addWidget(self.cluster_min)
        cluster_row.addWidget(QLabel("max:"))
        cluster_row.addWidget(self.cluster_max)
        cluster_row.addStretch(1)
        reduced_layout.addRow("Cluster-Range (falls nicht übersprungen)", cluster_row)
        self.reduced_mode_status = QLabel("")
        self.reduced_mode_status.setObjectName("ReducedModeWarning")
        self.reduced_mode_status.setVisible(False)
        reduced_layout.addRow("", self.reduced_mode_status)
        layout.addWidget(reduced_box)
        layout.addStretch(1)

    def get_assumptions(self) -> dict:
        return {
            "frames_min": self.frames_min.value(),
            "frames_optimal": self.frames_optimal.value(),
            "frames_reduced_threshold": self.frames_reduced.value(),
            "exposure_time_tolerance_percent": self.exposure_tolerance.value(),
            "registration_residual_warn_px": self.reg_warn.value(),
            "registration_residual_max_px": self.reg_max.value(),
            "elongation_warn": self.elong_warn.value(),
            "elongation_max": self.elong_max.value(),
            "tracking_error_max_px": self.tracking_error_max.value(),
            "reduced_mode_skip_clustering": self.skip_clustering.isChecked(),
            "reduced_mode_cluster_range": [self.cluster_min.value(), self.cluster_max.value()],
        }

    def set_assumptions(self, assumptions: dict):
        if "frames_min" in assumptions:
            self.frames_min.setValue(int(assumptions["frames_min"]))
        if "frames_optimal" in assumptions:
            self.frames_optimal.setValue(int(assumptions["frames_optimal"]))
        if "frames_reduced_threshold" in assumptions:
            self.frames_reduced.setValue(int(assumptions["frames_reduced_threshold"]))
        if "exposure_time_tolerance_percent" in assumptions:
            self.exposure_tolerance.setValue(float(assumptions["exposure_time_tolerance_percent"]))
        if "registration_residual_warn_px" in assumptions:
            self.reg_warn.setValue(float(assumptions["registration_residual_warn_px"]))
        if "registration_residual_max_px" in assumptions:
            self.reg_max.setValue(float(assumptions["registration_residual_max_px"]))
        if "elongation_warn" in assumptions:
            self.elong_warn.setValue(float(assumptions["elongation_warn"]))
        if "elongation_max" in assumptions:
            self.elong_max.setValue(float(assumptions["elongation_max"]))
        if "reduced_mode_skip_clustering" in assumptions:
            self.skip_clustering.setChecked(bool(assumptions["reduced_mode_skip_clustering"]))
        if "reduced_mode_cluster_range" in assumptions:
            rng = assumptions["reduced_mode_cluster_range"]
            if isinstance(rng, list) and len(rng) >= 2:
                self.cluster_min.setValue(int(rng[0]))
                self.cluster_max.setValue(int(rng[1]))

    def update_reduced_mode_status(self, frame_count: int):
        threshold = self.frames_reduced.value()
        minimum = self.frames_min.value()
        if frame_count < minimum:
            self.reduced_mode_status.setText(f"⛔ Frame-Anzahl ({frame_count}) unter Minimum ({minimum})")
            self.reduced_mode_status.setVisible(True)
        elif frame_count < threshold:
            self.reduced_mode_status.setText(f"⚠ Reduced Mode aktiv: {frame_count} Frames < {threshold}")
            self.reduced_mode_status.setVisible(True)
        else:
            self.reduced_mode_status.setVisible(False)


class MainWindow(QMainWindow):
    ui_call = Signal(object)

    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.constants = _read_gui_constants(project_root)
        self.setMinimumSize(1000, 700)
        self.resize(1300, 900)

        self.backend = BackendClient(project_root, self.constants)
        self.runner = RunnerProcess()

        self.last_scan: Optional[dict] = None
        self.confirmed_color_mode: Optional[str] = None
        self.config_validated_ok: bool = False
        self.current_run_id: Optional[str] = None
        self.current_run_dir: Optional[str] = None
        self._start_blocked_reason: str = ""
        self._frame_count: int = 0

        self.ui_call.connect(self._ui_exec)

        self.setWindowTitle("Tile Compile – Methodik v3")
        self._build_ui()
        self._load_styles()

        self._update_controls()
        self._load_gui_state()

    def _load_styles(self) -> None:
        p = self.project_root / "gui" / "styles.qss"
        try:
            self.setStyleSheet(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Tile Compile – Methodik v3")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch(1)
        self.lbl_header = QLabel("idle")
        self.lbl_header.setObjectName("StatusLabel")
        header.addWidget(self.lbl_header)
        root.addLayout(header)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        def wrap_scroll(content: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(content)
            return sa

        scan_page = QWidget()
        scan_page_layout = QVBoxLayout(scan_page)
        scan_page_layout.setContentsMargins(0, 0, 0, 0)
        scan_page_layout.setSpacing(10)

        scan_box = QGroupBox("Scan")
        scan_layout = QVBoxLayout(scan_box)
        scan_layout.setContentsMargins(12, 18, 12, 12)
        scan_layout.setSpacing(10)
        # Verwende FormLayout für einfachere und stabilere Layouts
        scan_form = QFormLayout()
        scan_form.setSpacing(10)
        scan_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Input dir row
        input_row = QHBoxLayout()
        self.scan_input_dir = QLineEdit()
        self.scan_input_dir.setMinimumHeight(30)
        self.scan_input_dir.setPlaceholderText("/path/to/frames")
        self.btn_browse_scan_dir = QPushButton("Browse")
        self.btn_browse_scan_dir.setMinimumHeight(30)
        self.btn_browse_scan_dir.setFixedWidth(100)
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setMinimumHeight(30)
        self.btn_scan.setFixedWidth(100)
        self.lbl_scan = QLabel("idle")
        self.lbl_scan.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        input_row.addWidget(self.scan_input_dir)
        input_row.addWidget(self.btn_browse_scan_dir)
        input_row.addWidget(self.btn_scan)
        input_row.addWidget(self.lbl_scan)
        
        scan_form.addRow("Input dir", input_row)
        scan_layout.addLayout(scan_form)

        row2 = QHBoxLayout()
        self.scan_frames_min = QSpinBox()
        self.scan_frames_min.setMinimum(1)
        self.scan_frames_min.setMaximum(1000000)
        self.scan_frames_min.setValue(1)
        self.scan_with_checksums = QCheckBox("With checksums")
        row2.addWidget(QLabel("Frames min"))
        row2.addWidget(self.scan_frames_min)
        row2.addWidget(self.scan_with_checksums)
        row2.addStretch(1)
        scan_layout.addLayout(row2)

        self.scan_msg = QLabel("")
        self.scan_msg.setWordWrap(True)
        self.scan_msg.setObjectName("StatusLabel")
        scan_layout.addWidget(self.scan_msg)

        row3 = QHBoxLayout()
        self.color_mode_select = QComboBox()
        self.btn_confirm_color = QPushButton("Confirm")
        self.lbl_confirm_hint = QLabel("")
        self.lbl_confirm_hint.setObjectName("StatusLabel")
        row3.addWidget(QLabel("Color mode"))
        row3.addWidget(self.color_mode_select)
        row3.addWidget(self.btn_confirm_color)
        row3.addStretch(1)
        scan_layout.addLayout(row3)
        scan_layout.addWidget(self.lbl_confirm_hint)

        self.scan_reduced_mode_hint = QLabel("")
        self.scan_reduced_mode_hint.setObjectName("ReducedModeWarning")
        self.scan_reduced_mode_hint.setVisible(False)
        scan_layout.addWidget(self.scan_reduced_mode_hint)

        scan_page_layout.addWidget(scan_box, 1)
        tabs.addTab(wrap_scroll(scan_page), "Scan")

        cfg_box = QGroupBox("Configuration")
        cfg_layout = QVBoxLayout(cfg_box)
        cfg_layout.setContentsMargins(12, 18, 12, 12)
        cfg_layout.setSpacing(10)
        # Verwende FormLayout für Config-Sektion
        cfg_form = QFormLayout()
        cfg_form.setSpacing(10)
        cfg_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Config path row
        config_row = QHBoxLayout()
        self.config_path = QLineEdit("tile_compile.yaml")
        self.config_path.setMinimumHeight(30)
        self.btn_browse_config = QPushButton("Browse")
        self.btn_browse_config.setMinimumHeight(30)
        self.btn_browse_config.setFixedWidth(100)
        
        config_row.addWidget(self.config_path)
        config_row.addWidget(self.btn_browse_config)
        
        cfg_form.addRow("Config path", config_row)
        
        # Buttons row
        button_row = QHBoxLayout()
        self.btn_cfg_load = QPushButton("Load")
        self.btn_cfg_load.setMinimumHeight(30)
        self.btn_cfg_load.setFixedWidth(100)
        self.btn_cfg_save = QPushButton("Save")
        self.btn_cfg_save.setMinimumHeight(30)
        self.btn_cfg_save.setFixedWidth(100)
        self.btn_cfg_validate = QPushButton("Validate")
        self.btn_cfg_validate.setMinimumHeight(30)
        self.btn_cfg_validate.setFixedWidth(100)
        self.lbl_cfg = QLabel("not validated")
        self.lbl_cfg.setObjectName("StatusLabel")
        
        self.btn_apply_assumptions = QPushButton("Apply Assumptions")
        self.btn_apply_assumptions.setMinimumHeight(30)
        self.btn_apply_assumptions.setFixedWidth(140)
        self.btn_apply_assumptions.setToolTip("Apply assumptions from Assumptions tab to config YAML")
        button_row.addWidget(self.btn_cfg_load)
        button_row.addWidget(self.btn_cfg_save)
        button_row.addWidget(self.btn_cfg_validate)
        button_row.addWidget(self.btn_apply_assumptions)
        button_row.addWidget(self.lbl_cfg)
        button_row.addStretch(1)
        
        cfg_form.addRow("", button_row)
        cfg_layout.addLayout(cfg_form)

        self.config_yaml = QTextEdit()
        self.config_yaml.setAcceptRichText(False)
        self.config_yaml.setPlaceholderText("# YAML config...")
        cfg_layout.addWidget(self.config_yaml)
        cfg_page = QWidget()
        cfg_page_layout = QVBoxLayout(cfg_page)
        cfg_page_layout.setContentsMargins(0, 0, 0, 0)
        cfg_page_layout.setSpacing(10)
        cfg_page_layout.addWidget(cfg_box, 1)
        tabs.addTab(wrap_scroll(cfg_page), "Configuration")

        assumptions_page = QWidget()
        assumptions_page_layout = QVBoxLayout(assumptions_page)
        assumptions_page_layout.setContentsMargins(0, 0, 0, 0)
        assumptions_page_layout.setSpacing(10)
        assumptions_box = QGroupBox("Methodik v3 Assumptions")
        assumptions_box_layout = QVBoxLayout(assumptions_box)
        assumptions_box_layout.setContentsMargins(12, 18, 12, 12)
        self.assumptions_widget = AssumptionsWidget()
        assumptions_box_layout.addWidget(self.assumptions_widget)
        assumptions_page_layout.addWidget(assumptions_box, 1)
        tabs.addTab(wrap_scroll(assumptions_page), "Assumptions")

        run_box = QGroupBox("Run")
        run_layout = QVBoxLayout(run_box)
        run_layout.setContentsMargins(12, 18, 12, 12)
        run_layout.setSpacing(10)
        rr0 = QHBoxLayout()
        self.working_dir = QLineEdit(str(self.project_root))
        self.btn_browse_working_dir = QPushButton("Browse")
        rr0.addWidget(QLabel("Working dir"))
        rr0.addWidget(self.working_dir, 1)
        rr0.addWidget(self.btn_browse_working_dir)
        run_layout.addLayout(rr0)

        rr1 = QHBoxLayout()
        self.input_dir = QLineEdit("")
        self.btn_browse_input_dir = QPushButton("Browse")
        rr1.addWidget(QLabel("Input dir"))
        rr1.addWidget(self.input_dir, 1)
        rr1.addWidget(self.btn_browse_input_dir)
        run_layout.addLayout(rr1)

        rr2 = QHBoxLayout()
        self.runs_dir = QLineEdit("runs")
        self.pattern = QLineEdit("*.fit*")
        self.dry_run = QCheckBox("Dry run")
        rr2.addWidget(QLabel("Runs dir"))
        rr2.addWidget(self.runs_dir)
        rr2.addWidget(QLabel("Pattern"))
        rr2.addWidget(self.pattern, 1)
        rr2.addWidget(self.dry_run)
        run_layout.addLayout(rr2)

        rr3 = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_abort = QPushButton("Abort")
        self.lbl_run = QLabel("idle")
        self.lbl_run.setObjectName("StatusLabel")
        rr3.addWidget(self.btn_start)
        rr3.addWidget(self.btn_abort)
        rr3.addStretch(1)
        rr3.addWidget(self.lbl_run)
        run_layout.addLayout(rr3)

        self.run_reduced_mode_hint = QLabel("")
        self.run_reduced_mode_hint.setObjectName("ReducedModeWarning")
        self.run_reduced_mode_hint.setVisible(False)
        run_layout.addWidget(self.run_reduced_mode_hint)

        run_page = QWidget()
        run_page_layout = QVBoxLayout(run_page)
        run_page_layout.setContentsMargins(0, 0, 0, 0)
        run_page_layout.setSpacing(10)
        run_page_layout.addWidget(run_box, 1)
        tabs.addTab(wrap_scroll(run_page), "Run")

        progress_page = QWidget()
        progress_page_layout = QVBoxLayout(progress_page)
        progress_page_layout.setContentsMargins(0, 0, 0, 0)
        progress_page_layout.setSpacing(10)
        progress_box = QGroupBox("Pipeline Progress (Methodik v3)")
        progress_box_layout = QVBoxLayout(progress_box)
        progress_box_layout.setContentsMargins(12, 18, 12, 12)
        self.phase_progress = PhaseProgressWidget()
        progress_box_layout.addWidget(self.phase_progress)
        progress_page_layout.addWidget(progress_box, 1)
        tabs.addTab(wrap_scroll(progress_page), "Pipeline Progress")

        current_box = QGroupBox("Current run")
        cur_layout = QVBoxLayout(current_box)
        cur_layout.setContentsMargins(12, 18, 12, 12)
        cur_layout.setSpacing(10)
        cur_row = QHBoxLayout()
        self.lbl_run_id = QLabel("-")
        self.lbl_run_id.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_run_dir = QLabel("-")
        self.lbl_run_dir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cur_row.addWidget(QLabel("run_id"))
        cur_row.addWidget(self.lbl_run_id, 1)
        cur_row.addWidget(QLabel("run_dir"))
        cur_row.addWidget(self.lbl_run_dir, 3)
        cur_layout.addLayout(cur_row)

        cur_btns = QHBoxLayout()
        self.btn_refresh_runs = QPushButton("Refresh runs")
        self.btn_refresh_status = QPushButton("Refresh status")
        self.btn_refresh_logs = QPushButton("Refresh logs")
        self.btn_refresh_artifacts = QPushButton("Refresh artifacts")
        self.logs_tail = QSpinBox()
        self.logs_tail.setMinimum(1)
        self.logs_tail.setMaximum(1000000)
        self.logs_tail.setValue(200)
        self.logs_filter = QLineEdit("")
        self.logs_filter.setPlaceholderText("filter text")
        cur_btns.addWidget(self.btn_refresh_runs)
        cur_btns.addWidget(self.btn_refresh_status)
        cur_btns.addWidget(QLabel("Tail"))
        cur_btns.addWidget(self.logs_tail)
        cur_btns.addWidget(QLabel("Filter"))
        cur_btns.addWidget(self.logs_filter, 1)
        cur_btns.addWidget(self.btn_refresh_logs)
        cur_btns.addWidget(self.btn_refresh_artifacts)
        cur_layout.addLayout(cur_btns)

        self.current_status = QLabel("idle")
        self.current_status.setObjectName("StatusLabel")
        cur_layout.addWidget(self.current_status)

        self.current_logs = QPlainTextEdit()
        self.current_logs.setReadOnly(True)
        self.current_artifacts = QPlainTextEdit()
        self.current_artifacts.setReadOnly(True)
        cur_layout.addWidget(self.current_logs)
        cur_layout.addWidget(self.current_artifacts)
        current_page = QWidget()
        current_page_layout = QVBoxLayout(current_page)
        current_page_layout.setContentsMargins(0, 0, 0, 0)
        current_page_layout.setSpacing(10)
        current_page_layout.addWidget(current_box, 1)
        tabs.addTab(wrap_scroll(current_page), "Current run")

        hist_box = QGroupBox("Run history")
        hist_layout = QVBoxLayout(hist_box)
        hist_layout.setContentsMargins(12, 18, 12, 12)
        hist_layout.setSpacing(10)
        self.runs_table = QTableWidget(0, 5)
        self.runs_table.setHorizontalHeaderLabels([
            "created_at",
            "run_id",
            "status",
            "config_hash",
            "frames_manifest_id",
        ])
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hist_layout.addWidget(self.runs_table)
        hist_page = QWidget()
        hist_page_layout = QVBoxLayout(hist_page)
        hist_page_layout.setContentsMargins(0, 0, 0, 0)
        hist_page_layout.setSpacing(10)
        hist_page_layout.addWidget(hist_box, 1)
        tabs.addTab(wrap_scroll(hist_page), "Run history")

        live_box = QGroupBox("Live log")
        live_layout = QVBoxLayout(live_box)
        live_layout.setContentsMargins(12, 18, 12, 12)
        live_layout.setSpacing(10)
        self.live_log = QPlainTextEdit()
        self.live_log.setReadOnly(True)
        live_layout.addWidget(self.live_log)
        live_page = QWidget()
        live_page_layout = QVBoxLayout(live_page)
        live_page_layout.setContentsMargins(0, 0, 0, 0)
        live_page_layout.setSpacing(10)
        live_page_layout.addWidget(live_box, 1)
        tabs.addTab(wrap_scroll(live_page), "Live log")

        self.setCentralWidget(central)

        self.btn_scan.clicked.connect(self._scan)
        self.btn_browse_scan_dir.clicked.connect(self._browse_scan_dir)
        self.btn_confirm_color.clicked.connect(self._confirm_color)

        self.btn_cfg_load.clicked.connect(self._load_config)
        self.btn_cfg_save.clicked.connect(self._save_config)
        self.btn_cfg_validate.clicked.connect(self._validate_config)
        self.btn_browse_config.clicked.connect(self._browse_config)
        self.btn_apply_assumptions.clicked.connect(self._apply_assumptions_to_config)
        self.config_yaml.textChanged.connect(self._on_config_edited)

        self.assumptions_widget.assumptions_changed.connect(self._on_assumptions_changed)

        self.btn_start.clicked.connect(self._start_run)
        self.btn_abort.clicked.connect(self._abort_run)

        self.btn_browse_working_dir.clicked.connect(self._browse_working_dir)
        self.btn_browse_input_dir.clicked.connect(self._browse_input_dir)

        self.btn_refresh_runs.clicked.connect(self._refresh_runs)
        self.btn_refresh_status.clicked.connect(self._refresh_current_status)
        self.btn_refresh_logs.clicked.connect(self._refresh_current_logs)
        self.btn_refresh_artifacts.clicked.connect(self._refresh_current_artifacts)
        self.runs_table.cellClicked.connect(self._select_run)

        self.scan_input_dir.editingFinished.connect(self._persist_last_input_dir_from_ui)
        self.input_dir.editingFinished.connect(self._persist_last_input_dir_from_ui)

    def _append_live(self, text: str) -> None:
        self.live_log.appendPlainText(text)
        sb = self.live_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _ui_exec(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self._append_live(f"[error] ui: {e}")

    def _log_backend_cmd(self, args: list[str]) -> None:
        cmd_dbg = " ".join(self.backend._backend_cmd() + args)
        self._append_live(f"[backend] {cmd_dbg}")

    def _run_bg(self, fn, on_ok=None, on_err=None) -> None:
        def worker():
            try:
                res = fn()
                if on_ok is not None:
                    self.ui_call.emit(lambda: on_ok(res))
            except Exception as e:  # noqa: BLE001
                if on_err is not None:
                    self.ui_call.emit(lambda: on_err(e))
                else:
                    self.ui_call.emit(lambda: self._append_live(f"[error] {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _browse_scan_dir(self) -> None:
        start = self.scan_input_dir.text().strip() or self.input_dir.text().strip() or str(self.project_root)
        p = QFileDialog.getExistingDirectory(self, "Select input directory", start)
        if p:
            self.scan_input_dir.setText(p)
            if not self.input_dir.text().strip():
                self.input_dir.setText(p)
            self._save_gui_state(last_input_dir=p)

    def _persist_last_input_dir_from_ui(self) -> None:
        p = self.scan_input_dir.text().strip() or self.input_dir.text().strip()
        if p:
            self._save_gui_state(last_input_dir=p)

    def _browse_input_dir(self) -> None:
        start = self.input_dir.text().strip() or self.scan_input_dir.text().strip() or str(self.project_root)
        p = QFileDialog.getExistingDirectory(self, "Select input directory", start)
        if p:
            self.input_dir.setText(p)
            if not self.scan_input_dir.text().strip():
                self.scan_input_dir.setText(p)
            self._save_gui_state(last_input_dir=p)

    def _load_gui_state(self) -> None:
        def do():
            wd = Path(self.project_root)
            args = [self.constants["CLI"]["sub"]["LOAD_GUI_STATE"]]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            if not isinstance(res, dict):
                return
            state = res.get("state")
            if not isinstance(state, dict):
                return
            last = str(state.get("lastInputDir") or "").strip()
            if last and not self.scan_input_dir.text().strip():
                self.scan_input_dir.setText(last)
            if last and not self.input_dir.text().strip():
                self.input_dir.setText(last)

        self._run_bg(do, ok)

    def _save_gui_state(self, last_input_dir: str | None = None) -> None:
        payload: dict[str, Any] = {}
        if last_input_dir is not None:
            payload["lastInputDir"] = last_input_dir

        def do():
            wd = Path(self.project_root)
            args = [self.constants["CLI"]["sub"]["SAVE_GUI_STATE"], "--stdin"]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args, stdin_text=json.dumps(payload, ensure_ascii=False))

        self._run_bg(do)

    def _browse_working_dir(self) -> None:
        start = self.working_dir.text().strip() or str(self.project_root)
        p = QFileDialog.getExistingDirectory(self, "Select working directory", start)
        if p:
            self.working_dir.setText(p)

    def _browse_config(self) -> None:
        start = self.config_path.text().strip() or str(self.project_root)
        if start and not Path(start).is_absolute():
            start = str((Path(self.working_dir.text().strip() or str(self.project_root)) / start).resolve())
        p, _ = QFileDialog.getOpenFileName(self, "Select config YAML", start, "YAML Files (*.yaml *.yml);;All Files (*)")
        if p:
            self.config_path.setText(p)

    def _update_controls(self) -> None:
        needs_confirm = bool(self.last_scan and self.last_scan.get("requires_user_confirmation"))
        has_confirm = bool(self.confirmed_color_mode)
        scan_ok = bool(self.last_scan and self.last_scan.get("ok"))

        self.color_mode_select.setEnabled(scan_ok)
        self.btn_confirm_color.setEnabled(scan_ok)

        blocked = ""
        if not scan_ok:
            blocked = "please run Scan first" if self.last_scan is None else "scan has errors"
        elif needs_confirm and not has_confirm:
            blocked = "please confirm color mode"
        elif not self.config_validated_ok:
            blocked = "please validate config"
        self._start_blocked_reason = blocked

        is_running = self.runner.is_running()
        start_enabled = (not is_running) and (not blocked)
        self.btn_start.setEnabled(start_enabled)
        self.btn_start.setToolTip(blocked or "")
        self.btn_abort.setEnabled(is_running)

        if (not is_running) and blocked:
            self.lbl_run.setText(f"blocked: {blocked}")

        threshold = self.assumptions_widget.frames_reduced.value()
        if self._frame_count > 0 and self._frame_count < threshold:
            self.run_reduced_mode_hint.setText(
                f"⚠ Reduced Mode: {self._frame_count} Frames < {threshold}\n"
                "STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen."
            )
            self.run_reduced_mode_hint.setVisible(True)
        else:
            self.run_reduced_mode_hint.setVisible(False)

    def _on_config_edited(self) -> None:
        self.config_validated_ok = False
        self.lbl_cfg.setText("not validated")
        self._update_controls()

    def _on_assumptions_changed(self) -> None:
        if self._frame_count > 0:
            self.assumptions_widget.update_reduced_mode_status(self._frame_count)
        self._update_controls()

    def _apply_assumptions_to_config(self) -> None:
        self._append_live("[ui] apply assumptions to config")
        yaml_text = self.config_yaml.toPlainText()
        try:
            cfg = yaml.safe_load(yaml_text) or {}
        except Exception as e:
            QMessageBox.warning(self, "YAML Parse Error", f"Cannot parse config YAML: {e}")
            return
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["assumptions"] = self.assumptions_widget.get_assumptions()
        new_yaml = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self.config_yaml.blockSignals(True)
        self.config_yaml.setPlainText(new_yaml)
        self.config_yaml.blockSignals(False)
        self.config_validated_ok = False
        self.lbl_cfg.setText("not validated (assumptions applied)")
        self._update_controls()

    def _extract_assumptions_from_yaml(self, yaml_text: str) -> None:
        try:
            cfg = yaml.safe_load(yaml_text)
            if isinstance(cfg, dict) and "assumptions" in cfg:
                self.assumptions_widget.set_assumptions(cfg["assumptions"])
        except Exception:
            pass

    def _force_layout_update(self) -> None:
        """Force layout update to fix initial rendering issues"""
        self.adjustSize()
        self.update()
        
    def _scan(self) -> None:
        self._append_live("[ui] scan button clicked")
        self.btn_scan.setEnabled(False)
        self.lbl_scan.setText("scanning...")
        QTimer.singleShot(100, lambda: self.btn_scan.setEnabled(True))
        
        input_path = self.scan_input_dir.text().strip() or self.input_dir.text().strip()
        if not input_path:
            self.lbl_scan.setText("error")
            self.scan_msg.setText("Input dir is required")
            QMessageBox.warning(self, "Scan Error", "Input directory is required")
            return
            
        # Check if directory exists
        if not Path(input_path).exists():
            self.lbl_scan.setText("error")
            self.scan_msg.setText(f"Directory not found: {input_path}")
            self._append_live(f"[error] scan directory not found: {input_path}")
            QMessageBox.warning(self, "Scan Error", f"Directory not found: {input_path}")
            return
            
        if not Path(input_path).is_dir():
            self.lbl_scan.setText("error")
            self.scan_msg.setText(f"Not a directory: {input_path}")
            self._append_live(f"[error] scan path is not a directory: {input_path}")
            QMessageBox.warning(self, "Scan Error", f"Not a directory: {input_path}")
            return

        self.last_scan = None
        self.confirmed_color_mode = None
        self._frame_count = 0
        self.color_mode_select.clear()
        self.lbl_confirm_hint.setText("")
        self.lbl_header.setText("scanning...")
        self.lbl_scan.setText("scanning...")
        self.scan_msg.setText("")
        self.scan_reduced_mode_hint.setVisible(False)
        self._update_controls()

        frames_min = int(self.scan_frames_min.value())
        with_checksums = bool(self.scan_with_checksums.isChecked())

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [
                self.constants["CLI"]["sub"]["SCAN"],
                input_path,
                "--frames-min",
                str(frames_min),
            ]
            if with_checksums:
                args.append("--with-checksums")
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args, timeout_s=120.0)

        def ok(res):
            self.last_scan = res
            self.lbl_scan.setText("ok" if res.get("ok") else "error")
            self.lbl_header.setText("idle")

            msg = ""
            if res.get("errors"):
                first = res["errors"][0]
                msg = f"{first.get('code','error')}: {first.get('message','')}".strip()
            elif res.get("warnings"):
                first = res["warnings"][0]
                msg = f"{first.get('code','warning')}: {first.get('message','')}".strip()
            elif res.get("ok"):
                frames = res.get("frames_detected")
                cm = res.get("color_mode")
                bp = res.get("bayer_pattern")
                parts: list[str] = []
                if frames is not None:
                    parts.append(f"frames_detected={frames}")
                if cm:
                    parts.append(f"color_mode={cm}")
                if bp:
                    parts.append(f"bayer_pattern={bp}")
                msg = ", ".join(parts)
            self.scan_msg.setText(msg)

            # Populate color mode candidates even if confirmation is not required.
            self.color_mode_select.blockSignals(True)
            self.color_mode_select.clear()
            candidates = []
            for c in (res.get("color_mode_candidates") or []):
                if c is None:
                    continue
                s = str(c).strip()
                if s and s not in candidates:
                    candidates.append(s)
            cm = str(res.get("color_mode") or "").strip()
            if cm and cm not in candidates and cm != "UNKNOWN":
                candidates.insert(0, cm)
            for c in candidates:
                self.color_mode_select.addItem(c)
            if candidates:
                self.color_mode_select.setCurrentIndex(0)
            self.color_mode_select.blockSignals(False)

            if res.get("requires_user_confirmation"):
                self.lbl_confirm_hint.setText("Scan could not determine color mode (missing/inconsistent BAYERPAT).")
            else:
                self.lbl_confirm_hint.setText("")

            if res.get("ok") and (not res.get("requires_user_confirmation")):
                cm2 = str(res.get("color_mode") or "").strip()
                if cm2 and cm2 != "UNKNOWN":
                    self.confirmed_color_mode = cm2

            frames_detected = res.get("frames_detected", 0)
            self._frame_count = frames_detected
            threshold = self.assumptions_widget.frames_reduced.value()
            minimum = self.assumptions_widget.frames_min.value()
            if frames_detected < minimum:
                self.scan_reduced_mode_hint.setText(
                    f"⛔ Frame-Anzahl ({frames_detected}) unter Minimum ({minimum})"
                )
                self.scan_reduced_mode_hint.setVisible(True)
            elif frames_detected < threshold:
                self.scan_reduced_mode_hint.setText(
                    f"⚠ Reduced Mode: {frames_detected} Frames < {threshold}\n"
                    "STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen."
                )
                self.scan_reduced_mode_hint.setVisible(True)
            else:
                self.scan_reduced_mode_hint.setVisible(False)
            self.assumptions_widget.update_reduced_mode_status(frames_detected)

            # Convenience: propagate input dir
            if res.get("ok"):
                self.input_dir.setText(input_path)
                self._save_gui_state(last_input_dir=input_path)

            self._update_controls()

        def err(e: Exception):
            self.lbl_scan.setText("error")
            self.scan_msg.setText(str(e))
            self.lbl_header.setText("idle")
            self._update_controls()

        self._run_bg(do, ok, err)

    def _confirm_color(self) -> None:
        if not self.last_scan:
            return
        sel = self.color_mode_select.currentText().strip()
        self.confirmed_color_mode = sel or None
        self._append_live(f"[ui] confirmed color_mode={self.confirmed_color_mode}")
        self._update_controls()

    def _load_config(self) -> None:
        self._append_live("[ui] load config button clicked")
        self.btn_cfg_load.setEnabled(False)
        QTimer.singleShot(100, lambda: self.btn_cfg_load.setEnabled(True))
        
        path = self.config_path.text().strip() or "tile_compile.yaml"
        self.lbl_header.setText("loading config...")
        self.lbl_cfg.setText("loading...")
        
        # Check if file exists
        wd = Path(self.working_dir.text().strip() or str(self.project_root))
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = wd / config_path
            
        if not config_path.exists():
            self.lbl_cfg.setText("error")
            self.lbl_header.setText("idle")
            err_msg = f"Config file not found: {config_path}"
            self._append_live(f"[error] {err_msg}")
            QMessageBox.critical(self, "Load config failed", err_msg)
            self._update_controls()
            return

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["LOAD_CONFIG"], path]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            yaml_text = str(res.get("yaml") or "")
            self.config_yaml.blockSignals(True)
            self.config_yaml.setPlainText(yaml_text)
            self.config_yaml.blockSignals(False)
            self.config_validated_ok = False
            self.lbl_cfg.setText("not validated")
            self.lbl_header.setText("idle")
            self._extract_assumptions_from_yaml(yaml_text)
            self._update_controls()

        def err(e: Exception):
            self.lbl_cfg.setText("error")
            self.lbl_header.setText("idle")
            QMessageBox.critical(self, "Load config failed", str(e))
            self._update_controls()

        self._run_bg(do, ok, err)

    def _save_config(self) -> None:
        self._append_live("[ui] save config button clicked")
        self.btn_cfg_save.setEnabled(False)
        QTimer.singleShot(100, lambda: self.btn_cfg_save.setEnabled(True))
        
        path = self.config_path.text().strip() or "tile_compile.yaml"
        yaml_text = self.config_yaml.toPlainText()
        self.lbl_header.setText("saving config...")
        self.lbl_cfg.setText("saving...")

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["SAVE_CONFIG"], path, "--stdin"]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args, stdin_text=yaml_text)

        def ok(_res):
            self.lbl_cfg.setText("saved")
            self.lbl_header.setText("idle")
            self._update_controls()

        def err(e: Exception):
            self.lbl_cfg.setText("error")
            self.lbl_header.setText("idle")
            QMessageBox.critical(self, "Save config failed", str(e))
            self._update_controls()

        self._run_bg(do, ok, err)

    def _validate_config(self) -> None:
        self._append_live("[ui] validate config button clicked")
        self.btn_cfg_validate.setEnabled(False)
        QTimer.singleShot(100, lambda: self.btn_cfg_validate.setEnabled(True))
        
        yaml_text = self.config_yaml.toPlainText()
        self.lbl_header.setText("validating config...")
        self.lbl_cfg.setText("validating...")

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["VALIDATE_CONFIG"], "--yaml", "-", "--stdin"]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args, stdin_text=yaml_text)

        def ok(res):
            valid = bool(res.get("valid"))
            self.config_validated_ok = valid
            self.lbl_cfg.setText("ok" if valid else "invalid")
            self.lbl_header.setText("idle")
            if not valid:
                errs = res.get("errors") or []
                msg = "\n".join([str(e) for e in errs[:8]])
                if msg:
                    QMessageBox.warning(self, "Config invalid", msg)
            self._update_controls()

        def err(e: Exception):
            self.config_validated_ok = False
            self.lbl_cfg.setText("error")
            self.lbl_header.setText("idle")
            QMessageBox.critical(self, "Validate config failed", str(e))
            self._update_controls()

        self._run_bg(do, ok, err)

    def _runner_cmd(self) -> list[str]:
        runner = self.constants.get("RUNNER", {})
        py = str(runner.get("python") or "").strip() or sys.executable
        script = str(runner.get("script") or "tile_compile_runner.py").strip()
        sub = str(runner.get("run_subcommand") or "run").strip()

        if script.endswith(".py"):
            return [py, str((self.project_root / script).resolve()), sub]
        return [py, script, sub]

    def _start_run(self) -> None:
        if self.runner.is_running():
            return

        if self._start_blocked_reason:
            self.lbl_run.setText(f"blocked: {self._start_blocked_reason}")
            self._append_live(f"[ui] {self._start_blocked_reason}")
            return

        input_dir = self.input_dir.text().strip() or self.scan_input_dir.text().strip()
        config_path = self.config_path.text().strip() or "tile_compile.yaml"
        runs_dir = self.runs_dir.text().strip() or "runs"
        pattern = self.pattern.text().strip() or "*.fit*"
        dry_run = bool(self.dry_run.isChecked())

        if not input_dir:
            QMessageBox.warning(self, "Start run", "Input dir is required")
            return

        self.lbl_run.setText("starting...")
        self._append_live("[ui] start run")

        self.phase_progress.reset()
        threshold = self.assumptions_widget.frames_reduced.value()
        if self._frame_count > 0 and self._frame_count < threshold:
            self.phase_progress.set_reduced_mode(True, self._frame_count)

        cmd = self._runner_cmd() + [
            "--config",
            config_path,
            "--input-dir",
            input_dir,
            "--runs-dir",
            runs_dir,
            "--pattern",
            pattern,
        ]
        self._append_live("[runner] " + " ".join(cmd))
        if dry_run:
            cmd.append("--dry-run")
        if self.confirmed_color_mode:
            cmd.extend(["--color-mode-confirmed", self.confirmed_color_mode])

        wd = Path(self.working_dir.text().strip() or str(self.project_root))

        try:
            proc = self.runner.start(cmd, wd)
        except Exception as e:  # noqa: BLE001
            self.lbl_run.setText("error")
            QMessageBox.critical(self, "Start failed", str(e))
            self._update_controls()
            return

        self.lbl_run.setText("running")
        self._update_controls()

        def read_stream(stream, prefix: str):
            while True:
                line = stream.readline()
                if not line:
                    break
                s = line.rstrip("\n")
                if not s:
                    continue

                def make_ui_append(line, pfx):
                    def ui_append():
                        self._append_live(f"{pfx}{line}")
                        if pfx == "":
                            try:
                                ev = json.loads(line)
                                if isinstance(ev, dict):
                                    ev_type = ev.get("type")
                                    if ev_type == "run_start":
                                        self.current_run_id = str(ev.get("run_id") or "") or None
                                        paths = ev.get("paths") if isinstance(ev.get("paths"), dict) else {}
                                        self.current_run_dir = str(paths.get("run_dir") or "") or None
                                        self.lbl_run_id.setText(self.current_run_id or "-")
                                        self.lbl_run_dir.setText(self.current_run_dir or "-")
                                        self.phase_progress.reset()
                                    elif ev_type == "phase_start":
                                        phase_name = ev.get("phase_name", "")
                                        self.phase_progress.update_phase(phase_name, "running")
                                    elif ev_type == "phase_progress":
                                        phase_name = ev.get("phase_name", "")
                                        current = int(ev.get("current", 0))
                                        total = int(ev.get("total", 0))
                                        self.phase_progress.update_phase(phase_name, "running", current, total)
                                    elif ev_type == "phase_end":
                                        phase_name = ev.get("phase_name", "")
                                        status = str(ev.get("status") or "ok").lower()
                                        if status == "skipped":
                                            self.phase_progress.update_phase(phase_name, "skipped")
                                        elif status in ("ok", "success"):
                                            self.phase_progress.update_phase(phase_name, "ok")
                                        else:
                                            self.phase_progress.update_phase(phase_name, "error")
                            except Exception:
                                pass
                    return ui_append

                self.ui_call.emit(make_ui_append(s, prefix))

        if proc.stdout is not None:
            threading.Thread(target=read_stream, args=(proc.stdout, ""), daemon=True).start()
        if proc.stderr is not None:
            threading.Thread(target=read_stream, args=(proc.stderr, "[stderr] "), daemon=True).start()

        def wait_end():
            code = proc.wait()

            def ui_end():
                self.lbl_run.setText(f"finished ({code})")
                self._update_controls()
                self._refresh_runs()
                self._refresh_current_status()
                self._refresh_phase_status_from_logs()

            self.ui_call.emit(ui_end)

        threading.Thread(target=wait_end, daemon=True).start()

    def _abort_run(self) -> None:
        if not self.runner.is_running():
            return
        self._append_live("[ui] abort requested")
        self.lbl_run.setText("aborting...")
        self.runner.stop()
        self._update_controls()

    def _refresh_runs(self) -> None:
        runs_dir = self.runs_dir.text().strip() or "runs"

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["LIST_RUNS"], runs_dir]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            if not isinstance(res, list):
                return
            self.runs_table.setRowCount(0)
            for item in res:
                if not isinstance(item, dict):
                    continue
                r = self.runs_table.rowCount()
                self.runs_table.insertRow(r)
                self.runs_table.setItem(r, 0, QTableWidgetItem(str(item.get("created_at") or "")))
                self.runs_table.setItem(r, 1, QTableWidgetItem(str(item.get("run_id") or "")))
                self.runs_table.setItem(r, 2, QTableWidgetItem(str(item.get("status") or "")))
                self.runs_table.setItem(r, 3, QTableWidgetItem(str(item.get("config_hash") or "")))
                self.runs_table.setItem(r, 4, QTableWidgetItem(str(item.get("frames_manifest_id") or "")))
                # store run_dir in vertical header for retrieval
                self.runs_table.setVerticalHeaderItem(r, QTableWidgetItem(str(item.get("run_dir") or "")))

            try:
                self.runs_table.resizeColumnsToContents()
            except Exception:
                pass
            self._append_live(f"[ui] runs: {len(res)}")

            if (not self.current_run_dir) and res:
                try:
                    self.runs_table.selectRow(0)
                    self._select_run(0, 0)
                except Exception:
                    pass

        def err(e: Exception):
            QMessageBox.critical(self, "List runs failed", str(e))

        self._run_bg(do, ok, err)

    def _select_run(self, row: int, _col: int) -> None:
        vh = self.runs_table.verticalHeaderItem(row)
        run_dir = (vh.text() if vh is not None else "").strip()
        if not run_dir:
            return
        self.current_run_dir = run_dir
        self.lbl_run_dir.setText(run_dir)
        run_id = self.runs_table.item(row, 1).text() if self.runs_table.item(row, 1) else ""
        self.current_run_id = run_id or None
        self.lbl_run_id.setText(self.current_run_id or "-")
        self._refresh_current_status()
        self._refresh_current_logs()
        self._refresh_current_artifacts()
        self._refresh_phase_status_from_logs()

    def _refresh_current_status(self) -> None:
        if not self.current_run_dir:
            return

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["GET_RUN_STATUS"], self.current_run_dir]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            st = str(res.get("status") or "") if isinstance(res, dict) else ""
            ph = res.get("phase_name") if isinstance(res, dict) else None
            self.current_status.setText(f"{st} ({ph})" if ph else st)

        def err(e: Exception):
            self.current_status.setText("error")
            self._append_live(f"[ui] get-run-status failed: {e}")

        self._run_bg(do, ok, err)

    def _refresh_current_logs(self) -> None:
        if not self.current_run_dir:
            return
        tail = int(self.logs_tail.value())
        flt = self.logs_filter.text().strip().lower()

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["GET_RUN_LOGS"], self.current_run_dir, "--tail", str(tail)]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            events = res.get("events") if isinstance(res, dict) else []
            out_lines: list[str] = []
            for ev in events if isinstance(events, list) else []:
                if not isinstance(ev, dict):
                    continue
                s = json.dumps(ev, ensure_ascii=False, sort_keys=True)
                if flt and flt not in s.lower():
                    continue
                out_lines.append(s)
            self.current_logs.setPlainText("\n".join(out_lines))

        def err(e: Exception):
            self._append_live(f"[ui] get-run-logs failed: {e}")

        self._run_bg(do, ok, err)

    def _refresh_current_artifacts(self) -> None:
        if not self.current_run_dir:
            return

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["LIST_ARTIFACTS"], self.current_run_dir]
            self.ui_call.emit(lambda: self._log_backend_cmd(args))
            return self.backend.run_json(wd, args)

        def ok(res):
            if not isinstance(res, dict):
                return
            lines: list[str] = []
            for k in ["artifacts", "outputs"]:
                items = res.get(k) or []
                lines.append(f"[{k}]")
                for it in items if isinstance(items, list) else []:
                    if not isinstance(it, dict):
                        continue
                    lines.append(f"{it.get('path')} ({it.get('size')})")
                lines.append("")
            self.current_artifacts.setPlainText("\n".join(lines).strip())

        def err(e: Exception):
            self._append_live(f"[ui] list-artifacts failed: {e}")

        self._run_bg(do, ok, err)

    def _refresh_phase_status_from_logs(self) -> None:
        """Load phase status from event logs and update the phase progress widget."""
        if not self.current_run_dir:
            return

        def do():
            wd = Path(self.working_dir.text().strip() or str(self.project_root))
            args = [self.constants["CLI"]["sub"]["GET_RUN_LOGS"], self.current_run_dir, "--tail", "1000"]
            return self.backend.run_json(wd, args)

        def ok(res):
            events = res.get("events") if isinstance(res, dict) else []
            if not isinstance(events, list):
                return
            
            # Reset phase widget first
            self.phase_progress.reset()
            
            # Track last status for each phase
            phase_status: dict[str, tuple[str, int, int]] = {}  # phase_name -> (status, current, total)
            
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ev_type = ev.get("type")
                phase_name = ev.get("phase_name", "")
                if not phase_name:
                    continue
                
                if ev_type == "phase_start":
                    phase_status[phase_name] = ("running", 0, 0)
                elif ev_type == "phase_progress":
                    current = int(ev.get("current", 0))
                    total = int(ev.get("total", 0))
                    phase_status[phase_name] = ("running", current, total)
                elif ev_type == "phase_end":
                    status = str(ev.get("status") or "ok").lower()
                    if status == "skipped":
                        phase_status[phase_name] = ("skipped", 0, 0)
                    elif status in ("ok", "success"):
                        phase_status[phase_name] = ("ok", 0, 0)
                    else:
                        phase_status[phase_name] = ("error", 0, 0)
            
            # Apply final status to widget
            for phase_name, (status, current, total) in phase_status.items():
                self.phase_progress.update_phase(phase_name, status, current, total)

        def err(e: Exception):
            self._append_live(f"[ui] refresh-phase-status failed: {e}")

        self._run_bg(do, ok, err)


def main() -> int:
    app = QApplication(sys.argv)
    project_root = _resolve_project_root(Path(__file__))
    
    # Setze einen vernünftigen Stil
    app.setStyle('Fusion')
    
    # Erstelle und zeige das Hauptfenster
    win = MainWindow(project_root)
    win.show()
    
    # Erzwinge Layout-Updates nach dem Anzeigen
    QTimer.singleShot(10, win.adjustSize)
    QTimer.singleShot(100, win.adjustSize)
    QTimer.singleShot(500, win.adjustSize)  # Nochmal nach längerer Zeit
    
    return app.exec()
    # w.resize(1100, 800)
    # w.show()
    # return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
