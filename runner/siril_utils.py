"""
Siril script validation and execution utilities.

Functions for validating Siril scripts against policy, running scripts, and extracting metadata.
"""

import re
import subprocess
import time
from pathlib import Path


def validate_siril_script(path: Path) -> tuple[bool, list[str]]:
    """Validate Siril script against policy."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, [f"cannot read script: {e}"]
    
    lines = raw.splitlines()
    violations = []
    
    forbidden_commands = {
        "cd", "rmdir", "rm", "unlink", "delete", "exec", "system", "shell",
        "wget", "curl", "download", "upload", "ftp", "http", "https",
    }
    
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        
        lower = stripped.lower()
        for cmd in forbidden_commands:
            if lower.startswith(cmd + " ") or lower == cmd:
                violations.append(f"line {i}: forbidden command '{cmd}'")
        
        if ".." in stripped:
            violations.append(f"line {i}: path traversal '..' not allowed")
        
        if stripped.startswith("/") and not stripped.startswith("//"):
            violations.append(f"line {i}: absolute path not allowed")
    
    return len(violations) == 0, sorted(set(violations))


def run_siril_script(
    siril_exe: str,
    work_dir: Path,
    script_path: Path,
    artifacts_dir: Path,
    log_name: str,
    quiet: bool = False,
) -> tuple[bool, dict]:
    """Run Siril script and return success status with metadata."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = artifacts_dir / log_name
    
    cmd = [siril_exe, "-s", str(script_path.resolve())]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
        elapsed = time.time() - start_time
        
        output = result.stdout or ""
        log_path.write_text(output, encoding="utf-8", errors="replace")
        
        success = result.returncode == 0
        
        return success, {
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "log_file": str(log_path),
            "output_lines": len(output.splitlines()) if not quiet else 0,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, {
            "error": "timeout",
            "elapsed_seconds": elapsed,
            "log_file": str(log_path),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return False, {
            "error": str(e),
            "elapsed_seconds": elapsed,
        }


def run_siril(
    siril_exe: str,
    work_dir: Path,
    script_text: str,
    artifacts_dir: Path,
    log_name: str,
    quiet: bool = False,
) -> tuple[bool, dict]:
    """Run Siril with inline script text."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = work_dir / "inline_script.ssf"
    script_path.write_text(script_text, encoding="utf-8")
    
    return run_siril_script(siril_exe, work_dir, script_path, artifacts_dir, log_name, quiet)


def extract_siril_save_targets(script_path: Path) -> list[str]:
    """Extract save target filenames from Siril script."""
    try:
        raw = script_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    
    lines = raw.splitlines()
    targets = []
    
    save_pattern = re.compile(r"^\s*save\s+(.+)", re.IGNORECASE)
    savebmp_pattern = re.compile(r"^\s*savebmp\s+(.+)", re.IGNORECASE)
    savetif_pattern = re.compile(r"^\s*savetif\s+(.+)", re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        
        for pattern in [save_pattern, savebmp_pattern, savetif_pattern]:
            m = pattern.match(stripped)
            if m:
                target = m.group(1).strip()
                target = target.split()[0] if target else ""
                if target:
                    targets.append(target)
    
    return targets
