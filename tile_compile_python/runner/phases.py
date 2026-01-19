"""
Pipeline orchestration wrapper for Methodik v3.

This module provides the main run_phases function that orchestrates the entire pipeline.
The actual implementation is in phases_impl.py (extracted from the monolithic runner).
"""

import signal
from pathlib import Path
from typing import Any

from .phases_impl import run_phases_impl


# Global stop flag for signal handling
_STOP = False


def _handle_signal(_signum, _frame):
    """Signal handler for graceful shutdown."""
    global _STOP
    _STOP = True


def run_phases(
    run_id: str,
    log_fp,
    dry_run: bool,
    run_dir: Path,
    project_root: Path,
    cfg: dict[str, Any],
    frames: list[Path],
    siril_exe: str | None,
) -> bool:
    """
    Run the Methodik v3 pipeline phases.
    
    Args:
        run_id: Unique run identifier
        log_fp: Log file handle for event logging
        dry_run: If True, only emit phase events without actual processing
        run_dir: Run directory for outputs/artifacts/work
        project_root: Project root directory
        cfg: Configuration dictionary
        frames: List of input frame paths
        siril_exe: Path to Siril executable (optional)
    
    Returns:
        True if pipeline completed successfully, False otherwise
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    
    # Call the actual implementation
    return run_phases_impl(
        run_id=run_id,
        log_fp=log_fp,
        dry_run=dry_run,
        run_dir=run_dir,
        project_root=project_root,
        cfg=cfg,
        frames=frames,
        siril_exe=siril_exe,
        stop_flag=_STOP,
    )
