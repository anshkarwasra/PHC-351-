"""Tee process output to a timestamped log file.

Output written directly by C/C++ code to the process file descriptors is not captured;
faulthandler covers native crashes and still writes a traceback to the log file.
"""

import atexit
import faulthandler
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

_LOG_STATE = {}


class TeeStream:
    """Mirror writes to the original stream and a file, while preserving tty behavior."""

    def __init__(self, original, log_handle):
        self._original = original
        self._log = log_handle

    def write(self, data):
        if data:
            try:
                self._original.write(data)
            except Exception:
                pass
            try:
                self._log.write(data)
                self._log.flush()
            except Exception:
                pass
        return len(data) if hasattr(data, "__len__") else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._original, "encoding", None) or "utf-8"

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _git_short_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip() or "unknown"
    except Exception:
        return "unknown"


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _warn_handler(message, category, filename, lineno, file=None, line=None):
    logger = logging.getLogger()
    logger.warning("%s:%s: %s: %s", filename, lineno, category.__name__, message)


def _make_log_path(script_name, log_dir):
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{script_name}_{ts}.log"


def _finalize_run():
    state = _LOG_STATE.get("active")
    if state is None:
        return
    start_ts = state["start_ts"]
    elapsed = time.time() - start_ts
    exit_status = state.get("exit_status", 0)
    log_path = Path(state["log_path"])
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n=== RUN END ===\n")
            fh.write(f"elapsed_sec={elapsed:.6f}\n")
            fh.write(f"exit_status={exit_status}\n")
    except Exception:
        pass
    try:
        index_path = Path(state["log_dir"]) / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not index_path.exists() or index_path.stat().st_size == 0
        with index_path.open("a", encoding="utf-8", newline="") as fh:
            if write_header:
                fh.write("timestamp,script,argv,log_path,exit_status,elapsed_sec\n")
            argv = "|".join(state["argv"])
            fh.write(f"{datetime.now().isoformat(timespec='seconds')},{state['script_name']},{argv},{log_path},{exit_status},{elapsed:.6f}\n")
    except Exception:
        pass


def _exception_hook(exctype, value, tb):
    state = _LOG_STATE.get("active")
    if state is not None:
        state["exit_status"] = 1
        log_path = Path(state["log_path"])
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write("\n=== UNCAUGHT EXCEPTION ===\n")
            traceback.print_exception(exctype, value, tb, file=fh)
    sys.__excepthook__(exctype, value, tb)


def start_run_log(script_name, log_dir="data/logs", extra=None):
    """Begin logging stdout/stderr and warnings to a timestamped file.

    Returns the path of the log file. A no-op when MESH_NO_LOG=1 is set.
    """
    if os.environ.get("MESH_NO_LOG") == "1":
        return None
    if _LOG_STATE.get("active") is not None:
        return Path(_LOG_STATE["active"]["log_path"])

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = _make_log_path(script_name, log_dir)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    tee_stdout = TeeStream(original_stdout, log_handle)
    tee_stderr = TeeStream(original_stderr, log_handle)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(tee_stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)
    warnings.showwarning = _warn_handler

    sys.excepthook = _exception_hook
    faulthandler.enable(file=log_handle)
    atexit.register(_finalize_run)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "=== RUN START ===",
        f"timestamp={start_time}",
        f"script={script_name}",
        f"argv={sys.argv}",
        f"cwd={os.getcwd()}",
        f"git_short_hash={_git_short_hash()}",
        f"python={platform.python_version()}",
        f"numpy={__import__('numpy').__version__}",
        f"torch={__import__('torch').__version__}",
        f"pyvista={_package_version('pyvista')}",
        f"tetgen={_package_version('tetgen')}",
        f"trimesh={_package_version('trimesh')}",
        f"scipy={_package_version('scipy')}",
    ]
    if extra is not None:
        header.append(f"extra={extra}")
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n".join(header) + "\n")

    _LOG_STATE["active"] = {
        "start_ts": time.time(),
        "script_name": script_name,
        "log_path": str(log_path),
        "log_dir": str(log_dir),
        "argv": list(sys.argv),
        "exit_status": 0,
        "log_handle": log_handle,
    }
    return log_path
