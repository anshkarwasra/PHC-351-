import logging
import os
import sys

from src.utils.run_logging import start_run_log


def test_run_log_records_stdout_stderr_and_warning(tmp_path):
    log_path = start_run_log("probe", log_dir=tmp_path)
    assert log_path is not None
    assert log_path.parent == tmp_path

    print("hello stdout")
    sys.stderr.write("hello stderr\n")
    logging.warning("hello warning")
    second = start_run_log("probe", log_dir=tmp_path)
    assert second == log_path

    text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "hello stdout" in text
    assert "hello stderr" in text
    assert "hello warning" in text


def test_run_log_unhandled_exception_written_to_log(tmp_path):
    log_path = start_run_log("exploder", log_dir=tmp_path)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "UNCAUGHT EXCEPTION" in text
    assert "RuntimeError: boom" in text


def test_run_log_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_NO_LOG", "1")
    disabled = start_run_log("disabled", log_dir=tmp_path)
    assert disabled is None
