"""log_setup.py and what the app logs."""
import logging
import logging.handlers
import os
import subprocess
import sys

import pytest

import downloads
import generator
import latents
import log_setup
from conftest import GENDER_FILE, DummyContext, FakeEstimator, FakeGs

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def isolated_logging(monkeypatch):
    """Let a test call log_setup.configure() and undo everything afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    names = log_setup.APP_LOGGERS + ("tensorflow",)
    saved_levels = {n: logging.getLogger(n).level for n in names}
    monkeypatch.setattr(log_setup, "_configured", False)
    monkeypatch.setattr(log_setup, "_error_reporter", None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    yield
    for h in root.handlers[:]:
        if h not in saved_handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved_level)
    for n, level in saved_levels.items():
        logging.getLogger(n).setLevel(level)


def added_handlers():
    return [h for h in logging.getLogger().handlers
            if type(h) in (logging.StreamHandler, logging.handlers.RotatingFileHandler)
            and getattr(h, "formatter", None) is not None
            and (isinstance(h, logging.handlers.RotatingFileHandler)
                 or isinstance(h.formatter, log_setup._ConsoleFormatter))]


# ── log_setup.configure ────────────────────────────────────────────────────
def test_configure_writes_a_rotating_log_file(isolated_logging, tmp_path):
    path = str(tmp_path / "logs" / "facegen.log")
    assert log_setup.configure(log_path=path) == path
    file_handler, = [h for h in added_handlers() if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert (file_handler.maxBytes, file_handler.backupCount) == (1_000_000, 3)

    logging.getLogger("generator").debug("a debug detail")
    logging.getLogger("latents").info("an info line")
    logging.getLogger("PIL.PngImagePlugin").debug("third-party chatter")
    file_handler.flush()
    text = open(path, encoding="utf-8").read()
    assert "DEBUG   [generator] a debug detail" in text
    assert "INFO    [latents] an info line" in text
    assert "third-party chatter" not in text          # other libraries stay at INFO
    assert log_setup.active_log_file() == path
    assert path in log_setup.details_hint()


def test_configure_twice_adds_nothing(isolated_logging, tmp_path):
    path = str(tmp_path / "f.log")
    log_setup.configure(log_path=path)
    before = len(added_handlers())
    assert log_setup.configure(log_path=str(tmp_path / "other.log")) == path
    assert len(added_handlers()) == before == 2


def test_console_only_for_scripts(isolated_logging):
    assert log_setup.configure(log_file=False) is None
    assert log_setup.active_log_file() is None
    assert log_setup.details_hint() == ""


def test_tensorflow_is_quieted(isolated_logging, tmp_path):
    log_setup.configure(log_path=str(tmp_path / "f.log"))
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert logging.getLogger("tensorflow").level == logging.ERROR


def test_console_lines_name_the_level_only_for_problems():
    fmt = log_setup._ConsoleFormatter()
    make = lambda level: logging.LogRecord("latents", level, __file__, 1, "hello", None, None)
    assert fmt.format(make(logging.INFO)) == "[latents] hello"
    assert fmt.format(make(logging.WARNING)) == "[latents] WARNING: hello"


# ── what gets logged ───────────────────────────────────────────────────────
def test_age_mode_is_logged(regression_cache, caplog):
    caplog.set_level(logging.DEBUG)
    latents.LatentDirections()
    assert ("latents", logging.INFO, "Using calibrated ridge-regression age direction.") in caplog.record_tuples


def test_fallback_mode_is_a_warning(tmp_cache, caplog):
    latents.LatentDirections()
    assert any(r.name == "latents" and r.levelno == logging.WARNING and "fallback" in r.getMessage()
               for r in caplog.records)


def test_run_summary_and_retry_details(regression_cache, tmp_path, use_estimator, caplog):
    caplog.set_level(logging.DEBUG)
    use_estimator(FakeEstimator([None, -100.0, 30.0]))   # no face, rejected, accepted
    generator.generate_batch(FakeGs(), DummyContext(), DummyContext(), 1, "26-35 years old", "Any",
                             str(tmp_path / "out"), {})
    messages = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "generator"]
    assert (logging.DEBUG, "age=26-35 years old: no face detected, attempt 1/5") in messages
    assert (logging.DEBUG, "age=26-35 years old target=[26,35] got=-100.0, attempt 2/5") in messages
    summary = [m for lvl, m in messages if lvl == logging.INFO and m.startswith("Saved")]
    assert len(summary) == 1
    assert summary[0].startswith("Saved 1 of 1 image(s) in ")
    assert summary[0].endswith("(3 attempts: 1 rejected by the age check, 1 with no face detected)")


def test_run_summary_notes_an_early_stop(regression_cache, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    gs = FakeGs()
    generator.generate_batch(gs, DummyContext(), DummyContext(), 4, "Any", "Any", str(tmp_path / "out"),
                             {}, should_stop=lambda: len(gs.calls) >= 2)
    assert any(r.getMessage().startswith("Saved 2 of 4") and r.getMessage().endswith("; stopped early")
               for r in caplog.records)


def test_missing_insightface_is_a_warning(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "insightface.app", None)   # import fails
    est = generator.AgeEstimator()
    assert not est._available
    assert any(r.levelno == logging.WARNING and "InsightFace not available" in r.getMessage()
               for r in caplog.records)


def test_failed_estimate_is_logged_with_details(caplog):
    caplog.set_level(logging.DEBUG)

    class BrokenApp:
        def get(self, img):
            raise RuntimeError("detector crashed")
    est = generator.AgeEstimator.__new__(generator.AgeEstimator)
    est._available, est._app = True, BrokenApp()
    import numpy as np
    assert est.estimate(np.zeros((32, 32, 3), np.uint8)) is None
    warning, = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "detector crashed" in warning.getMessage()
    assert any(r.levelno == logging.DEBUG and r.exc_info for r in caplog.records)


def test_worker_errors_are_logged(qapp, regression_cache, tmp_path, caplog):
    import engine

    class BrokenGs(FakeGs):
        def run(self, *a, **k):
            raise ValueError("network exploded")
    t = engine.GeneratorThread(1, "Any", "Any", str(tmp_path / "out"), BrokenGs(), DummyContext(), DummyContext())
    t.run()
    record, = [r for r in caplog.records if r.name == "engine"]
    assert record.levelno == logging.ERROR and record.getMessage() == "Generation failed"
    assert record.exc_info and "network exploded" in str(record.exc_info[1])

    caplog.clear()
    with open(str(regression_cache / GENDER_FILE), "r+b") as f:
        f.seek(300)
        f.write(b"\xff\xff")
    engine.GeneratorThread(1, "Any", "Any", str(tmp_path / "out"), FakeGs(), DummyContext(), DummyContext()).run()
    record, = [r for r in caplog.records if r.name == "engine"]
    assert record.levelno == logging.ERROR and "doesn't match the expected checksum" in record.getMessage()
    assert not record.exc_info                         # a known problem: no traceback needed


# ── unexpected errors in the UI ────────────────────────────────────────────
SLOT_ERROR_SCRIPT = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
import log_setup
log_setup.configure(log_path=sys.argv[2])
reported = []
log_setup.set_error_reporter(lambda: reported.append(1))
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtCore import QTimer
app = QApplication([])
button = QPushButton()
def handler():
    raise ValueError("bug in a button handler")
button.clicked.connect(handler)
QTimer.singleShot(50, button.click)
QTimer.singleShot(800, lambda: (print("still running; reported", len(reported), flush=True), app.quit()))
sys.exit(app.exec_())
"""


def test_error_in_a_button_handler_is_logged_and_the_app_keeps_running(tmp_path):
    log_path = str(tmp_path / "app.log")
    r = subprocess.run([sys.executable, "-c", SLOT_ERROR_SCRIPT, PROJECT, log_path],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0                          # without the hook PyQt5 aborts (exit 127)
    assert "still running; reported 1" in r.stdout
    text = open(log_path, encoding="utf-8").read()
    assert "CRITICAL [app] Unexpected error" in text
    assert "ValueError: bug in a button handler" in text and "Traceback" in text
