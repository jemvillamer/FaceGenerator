"""engine.py (Qt threads) and tf_compat.py. Imports TensorFlow (~5 s)."""
import os

import numpy as np
import pytest

import downloads
import engine
import generator
import latents
from conftest import GENDER_FILE, DummyContext, FakeGs


def test_previous_names_are_still_exported():
    for name in ["AgeBucket", "AGE_BUCKETS", "AGE_BUCKETS_BY_KEY", "AGE_RANGES", "AGE_RESPONSE",
                 "DEFAULT_TRUNCATION_PSI", "LatentDirections", "_AgeEstimator", "next_output_index",
                 "load_model", "OUTPUT_TRANSFORM", "GeneratorThread", "ModelLoaderThread"]:
        assert hasattr(engine, name), name
    assert engine.LatentDirections is latents.LatentDirections
    assert engine._AgeEstimator is generator.AgeEstimator


def run_thread(qapp, out_dir, n, gs=None, stop_after=None):
    """Run GeneratorThread.run() synchronously and record its signals."""
    events = []
    t = engine.GeneratorThread(n, "Any", "Any", str(out_dir), gs or FakeGs(), DummyContext(), DummyContext())
    t.progress.connect(lambda v: events.append(("progress", v)))
    t.preview.connect(lambda img, path: events.append(("preview", os.path.basename(path))))
    t.completed.connect(lambda saved, cancelled: events.append(("completed", saved, cancelled)))
    t.error.connect(lambda msg: events.append(("error", msg)))
    if stop_after is not None:
        t.isInterruptionRequested = lambda: sum(e[0] == "progress" for e in events) >= stop_after
    t.run()
    return events


def test_generator_thread_signal_order(qapp, regression_cache, tmp_path):
    assert run_thread(qapp, tmp_path / "out", 2) == [
        ("progress", 1), ("preview", "generated_0000.png"),
        ("progress", 2), ("preview", "generated_0001.png"),
        ("completed", 2, False)]


def test_generator_thread_reports_cancel(qapp, regression_cache, tmp_path):
    events = run_thread(qapp, tmp_path / "out", 5, stop_after=2)
    assert events[-1] == ("completed", 2, True)


def test_checksum_error_is_shown_as_a_plain_message(qapp, regression_cache, tmp_path):
    with open(str(regression_cache / GENDER_FILE), "r+b") as f:
        f.seek(300)
        f.write(b"\xff\xff")
    (kind, msg), = run_thread(qapp, tmp_path / "out", 1)
    assert kind == "error"
    assert "doesn't match the expected checksum" in msg and "Traceback" not in msg


def test_other_errors_include_the_traceback(qapp, regression_cache, tmp_path):
    class BrokenGs(FakeGs):
        def run(self, *a, **k):
            raise ValueError("network exploded")
    (kind, msg), = run_thread(qapp, tmp_path / "out", 1, gs=BrokenGs())
    assert kind == "error" and "Traceback" in msg and "network exploded" in msg


def test_model_loader_refuses_a_tampered_model(qapp, tmp_cache, monkeypatch):
    import pickle
    calls = []
    monkeypatch.setattr(pickle, "load", lambda f: calls.append(f))
    open(str(tmp_cache / downloads.MODEL_FILE), "wb").write(b"cos\nsystem\n(S'echo pwned'\ntR.")
    messages = []
    t = engine.ModelLoaderThread()
    t.error.connect(messages.append)
    t.run()
    assert len(messages) == 1 and "doesn't match the expected checksum" in messages[0]
    assert calls == []  # never unpickled


def test_tf_compat_install_is_idempotent():
    import tensorflow as tf
    import tf_compat
    names = ("conv2d", "conv2d_transpose", "depthwise_conv2d", "avg_pool")
    before = {n: getattr(tf.nn, n) for n in names}
    assert all(getattr(f, "_facegen_cpu_patch", False) for f in before.values())
    tf_compat.install()
    tf_compat.install()
    assert all(getattr(tf.nn, n) is before[n] for n in names)
