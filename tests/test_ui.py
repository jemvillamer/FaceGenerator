"""The window, offscreen: generation runs on a real thread against FakeGs."""
import os
import time

import numpy as np
import PIL.Image
import pytest
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from components.constants import DEFAULT_AGE, DEFAULT_GENDER, DEFAULT_IMAGE_COUNT, GENDERS
from conftest import DummyContext, FakeEstimator, FakeGs, pump_until
from latents import AGE_BUCKETS


def filter_state(w):
    """What the filters are set to: (age, gender sent to the engine, gender button
    shown as selected, image count, count shown in the stepper)."""
    return (w.age_combo.currentData(), w._selected_gender(), w.gender_group.checkedId(),
            w._image_count, w.stepper_input.text())


class NoLoad(QThread):
    """Replaces model loading: the tests set window.Gs themselves."""
    model_loaded = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def run(self):
        pass


@pytest.fixture
def window(qapp, regression_cache, tmp_path, monkeypatch):
    import components.generation as gen
    monkeypatch.setattr(gen, "ModelLoaderThread", NoLoad)
    from components.window import FaceGeneratorApp
    w = FaceGeneratorApp()
    w.startup_state = filter_state(w)   # before the fixture changes anything
    w.out = tmp_path / "out"
    w.out.mkdir()
    w._output_dir = str(w.out)
    w.path_input.setText(str(w.out))
    w.Gs, w.tf_session, w.tf_graph = FakeGs(size=32, delay=0.05), DummyContext(), DummyContext()
    w.gen_btn.setEnabled(True)
    w.age_combo.setCurrentIndex(0)             # Any
    w.gender_group.button(0).setChecked(True)  # Any
    yield w
    t = w.generator_thread
    if t is not None and t.isRunning():
        t.requestInterruption()
        t.wait()
    w.close()
    w.deleteLater()
    qapp.processEvents()


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text))))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    return shown


def start(w, n):
    w._set_image_count(n)
    w.on_gen_btn_clicked()


def wait_done(qapp, w):
    pump_until(qapp, lambda: not w._generating and not w.generator_thread.isRunning())


def to_array(img):
    img = img.convertToFormat(QImage.Format_RGB888)
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    rows = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())
    return rows[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy()


def file_array(path):
    return np.array(PIL.Image.open(path))


# ── controls ───────────────────────────────────────────────────────────────
def test_age_dropdown_is_built_from_the_bucket_table(window):
    items = [(window.age_combo.itemText(i), window.age_combo.itemData(i))
             for i in range(window.age_combo.count())]
    assert items == [("Any", "Any")] + [(b.label, b.key) for b in AGE_BUCKETS]


def test_startup_uses_the_filter_defaults(window):
    assert window.startup_state == (DEFAULT_AGE, DEFAULT_GENDER, GENDERS.index(DEFAULT_GENDER),
                                    DEFAULT_IMAGE_COUNT, str(DEFAULT_IMAGE_COUNT))
    assert window.startup_state[:2] == ("Any", "Any")   # no filters until the user picks some


def test_reset_restores_exactly_the_startup_state(window):
    window.age_combo.setCurrentIndex(window.age_combo.findData("50+ years old"))
    window.gender_group.button(GENDERS.index("Male")).setChecked(True)
    window._set_image_count(137)
    assert filter_state(window) != window.startup_state
    window.reset_filters()
    assert filter_state(window) == window.startup_state


def test_defaults_are_valid_choices(window):
    assert window.age_combo.findData(DEFAULT_AGE) >= 0
    assert DEFAULT_GENDER in GENDERS


def test_gender_sent_to_the_engine_does_not_depend_on_button_labels(qapp, window):
    for i, value in enumerate(GENDERS):
        window.gender_group.button(i).setText(f"Label {i}")   # e.g. renamed or translated
        window.gender_group.button(i).setChecked(True)
        assert window._selected_gender() == value
    window.gender_group.button(GENDERS.index("Female")).setChecked(True)
    start(window, 1)
    assert window.generator_thread.gender == "Female"
    wait_done(qapp, window)


def test_a_run_locks_browse_and_turns_generate_into_cancel(qapp, window):
    start(window, 3)
    assert not window.btn_browse.isEnabled()
    assert window.gen_btn.property("mode") == "cancel" and window.gen_btn.isEnabled()
    wait_done(qapp, window)
    assert window.btn_browse.isEnabled()
    assert window.gen_btn.property("mode") == "generate"
    assert window.status_text.text() == "Generated 3 images"


# ── gallery paths and memory ───────────────────────────────────────────────
def test_deleting_mid_run_removes_the_right_file(qapp, window, warnings):
    emitted = []
    original = window.on_image_generated

    def hooked(arr, path):
        emitted.append(arr.copy())
        original(arr, path)
        if len(window._images) == 3 and len(emitted) == 3:
            window._show_image_at(1)
            window.delete_current_image()   # deletes generated_0001 mid-run
    window.on_image_generated = hooked
    start(window, 6)
    wait_done(qapp, window)

    assert not (window.out / "generated_0001.png").exists()
    expected = emitted[:1] + emitted[2:]
    assert len(window._images) == 5
    for entry, arr in zip(window._images, expected):
        assert np.array_equal(file_array(entry["path"]), arr)

    last = window._images[-1]["path"]
    window._show_image_at(len(window._images) - 1)
    window.delete_current_image()
    assert not os.path.exists(last) and len(os.listdir(str(window.out))) == 4


def test_gallery_keeps_only_paths_and_thumbnails(qapp, window):
    start(window, 3)
    wait_done(qapp, window)
    assert all(set(entry) == {"path", "icon"} for entry in window._images)


def test_preview_shows_each_image_from_its_file(qapp, window):
    start(window, 3)
    wait_done(qapp, window)
    for i, entry in enumerate(window._images):
        window._show_image_at(i)
        assert np.array_equal(to_array(window.preview_canvas._pixmap.toImage()), file_array(entry["path"]))


# ── copy and save as ───────────────────────────────────────────────────────
def test_copy_puts_the_current_image_on_the_clipboard(qapp, window, warnings):
    start(window, 2)
    wait_done(qapp, window)
    window._show_image_at(0)
    window.copy_current_image()
    assert np.array_equal(to_array(QApplication.clipboard().image()), file_array(window._images[0]["path"]))
    assert warnings == []


def test_save_as(qapp, window, warnings, monkeypatch, tmp_path):
    start(window, 1)
    wait_done(qapp, window)
    target = {}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target["path"], "")))
    source = file_array(window._images[0]["path"])

    target["path"] = str(tmp_path / "copy.png")
    window.save_current_as()
    assert np.array_equal(file_array(target["path"]), source) and warnings == []

    target["path"] = str(tmp_path / "copy.jpg")
    window.save_current_as()
    assert PIL.Image.open(target["path"]).format == "JPEG" and warnings == []

    target["path"] = str(tmp_path / "no_extension")
    window.save_current_as()
    assert warnings[-1][0] == "Save Failed" and not os.path.exists(target["path"])

    target["path"] = str(tmp_path / "missing_folder" / "x.png")
    window.save_current_as()
    assert warnings[-1][0] == "Save Failed"

    warnings.clear()
    target["path"] = ""          # dialog cancelled
    window.save_current_as()
    assert warnings == []


def test_image_file_deleted_outside_the_app(qapp, window, warnings, monkeypatch, tmp_path):
    from components.gallery import MISSING_FILE_TEXT
    start(window, 3)
    wait_done(qapp, window)
    os.remove(window._images[1]["path"])
    window._show_image_at(1)
    assert window.preview_canvas._pixmap is None
    assert window.preview_canvas._message == MISSING_FILE_TEXT

    window.copy_current_image()
    assert warnings[-1][0] == "Copy Failed"
    target = str(tmp_path / "from_missing.png")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target, "")))
    window.save_current_as()
    assert warnings[-1][0] == "Save Failed" and not os.path.exists(target)

    window._show_image_at(2)
    assert window.preview_canvas._pixmap is not None
    window._show_image_at(1)
    warnings.clear()
    window.delete_current_image()
    assert len(window._images) == 2 and warnings == []


# ── error dialogs ──────────────────────────────────────────────────────────
def test_unexpected_error_dialog_is_throttled_and_never_stacked(window, monkeypatch):
    import log_setup
    monkeypatch.setattr(log_setup, "active_log_file", lambda: "C:/project/logs/facegen.log")
    shown = []

    def critical(parent, title, text):
        shown.append(text)
        window.report_unexpected_error()     # another error while the dialog is open
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(critical))
    window.report_unexpected_error()
    window.report_unexpected_error()         # repeats right away
    assert len(shown) == 1
    assert "C:/project/logs/facegen.log" in shown[0] and "restarting is safest" in shown[0]


def test_error_dialogs_point_to_the_log_file(window, monkeypatch):
    import log_setup
    monkeypatch.setattr(log_setup, "active_log_file", lambda: "C:/project/logs/facegen.log")
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda p, title, text: shown.append(text)))
    window.on_model_load_error("model broke")
    assert window.status_subtext.text() == "Details in logs\\facegen.log"
    window.generator_thread = None
    window._generating = True
    window.generation_error("generation broke")
    assert all("Details were saved to:\nC:/project/logs/facegen.log" in t for t in shown) and len(shown) == 2


# ── cancel and close ───────────────────────────────────────────────────────
def test_cancel_stops_the_run_early(qapp, window):
    start(window, 6)
    pump_until(qapp, lambda: len(window._images) >= 2)
    window.on_gen_btn_clicked()                       # Cancel
    assert not window.gen_btn.isEnabled() and "Stopping" in window.gen_btn.text()
    assert window.status_subtext.text() == "Stopping after the current image"
    window.on_gen_btn_clicked()                       # a second click is harmless
    wait_done(qapp, window)

    saved = len(window._images)
    assert 2 <= saved < 6
    assert len(os.listdir(str(window.out))) == saved
    assert window.status_text.text() == f"Stopped at {saved} of 6"
    assert window.gen_btn.property("mode") == "generate" and window.gen_btn.isEnabled()

    start(window, 2)                                  # the next run works normally
    wait_done(qapp, window)
    assert window.status_text.text() == "Generated 2 images"


def test_cancel_during_age_retries_is_quick(qapp, window, use_estimator):
    use_estimator(FakeEstimator(-100.0))              # every attempt is rejected
    window.Gs = FakeGs(size=32, delay=0.2)
    window.age_combo.setCurrentIndex(window.age_combo.findData("26-35 years old"))
    start(window, 3)
    pump_until(qapp, lambda: len(window.Gs.calls) >= 1)
    t0 = time.time()
    window.on_gen_btn_clicked()
    wait_done(qapp, window)
    assert time.time() - t0 < 2 * 0.2 + 0.3           # not the full 5 x 0.2 s of retries
    assert os.listdir(str(window.out)) == []


def test_closing_mid_run_finishes_the_current_image(qapp, window):
    start(window, 20)
    pump_until(qapp, lambda: len(window._images) >= 1)
    window.close()
    assert not window.generator_thread.isRunning()
    files = os.listdir(str(window.out))
    assert 1 <= len(files) < 20
    for name in files:
        PIL.Image.open(str(window.out / name)).load()  # every file is a complete PNG
