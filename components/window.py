"""Main window: wires the sidebar, preview panel and the behaviour mixins together."""
import time

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox

import log_setup
from .constants import DEFAULT_IMAGE_COUNT
from .controls import ControlsMixin
from .gallery import GalleryMixin
from .generation import GenerationMixin
from .image_actions import ImageActionsMixin
from .paths import default_output_dir
from .preview_panel import build_preview_panel
from .sidebar import build_sidebar
from .status import StatusMixin
from .theme import load_stylesheet


class FaceGeneratorApp(ControlsMixin, StatusMixin, GenerationMixin, GalleryMixin,
                       ImageActionsMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StyleGAN Face Generator")
        self.setMinimumSize(900, 620)
        self.resize(1000, 720)
        self.Gs = None
        self.tf_session = None
        self.tf_graph = None
        self.generator_thread = None
        self._error_dialog_open = False
        self._last_error_dialog = float("-inf")
        self._total_count = 0
        self._generating = False
        self._cancelling = False
        self._follow_latest = True  # big preview follows each new image
        self._images = []  # list of {"path": str, "icon": QIcon}; full image is read from path
        self._current_index = 0
        self._thumb_start = 0
        self._image_count = DEFAULT_IMAGE_COUNT
        self._output_dir = default_output_dir()
        self._status_phase = "loading"

        self.setStyleSheet(load_stylesheet())
        self.initUI()
        self.load_model_async()
        self._refresh_gallery()
        self._update_nav_and_toolbar()

    def initUI(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        hl.addWidget(build_sidebar(self))
        hl.addWidget(build_preview_panel(self), 1)
        self._apply_filter_defaults()  # the same defaults Reset restores
        self._set_status("loading", "Loading model...", "Please wait", show_check=False)

    def report_unexpected_error(self):
        """Tell the user about an unexpected error (already logged by log_setup).
        Shown at most every few seconds and never stacked, so an error that
        repeats (e.g. on every repaint) can't flood the screen with dialogs."""
        now = time.monotonic()
        if self._error_dialog_open or now - self._last_error_dialog < 5.0:
            return
        self._error_dialog_open = True
        try:
            QMessageBox.critical(
                self, "Unexpected Error",
                "An unexpected error occurred." + log_setup.details_hint() +
                "\n\nThe app may be unstable; restarting is safest.")
        finally:
            self._error_dialog_open = False
            self._last_error_dialog = time.monotonic()

    def closeEvent(self, event):
        # Stop a running generation and wait for it: exiting with the worker
        # still running can kill it mid-save and leave a truncated PNG.
        # The wait is at most one render.
        thread = self.generator_thread
        if thread is not None and thread.isRunning():
            thread.requestInterruption()
            thread.wait()
        super().closeEvent(event)
