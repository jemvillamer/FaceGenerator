"""Main window: wires the sidebar, preview panel and the behaviour mixins together."""
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

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
        self._total_count = 0
        self._generating = False
        self._follow_latest = True  # big preview follows each new image
        self._images = []  # list of {"path": str, "array": ndarray, "icon": QIcon}
        self._current_index = 0
        self._thumb_start = 0
        self._image_count = 20
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
        self._set_status("loading", "Loading model...", "Please wait", show_check=False)
