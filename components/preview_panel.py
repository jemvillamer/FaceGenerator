"""Main panel: toolbar (nav / Save as / Copy / Delete), preview canvas and thumbnail strip.

build_preview_panel(window) creates the widgets and stores the ones the rest of
the app needs on `window` (btn_prev, nav_label, preview_canvas, thumb_buttons, ...)."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QSize

from .constants import THUMB_VISIBLE, THUMB_SIZE
from .icons import make_icon
from .widgets import PreviewCanvas


def build_preview_panel(window):
    preview = QWidget()
    preview.setObjectName("preview_area")
    pl = QVBoxLayout(preview)
    pl.setContentsMargins(24, 20, 24, 20)
    pl.setSpacing(16)

    toolbar = QWidget()
    toolbar.setObjectName("preview_toolbar")
    tb = QHBoxLayout(toolbar)
    tb.setContentsMargins(0, 0, 0, 0)
    nav = QHBoxLayout()
    nav.setSpacing(10)
    window.btn_prev = QPushButton()
    window.btn_prev.setObjectName("nav_btn")
    window.btn_prev.setIcon(make_icon("chevron_left", "#e8eaf0", "#4a5266", size=20))
    window.btn_prev.setIconSize(QSize(20, 20))
    window.btn_prev.setCursor(Qt.PointingHandCursor)
    window.btn_prev.clicked.connect(window.show_previous_image)
    window.nav_label = QLabel("Image 0 of 0")
    window.nav_label.setObjectName("nav_counter")
    window.nav_label.setAlignment(Qt.AlignCenter)
    window.btn_next = QPushButton()
    window.btn_next.setObjectName("nav_btn")
    window.btn_next.setIcon(make_icon("chevron_right", "#e8eaf0", "#4a5266", size=20))
    window.btn_next.setIconSize(QSize(20, 20))
    window.btn_next.setCursor(Qt.PointingHandCursor)
    window.btn_next.clicked.connect(window.show_next_image)
    nav.addWidget(window.btn_prev)
    nav.addWidget(window.nav_label)
    nav.addWidget(window.btn_next)
    tb.addLayout(nav)
    tb.addStretch(1)
    actions = QHBoxLayout()
    actions.setSpacing(9)
    window.btn_save_as = QPushButton("Save as...")
    window.btn_save_as.setObjectName("toolbar_btn")
    window.btn_save_as.setCursor(Qt.PointingHandCursor)
    window.btn_save_as.clicked.connect(window.save_current_as)
    window.btn_copy = QPushButton("Copy")
    window.btn_copy.setObjectName("toolbar_btn")
    window.btn_copy.setCursor(Qt.PointingHandCursor)
    window.btn_copy.clicked.connect(window.copy_current_image)
    window.btn_delete = QPushButton()
    window.btn_delete.setObjectName("delete_btn")
    window.btn_delete.setIcon(make_icon("trash", "#ff9a9a", "#5b4350", size=18))
    window.btn_delete.setIconSize(QSize(18, 18))
    window.btn_delete.setToolTip("Delete")
    window.btn_delete.setCursor(Qt.PointingHandCursor)
    window.btn_delete.clicked.connect(window.delete_current_image)
    actions.addWidget(window.btn_save_as)
    actions.addWidget(window.btn_copy)
    actions.addWidget(window.btn_delete)
    tb.addLayout(actions)
    pl.addWidget(toolbar)

    window.preview_canvas = PreviewCanvas()
    pl.addWidget(window.preview_canvas, 1)

    # Thumbnail strip: fixed slots that fill in live while generating
    thumb_wrap = QWidget()
    thumb_wrap.setObjectName("thumb_strip")
    window.thumb_layout = QHBoxLayout(thumb_wrap)
    window.thumb_layout.setContentsMargins(0, 0, 0, 0)
    window.thumb_layout.setSpacing(10)
    window.thumb_buttons = []
    for slot in range(THUMB_VISIBLE):
        btn = QPushButton()
        btn.setObjectName("thumb_btn")
        btn.setCheckable(True)
        btn.setEnabled(False)
        btn.setIconSize(QSize(THUMB_SIZE - 4, THUMB_SIZE - 4))
        btn.clicked.connect(lambda _checked=False, s=slot: window._on_thumb_clicked(s))
        window.thumb_layout.addWidget(btn)
        window.thumb_buttons.append(btn)
    window.thumb_overflow = QPushButton()
    window.thumb_overflow.setObjectName("thumb_overflow")
    window.thumb_overflow.setCursor(Qt.PointingHandCursor)
    window.thumb_overflow.clicked.connect(window._on_overflow_clicked)
    window.thumb_overflow.hide()
    window.thumb_layout.addWidget(window.thumb_overflow)
    window.thumb_layout.addStretch(1)
    pl.addWidget(thumb_wrap)

    return preview
