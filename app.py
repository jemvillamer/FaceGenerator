import sys
import os
import subprocess
import tempfile
import numpy as np
from engine import ModelLoaderThread, GeneratorThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QProgressBar, QMessageBox, QFileDialog, QFrame,
    QSizePolicy, QLineEdit, QButtonGroup, QListView, QToolTip,
)
from PyQt5.QtCore import (
    Qt, QSize, QRectF, QPoint, QRect, QByteArray, QBuffer, QIODevice,
    QMimeData, QRegExp,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPainterPath, QColor, QFont,
    QRegExpValidator,
)
from PyQt5.QtSvg import QSvgRenderer

THUMB_VISIBLE = 6          # thumbnail slots shown before the "+N" tile
THUMB_SIZE = 68
MAX_IMAGES = 1000

# ----------------------------------------------------------------------
# Icons (Lucide-style outline SVGs, drawn at runtime so no asset files
# need to ship with the project)
# ----------------------------------------------------------------------
ICONS = {
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4.5 21a7.5 7.5 0 0 1 15 0"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "chevron_down": '<path d="m6 9 6 6 6-6"/>',
    "chevron_left": '<path d="m15 18-6-6 6-6"/>',
    "chevron_right": '<path d="m9 18 6-6-6-6"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
    "sparkle": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "dot": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="1.5"/>',
}


def _svg_markup(name, color, stroke):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def svg_pixmap(name, color, size, stroke=2.0, scale=2.0):
    """Render an icon to a crisp pixmap (size is in logical pixels)."""
    renderer = QSvgRenderer(QByteArray(_svg_markup(name, color, stroke).encode("utf-8")))
    px = int(size * scale)
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return pm


def make_icon(name, color, disabled_color=None, stroke=2.0, size=20):
    icon = QIcon()
    icon.addPixmap(svg_pixmap(name, color, size, stroke), QIcon.Normal)
    if disabled_color:
        icon.addPixmap(svg_pixmap(name, disabled_color, size, stroke), QIcon.Disabled)
    return icon


def _repolish(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ----------------------------------------------------------------------
# Stylesheet / paths / image helpers
# ----------------------------------------------------------------------
def prepare_icon_files():
    """QSS can only reference icons by file path, so write the dropdown
    chevron (1x + @2x) into a temp folder and return its path."""
    folder = os.path.join(tempfile.gettempdir(), "facegen_icons")
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, "chevron_down.png")
    svg_pixmap("chevron_down", "#9aa3b5", 16, stroke=2.0, scale=1).save(base, "PNG")
    svg_pixmap("chevron_down", "#9aa3b5", 16, stroke=2.0, scale=2).save(
        os.path.join(folder, "chevron_down@2x.png"), "PNG"
    )
    return base.replace("\\", "/")


def load_stylesheet():
    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            qss = f.read()
    except FileNotFoundError:
        print(f"[Warning] style.qss not found at {qss_path}. Running without stylesheet.")
        return ""
    return qss.replace("@CHEVRON_DOWN@", prepare_icon_files())


def default_output_dir():
    path = os.path.join(os.path.expanduser("~"), "FaceGenerator", "output")
    os.makedirs(path, exist_ok=True)
    return path


def array_to_qimage(img_array):
    arr = np.asarray(img_array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr)
    h, w, ch = arr.shape
    fmt = QImage.Format_RGBA8888 if ch == 4 else QImage.Format_RGB888
    return QImage(arr.data, w, h, ch * w, fmt).copy()


def array_to_pixmap(img_array, max_size=None):
    pixmap = QPixmap.fromImage(array_to_qimage(img_array))
    if max_size:
        pixmap = pixmap.scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


def _rounded(src, size, radius, scale=2.0):
    px = int(size * scale)
    out = QPixmap(px, px)
    out.fill(Qt.transparent)
    fitted = src.scaled(px, px, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    path = QPainterPath()
    path.addRoundedRect(0, 0, px, px, radius * scale, radius * scale)
    p.setClipPath(path)
    p.drawPixmap((px - fitted.width()) // 2, (px - fitted.height()) // 2, fitted)
    p.end()
    out.setDevicePixelRatio(scale)
    return out


def make_thumb_icon(src):
    """Thumbnail icon: full-bleed when idle, inset (ring gap) when selected."""
    inner = THUMB_SIZE - 4  # 2px border on each side
    off = _rounded(src, inner, 10)
    selected = QPixmap(int(inner * 2), int(inner * 2))
    selected.fill(Qt.transparent)
    selected.setDevicePixelRatio(2.0)
    inset = _rounded(src, inner - 6, 8)
    p = QPainter(selected)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawPixmap(3, 3, inset)  # logical px: centered inside the ring gap
    p.end()
    icon = QIcon()
    icon.addPixmap(off, QIcon.Normal, QIcon.Off)
    icon.addPixmap(selected, QIcon.Normal, QIcon.On)
    return icon


# ----------------------------------------------------------------------
# Custom widgets
# ----------------------------------------------------------------------
class ElidedLabel(QLabel):
    """Single-line label that shows 'C:\\Users\\name\\...' style truncation."""

    def __init__(self, text="", pad=28):
        super().__init__()
        self._pad = pad
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setText(text)

    def setText(self, text):
        self._full = text
        self.setToolTip(text)
        self._apply()

    def text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self):
        avail = max(self.width() - self._pad, 20)
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.ElideRight, avail))


class StepperEdit(QLineEdit):
    """Number field that lights up its parent box while focused."""

    def _set_focus_flag(self, on):
        parent = self.parentWidget()
        if parent is not None:
            parent.setProperty("focused", on)
            _repolish(parent)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._set_focus_flag(True)
        self.selectAll()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._set_focus_flag(False)


class PreviewCanvas(QFrame):
    """Dark canvas with a centered square card: placeholder or the image."""

    MARGIN = 50
    CARD_COLOR = QColor("#242833")
    CARD_RADIUS = 20

    def __init__(self):
        super().__init__()
        self.setObjectName("preview_canvas")
        self.setMinimumSize(260, 260)
        self._pixmap = None
        self._scaled = None
        self._icon_renderer = QSvgRenderer(
            QByteArray(_svg_markup("user", "#4a5266", 0.95).encode("utf-8"))
        )

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._scaled = None
        self.update()

    def clear_image(self):
        self.set_image(None)

    def _card_rect(self):
        side = max(min(self.width(), self.height()) - 2 * self.MARGIN, 140)
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self._card_rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(rect, self.CARD_RADIUS, self.CARD_RADIUS)
        p.fillPath(path, self.CARD_COLOR)

        if self._pixmap is not None and not self._pixmap.isNull():
            target = rect.size().toSize()
            if self._scaled is None or self._scaled.size() != target:
                self._scaled = self._pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            p.setClipPath(path)
            x = rect.x() + (rect.width() - self._scaled.width()) / 2
            y = rect.y() + (rect.height() - self._scaled.height()) / 2
            p.drawPixmap(int(x), int(y), self._scaled)
            return

        # placeholder: person glyph + hint text
        box = 150.0
        cx, cy = rect.center().x(), rect.center().y()
        self._icon_renderer.render(
            p, QRectF(cx - box / 2, cy - 13 - box / 2, box, box)
        )
        font = QFont(self.font())
        font.setPixelSize(14)
        p.setFont(font)
        p.setPen(QColor("#9aa3b5"))
        p.drawText(
            QRectF(rect.x(), cy + 62, rect.width(), 28),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "Generated face appears here, scaled to fit",
        )


# ----------------------------------------------------------------------
class FaceGeneratorApp(QMainWindow):
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

    # ------------------------------------------------------------------
    def _section_row(self, title, action_btn=None):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(title)
        lbl.setObjectName("section_heading")
        f = lbl.font()
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
        lbl.setFont(f)
        row.addWidget(lbl)
        row.addStretch(1)
        if action_btn:
            row.addWidget(action_btn)
        return row

    def _field_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("field_label")
        return lbl

    def initUI(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        # ── SIDEBAR ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(324)
        vl = QVBoxLayout(sidebar)
        vl.setContentsMargins(22, 25, 22, 24)
        vl.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(12)
        badge = QLabel()
        badge.setObjectName("app_icon_badge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setPixmap(svg_pixmap("user", "#ffffff", 20, stroke=2.0))
        header.addWidget(badge)
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("StyleGAN")
        title.setObjectName("app_title")
        subtitle = QLabel("Face Generator")
        subtitle.setObjectName("app_subtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)
        vl.addLayout(header)

        vl.addSpacing(20)

        # OUTPUT
        vl.addLayout(self._section_row("OUTPUT"))
        vl.addSpacing(14)
        vl.addWidget(self._field_label("Save to"))
        vl.addSpacing(4)
        row_out = QHBoxLayout()
        row_out.setSpacing(8)
        self.path_input = ElidedLabel(self._output_dir)
        self.path_input.setObjectName("path_input")
        btn_browse = QPushButton("  Browse")
        btn_browse.setObjectName("browse_btn")
        btn_browse.setIcon(make_icon("folder", "#8ab8ff", size=18))
        btn_browse.setIconSize(QSize(18, 18))
        btn_browse.setCursor(Qt.PointingHandCursor)
        btn_browse.clicked.connect(self.browse_output)
        row_out.addWidget(self.path_input, 1)
        row_out.addWidget(btn_browse)
        vl.addLayout(row_out)

        vl.addSpacing(16)
        vl.addWidget(self._field_label("Number of images"))
        vl.addSpacing(4)

        # one bordered control:  [ − | 20 | + ]
        stepper_box = QFrame()
        stepper_box.setObjectName("stepper_box")
        stepper_row = QHBoxLayout(stepper_box)
        stepper_row.setContentsMargins(0, 0, 0, 0)
        stepper_row.setSpacing(0)
        self.btn_minus = QPushButton("\u2212")
        self.btn_minus.setObjectName("stepper_btn")
        self.btn_minus.setProperty("side", "left")
        self.btn_minus.setCursor(Qt.PointingHandCursor)
        self.btn_minus.setAutoRepeat(True)
        self.btn_minus.clicked.connect(lambda: self._adjust_image_count(-1))
        self.stepper_input = StepperEdit(str(self._image_count))
        self.stepper_input.setObjectName("stepper_input")
        self.stepper_input.setAlignment(Qt.AlignCenter)
        self.stepper_input.setMaxLength(4)
        self.stepper_input.setValidator(QRegExpValidator(QRegExp(r"\d{0,4}"), self))
        self.stepper_input.setToolTip(f"Type a number (1\u2013{MAX_IMAGES})")
        self.stepper_input.editingFinished.connect(self._commit_image_count)
        self.stepper_input.returnPressed.connect(self.stepper_input.clearFocus)
        self.btn_plus = QPushButton("+")
        self.btn_plus.setObjectName("stepper_btn")
        self.btn_plus.setProperty("side", "right")
        self.btn_plus.setCursor(Qt.PointingHandCursor)
        self.btn_plus.setAutoRepeat(True)
        self.btn_plus.clicked.connect(lambda: self._adjust_image_count(1))
        stepper_row.addWidget(self.btn_minus)
        stepper_row.addWidget(self.stepper_input, 1)
        stepper_row.addWidget(self.btn_plus)
        vl.addWidget(stepper_box)

        vl.addSpacing(25)

        # FILTERS
        btn_reset = QPushButton("Reset")
        btn_reset.setObjectName("link_btn")
        btn_reset.setCursor(Qt.PointingHandCursor)
        btn_reset.clicked.connect(self.reset_filters)
        vl.addLayout(self._section_row("FILTERS", btn_reset))
        vl.addSpacing(16)

        vl.addWidget(self._field_label("Age range"))
        vl.addSpacing(4)
        self.age_combo = QComboBox()
        self.age_combo.setView(QListView())  # lets QSS style the popup items
        # (label shown to the user, exact value the engine already expects)
        for shown, value in [
            ("Any", "Any"),
            ("1\u201311 years old", "1-11 years old"),
            ("12\u201318 years old", "12-18 years old"),
            ("19\u201325 years old", "19-25 years old"),
            ("26\u201335 years old", "26-35 years old"),
            ("35\u201350 years old", "35-50 years old"),
            ("50+ years old", "50+ years old"),
        ]:
            self.age_combo.addItem(shown, value)
        self.age_combo.setCurrentIndex(self.age_combo.findData("35-50 years old"))
        self.age_combo.setCursor(Qt.PointingHandCursor)
        vl.addWidget(self.age_combo)

        vl.addSpacing(16)
        vl.addWidget(self._field_label("Gender"))
        vl.addSpacing(4)
        gender_wrap = QFrame()
        gender_wrap.setObjectName("gender_toggle")
        g_row = QHBoxLayout(gender_wrap)
        g_row.setContentsMargins(4, 4, 4, 4)
        g_row.setSpacing(0)
        self.gender_group = QButtonGroup(self)
        self.gender_group.setExclusive(True)
        for i, label in enumerate(["Any", "Female", "Male"]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            self.gender_group.addButton(btn, i)
            g_row.addWidget(btn, 1)
        self.gender_group.button(1).setChecked(True)
        vl.addWidget(gender_wrap)

        vl.addStretch(1)

        # Status card
        self.status_card = QFrame()
        self.status_card.setObjectName("status_card")
        status_layout = QVBoxLayout(self.status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(7)
        status_top = QHBoxLayout()
        status_top.setSpacing(8)
        self.status_icon = QLabel()
        self.status_icon.setObjectName("status_icon")
        self.status_icon.setFixedSize(18, 18)
        self.status_text = ElidedLabel("Loading model...", pad=2)
        self.status_text.setObjectName("status_text")
        status_top.addWidget(self.status_icon)
        status_top.addWidget(self.status_text, 1)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.setObjectName("open_folder_btn")
        self.open_folder_btn.setCursor(Qt.PointingHandCursor)
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        status_top.addWidget(self.open_folder_btn)
        status_layout.addLayout(status_top)
        self.status_subtext = QLabel("Please wait")
        self.status_subtext.setObjectName("status_subtext")
        status_layout.addWidget(self.status_subtext)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        vl.addWidget(self.status_card)

        vl.addSpacing(14)

        self.gen_btn = QPushButton("  Generate images")
        self.gen_btn.setObjectName("gen_btn")
        self.gen_btn.setIcon(make_icon("sparkle", "#ffffff", "#4a5266", size=20))
        self.gen_btn.setIconSize(QSize(20, 20))
        self.gen_btn.setEnabled(False)
        self.gen_btn.setCursor(Qt.PointingHandCursor)
        self.gen_btn.clicked.connect(self.generate_images)
        vl.addWidget(self.gen_btn)

        hl.addWidget(sidebar)

        # ── MAIN PANEL ─────────────────────────────────────────────────
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
        self.btn_prev = QPushButton()
        self.btn_prev.setObjectName("nav_btn")
        self.btn_prev.setIcon(make_icon("chevron_left", "#e8eaf0", "#4a5266", size=20))
        self.btn_prev.setIconSize(QSize(20, 20))
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.nav_label = QLabel("Image 0 of 0")
        self.nav_label.setObjectName("nav_counter")
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton()
        self.btn_next.setObjectName("nav_btn")
        self.btn_next.setIcon(make_icon("chevron_right", "#e8eaf0", "#4a5266", size=20))
        self.btn_next.setIconSize(QSize(20, 20))
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self.show_next_image)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.nav_label)
        nav.addWidget(self.btn_next)
        tb.addLayout(nav)
        tb.addStretch(1)
        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.btn_save_as = QPushButton("Save as...")
        self.btn_save_as.setObjectName("toolbar_btn")
        self.btn_save_as.setCursor(Qt.PointingHandCursor)
        self.btn_save_as.clicked.connect(self.save_current_as)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setObjectName("toolbar_btn")
        self.btn_copy.setCursor(Qt.PointingHandCursor)
        self.btn_copy.clicked.connect(self.copy_current_image)
        self.btn_delete = QPushButton()
        self.btn_delete.setObjectName("delete_btn")
        self.btn_delete.setIcon(make_icon("trash", "#ff9a9a", "#5b4350", size=18))
        self.btn_delete.setIconSize(QSize(18, 18))
        self.btn_delete.setToolTip("Delete")
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.clicked.connect(self.delete_current_image)
        actions.addWidget(self.btn_save_as)
        actions.addWidget(self.btn_copy)
        actions.addWidget(self.btn_delete)
        tb.addLayout(actions)
        pl.addWidget(toolbar)

        self.preview_canvas = PreviewCanvas()
        pl.addWidget(self.preview_canvas, 1)

        # Thumbnail strip: fixed slots that fill in live while generating
        thumb_wrap = QWidget()
        thumb_wrap.setObjectName("thumb_strip")
        self.thumb_layout = QHBoxLayout(thumb_wrap)
        self.thumb_layout.setContentsMargins(0, 0, 0, 0)
        self.thumb_layout.setSpacing(10)
        self.thumb_buttons = []
        for slot in range(THUMB_VISIBLE):
            btn = QPushButton()
            btn.setObjectName("thumb_btn")
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.setIconSize(QSize(THUMB_SIZE - 4, THUMB_SIZE - 4))
            btn.clicked.connect(lambda _checked=False, s=slot: self._on_thumb_clicked(s))
            self.thumb_layout.addWidget(btn)
            self.thumb_buttons.append(btn)
        self.thumb_overflow = QPushButton()
        self.thumb_overflow.setObjectName("thumb_overflow")
        self.thumb_overflow.setCursor(Qt.PointingHandCursor)
        self.thumb_overflow.clicked.connect(self._on_overflow_clicked)
        self.thumb_overflow.hide()
        self.thumb_layout.addWidget(self.thumb_overflow)
        self.thumb_layout.addStretch(1)
        pl.addWidget(thumb_wrap)

        hl.addWidget(preview, 1)
        self._set_status("loading", "Loading model...", "Please wait", show_check=False)

    # ------------------------------------------------------------------
    # Number-of-images control (buttons or typed)
    def _read_image_count(self):
        txt = self.stepper_input.text().strip()
        try:
            val = int(txt)
        except ValueError:
            return self._image_count
        return int(np.clip(val, 1, MAX_IMAGES))

    def _set_image_count(self, n):
        self._image_count = int(np.clip(n, 1, MAX_IMAGES))
        self.stepper_input.setText(str(self._image_count))

    def _commit_image_count(self):
        self._set_image_count(self._read_image_count())

    def _image_count_value(self):
        self._commit_image_count()
        return self._image_count

    def _adjust_image_count(self, delta):
        self._set_image_count(self._read_image_count() + delta)

    def _selected_gender(self):
        checked = self.gender_group.checkedButton()
        return checked.text() if checked else "Any"

    def reset_filters(self):
        self.age_combo.setCurrentIndex(0)
        self.gender_group.button(0).setChecked(True)
        self._set_image_count(20)

    # ------------------------------------------------------------------
    STATUS_ICONS = {
        "complete": ("check", "#7fe0a6"),
        "error": ("alert", "#ff9a9a"),
        "loading": ("clock", "#fbbf24"),
        "generating": ("clock", "#8ab8ff"),
        "idle": ("dot", "#9aa3b5"),
    }

    def _set_status(self, phase, title, subtitle="", show_check=True, progress=None):
        self._status_phase = phase
        for w in (self.status_card, self.status_icon, self.status_text):
            w.setProperty("status", phase)
            _repolish(w)
        name, color = self.STATUS_ICONS.get(phase, self.STATUS_ICONS["idle"])
        if phase == "complete" and not show_check:
            name, color = self.STATUS_ICONS["idle"]
        self.status_icon.setPixmap(svg_pixmap(name, color, 18, stroke=2.0))
        self.status_text.setText(title)
        self.status_subtext.setText(subtitle)
        self.status_subtext.setVisible(bool(subtitle) and phase in ("loading", "generating", "idle", "error"))
        self.open_folder_btn.setVisible(phase == "complete")
        if progress is not None:
            self.progress_bar.setValue(progress)
        self.progress_bar.setProperty("phase", "generating" if phase == "generating" else "idle")
        _repolish(self.progress_bar)

    def load_model_async(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_load_error)
        self.loader_thread.start()

    def on_model_loaded(self, Gs, session, graph):
        self.Gs = Gs
        self.tf_session = session
        self.tf_graph = graph
        self.gen_btn.setEnabled(True)
        self._set_status("idle", "Ready to generate", "Choose filters and click Generate images")

    def on_model_load_error(self, err):
        self._set_status("error", "Failed to load model", "Check console for details")
        QMessageBox.critical(self, "Model Load Error", f"Failed to load the model:\n\n{err}")

    def browse_output(self):
        start = self.path_input.text() or self._output_dir
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start)
        if dir_path:
            self._output_dir = dir_path
            self.path_input.setText(dir_path)

    def open_output_folder(self):
        path = self.path_input.text().strip() or self._output_dir
        if not os.path.isdir(path):
            return
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def generate_images(self):
        if self.Gs is None:
            QMessageBox.warning(self, "Not Ready", "Model is still loading. Please wait.")
            return

        num_images = self._image_count_value()
        age = self.age_combo.currentData() or self.age_combo.currentText()
        gender = self._selected_gender()
        out_dir = self.path_input.text().strip() or self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        self._output_dir = out_dir

        self._images.clear()
        self._current_index = 0
        self._total_count = num_images
        self._generating = True
        self._follow_latest = True
        self.progress_bar.setMaximum(num_images)
        self.progress_bar.setValue(0)
        self.gen_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.preview_canvas.clear_image()
        self._refresh_gallery()
        self._update_nav_and_toolbar()
        self._set_status(
            "generating",
            f"Generating {num_images} image{'s' if num_images > 1 else ''}...",
            "This may take a while on CPU",
            show_check=False,
            progress=0,
        )

        self.thread = GeneratorThread(
            num_images, age, gender, out_dir, self.Gs, self.tf_session, self.tf_graph
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.preview.connect(self.on_image_generated)
        self.thread.finished.connect(self.generation_finished)
        self.thread.error.connect(self.generation_error)
        self.thread.start()

    def on_image_generated(self, img_array):
        idx = len(self._images)
        path = os.path.join(self._output_dir, f"generated_{idx:04d}.png")
        arr = img_array.copy()
        icon = make_thumb_icon(array_to_pixmap(arr))
        self._images.append({"path": path, "array": arr, "icon": icon})
        if self._follow_latest:
            # show the newest image in the big preview right away
            self._show_image_at(idx)        # also refreshes the thumbnails
            self.preview_canvas.repaint()   # force a paint so every image is seen
        else:
            self._refresh_gallery()         # user is browsing: just fill the slot
            self._update_nav_and_toolbar()

    def update_progress(self, val):
        self.progress_bar.setValue(val)
        self._set_status(
            "generating",
            f"Generating {self._total_count} image{'s' if self._total_count > 1 else ''}...",
            f"{val} of {self._total_count} complete",
            show_check=False,
            progress=val,
        )

    def generation_finished(self, _msg):
        self._generating = False
        self.gen_btn.setEnabled(True)
        count = len(self._images)
        self.open_folder_btn.setEnabled(count > 0)
        if count:
            self._current_index = min(self._current_index, count - 1)
            self._show_image_at(self._current_index)
        self._refresh_gallery()
        self._set_status(
            "complete",
            f"Generated {count} image{'s' if count != 1 else ''}",
            "",
            show_check=True,
            progress=self.progress_bar.maximum(),
        )

    def generation_error(self, err):
        self._generating = False
        self.gen_btn.setEnabled(True)
        self._refresh_gallery()
        self._set_status("error", "Error during generation", "See details in the dialog")
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{err}")

    # ------------------------------------------------------------------
    def _show_image_at(self, index):
        if not self._images:
            self.preview_canvas.clear_image()
            return
        index = max(0, min(index, len(self._images) - 1))
        self._current_index = index
        self.preview_canvas.set_image(array_to_pixmap(self._images[index]["array"]))
        self._update_nav_and_toolbar()
        self._refresh_gallery()

    def show_previous_image(self):
        if not self._images:
            return
        self._follow_latest = False
        self._show_image_at((self._current_index - 1) % len(self._images))

    def show_next_image(self):
        if not self._images:
            return
        self._follow_latest = False
        self._show_image_at((self._current_index + 1) % len(self._images))

    def _update_nav_and_toolbar(self):
        n = len(self._images)
        if n:
            self.nav_label.setText(f"Image {self._current_index + 1} of {n}")
        else:
            self.nav_label.setText("Image 0 of 0")
        enabled = n > 0
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_save_as.setEnabled(enabled)
        self.btn_copy.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Thumbnail strip
    def _gallery_total(self):
        n = len(self._images)
        return max(n, self._total_count) if self._generating else n

    def _refresh_gallery(self):
        n = len(self._images)
        cur = self._current_index
        start = 0 if cur < THUMB_VISIBLE else cur - THUMB_VISIBLE + 1
        self._thumb_start = start
        for slot, btn in enumerate(self.thumb_buttons):
            idx = start + slot
            has = idx < n
            btn.setEnabled(has)
            btn.setCursor(Qt.PointingHandCursor if has else Qt.ArrowCursor)
            btn.setIcon(self._images[idx]["icon"] if has else QIcon())
            btn.setChecked(has and idx == cur)
        remaining = self._gallery_total() - (start + THUMB_VISIBLE)
        if remaining > 0:
            self.thumb_overflow.setText(f"+{remaining}")
            self.thumb_overflow.show()
        else:
            self.thumb_overflow.hide()

    def _on_thumb_clicked(self, slot):
        self._follow_latest = False
        idx = self._thumb_start + slot
        if idx < len(self._images):
            self._show_image_at(idx)
        else:
            self._refresh_gallery()

    def _on_overflow_clicked(self):
        self._follow_latest = False
        target = self._thumb_start + THUMB_VISIBLE
        if target < len(self._images):
            self._show_image_at(target)

    def _current_entry(self):
        if not self._images:
            return None
        return self._images[self._current_index]

    # ------------------------------------------------------------------
    def save_current_as(self):
        entry = self._current_entry()
        if not entry:
            return
        default_name = os.path.basename(entry["path"])
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", default_name, "PNG Images (*.png);;All Files (*)"
        )
        if path:
            array_to_pixmap(entry["array"]).save(path)

    def copy_current_image(self):
        entry = self._current_entry()
        if not entry:
            return
        try:
            image = array_to_qimage(entry["array"]).convertToFormat(QImage.Format_RGB32)
            png = QByteArray()
            buf = QBuffer(png)
            buf.open(QIODevice.WriteOnly)
            image.save(buf, "PNG")
            buf.close()
            mime = QMimeData()
            mime.setImageData(image)
            mime.setData("image/png", png)
            QApplication.clipboard().setMimeData(mime)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Copy Failed", str(e))
            return
        QToolTip.showText(
            self.btn_copy.mapToGlobal(QPoint(0, self.btn_copy.height() + 4)),
            "Copied to clipboard",
            self.btn_copy,
            QRect(),
            1500,
        )

    def delete_current_image(self):
        entry = self._current_entry()
        if not entry:
            return
        reply = QMessageBox.question(
            self,
            "Delete Image",
            "Remove this image from the gallery and delete the file on disk?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        path = entry["path"]
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Failed", str(e))
                return
        del self._images[self._current_index]
        if not self._images:
            self._current_index = 0
            self.preview_canvas.clear_image()
            self.open_folder_btn.setEnabled(False)
        else:
            self._current_index = min(self._current_index, len(self._images) - 1)
            self._show_image_at(self._current_index)
        self._refresh_gallery()
        self._update_nav_and_toolbar()
        if self._images and not self._generating:
            self._set_status(
                "complete",
                f"Generated {len(self._images)} image{'s' if len(self._images) != 1 else ''}",
                "",
                show_check=True,
                progress=len(self._images),
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = FaceGeneratorApp()
    ex.show()
    sys.exit(app.exec_())