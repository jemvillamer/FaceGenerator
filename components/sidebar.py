"""Left sidebar: output, filters, status card and the Generate button.

build_sidebar(window) creates the widgets and stores the ones the rest of the
app needs on `window` (path_input, stepper_input, age_combo, gen_btn, ...)."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QProgressBar, QFrame, QButtonGroup, QListView,
)
from PyQt5.QtCore import Qt, QSize, QRegExp
from PyQt5.QtGui import QFont, QRegExpValidator

from .constants import MAX_IMAGES
from .icons import svg_pixmap, make_icon
from .widgets import ElidedLabel, StepperEdit


def _section_row(title, action_btn=None):
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


def _field_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("field_label")
    return lbl


def build_sidebar(window):
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
    vl.addLayout(_section_row("OUTPUT"))
    vl.addSpacing(14)
    vl.addWidget(_field_label("Save to"))
    vl.addSpacing(4)
    row_out = QHBoxLayout()
    row_out.setSpacing(8)
    window.path_input = ElidedLabel(window._output_dir)
    window.path_input.setObjectName("path_input")
    btn_browse = QPushButton("  Browse")
    btn_browse.setObjectName("browse_btn")
    btn_browse.setIcon(make_icon("folder", "#8ab8ff", size=18))
    btn_browse.setIconSize(QSize(18, 18))
    btn_browse.setCursor(Qt.PointingHandCursor)
    btn_browse.clicked.connect(window.browse_output)
    row_out.addWidget(window.path_input, 1)
    row_out.addWidget(btn_browse)
    vl.addLayout(row_out)

    vl.addSpacing(16)
    vl.addWidget(_field_label("Number of images"))
    vl.addSpacing(4)

    # one bordered control:  [ − | 20 | + ]
    stepper_box = QFrame()
    stepper_box.setObjectName("stepper_box")
    stepper_row = QHBoxLayout(stepper_box)
    stepper_row.setContentsMargins(0, 0, 0, 0)
    stepper_row.setSpacing(0)
    window.btn_minus = QPushButton("\u2212")
    window.btn_minus.setObjectName("stepper_btn")
    window.btn_minus.setProperty("side", "left")
    window.btn_minus.setCursor(Qt.PointingHandCursor)
    window.btn_minus.setAutoRepeat(True)
    window.btn_minus.clicked.connect(lambda: window._adjust_image_count(-1))
    window.stepper_input = StepperEdit(str(window._image_count))
    window.stepper_input.setObjectName("stepper_input")
    window.stepper_input.setAlignment(Qt.AlignCenter)
    window.stepper_input.setMaxLength(4)
    window.stepper_input.setValidator(QRegExpValidator(QRegExp(r"\d{0,4}"), window))
    window.stepper_input.setToolTip(f"Type a number (1\u2013{MAX_IMAGES})")
    window.stepper_input.editingFinished.connect(window._commit_image_count)
    window.stepper_input.returnPressed.connect(window.stepper_input.clearFocus)
    window.btn_plus = QPushButton("+")
    window.btn_plus.setObjectName("stepper_btn")
    window.btn_plus.setProperty("side", "right")
    window.btn_plus.setCursor(Qt.PointingHandCursor)
    window.btn_plus.setAutoRepeat(True)
    window.btn_plus.clicked.connect(lambda: window._adjust_image_count(1))
    stepper_row.addWidget(window.btn_minus)
    stepper_row.addWidget(window.stepper_input, 1)
    stepper_row.addWidget(window.btn_plus)
    vl.addWidget(stepper_box)

    vl.addSpacing(25)

    # FILTERS
    btn_reset = QPushButton("Reset")
    btn_reset.setObjectName("link_btn")
    btn_reset.setCursor(Qt.PointingHandCursor)
    btn_reset.clicked.connect(window.reset_filters)
    vl.addLayout(_section_row("FILTERS", btn_reset))
    vl.addSpacing(16)

    vl.addWidget(_field_label("Age range"))
    vl.addSpacing(4)
    window.age_combo = QComboBox()
    window.age_combo.setView(QListView())  # lets QSS style the popup items
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
        window.age_combo.addItem(shown, value)
    window.age_combo.setCurrentIndex(window.age_combo.findData("35-50 years old"))
    window.age_combo.setCursor(Qt.PointingHandCursor)
    vl.addWidget(window.age_combo)

    vl.addSpacing(16)
    vl.addWidget(_field_label("Gender"))
    vl.addSpacing(4)
    gender_wrap = QFrame()
    gender_wrap.setObjectName("gender_toggle")
    g_row = QHBoxLayout(gender_wrap)
    g_row.setContentsMargins(4, 4, 4, 4)
    g_row.setSpacing(0)
    window.gender_group = QButtonGroup(window)
    window.gender_group.setExclusive(True)
    for i, label in enumerate(["Any", "Female", "Male"]):
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        window.gender_group.addButton(btn, i)
        g_row.addWidget(btn, 1)
    window.gender_group.button(1).setChecked(True)
    vl.addWidget(gender_wrap)

    vl.addStretch(1)

    # Status card
    window.status_card = QFrame()
    window.status_card.setObjectName("status_card")
    status_layout = QVBoxLayout(window.status_card)
    status_layout.setContentsMargins(14, 12, 14, 12)
    status_layout.setSpacing(7)
    status_top = QHBoxLayout()
    status_top.setSpacing(8)
    window.status_icon = QLabel()
    window.status_icon.setObjectName("status_icon")
    window.status_icon.setFixedSize(18, 18)
    window.status_text = ElidedLabel("Loading model...", pad=2)
    window.status_text.setObjectName("status_text")
    status_top.addWidget(window.status_icon)
    status_top.addWidget(window.status_text, 1)
    window.open_folder_btn = QPushButton("Open folder")
    window.open_folder_btn.setObjectName("open_folder_btn")
    window.open_folder_btn.setCursor(Qt.PointingHandCursor)
    window.open_folder_btn.setEnabled(False)
    window.open_folder_btn.clicked.connect(window.open_output_folder)
    status_top.addWidget(window.open_folder_btn)
    status_layout.addLayout(status_top)
    window.status_subtext = QLabel("Please wait")
    window.status_subtext.setObjectName("status_subtext")
    status_layout.addWidget(window.status_subtext)
    window.progress_bar = QProgressBar()
    window.progress_bar.setTextVisible(False)
    window.progress_bar.setValue(0)
    status_layout.addWidget(window.progress_bar)
    vl.addWidget(window.status_card)

    vl.addSpacing(14)

    window.gen_btn = QPushButton("  Generate images")
    window.gen_btn.setObjectName("gen_btn")
    window.gen_btn.setIcon(make_icon("sparkle", "#ffffff", "#4a5266", size=20))
    window.gen_btn.setIconSize(QSize(20, 20))
    window.gen_btn.setEnabled(False)
    window.gen_btn.setCursor(Qt.PointingHandCursor)
    window.gen_btn.clicked.connect(window.generate_images)
    vl.addWidget(window.gen_btn)

    return sidebar
