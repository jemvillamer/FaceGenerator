import sys
import os
import numpy as np
from engine import ModelLoaderThread, GeneratorThread
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSpinBox, QComboBox,
                             QPushButton, QProgressBar, QMessageBox, QFileDialog,
                             QFrame, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

def load_stylesheet():
    """Load the QSS stylesheet from style.qss next to this script."""
    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.qss')
    try:
        with open(qss_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[Warning] style.qss not found at {qss_path}. Running without stylesheet.")
        return ""


class FaceGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StyleGAN - Face Generator")
        self.setMinimumSize(860, 620)
        self.resize(1000, 680)
        self.Gs         = None
        self.tf_session = None
        self.tf_graph   = None
        self._total_count = 0

        self.setStyleSheet(load_stylesheet())
        self.initUI()
        self.load_model_async()

    # ------------------------------------------------------------------
    def _divider(self):
        f = QFrame()
        f.setObjectName("divider")
        f.setFrameShape(QFrame.HLine)
        return f

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
        sidebar.setFixedWidth(300)
        vl = QVBoxLayout(sidebar)
        vl.setContentsMargins(20, 24, 20, 20)
        vl.setSpacing(10)

        # Title
        title = QLabel("StyleGAN")
        title.setObjectName("app_title")
        title.setAlignment(Qt.AlignCenter)
        vl.addWidget(title)
        vl.addWidget(self._divider())

        # Output Directory
        lbl_out = QLabel("Output Directory:")
        lbl_out.setObjectName("field_label")
        vl.addWidget(lbl_out)
        row_out = QHBoxLayout()
        row_out.setSpacing(6)
        self.out_label = QLabel("output/")
        self.out_label.setObjectName("path_label")
        btn_browse = QPushButton("Browse")
        btn_browse.setObjectName("browse_btn")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self.browse_output)
        row_out.addWidget(self.out_label, 1)
        row_out.addWidget(btn_browse)
        vl.addLayout(row_out)

        # Number of Images
        lbl_num = QLabel("Number of Images:")
        lbl_num.setObjectName("field_label")
        vl.addWidget(lbl_num)
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 1000)
        self.num_spin.setValue(1)
        vl.addWidget(self.num_spin)

        vl.addWidget(self._divider())

        # Filters
        lbl_filters = QLabel("Filters")
        lbl_filters.setObjectName("section_title")
        lbl_filters.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl_filters)

        lbl_age = QLabel("Age Range:")
        lbl_age.setObjectName("field_label")
        vl.addWidget(lbl_age)
        self.age_combo = QComboBox()
        self.age_combo.addItems(["Any", "1-11 years old", "12-18 years old", "19-25 years old", "26-35 years old", "35-50 years old", "50+ years old"])
        vl.addWidget(self.age_combo)

        lbl_gender = QLabel("Gender:")
        lbl_gender.setObjectName("field_label")
        vl.addWidget(lbl_gender)
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Any", "Male", "Female"])
        vl.addWidget(self.gender_combo)



        vl.addWidget(self._divider())

        # Generate button
        self.gen_btn = QPushButton("Generate Images")
        self.gen_btn.setObjectName("gen_btn")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self.generate_images)
        vl.addWidget(self.gen_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        vl.addWidget(self.progress_bar)

        # Push status to bottom
        vl.addStretch(1)

        self.status_label = QLabel("Status: Loading model...")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        vl.addWidget(self.status_label)

        hl.addWidget(sidebar)

        # ── PREVIEW PANEL ──────────────────────────────────────────────
        preview = QWidget()
        preview.setObjectName("preview_area")
        pl = QVBoxLayout(preview)
        pl.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("No image generated yet")
        self.preview_label.setObjectName("preview_placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pl.addWidget(self.preview_label)

        hl.addWidget(preview, 1)

    # ------------------------------------------------------------------
    def load_model_async(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_load_error)
        self.loader_thread.start()

    def on_model_loaded(self, Gs, session, graph):
        self.Gs         = Gs
        self.tf_session = session
        self.tf_graph   = graph
        self.status_label.setText("Status: Model ready")
        self.status_label.setStyleSheet("color: #4CAF50; font-size:12px;")
        self.gen_btn.setEnabled(True)

    def on_model_load_error(self, err):
        self.status_label.setText("Status: Failed to load model")
        self.status_label.setStyleSheet("color: #f44336; font-size:12px;")
        QMessageBox.critical(self, "Model Load Error", f"Failed to load the model:\n\n{err}")

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.out_label.setText(dir_path)

    def generate_images(self):
        if self.Gs is None:
            QMessageBox.warning(self, "Not Ready", "Model is still loading. Please wait.")
            return

        num_images = self.num_spin.value()
        age        = self.age_combo.currentText()
        gender     = self.gender_combo.currentText()
        out_dir    = self.out_label.text()

        if out_dir in ("output/", ""):
            out_dir = os.path.join(os.getcwd(), "output")

        self._total_count = num_images
        self.progress_bar.setMaximum(num_images)
        self.progress_bar.setValue(0)
        self.gen_btn.setEnabled(False)
        self.status_label.setStyleSheet("color: #FFC107; font-size:12px;")
        self.status_label.setText(
            f"Status: Generating {num_images} image{'s' if num_images > 1 else ''}...")

        self.thread = GeneratorThread(
            num_images, age, gender, out_dir,
            self.Gs, self.tf_session, self.tf_graph)
        self.thread.progress.connect(self.update_progress)
        self.thread.preview.connect(self.update_preview)
        self.thread.finished.connect(self.generation_finished)
        self.thread.error.connect(self.generation_error)
        self.thread.start()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def update_preview(self, img_array):
        img_copy = np.ascontiguousarray(img_array)
        h, w, ch = img_copy.shape
        q_img = QImage(img_copy.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pw, ph = self.preview_label.width(), self.preview_label.height()
        self.preview_label.setPixmap(
            pixmap.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def generation_finished(self, msg):
        self.gen_btn.setEnabled(True)
        self.status_label.setStyleSheet("color: #4CAF50; font-size:12px;")
        self.status_label.setText(
            f"Status: Generated {self._total_count} image{'s' if self._total_count > 1 else ''}")

    def generation_error(self, err):
        self.gen_btn.setEnabled(True)
        self.status_label.setStyleSheet("color: #f44336; font-size:12px;")
        self.status_label.setText("Status: Error during generation")
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{err}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = FaceGeneratorApp()
    ex.show()
    sys.exit(app.exec_())
