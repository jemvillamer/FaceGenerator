import sys
import os
import pickle
import numpy as np
import PIL.Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSpinBox, QComboBox,
                             QPushButton, QProgressBar, QMessageBox, QFileDialog,
                             QFrame, QSizePolicy)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import tensorflow as tf

# ---------------------------------------------------------------------------
# CPU compatibility monkey-patches
# The pretrained StyleGAN pkl contains frozen graph nodes that use NCHW format,
# which TF does NOT support on CPU for several ops. We intercept these calls
# and transparently redirect them to NHWC by transposing around the op.
# This must happen BEFORE any model loading or TF graph construction.
# ---------------------------------------------------------------------------

_orig_conv2d_transpose = tf.nn.conv2d_transpose
def _cpu_conv2d_transpose(value, filters=None, output_shape=None, strides=None,
                           padding='SAME', data_format='NHWC', dilations=None,
                           name=None, filter=None):
    filters = filters if filters is not None else filter
    if data_format == 'NCHW':
        value = tf.transpose(value, [0, 2, 3, 1])
        if isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
        else:
            output_shape = tf.stack([output_shape[0], output_shape[2], output_shape[3], output_shape[1]])
        if strides is not None and len(strides) == 4:
            strides = [strides[0], strides[2], strides[3], strides[1]]
        result = _orig_conv2d_transpose(value, filters, output_shape, strides,
                                        padding, 'NHWC', dilations, name)
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_conv2d_transpose(value, filters, output_shape, strides,
                                   padding, data_format, dilations, name)
tf.nn.conv2d_transpose = _cpu_conv2d_transpose

_orig_depthwise_conv2d = tf.nn.depthwise_conv2d
def _cpu_depthwise_conv2d(input, filter, strides, padding, rate=None,
                           name=None, data_format=None):
    if data_format == 'NCHW':
        input = tf.transpose(input, [0, 2, 3, 1])
        strides = [strides[0], strides[2], strides[3], strides[1]]
        result = _orig_depthwise_conv2d(input, filter, strides, padding,
                                         rate=rate, name=name, data_format='NHWC')
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_depthwise_conv2d(input, filter, strides, padding,
                                   rate=rate, name=name, data_format=data_format)
tf.nn.depthwise_conv2d = _cpu_depthwise_conv2d

_orig_avg_pool = tf.nn.avg_pool
def _cpu_avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None, input=None):
    val = value if value is not None else input
    if data_format == 'NCHW':
        val = tf.transpose(val, [0, 2, 3, 1])
        ksize   = [ksize[0],   ksize[2],   ksize[3],   ksize[1]]
        strides = [strides[0], strides[2], strides[3], strides[1]]
        result = _orig_avg_pool(val, ksize, strides, padding, 'NHWC', name)
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_avg_pool(val, ksize, strides, padding, data_format, name)
tf.nn.avg_pool = _cpu_avg_pool

_orig_conv2d = tf.nn.conv2d
def _cpu_conv2d(input, filter=None, strides=None, padding=None, use_cudnn_on_gpu=True,
                data_format='NHWC', dilations=None, name=None, filters=None):
    filt = filter if filter is not None else filters
    if data_format == 'NCHW':
        input = tf.transpose(input, [0, 2, 3, 1])
        if strides is not None and len(strides) == 4:
            strides = [strides[0], strides[2], strides[3], strides[1]]
        result = _orig_conv2d(input, filt, strides, padding,
                              use_cudnn_on_gpu, 'NHWC', dilations, name)
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_conv2d(input, filt, strides, padding,
                        use_cudnn_on_gpu, data_format, dilations, name)
tf.nn.conv2d = _cpu_conv2d

# ---------------------------------------------------------------------------

import dnnlib
import dnnlib.tflib as tflib
import config


# Define latent directions stub
class LatentDirections:
    def __init__(self):
        # Stub for latent directions
        # Here you would load your latent direction vectors (e.g., numpy arrays)
        # self.age_direction = np.load('age_direction.npy')
        pass

    def apply_filters(self, latents, age, gender, ethnicity):
        """
        Stub function: modify latents based on filter values.
        You can plug in your own latent direction vectors here later.
        For example:
            if gender == 'Female':
                latents += self.gender_direction * weight
        """
        return latents


def load_model():
    """Initialize TF and load the pretrained StyleGAN model."""
    import gdown

    tflib.init_tf({"allow_soft_placement": True, "log_device_placement": False})

    os.makedirs(config.cache_dir, exist_ok=True)
    model_path = os.path.join(config.cache_dir, 'karras2019stylegan-ffhq-1024x1024.pkl')

    if not os.path.exists(model_path):
        print("Downloading pretrained model (~350MB)...")
        gdown.download(id='1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', output=model_path, quiet=False)

    print("Loading model...")
    with open(model_path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    print("Model loaded successfully.")
    return Gs, tf.get_default_session(), tf.get_default_graph()


class GeneratorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    preview  = pyqtSignal(np.ndarray)
    error    = pyqtSignal(str)

    def __init__(self, num_images, age, gender, ethnicity, output_dir, Gs, session, graph):
        super().__init__()
        self.num_images = num_images
        self.age        = age
        self.gender     = gender
        self.ethnicity  = ethnicity
        self.output_dir = output_dir
        self.Gs         = Gs
        self.session    = session
        self.graph      = graph

    def run(self):
        try:
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            latent_modifier = LatentDirections()
            os.makedirs(self.output_dir, exist_ok=True)

            with self.graph.as_default():
                with self.session.as_default():
                    for i in range(self.num_images):
                        if self.isInterruptionRequested():
                            break
                        rnd = np.random.RandomState(None)
                        latents = rnd.randn(1, self.Gs.input_shape[1])
                        latents = latent_modifier.apply_filters(
                            latents, self.age, self.gender, self.ethnicity)
                        images = self.Gs.run(latents, None, truncation_psi=0.7,
                                            randomize_noise=True, output_transform=fmt)
                        img_data = images[0]
                        png_filename = os.path.join(self.output_dir, f'generated_{i:04d}.png')
                        PIL.Image.fromarray(img_data, 'RGB').save(png_filename)
                        self.progress.emit(i + 1)
                        self.preview.emit(img_data)

            self.finished.emit(
                f"Successfully generated {self.num_images} images in:\n{self.output_dir}")
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


class ModelLoaderThread(QThread):
    """Loads the model in a background thread so the UI doesn't freeze."""
    model_loaded = pyqtSignal(object, object, object)
    error        = pyqtSignal(str)

    def run(self):
        try:
            Gs, session, graph = load_model()
            self.model_loaded.emit(Gs, session, graph)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


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
        self.age_combo.addItems(["Any", "Child", "Young Adult", "Middle Aged", "Senior"])
        vl.addWidget(self.age_combo)

        lbl_gender = QLabel("Gender:")
        lbl_gender.setObjectName("field_label")
        vl.addWidget(lbl_gender)
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Any", "Male", "Female"])
        vl.addWidget(self.gender_combo)

        lbl_eth = QLabel("Ethnicity:")
        lbl_eth.setObjectName("field_label")
        vl.addWidget(lbl_eth)
        self.eth_combo = QComboBox()
        self.eth_combo.addItems(["Any", "Asian", "Black", "White", "Hispanic", "Indian"])
        vl.addWidget(self.eth_combo)

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
        ethnicity  = self.eth_combo.currentText()
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
            num_images, age, gender, ethnicity, out_dir,
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
