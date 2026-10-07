import os
import pickle
import numpy as np
import PIL.Image
from PyQt5.QtCore import QThread, pyqtSignal
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


# ---------------------------------------------------------------------------
# Age range definitions  (lo, hi) in real years
# ---------------------------------------------------------------------------
AGE_RANGES = {
    "1-11 years old":  ( 1, 11),
    "12-18 years old": (12, 18),
    "19-25 years old": (19, 25),
    "26-35 years old": (26, 35),
    "35-50 years old": (36, 50),
    "50+ years old":   (51, 75),
}

# Layers used for age editing in W+ space (coarse+mid for 1024px StyleGAN v1)
# StyleGAN v1 @ 1024px has 18 style layers (0..17). We edit 0-11 (coarse+mid).
AGE_EDIT_LAYERS = list(range(12))   # indices 0-11


class LatentDirections:
    """
    Handles age and gender edits in Z space.

    Age direction priority:
      1. cache/age_regression.npz  — calibrated ridge-regression direction
         (run scripts/fit_age_direction.py once to produce this).
      2. Fallback: InterFaceGAN SVM boundary (less accurate, still useful
         as a starting point until the regression file is ready).
    """

    # ── construction ───────────────────────────────────────────────────────
    def __init__(self):
        self.gender_boundary = self._load_or_download('stylegan_ffhq_gender_boundary.npy')
        self.gender_boundary /= np.linalg.norm(self.gender_boundary)

        reg_path = os.path.join(config.cache_dir, 'age_regression.npz')
        if os.path.exists(reg_path):
            data = np.load(reg_path)
            self.age_coef = data['age_coef'].astype('float64')
            self.age_bias = float(data['age_bias'])
            self.age_dir  = data['age_dir'].astype('float64')
            self._age_mode = 'regression'
            print("[LatentDirections] Using calibrated ridge-regression age direction.")
        else:
            # Fallback: InterFaceGAN SVM boundary.
            # We stay in the simple offset world here — the regression path
            # (_set_age) is only used when age_regression.npz exists.
            raw = self._load_or_download('stylegan_ffhq_age_boundary.npy')
            raw = raw.astype('float64').ravel()
            raw /= np.linalg.norm(raw)
            self.age_coef = None   # signals: use _apply_age_svm() instead
            self.age_bias = None
            self.age_dir  = raw    # unit-norm boundary vector
            self._age_mode = 'svm_fallback'
            # Safe per-range scalars (empirically tuned, max 3.0 to stay in-distribution)
            self._svm_offsets = {
                "1-11 years old":  -2.5,
                "12-18 years old": -1.2,
                "19-25 years old": -0.3,
                "26-35 years old":  0.4,
                "35-50 years old":  1.5,
                "50+ years old":    3.5,   # raised: model skews young
            }
            print("[LatentDirections] age_regression.npz not found — using "
                  "InterFaceGAN SVM boundary as fallback. "
                  "Run scripts/fit_age_direction.py for better accuracy.")

        # Orthogonalize age direction against gender to reduce entanglement
        g = self.gender_boundary.ravel()
        d = self.age_dir.ravel()
        d = d - (d @ g) / (g @ g) * g
        if np.linalg.norm(d) > 1e-8:
            d /= np.linalg.norm(d)
        self.age_dir = d
        # age_coef must also be updated so predicted age is consistent
        if self._age_mode == 'regression':
            c = self.age_coef.ravel()
            c = c - (c @ g) / (g @ g) * g
            self.age_coef = c

    # ── helpers ─────────────────────────────────────────────────────────────
    def _load_or_download(self, filename):
        import urllib.request
        os.makedirs(config.cache_dir, exist_ok=True)
        filepath = os.path.join(config.cache_dir, filename)
        if not os.path.exists(filepath):
            url = (f"https://raw.githubusercontent.com/genforce/interfacegan"
                   f"/master/boundaries/{filename}")
            print(f"[LatentDirections] Downloading {filename}…")
            urllib.request.urlretrieve(url, filepath)
        return np.load(filepath)

    def _predict_age(self, z):
        """Predict age for a single z vector (shape (512,))."""
        return float(z.ravel() @ self.age_coef + self.age_bias)

    def _set_age(self, z, target_age, max_shift=4.0):
        """
        Move z so that its predicted age equals target_age.
        Only called in regression mode. Clamps the shift to ±max_shift
        so the latent never leaves the model's training distribution.
        Extreme age buckets pass a higher max_shift (up to 7.0).
        """
        z = z.astype('float64')
        current = float(z.ravel() @ self.age_coef + self.age_bias)
        delta   = float(np.clip(target_age - current, -max_shift, max_shift))
        return z + delta * self.age_dir.reshape(1, -1)

    def apply_filters(self, latents, age, gender):
        """
        Apply age and gender edits to Z latents.
        latents: (1, 512) float32.  Returns: (1, 512) float32.
        """
        z = latents.astype('float64')

        # ── Age filter ───────────────────────────────────────────────────
        if age != "Any" and age in AGE_RANGES:
            # Extreme buckets push harder to overcome FFHQ's adult bias
            extreme = age in ("1-11 years old", "50+ years old")
            if self._age_mode == 'regression':
                lo, hi = AGE_RANGES[age]
                target = float(np.random.uniform(lo, hi))
                z = self._set_age(z, target, max_shift=7.0 if extreme else 4.0)
            else:
                # SVM fallback: safe direct offset, no regression math
                scalar = self._svm_offsets.get(age, 0.0)
                d = self.age_dir.reshape(1, -1)
                proj = np.sum(z * d, axis=1, keepdims=True)
                z = z - proj * d              # remove existing age component
                z = z + d * scalar            # place at target scalar

        if gender != "Any":
            # Neutralize inherent gender variance, then set absolute target
            g = self.gender_boundary.reshape(1, -1)
            proj = np.sum(z * g, axis=1, keepdims=True)
            z = z - proj * g
            if gender == "Male":
                z = z + g * 2.0
            elif gender == "Female":
                z = z - g * 2.0

        return z.astype('float32')


# ---------------------------------------------------------------------------
# Age estimator (used for closed-loop validation inside GeneratorThread)
# ---------------------------------------------------------------------------
class _AgeEstimator:
    """
    Thin wrapper around InsightFace genderage.
    Created lazily so the main app still starts if InsightFace isn't installed.
    """
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_l',
                               allowed_modules=['detection', 'genderage'],
                               providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(160, 160))
            self._app = app
            self._available = True
        except Exception as e:
            print(f"[AgeEstimator] InsightFace not available ({e}). "
                  "Closed-loop validation disabled.")
            self._available = False

    def estimate(self, img_rgb):
        """img_rgb: HxWx3 uint8.  Returns float age or None."""
        if not self._available:
            return None
        try:
            import cv2
            img_bgr = cv2.cvtColor(
                np.array(PIL.Image.fromarray(img_rgb).resize((160, 160))),
                cv2.COLOR_RGB2BGR)
            faces = self._app.get(img_bgr)
            if not faces:
                return None
            areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
            return float(faces[int(np.argmax(areas))].age)
        except Exception as e:
            print(f"[AgeEstimator] estimate() failed: {e}")
            return None


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

    def __init__(self, num_images, age, gender, output_dir, Gs, session, graph):
        super().__init__()
        self.num_images = num_images
        self.age        = age
        self.gender     = gender
        self.output_dir = output_dir
        self.Gs         = Gs
        self.session    = session
        self.graph      = graph

    def _generate_one(self, latent_modifier, fmt):
        """
        Closed-loop generation: keep retrying until the generated image's
        estimated age falls within the requested range (with per-bucket
        tolerance), or until max_tries is exhausted.
        Returns (img_data, latents).
        """
        lo, hi = AGE_RANGES.get(self.age, (None, None))
        use_validation = (lo is not None and
                         _AgeEstimator.get()._available)

        # FFHQ skews 20-40 y/o; extreme buckets need more attempts + wider window
        extreme = self.age in ("1-11 years old", "50+ years old")
        max_tries = 10 if extreme else 5

        # Per-bucket acceptance tolerance (lo - tol_lo, hi + tol_hi)
        TOL = {
            "1-11 years old":  (2,  12),  # accept up to ~23  (model hard floor ~18)
            "12-18 years old": (3,   5),
            "19-25 years old": (3,   3),
            "26-35 years old": (3,   3),
            "35-50 years old": (3,   3),
            "50+ years old":   (10,  5),  # accept from ~41   (model hard ceiling ~50)
        }
        tol_lo, tol_hi = TOL.get(self.age, (3, 3))

        for attempt in range(max_tries):
            rnd     = np.random.RandomState(None)
            latents = rnd.randn(1, self.Gs.input_shape[1]).astype('float32')
            latents = latent_modifier.apply_filters(
                latents, self.age, self.gender)

            # Extreme buckets use lower truncation for more variety
            trunc = 0.5 if extreme else 0.7

            images = self.Gs.run(latents, None,
                                 truncation_psi=trunc,
                                 randomize_noise=True,
                                 output_transform=fmt)
            img_data = images[0]

            if not use_validation:
                break  # no estimator — just return first result

            est_age = _AgeEstimator.get().estimate(img_data)
            if est_age is None:
                continue  # no face detected — retry

            if (lo - tol_lo) <= est_age <= (hi + tol_hi):
                break  # within tolerance — accept

            print(f"[GeneratorThread] age={self.age} target=[{lo},{hi}] "
                  f"got={est_age:.1f}  retry {attempt+1}/{max_tries}")

        return img_data, latents

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
                        img_data, _ = self._generate_one(latent_modifier, fmt)
                        png_filename = os.path.join(
                            self.output_dir, f'generated_{i:04d}.png')
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
