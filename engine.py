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
WPLUS_TRUNCATION_CUTOFF = 8          # StyleGAN v1 G_style default

# Closed-loop tuning -------------------------------------------------------
EXTREME_BUCKETS = ("1-11 years old", "50+ years old")

# Truncation per bucket. Truncation pulls W toward the average face (an adult
# around 30), which also shrinks the age edit. Extremes keep the low value
# that already works for them; middle buckets get a higher value so more of
# the edit survives.
TRUNC_EXTREME = 0.5
TRUNC_MIDDLE  = 0.8
TRUNC_ANY     = 0.7

# Acceptance window = (lo - tol_lo, hi + tol_hi) on the *estimated* age.
# Middle buckets are kept tight and no longer overlap their neighbours.
# Extremes keep their original wide windows (the model can't reach the very
# edges of FFHQ's age distribution).
TOL = {
    "1-11 years old":  (2, 12),
    "12-18 years old": (1,  2),
    "19-25 years old": (2,  2),
    "26-35 years old": (2,  2),
    "35-50 years old": (2,  2),
    "50+ years old":   (10, 5),
}

# Per-image refinement limits.  They deliberately bound the feedback loop: a
# noisy age estimate should not be able to move a face far off-distribution.
REFINE_STEP_MAX = 1.0
REFINE_TOTAL_MAX = 4.0

# SVM-fallback only: the old per-bucket scalars, treated as (mid-age, scalar)
# points. They are close to linear (~0.105 scalar units per year), so we
# interpolate to get a scalar for any target age and assume ~9.5 years/unit.
_SVM_POINTS_AGE    = [6.0, 15.0, 22.0, 30.5, 43.0, 63.0]
_SVM_POINTS_SCALAR = [-2.5, -1.2, -0.3, 0.4, 1.5, 3.5]
_SVM_YEARS_PER_UNIT = 9.5


class LatentDirections:
    """
    Handles age and gender edits in Z space.

    Age direction priority:
      1. cache/age_wplus_regression.npz — calibrated W/W+ direction.
      2. cache/age_regression.npz — legacy calibrated Z direction.
      3. Fallback: InterFaceGAN SVM boundary (less accurate, still useful
         as a starting point until the regression file is ready).

    Key invariants (these are what keep the middle age ranges accurate):
      * age_coef / age_bias are the ORIGINAL fitted regression. They are never
        modified, so predicted age matches what the regression was fitted on.
      * age_dir is the unit direction we move along (orthogonal to gender).
      * age_slope = age_coef . age_dir  is "predicted years per latent unit".
        Converting a years-difference into a latent shift MUST divide by it.
    """

    # ── construction ───────────────────────────────────────────────────────
    def __init__(self):
        self.gender_boundary = self._load_or_download('stylegan_ffhq_gender_boundary.npy')
        self.gender_boundary /= np.linalg.norm(self.gender_boundary)

        wplus_path = os.path.join(config.cache_dir, 'age_wplus_regression.npz')
        reg_path = os.path.join(config.cache_dir, 'age_regression.npz')
        self._wplus_data = None
        if os.path.exists(wplus_path):
            data = np.load(wplus_path)
            self.age_coef = data['age_coef'].astype('float64').ravel()
            self.age_bias = float(data['age_bias'])
            self.age_dir = data['age_dir'].astype('float64').ravel()
            self._wplus_data = True
            self._age_mode = 'wplus_regression'
            print("[LatentDirections] Using calibrated W/W+ age direction.")
        elif os.path.exists(reg_path):
            data = np.load(reg_path)
            self.age_coef = data['age_coef'].astype('float64').ravel()
            self.age_bias = float(data['age_bias'])
            self.age_dir  = data['age_dir'].astype('float64').ravel()
            self._age_mode = 'regression'
            print("[LatentDirections] Using calibrated ridge-regression age direction.")
        else:
            raw = self._load_or_download('stylegan_ffhq_age_boundary.npy')
            raw = raw.astype('float64').ravel()
            raw /= np.linalg.norm(raw)
            self.age_coef = None   # signals: use SVM path instead
            self.age_bias = None
            self.age_dir  = raw    # unit-norm boundary vector
            self._age_mode = 'svm_fallback'
            print("[LatentDirections] age_regression.npz not found — using "
                  "InterFaceGAN SVM boundary as fallback. "
                  "Run scripts/fit_age_direction.py for better accuracy.")

        # Z and W live in different coordinate systems. Orthogonalizing a W
        # direction against the Z-space gender boundary is invalid, so do it
        # only for the legacy Z and SVM paths.
        g = self.gender_boundary.ravel()
        d = self.age_dir.ravel()
        if self._age_mode != 'wplus_regression':
            d = d - (d @ g) / (g @ g) * g
        if np.linalg.norm(d) > 1e-8:
            d /= np.linalg.norm(d)
        self.age_dir = d

        # Years of predicted age per unit of movement along age_dir.
        if self._age_mode in ('regression', 'wplus_regression'):
            slope = float(self.age_coef @ self.age_dir)
            if abs(slope) < 1e-3:
                print(f"[LatentDirections] WARNING: age slope is ~0 ({slope:.5f}); "
                      "falling back to 1.0. Check age_regression.npz.")
                slope = 1.0
            self.age_slope = slope
            print(f"[LatentDirections] age slope = {slope:.2f} years per latent unit")
        else:
            self.age_slope = _SVM_YEARS_PER_UNIT

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
        """Predict age from a single fitted-space vector (shape (512,))."""
        return float(z.ravel() @ self.age_coef + self.age_bias)

    def _set_age(self, z, target_age, max_shift=4.0):
        """
        Move z so that its predicted age equals target_age (regression mode).

        delta_years / slope converts years into latent units (the old code
        used years directly as latent units, which is only right when the
        slope happens to be exactly 1). The shift is clamped so the latent
        stays in-distribution.
        """
        z = z.astype('float64')
        current = self._predict_age(z)
        delta = (target_age - current) / self.age_slope
        delta = float(np.clip(delta, -max_shift, max_shift))
        return z + delta * self.age_dir.reshape(1, -1)

    def _svm_scalar(self, target_age):
        scalar = float(np.interp(target_age, _SVM_POINTS_AGE, _SVM_POINTS_SCALAR))
        return float(np.clip(scalar, -4.0, 4.5))

    def apply_filters(self, latents, age, gender, target_age=None):
        """
        Apply gender then age edits to Z latents.
        latents: (1, 512) float32.  Returns: (1, 512) float32.

        target_age : aimed age in years (drawn from the bucket if None).
        Gender is applied FIRST. The gender edit changes the predicted age
        (the regression has a gender component), so the age step has to see
        the post-gender latent to land on the right age. age_dir is orthogonal
        to the gender direction, so the age step does not undo the gender edit.
        """
        z = latents.astype('float64')

        # ── Gender filter ────────────────────────────────────────────────
        if gender != "Any":
            # Neutralize inherent gender variance, then set absolute target
            g = self.gender_boundary.reshape(1, -1)
            proj = np.sum(z * g, axis=1, keepdims=True)
            z = z - proj * g
            if gender == "Male":
                z = z + g * 2.0
            elif gender == "Female":
                z = z - g * 2.0

        # ── Age filter ───────────────────────────────────────────────────
        if (age != "Any" and age in AGE_RANGES and
                self._age_mode != 'wplus_regression'):
            extreme = age in EXTREME_BUCKETS
            lo, hi = AGE_RANGES[age]
            if target_age is None:
                target_age = float(np.random.uniform(lo, hi))
            if self._age_mode == 'regression':
                z = self._set_age(z, target_age,
                                  max_shift=7.0 if extreme else 4.0)
            else:
                d = self.age_dir.reshape(1, -1)
                proj = np.sum(z * d, axis=1, keepdims=True)
                z = z - proj * d                                   # remove existing age component
                z = z + d * self._svm_scalar(target_age)   # place at target scalar

        return z.astype('float32')

    def prepare_wplus(self, Gs, z, truncation_psi):
        """Map Z to the truncated W+ tensor used by StyleGAN synthesis."""
        dlatents = Gs.components.mapping.run(z, None).astype('float64')
        cutoff = Gs.static_kwargs.get('truncation_cutoff', WPLUS_TRUNCATION_CUTOFF)
        if cutoff is not None and cutoff > 0:
            average = Gs.get_var('dlatent_avg').astype('float64')
            dlatents[:, :cutoff, :] = (average + truncation_psi *
                                        (dlatents[:, :cutoff, :] - average))
        return dlatents

    def set_wplus_age(self, dlatents, target_age, max_shift=4.0):
        """Set the fitted W age and apply that direction to coarse/mid W+."""
        w = dlatents[:, 0, :]
        delta = (target_age - self._predict_age(w)) / self.age_slope
        delta = float(np.clip(delta, -max_shift, max_shift))
        result = dlatents.copy()
        result[:, AGE_EDIT_LAYERS, :] += delta * self.age_dir.reshape(1, 1, -1)
        return result


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
        Closed-loop generation. For each attempt:
          1. draw one target age and build one latent,
          2. generate and estimate the age,
          3. refine that same latent along the age direction,
          4. accept if inside the tolerance window.
        If nothing is accepted, return the attempt that came CLOSEST to the
        range (not simply the last one). Returns the uint8 image.
        """
        lo, hi = AGE_RANGES.get(self.age, (None, None))
        estimator = _AgeEstimator.get() if lo is not None else None
        use_validation = estimator is not None and estimator._available

        extreme = self.age in EXTREME_BUCKETS
        max_tries = 10 if extreme else 6
        tol_lo, tol_hi = TOL.get(self.age, (2, 2))

        if lo is None:
            trunc = TRUNC_ANY
        else:
            trunc = TRUNC_EXTREME if extreme else TRUNC_MIDDLE

        best = None       # (miss_in_years, image) among attempts with an estimate
        last_img = None   # fallback when no face was ever detected

        # A retry is a measurement-and-correction cycle for one face, not a
        # new random draw.  This lets its observed response set the direction
        # and scale of the next correction without leaking noise to later
        # images.
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, self.Gs.input_shape[1]).astype('float32')
        target = float(np.random.uniform(lo, hi)) if lo is not None else None
        latents = latent_modifier.apply_filters(
            latents, self.age, self.gender, target_age=target)
        use_wplus = latent_modifier._age_mode == 'wplus_regression'
        if use_wplus:
            base_latents = latent_modifier.prepare_wplus(Gs=self.Gs, z=latents,
                                                         truncation_psi=trunc)
            base_latents = latent_modifier.set_wplus_age(
                base_latents, target,
                max_shift=7.0 if extreme else 4.0)
        else:
            base_latents = latents.astype('float64')
        refinement = 0.0
        previous = None  # (refinement, estimated_age)

        for attempt in range(max_tries):
            if use_wplus:
                candidate = base_latents.copy()
                candidate[:, AGE_EDIT_LAYERS, :] += (
                    refinement * latent_modifier.age_dir.reshape(1, 1, -1))
                images = self.Gs.components.synthesis.run(
                    candidate.astype('float32'), output_transform=fmt)
            else:
                candidate = (base_latents + refinement *
                             latent_modifier.age_dir.reshape(1, -1)).astype('float32')
                images = self.Gs.run(candidate, None,
                                     truncation_psi=trunc,
                                     # Keep stochastic noise fixed too: otherwise a retry
                                     # changes two inputs and corrupts the local slope.
                                     randomize_noise=False,
                                     output_transform=fmt)
            img_data = images[0]
            last_img = img_data

            if not use_validation:
                return img_data  # no estimator — just return first result

            est_age = estimator.estimate(img_data)
            if est_age is None:
                print(f"[GeneratorThread] age={self.age} target={target:.1f} "
                      f"attempt {attempt+1}/{max_tries}: no face detected")
                continue  # no face detected — retry

            miss = max(lo - est_age, est_age - hi, 0.0)
            if best is None or miss < best[0]:
                best = (miss, img_data)

            if (lo - tol_lo) <= est_age <= (hi + tol_hi):
                print(f"[GeneratorThread] age={self.age} target={target:.1f} "
                      f"got={est_age:.1f} accepted attempt {attempt+1}/{max_tries}")
                return img_data  # within tolerance — accept

            # First correction uses the fitted slope only as a small probe.
            # Thereafter use the face's own two measurements (a secant slope).
            # If the local response is too small or reverses, retain the safe
            # fitted-slope probe instead of amplifying estimator noise.
            if previous is None:
                slope = latent_modifier.age_slope
            else:
                delta_refinement = refinement - previous[0]
                local_slope = ((est_age - previous[1]) / delta_refinement
                               if abs(delta_refinement) > 1e-6 else 0.0)
                slope = (local_slope if local_slope * latent_modifier.age_slope > 0.5
                         else latent_modifier.age_slope)
            step = float(np.clip((target - est_age) / slope,
                                 -REFINE_STEP_MAX, REFINE_STEP_MAX))
            previous = (refinement, est_age)
            refinement = float(np.clip(refinement + step,
                                       -REFINE_TOTAL_MAX, REFINE_TOTAL_MAX))
            print(f"[GeneratorThread] age={self.age} target={target:.1f} "
                  f"got={est_age:.1f} refine={refinement:+.2f} "
                  f"retry {attempt+1}/{max_tries}")

        if best is not None:
            print(f"[GeneratorThread] age={self.age} exhausted {max_tries} attempts; "
                  f"returning closest miss={best[0]:.1f} years")
            return best[1]
        return last_img

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
                        img_data = self._generate_one(latent_modifier, fmt)
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
