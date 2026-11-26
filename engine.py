import os
import re
from dataclasses import dataclass

import numpy as np
import PIL.Image
from PyQt5.QtCore import QThread, pyqtSignal

# model.py installs the CPU compatibility patches (tf_compat) on import.
from model import load_model, OUTPUT_TRANSFORM  # noqa: F401 (load_model re-exported)
import config
import downloads
from downloads import ChecksumError


# ---------------------------------------------------------------------------
# Age buckets: every rule tied to an age choice, in one place
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgeBucket:
    key: str            # value used in code and passed from the UI
    label: str          # text shown in the age dropdown
    lo: int             # target age range in years (inclusive)
    hi: int
    svm_offset: float   # SVM-fallback position along the age boundary, in latent
                        # units (empirically tuned; kept small to stay in distribution)
    tolerance: tuple    # (years below lo, years above hi) the age check accepts
    # FFHQ skews 20-40 years old, so the extreme buckets push harder:
    # more attempts, lower truncation (more variety) and a larger edit cap.
    extreme: bool = False

    @property
    def truncation_psi(self):
        return 0.5 if self.extreme else 0.7

    @property
    def max_tries(self):
        return 10 if self.extreme else 5

    @property
    def max_shift(self):
        """Largest calibrated age edit, in latent units."""
        return 7.0 if self.extreme else 4.0


AGE_BUCKETS = [
    # tolerance (2, 12): accept up to ~23 (model hard floor ~18)
    AgeBucket("1-11 years old",  "1–11 years old",   1, 11, -2.5, (2, 12), extreme=True),
    AgeBucket("12-18 years old", "12–18 years old", 12, 18, -1.2, (3, 5)),
    AgeBucket("19-25 years old", "19–25 years old", 19, 25, -0.3, (3, 3)),
    AgeBucket("26-35 years old", "26–35 years old", 26, 35,  0.4, (3, 3)),
    AgeBucket("35-50 years old", "35–50 years old", 36, 50,  1.5, (3, 3)),
    # svm_offset raised to 3.5: the model skews young.
    # tolerance (10, 5): accept from ~41 (model hard ceiling ~50)
    AgeBucket("50+ years old",   "50+ years old",        51, 75,  3.5, (10, 5), extreme=True),
]
AGE_BUCKETS_BY_KEY = {b.key: b for b in AGE_BUCKETS}

# (lo, hi) per bucket key; derived from AGE_BUCKETS, kept for existing callers
AGE_RANGES = {b.key: (b.lo, b.hi) for b in AGE_BUCKETS}

# Truncation used when no age bucket is chosen ("Any")
DEFAULT_TRUNCATION_PSI = 0.7

# Measured response of rendered faces to the calibrated age edit, per
# truncation_psi:  InsightFace age ≈ intercept + slope × calibrated predicted age.
# Truncation pulls faces toward the average face, and the ridge fit overstates
# the age signal, so real faces move only ~0.4-0.5 years per predicted year.
# _set_age inverts this to aim the edit. Fitted on 360 rendered faces against
# the current cache/age_regression.npz; re-measure after recalibrating.
AGE_RESPONSE = {
    0.7: (20.28, 0.515),
    0.5: (26.13, 0.414),
}


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
            self._age_mode = 'svm_fallback'  # offsets: AgeBucket.svm_offset
            print("[LatentDirections] age_regression.npz not found — using "
                  "InterFaceGAN SVM boundary as fallback. "
                  "Run scripts/fit_age_direction.py for better accuracy.")

        # Orthogonalize age against gender to reduce entanglement
        g = self.gender_boundary.ravel()
        if self._age_mode == 'regression':
            # Remove the gender component from the regression weights, then
            # rebuild the direction as c / (c·c): moving 1.0 along it changes
            # the predicted age by exactly 1 year. (Normalising it to unit
            # length instead makes each latent unit worth ||c|| years, ~10 with
            # a typical calibration, so every edit overshoots its target.)
            c = self.age_coef.ravel()
            c = c - (c @ g) / (g @ g) * g
            self.age_coef = c
            self.age_dir = c / (c @ c)
        else:
            # SVM fallback: offsets are in latent units, so keep a unit vector
            d = self.age_dir.ravel()
            d = d - (d @ g) / (g @ g) * g
            if np.linalg.norm(d) > 1e-8:
                d /= np.linalg.norm(d)
            self.age_dir = d

    # ── helpers ─────────────────────────────────────────────────────────────
    def _load_or_download(self, filename):
        import urllib.request
        os.makedirs(config.cache_dir, exist_ok=True)
        filepath = os.path.join(config.cache_dir, filename)
        expected = downloads.BOUNDARY_SHA256[filename]
        if not os.path.exists(filepath):
            print(f"[LatentDirections] Downloading {filename}…")
            downloads.fetch_verified(
                lambda tmp: urllib.request.urlretrieve(downloads.boundary_url(filename), tmp),
                filepath, expected)
        else:
            downloads.verify_existing(filepath, expected)
        return np.load(filepath)

    def _set_age(self, z, target_age, max_shift=4.0, truncation_psi=None):
        """
        Move z so that its predicted age equals target_age.
        Only called in regression mode. age_dir is scaled so that one unit
        along it is one year; the latent step is clamped to max_shift latent
        units (Euclidean length) so the latent never leaves the model's
        training distribution. Callers pass the bucket's AgeBucket.max_shift.
        If truncation_psi has a measured response in AGE_RESPONSE, the target
        is converted to the predicted age that renders at target_age.
        """
        z = z.astype('float64')
        if truncation_psi in AGE_RESPONSE:
            intercept, slope = AGE_RESPONSE[truncation_psi]
            target_age = (target_age - intercept) / slope
        current = float(z.ravel() @ self.age_coef + self.age_bias)
        max_years = max_shift / np.linalg.norm(self.age_dir)  # latent cap -> years
        delta   = float(np.clip(target_age - current, -max_years, max_years))
        return z + delta * self.age_dir.reshape(1, -1)

    def apply_filters(self, latents, age, gender, truncation_psi=None):
        """
        Apply age and gender edits to Z latents.
        latents: (1, 512) float32.  Returns: (1, 512) float32.
        truncation_psi: the value the latents will be rendered with; lets the
        calibrated age edit correct for it (see AGE_RESPONSE).
        """
        z = latents.astype('float64')

        # ── Age filter ───────────────────────────────────────────────────
        bucket = AGE_BUCKETS_BY_KEY.get(age)  # None for "Any"
        if bucket is not None:
            if self._age_mode == 'regression':
                target = float(np.random.uniform(bucket.lo, bucket.hi))
                z = self._set_age(z, target, max_shift=bucket.max_shift,
                                  truncation_psi=truncation_psi)
            else:
                # SVM fallback: safe direct offset, no regression math
                scalar = bucket.svm_offset
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


_OUTPUT_NAME_RE = re.compile(r'^generated_(\d+)\.png$', re.IGNORECASE)


def next_output_index(output_dir):
    """Return the number after the highest existing generated_NNNN.png in
    output_dir (0 if there are none), so a new run never overwrites old images."""
    try:
        names = os.listdir(output_dir)
    except FileNotFoundError:
        return 0
    numbers = [int(m.group(1)) for m in map(_OUTPUT_NAME_RE.match, names) if m]
    return max(numbers) + 1 if numbers else 0


class GeneratorThread(QThread):
    progress  = pyqtSignal(int)
    # (images saved, cancelled). Not named `finished`: that would hide
    # QThread's own finished signal.
    completed = pyqtSignal(int, bool)
    preview   = pyqtSignal(np.ndarray, str)   # (image, path it was saved to)
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
        Returns (img_data, latents), or (None, None) if the run was cancelled
        before an image was accepted.
        """
        bucket = AGE_BUCKETS_BY_KEY.get(self.age)  # None for "Any": no age check
        use_validation = (bucket is not None and
                         _AgeEstimator.get()._available)
        if bucket is not None:
            lo, hi = bucket.lo, bucket.hi
            tol_lo, tol_hi = bucket.tolerance
            max_tries = bucket.max_tries
            trunc = bucket.truncation_psi
        else:
            max_tries = 1  # nothing to validate, so the first image is used
            trunc = DEFAULT_TRUNCATION_PSI

        for attempt in range(max_tries):
            if self.isInterruptionRequested():
                return None, None  # cancelled: don't wait out the retries
            rnd     = np.random.RandomState(None)
            latents = rnd.randn(1, self.Gs.input_shape[1]).astype('float32')
            latents = latent_modifier.apply_filters(
                latents, self.age, self.gender, truncation_psi=trunc)

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
            fmt = OUTPUT_TRANSFORM
            latent_modifier = LatentDirections()
            os.makedirs(self.output_dir, exist_ok=True)
            file_index = next_output_index(self.output_dir)
            saved = 0

            with self.graph.as_default():
                with self.session.as_default():
                    for i in range(self.num_images):
                        if self.isInterruptionRequested():
                            break
                        img_data, _ = self._generate_one(latent_modifier, fmt)
                        if img_data is None:
                            break  # cancelled mid-retry; nothing to save
                        png_filename = os.path.join(
                            self.output_dir, f'generated_{file_index:04d}.png')
                        while os.path.exists(png_filename):  # file appeared mid-run
                            file_index += 1
                            png_filename = os.path.join(
                                self.output_dir, f'generated_{file_index:04d}.png')
                        file_index += 1
                        PIL.Image.fromarray(img_data, 'RGB').save(png_filename)
                        saved += 1
                        self.progress.emit(i + 1)
                        self.preview.emit(img_data, png_filename)

            self.completed.emit(saved, saved < self.num_images)
        except ChecksumError as e:
            self.error.emit(str(e))  # readable message instead of a traceback
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
        except ChecksumError as e:
            self.error.emit(str(e))  # readable message instead of a traceback
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())
