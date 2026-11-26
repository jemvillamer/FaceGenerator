"""Age buckets and latent-space age/gender edits.

Pure numpy (no Qt, no TensorFlow): AGE_BUCKETS holds every rule tied to an
age choice, and LatentDirections applies the age and gender edits to Z
latents. Used by generator.py and, for the dropdown, by the sidebar.
"""
import logging
import os
from dataclasses import dataclass

import numpy as np

import config
import downloads

log = logging.getLogger(__name__)


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

        reg_path = config.data_file('age_regression.npz')
        if os.path.exists(reg_path):
            data = np.load(reg_path)
            self.age_coef = data['age_coef'].astype('float64')
            self.age_bias = float(data['age_bias'])
            self.age_dir  = data['age_dir'].astype('float64')
            self._age_mode = 'regression'
            log.info("Using calibrated ridge-regression age direction.")
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
            log.warning("age_regression.npz not found: using the InterFaceGAN SVM "
                        "boundary as a fallback (less accurate). "
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
        filepath = config.data_file(filename)
        expected = downloads.BOUNDARY_SHA256[filename]
        if not os.path.exists(filepath):
            log.info("Downloading %s ...", filename)
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
