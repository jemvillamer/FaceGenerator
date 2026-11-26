"""Face generation core: the age-check retry loop, a batch run with output
numbering, and the InsightFace age estimator.

No Qt and no TensorFlow import: the model (Gs, session, graph) and its output
transform are passed in, and progress/cancellation go through plain callbacks.
engine.py wraps generate_batch in a QThread for the app.
"""
import logging
import os
import re
import time

import numpy as np
import PIL.Image

from latents import AGE_BUCKETS_BY_KEY, DEFAULT_TRUNCATION_PSI, LatentDirections

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Age estimator (used for closed-loop validation in generate_one)
# ---------------------------------------------------------------------------
class AgeEstimator:
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
            log.warning("InsightFace not available (%s); the age check is disabled.", e)
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
            log.warning("Age estimate failed: %s", e)
            log.debug("Age estimate failure details", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Output file numbering
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_one(Gs, latent_dirs, age, gender, output_transform,
                 should_stop=lambda: False, stats=None):
    """
    Closed-loop generation: keep retrying until the generated image's
    estimated age falls within the requested range (with per-bucket
    tolerance), or until max_tries is exhausted.
    Returns (img_data, latents), or (None, None) if should_stop() became true
    before an image was accepted.
    stats, if given, is a dict whose 'attempts', 'rejected' and 'no_face'
    counts are increased (used for the per-run summary in the log).
    """
    if stats is None:
        stats = {}
    bucket = AGE_BUCKETS_BY_KEY.get(age)  # None for "Any": no age check
    use_validation = (bucket is not None and
                      AgeEstimator.get()._available)
    if bucket is not None:
        lo, hi = bucket.lo, bucket.hi
        tol_lo, tol_hi = bucket.tolerance
        max_tries = bucket.max_tries
        trunc = bucket.truncation_psi
    else:
        max_tries = 1  # nothing to validate, so the first image is used
        trunc = DEFAULT_TRUNCATION_PSI

    for attempt in range(max_tries):
        if should_stop():
            return None, None  # cancelled: don't wait out the retries
        rnd     = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1]).astype('float32')
        latents = latent_dirs.apply_filters(
            latents, age, gender, truncation_psi=trunc)

        images = Gs.run(latents, None,
                        truncation_psi=trunc,
                        randomize_noise=True,
                        output_transform=output_transform)
        img_data = images[0]
        stats['attempts'] = stats.get('attempts', 0) + 1

        if not use_validation:
            break  # no estimator — just return first result

        est_age = AgeEstimator.get().estimate(img_data)
        if est_age is None:
            stats['no_face'] = stats.get('no_face', 0) + 1
            log.debug("age=%s: no face detected, attempt %d/%d", age, attempt + 1, max_tries)
            continue  # no face detected — retry

        if (lo - tol_lo) <= est_age <= (hi + tol_hi):
            break  # within tolerance — accept

        stats['rejected'] = stats.get('rejected', 0) + 1
        log.debug("age=%s target=[%d,%d] got=%.1f, attempt %d/%d",
                  age, lo, hi, est_age, attempt + 1, max_tries)

    return img_data, latents


def generate_batch(Gs, session, graph, num_images, age, gender, output_dir,
                   output_transform, should_stop=lambda: False,
                   on_saved=lambda count, img, path: None):
    """
    Generate up to num_images faces into output_dir as generated_NNNN.png,
    numbering after any existing images so nothing is overwritten.

    Calls on_saved(count_so_far, image, path) after each image is written,
    and stops early (between images or retries) once should_stop() is true.
    Returns the number of images saved. Errors propagate to the caller.
    """
    log.info("Generating %d image(s): age=%s, gender=%s, into %s",
             num_images, age, gender, output_dir)
    started = time.monotonic()
    stats = {'attempts': 0, 'rejected': 0, 'no_face': 0}
    latent_dirs = LatentDirections()
    os.makedirs(output_dir, exist_ok=True)
    file_index = next_output_index(output_dir)
    saved = 0

    with graph.as_default():
        with session.as_default():
            for _ in range(num_images):
                if should_stop():
                    break
                img_data, _latents = generate_one(
                    Gs, latent_dirs, age, gender, output_transform, should_stop, stats)
                if img_data is None:
                    break  # cancelled mid-retry; nothing to save
                png_filename = os.path.join(
                    output_dir, f'generated_{file_index:04d}.png')
                while os.path.exists(png_filename):  # file appeared mid-run
                    file_index += 1
                    png_filename = os.path.join(
                        output_dir, f'generated_{file_index:04d}.png')
                file_index += 1
                PIL.Image.fromarray(img_data, 'RGB').save(png_filename)
                saved += 1
                on_saved(saved, img_data, png_filename)

    log.info("Saved %d of %d image(s) in %.1f s (%d attempts: %d rejected by the "
             "age check, %d with no face detected)%s",
             saved, num_images, time.monotonic() - started, stats['attempts'],
             stats['rejected'], stats['no_face'],
             "; stopped early" if saved < num_images else "")
    return saved
