"""Qt layer: background threads that load the model and run generation, and
report back to the UI through signals.

The logic lives in Qt-free modules: model.py (loading), latents.py (age buckets
and latent edits) and generator.py (retry loop, batch run, output numbering).
"""
import logging

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# model.py installs the CPU compatibility patches (tf_compat) on import.
from model import load_model, OUTPUT_TRANSFORM
from downloads import ChecksumError
import generator

# Re-exported so existing `from engine import ...` code keeps working.
from latents import (AgeBucket, AGE_BUCKETS, AGE_BUCKETS_BY_KEY, AGE_RANGES,  # noqa: F401
                     AGE_RESPONSE, DEFAULT_TRUNCATION_PSI, LatentDirections)
from generator import AgeEstimator, next_output_index  # noqa: F401
_AgeEstimator = AgeEstimator  # previous name (same class)

log = logging.getLogger(__name__)


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

    def _on_saved(self, count, img_data, path):
        self.progress.emit(count)
        self.preview.emit(img_data, path)

    def run(self):
        try:
            saved = generator.generate_batch(
                self.Gs, self.session, self.graph, self.num_images,
                self.age, self.gender, self.output_dir, OUTPUT_TRANSFORM,
                should_stop=self.isInterruptionRequested,
                on_saved=self._on_saved)
            self.completed.emit(saved, saved < self.num_images)
        except ChecksumError as e:
            log.error("Generation stopped: %s", e)
            self.error.emit(str(e))  # readable message instead of a traceback
        except Exception as e:
            log.exception("Generation failed")
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
            log.error("Model not loaded: %s", e)
            self.error.emit(str(e))  # readable message instead of a traceback
        except Exception as e:
            log.exception("Model loading failed")
            import traceback
            self.error.emit(traceback.format_exc())
