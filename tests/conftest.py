"""Shared fixtures.

The default test run needs neither the real model nor the internet: tests use a
temporary cache folder with small synthetic boundary/calibration files (their
checksums patched in), a fake generator network (FakeGs) and fake age estimators.
"""
import contextlib
import hashlib
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any Qt import

import numpy as np
import pytest

import config
import downloads
import generator

GENDER_FILE = "stylegan_ffhq_gender_boundary.npy"
AGE_FILE = "stylegan_ffhq_age_boundary.npy"


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeGs:
    """Stands in for the StyleGAN network: records every run() call and returns
    a small image whose pixels encode the call number (1, 2, 3, ...)."""
    input_shape = [None, 512]

    def __init__(self, size=16, delay=0.0):
        self.size, self.delay, self.calls = size, delay, []

    def run(self, latents, labels, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"latents": latents.copy(), **kwargs})
        n = len(self.calls)
        img = np.zeros((1, self.size, self.size, 3), np.uint8)
        img[0, :, :, 0] = n % 256
        img[0, :, :, 1] = np.arange(self.size, dtype=np.uint8)[None, :] * 7 % 256
        img[0, :, :, 2] = (n * 37) % 256
        return img


class FakeEstimator:
    """Age estimator returning fixed ages: a constant, or values from a list."""
    def __init__(self, ages, available=True):
        self._ages = ages
        self._available = available
        self.calls = 0

    def estimate(self, img):
        self.calls += 1
        if isinstance(self._ages, list):
            return self._ages[min(self.calls - 1, len(self._ages) - 1)]
        return self._ages


class DummyContext:
    """Stands in for the TF graph/session: generate_batch only enters as_default()."""
    def as_default(self):
        return contextlib.nullcontext()


# ── fixtures ───────────────────────────────────────────────────────────────
def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point config.cache_dir at a temp folder holding synthetic boundary files,
    and patch their checksums in so LatentDirections accepts them."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.RandomState(0)
    for name in (GENDER_FILE, AGE_FILE):
        v = rng.randn(1, 512)
        np.save(str(cache / name), v / np.linalg.norm(v))
        monkeypatch.setitem(downloads.BOUNDARY_SHA256, name, _sha256(str(cache / name)))
    monkeypatch.setattr(config, "cache_dir", str(cache))
    return cache


@pytest.fixture
def regression_cache(tmp_cache):
    """tmp_cache plus a synthetic age_regression.npz (about 10 years per latent
    unit, like the real calibration), so LatentDirections uses regression mode."""
    rng = np.random.RandomState(1)
    coef = rng.randn(512) * 0.45
    np.savez(str(tmp_cache / "age_regression.npz"), age_coef=coef, age_bias=np.array(36.0),
             age_dir=coef / (coef @ coef), age_mean=np.array(36.0))
    return tmp_cache


@pytest.fixture(autouse=True)
def no_real_downloads_or_insightface(request, monkeypatch):
    """Default tests must not reach the internet or load InsightFace."""
    if request.node.get_closest_marker("network") is None:
        import urllib.request
        def blocked(*a, **k):
            raise AssertionError("test tried to download: " + str(a[:1]))
        monkeypatch.setattr(urllib.request, "urlretrieve", blocked)
        try:
            import gdown
            monkeypatch.setattr(gdown, "download", blocked)
        except ImportError:
            pass
    if request.node.get_closest_marker("model") is None:
        no_estimator = FakeEstimator(None, available=False)
        monkeypatch.setattr(generator.AgeEstimator, "get", classmethod(lambda cls: no_estimator))


@pytest.fixture
def use_estimator(monkeypatch):
    """use_estimator(FakeEstimator(...)) makes generate_one use that estimator."""
    def install(est):
        monkeypatch.setattr(generator.AgeEstimator, "get", classmethod(lambda cls: est))
        return est
    return install


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def pump_until(qapp, condition, timeout=20.0):
    """Process Qt events until condition() is true (or fail after timeout)."""
    end = time.time() + timeout
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.time() > end:
            raise AssertionError("timed out waiting for condition")
    qapp.processEvents()
