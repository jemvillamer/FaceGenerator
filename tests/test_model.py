"""Real-model tests: need cache/karras2019stylegan-ffhq-1024x1024.pkl.
Skipped by default; run with:  pytest -m model"""
import os

import pytest

import config
import downloads
import engine
import generator

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def loaded_model():
    if not os.path.exists(os.path.join(config.cache_dir, downloads.MODEL_FILE)):
        pytest.skip("model not downloaded; run the app once first")
    return engine.load_model(download=False)


def test_real_model_passes_its_checksum_and_loads(loaded_model):
    Gs, session, graph = loaded_model
    assert Gs.input_shape[1] == 512


def test_generate_batch_saves_real_faces(loaded_model, tmp_path):
    Gs, session, graph = loaded_model
    seen = []
    saved = generator.generate_batch(
        Gs, session, graph, 2, "Any", "Male", str(tmp_path), engine.OUTPUT_TRANSFORM,
        on_saved=lambda count, img, path: seen.append((count, img.shape, os.path.basename(path))))
    assert saved == 2
    assert seen == [(1, (1024, 1024, 3), "generated_0000.png"),
                    (2, (1024, 1024, 3), "generated_0001.png")]
