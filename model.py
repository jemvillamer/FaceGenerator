"""The pretrained StyleGAN generator: CPU patches, download/verify, loading.

Shared by the app (engine.py) and scripts/fit_age_direction.py.
"""
import os
import pickle

import tensorflow as tf

import tf_compat
tf_compat.install()  # must happen before the model is unpickled

import dnnlib.tflib as tflib  # noqa: E402
import config                 # noqa: E402
import downloads              # noqa: E402

# Gs.run(..., output_transform=OUTPUT_TRANSFORM) returns uint8 NHWC images.
# (Network.run copies this dict, so sharing one instance is safe.)
OUTPUT_TRANSFORM = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


def model_path():
    return os.path.join(config.cache_dir, downloads.MODEL_FILE)


def load_model(download=True):
    """Initialize TF and load the pretrained StyleGAN model.

    Downloads it first if missing (unless download=False, in which case a
    missing file raises FileNotFoundError). The file is always checksummed
    before unpickling, because unpickling runs code; a mismatch raises
    downloads.ChecksumError. Returns (Gs, session, graph).
    """
    tflib.init_tf({"allow_soft_placement": True, "log_device_placement": False})

    os.makedirs(config.cache_dir, exist_ok=True)
    path = model_path()

    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        import gdown
        print("Downloading pretrained model (~325MB)...")
        downloads.fetch_verified(
            lambda tmp: gdown.download(id=downloads.MODEL_GDRIVE_ID, output=tmp, quiet=False),
            path, downloads.MODEL_SHA256)
    else:
        downloads.verify_existing(path, downloads.MODEL_SHA256)  # ~0.5 s

    print("Loading model...")
    with open(path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    print("Model loaded successfully.")
    return Gs, tf.get_default_session(), tf.get_default_graph()
