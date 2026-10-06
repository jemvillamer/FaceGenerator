"""
fit_age_direction.py
====================
One-time offline script that fits a ridge-regression age direction in Z space
using InsightFace (buffalo_l) as the age estimator.

It produces `cache/age_regression.npz` containing:
  age_coef  (512,)   — regression weights
  age_bias  float    — regression intercept
  age_dir   (512,)   — the move-1-year direction (= coef / (coef @ coef))
  age_mean  float    — mean predicted age over the sample (sanity check)

Usage (from the FaceGenerator root, in the venv37):
  pip install insightface onnxruntime scikit-learn
  python scripts/fit_age_direction.py

The script samples N_SAMPLES latents, generates 64x64 thumbnails (fast on
CPU), runs InsightFace genderage, fits Ridge regression, then saves the npz.

Typical wall-clock time on CPU: ~25 min for 2000 samples (recommended).
For a quick smoke-test use --samples 200 (less accurate but finishes in ~3 min).
"""

import argparse
import os
import sys
import pickle
import numpy as np

# ── make sure the project root is on sys.path ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config                                # noqa: E402 – project config

# ── monkey-patches (same as app.py) so the pkl loads on CPU ────────────────
import tensorflow as tf

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
        result = _orig_conv2d_transpose(value, filters, output_shape, strides, padding, 'NHWC', dilations, name)
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_conv2d_transpose(value, filters, output_shape, strides, padding, data_format, dilations, name)
tf.nn.conv2d_transpose = _cpu_conv2d_transpose

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
        result = _orig_conv2d(input, filt, strides, padding, use_cudnn_on_gpu, 'NHWC', dilations, name)
        return tf.transpose(result, [0, 3, 1, 2])
    return _orig_conv2d(input, filt, strides, padding, use_cudnn_on_gpu, data_format, dilations, name)
tf.nn.conv2d = _cpu_conv2d

import dnnlib.tflib as tflib  # noqa: E402


# ───────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=2000,
                   help='Number of random Z vectors to sample (default 2000)')
    p.add_argument('--seed',    type=int, default=0,
                   help='Master RNG seed for reproducibility')
    p.add_argument('--thumb',   type=int, default=112,
                   help='Thumbnail size passed to InsightFace (112 is its native input)')
    p.add_argument('--alpha',   type=float, default=1e-3,
                   help='Ridge regularisation strength')
    p.add_argument('--out',     default=os.path.join('cache', 'age_regression.npz'),
                   help='Output .npz path')
    return p.parse_args()


def load_stylegan(model_path):
    tflib.init_tf({'allow_soft_placement': True, 'log_device_placement': False})
    with open(model_path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    return Gs


def build_age_estimator(thumb_size):
    """Return a callable img_bgr -> age (float)."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        raise SystemExit(
            "InsightFace not installed.\n"
            "Run:  pip install insightface onnxruntime\n"
            "then re-run this script."
        )
    app = FaceAnalysis(name='buffalo_l',
                       allowed_modules=['detection', 'genderage'],
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(thumb_size, thumb_size))

    def estimate_age(img_rgb_uint8):
        """img_rgb_uint8: HxWx3 uint8 numpy array. Returns float age or None."""
        import cv2
        img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        faces = app.get(img_bgr)
        if not faces:
            return None
        # pick largest face
        areas = [(f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]) for f in faces]
        best  = faces[int(np.argmax(areas))]
        return float(best.age)

    return estimate_age


def main():
    args = parse_args()
    rng  = np.random.RandomState(args.seed)

    model_path = os.path.join(config.cache_dir,
                               'karras2019stylegan-ffhq-1024x1024.pkl')
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found at {model_path}. "
                          "Run the main app once to download it.")

    print(f"[fit_age_direction] Loading StyleGAN from {model_path} …")
    Gs = load_stylegan(model_path)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    print("[fit_age_direction] Building InsightFace age estimator …")
    estimate_age = build_age_estimator(args.thumb)

    # We generate at a lower resolution to stay fast on CPU.
    # StyleGAN v1 / FFHQ has a fixed output size, so we just downscale the image.
    import PIL.Image as PILImage

    z_list   = []
    age_list = []
    skipped  = 0

    print(f"[fit_age_direction] Sampling {args.samples} latents …")
    for i in range(args.samples):
        z = rng.randn(1, Gs.input_shape[1]).astype('float32')
        img = Gs.run(z, None, truncation_psi=0.7,
                     randomize_noise=False, output_transform=fmt)[0]

        # Downscale to thumb_size x thumb_size for faster InsightFace inference
        pil = PILImage.fromarray(img, 'RGB').resize(
            (args.thumb, args.thumb), PILImage.BILINEAR)
        img_small = np.array(pil)

        age = estimate_age(img_small)
        if age is None:
            skipped += 1
            if i % 100 == 0:
                print(f"  [{i}/{args.samples}] no face detected, skipping")
            continue

        z_list.append(z[0])
        age_list.append(age)

        if i % 100 == 0:
            n = len(age_list)
            print(f"  [{i}/{args.samples}] collected {n}, "
                  f"mean_age={np.mean(age_list):.1f}, skipped={skipped}")

    Z   = np.stack(z_list)           # (N, 512)
    ages = np.array(age_list)        # (N,)
    print(f"\n[fit_age_direction] Collected {len(Z)} samples "
          f"(skipped {skipped} / {args.samples}). "
          f"Age range: {ages.min():.0f}–{ages.max():.0f}, "
          f"mean={ages.mean():.1f}")

    # ── Ridge regression: age ≈ Z @ coef + bias ────────────────────────────
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=args.alpha, fit_intercept=True)
    reg.fit(Z, ages)
    coef = reg.coef_.astype('float64')    # (512,)
    bias = float(reg.intercept_)

    # age_dir: moving along this by 1.0 changes predicted age by 1 year
    age_dir = coef / (coef @ coef)

    pred = Z @ coef + bias
    rmse = float(np.sqrt(np.mean((pred - ages)**2)))
    print(f"[fit_age_direction] Ridge fit RMSE: {rmse:.2f} years")
    print(f"[fit_age_direction] bias (mean predicted age): {bias:.1f}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out,
             age_coef=coef,
             age_bias=np.array(bias),
             age_dir=age_dir,
             age_mean=np.array(ages.mean()))
    print(f"[fit_age_direction] Saved → {args.out}")
    print("You can now run the main app — it will pick up the new regression direction automatically.")


if __name__ == '__main__':
    main()
