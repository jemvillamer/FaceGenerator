"""
measure_age_response.py
=======================
Measure how rendered faces respond to the calibrated age edit, and how
accurately each age bucket is hit.

Run this after recalibrating with scripts/fit_age_direction.py. It renders
real faces, estimates their ages with InsightFace, and prints:
  1. a fitted AGE_RESPONSE table to paste into latents.py
     (InsightFace age ~= intercept + slope x calibrated predicted age,
      per truncation_psi), and
  2. a per-bucket report: how many single-shot faces (no retries) land in
     the bucket, or within the age check's tolerance, with the current settings.

Needs: the model (run the app once), cache/age_regression.npz, and
InsightFace (pip install insightface onnxruntime).

Usage (from the FaceGenerator root, in the venv37):
  python scripts/measure_age_response.py
  python scripts/measure_age_response.py --fit-samples 60 --report-samples 0

Time: ~2 s per rendered face on CPU. The defaults render 2 x 120 + 6 x 20 =
360 faces, about 12 minutes.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import log_setup                     # noqa: E402
log_setup.configure(log_file=False)  # console only; quiets TensorFlow (before it's imported)
import downloads                     # noqa: E402
import model                         # noqa: E402  (installs the TF CPU patches)
from generator import AgeEstimator   # noqa: E402
from latents import AGE_BUCKETS, AGE_RESPONSE, LatentDirections  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument('--fit-samples', type=int, default=120,
                   help='faces per truncation value used to fit AGE_RESPONSE (default 120; 0 skips)')
    p.add_argument('--report-samples', type=int, default=20,
                   help='faces per age bucket for the accuracy report (default 20; 0 skips)')
    p.add_argument('--seed', type=int, default=0, help='RNG seed (default 0)')
    return p.parse_args()


def render(Gs, z, psi):
    return Gs.run(z, None, truncation_psi=psi, randomize_noise=False,
                  output_transform=model.OUTPUT_TRANSFORM)[0]


def fit_response(Gs, ld, estimator, rng, samples):
    """Spread predicted ages widely (uncorrected edits, 7-unit cap), render,
    estimate, and fit estimated = intercept + slope * predicted."""
    psis = sorted({b.truncation_psi for b in AGE_BUCKETS})
    fitted = {}
    for psi in psis:
        X, Y = [], []
        for i in range(samples):
            z = rng.randn(1, Gs.input_shape[1])
            z = ld._set_age(z, rng.uniform(-30, 100), max_shift=7.0).astype('float32')
            age = estimator.estimate(render(Gs, z, psi))
            if age is not None:
                X.append(float(z.ravel().astype('float64') @ ld.age_coef + ld.age_bias))
                Y.append(age)
            if (i + 1) % 20 == 0:
                print(f"  psi={psi}: {i + 1}/{samples} faces", flush=True)
        if len(X) < 3:
            print(f"  psi={psi}: too few faces detected ({len(X)}) to fit")
            continue
        slope, intercept = np.polyfit(X, Y, 1)
        r = float(np.corrcoef(X, Y)[0, 1])
        resid = np.std(np.array(Y) - (intercept + slope * np.array(X)))
        fitted[psi] = (round(float(intercept), 2), round(float(slope), 3))
        print(f"  psi={psi}: estimated = {intercept:.2f} + {slope:.3f} x predicted   "
              f"(r={r:.2f}, residual sd {resid:.1f} years, n={len(X)}; "
              f"current: {AGE_RESPONSE.get(psi)})")
    return fitted


def report_buckets(Gs, ld, estimator, rng, samples):
    print(f"\n  {'bucket':16} {'in bucket':>10} {'in tolerance':>13} {'mean est':>9}")
    for b in AGE_BUCKETS:
        ages = []
        for _ in range(samples):
            z = rng.randn(1, Gs.input_shape[1]).astype('float32')
            z = ld.apply_filters(z, b.key, "Any", truncation_psi=b.truncation_psi)
            ages.append(estimator.estimate(render(Gs, z, b.truncation_psi)))
        found = [a for a in ages if a is not None]
        in_bucket = sum(b.lo <= a <= b.hi for a in found)
        in_tol = sum(b.lo - b.tolerance[0] <= a <= b.hi + b.tolerance[1] for a in found)
        mean = f"{np.mean(found):9.1f}" if found else "      n/a"
        print(f"  {b.key:16} {in_bucket:>5}/{samples:<4} {in_tol:>8}/{samples:<4} {mean}", flush=True)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    try:
        Gs, _session, _graph = model.load_model(download=False)
    except FileNotFoundError:
        raise SystemExit(f"Model not found at {model.model_path()}. "
                         "Run the main app once to download it.")
    except downloads.ChecksumError as e:
        raise SystemExit(str(e))

    ld = LatentDirections()
    if ld._age_mode != 'regression':
        raise SystemExit("No calibration found (cache/age_regression.npz). "
                         "Run scripts/fit_age_direction.py first.")
    estimator = AgeEstimator.get()
    if not estimator._available:
        raise SystemExit("InsightFace is not available. "
                         "Run: pip install insightface onnxruntime")

    if args.fit_samples > 0:
        print(f"\nFitting AGE_RESPONSE ({args.fit_samples} faces per truncation value)...")
        fitted = fit_response(Gs, ld, estimator, rng, args.fit_samples)
        if fitted:
            print("\nSuggested table for latents.py:\nAGE_RESPONSE = {")
            for psi in sorted(fitted, reverse=True):
                print(f"    {psi}: {fitted[psi]},")
            print("}")
    if args.report_samples > 0:
        print(f"\nBucket accuracy with the current settings "
              f"({args.report_samples} single-shot faces each, no retries):")
        report_buckets(Gs, ld, estimator, rng, args.report_samples)


if __name__ == '__main__':
    main()
