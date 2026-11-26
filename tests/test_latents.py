"""latents.py: the age-bucket table and the latent age/gender edits."""
import os

import numpy as np
import pytest

import downloads
import latents
from conftest import AGE_FILE, GENDER_FILE
from latents import AGE_BUCKETS, AGE_BUCKETS_BY_KEY, AGE_RESPONSE, LatentDirections


def random_latents(n, seed=0):
    return np.random.RandomState(seed).randn(n, 1, 512).astype("float32")


def predicted_age(ld, z):
    return float(z.ravel().astype("float64") @ ld.age_coef + ld.age_bias)


def target_for(bucket, seed):
    """The target age apply_filters draws after np.random.seed(seed)."""
    np.random.seed(seed)
    return float(np.random.uniform(bucket.lo, bucket.hi))


# ── the table ──────────────────────────────────────────────────────────────
def test_buckets_are_ordered_non_overlapping_and_unique():
    assert len({b.key for b in AGE_BUCKETS}) == len(AGE_BUCKETS)
    for b in AGE_BUCKETS:
        assert b.lo <= b.hi
    for prev, nxt in zip(AGE_BUCKETS, AGE_BUCKETS[1:]):
        assert prev.hi < nxt.lo


def test_dropdown_labels_and_keys():
    assert [(b.label, b.key) for b in AGE_BUCKETS] == [
        ("1–11 years old", "1-11 years old"), ("12–18 years old", "12-18 years old"),
        ("19–25 years old", "19-25 years old"), ("26–35 years old", "26-35 years old"),
        ("35–50 years old", "35-50 years old"), ("50+ years old", "50+ years old")]


def test_tuned_values_are_unchanged():
    """The ranges, fallback offsets and tolerances were tuned by hand. This
    catches accidental edits; when re-tuning on purpose, update it too."""
    assert [(b.key, b.lo, b.hi, b.svm_offset, b.tolerance, b.extreme) for b in AGE_BUCKETS] == [
        ("1-11 years old",   1, 11, -2.5, (2, 12), True),
        ("12-18 years old", 12, 18, -1.2, (3, 5),  False),
        ("19-25 years old", 19, 25, -0.3, (3, 3),  False),
        ("26-35 years old", 26, 35,  0.4, (3, 3),  False),
        ("35-50 years old", 36, 50,  1.5, (3, 3),  False),
        ("50+ years old",   51, 75,  3.5, (10, 5), True),
    ]
    assert AGE_RESPONSE == {0.7: (20.28, 0.515), 0.5: (26.13, 0.414)}


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_derived_settings(bucket):
    if bucket.extreme:
        assert (bucket.truncation_psi, bucket.max_tries, bucket.max_shift) == (0.5, 10, 7.0)
    else:
        assert (bucket.truncation_psi, bucket.max_tries, bucket.max_shift) == (0.7, 5, 4.0)


def test_only_the_youngest_and_oldest_buckets_are_extreme():
    assert [b.key for b in AGE_BUCKETS if b.extreme] == ["1-11 years old", "50+ years old"]


def test_lookup_tables_are_derived_from_the_buckets():
    assert latents.AGE_RANGES == {b.key: (b.lo, b.hi) for b in AGE_BUCKETS}
    assert all(AGE_BUCKETS_BY_KEY[b.key] is b for b in AGE_BUCKETS)


def test_every_bucket_truncation_has_a_measured_response():
    assert {b.truncation_psi for b in AGE_BUCKETS} <= set(AGE_RESPONSE)


# ── calibrated (regression) mode ───────────────────────────────────────────
def test_regression_direction_is_one_year_per_unit_and_gender_neutral(regression_cache):
    ld = LatentDirections()
    assert ld._age_mode == "regression"
    assert ld.age_coef @ ld.age_dir == pytest.approx(1.0)
    assert ld.age_dir @ ld.gender_boundary.ravel() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_uncapped_edits_land_on_the_target_age(regression_cache, bucket):
    ld = LatentDirections()
    hits = 0
    for i, z in enumerate(random_latents(200)):
        target = target_for(bucket, i)
        np.random.seed(i)
        out = ld.apply_filters(z, bucket.key, "Any")  # no truncation: no correction
        step = np.linalg.norm(out.astype("float64") - z)
        if step < bucket.max_shift - 1e-3:
            hits += 1
            assert predicted_age(ld, out) == pytest.approx(target, abs=0.01)
    assert hits > 100  # most edits are uncapped, so the check is meaningful


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_corrected_edits_aim_for_the_measured_response(regression_cache, bucket):
    ld = LatentDirections()
    intercept, slope = AGE_RESPONSE[bucket.truncation_psi]
    for i, z in enumerate(random_latents(200)):
        target = target_for(bucket, i)
        np.random.seed(i)
        out = ld.apply_filters(z, bucket.key, "Any", truncation_psi=bucket.truncation_psi)
        if np.linalg.norm(out.astype("float64") - z) < bucket.max_shift - 1e-3:
            assert intercept + slope * predicted_age(ld, out) == pytest.approx(target, abs=0.01)


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_edits_never_exceed_the_bucket_cap(regression_cache, bucket):
    ld = LatentDirections()
    for i, z in enumerate(random_latents(300, seed=4)):
        np.random.seed(i)
        out = ld.apply_filters(z, bucket.key, "Any", truncation_psi=bucket.truncation_psi)
        assert np.linalg.norm(out.astype("float64") - z) <= bucket.max_shift + 1e-4


def test_unmeasured_truncation_means_no_correction(regression_cache):
    ld = LatentDirections()
    z = random_latents(1)[0]
    np.random.seed(0)
    plain = ld.apply_filters(z, "26-35 years old", "Any")
    np.random.seed(0)
    unmeasured = ld.apply_filters(z, "26-35 years old", "Any", truncation_psi=0.9)
    assert np.array_equal(plain, unmeasured)


@pytest.mark.parametrize("gender,sign", [("Male", 1), ("Female", -1)])
def test_gender_edit_sets_gender_without_changing_age(regression_cache, gender, sign):
    ld = LatentDirections()
    g = ld.gender_boundary.ravel()
    for i, z in enumerate(random_latents(20)):
        np.random.seed(i)
        aged = ld.apply_filters(z, "35-50 years old", "Any")
        np.random.seed(i)
        out = ld.apply_filters(z, "35-50 years old", gender)
        assert predicted_age(ld, out) == pytest.approx(predicted_age(ld, aged), abs=1e-3)
        assert out.ravel().astype("float64") @ g == pytest.approx(sign * 2.0, abs=1e-4)


@pytest.mark.parametrize("age", ["Any", "not-a-bucket"])
def test_no_age_bucket_means_no_age_edit(regression_cache, age):
    ld = LatentDirections()
    z = random_latents(1)[0]
    out = ld.apply_filters(z, age, "Any")
    assert out.dtype == np.float32 and out.shape == (1, 512)
    assert np.array_equal(out, z)


# ── SVM fallback mode ──────────────────────────────────────────────────────
@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_fallback_places_the_face_at_the_bucket_offset(tmp_cache, bucket):
    ld = LatentDirections()
    assert ld._age_mode == "svm_fallback"
    assert np.linalg.norm(ld.age_dir) == pytest.approx(1.0)
    for z in random_latents(10):
        out = ld.apply_filters(z, bucket.key, "Any")
        assert out.ravel().astype("float64") @ ld.age_dir == pytest.approx(bucket.svm_offset, abs=1e-4)


# ── boundary files ─────────────────────────────────────────────────────────
def test_corrupt_cached_boundary_is_refused(tmp_cache):
    with open(str(tmp_cache / GENDER_FILE), "r+b") as f:
        f.seek(200)
        f.write(b"\x00\x01")
    with pytest.raises(downloads.ChecksumError, match=GENDER_FILE):
        LatentDirections()


def test_missing_boundaries_are_downloaded_from_the_pinned_commit(tmp_cache, monkeypatch):
    originals = {n: open(str(tmp_cache / n), "rb").read() for n in (GENDER_FILE, AGE_FILE)}
    for n in originals:
        os.remove(str(tmp_cache / n))
    requested = []

    def fake_urlretrieve(url, filename, *a, **k):
        requested.append(url)
        open(filename, "wb").write(originals[url.rsplit("/", 1)[1]])
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    assert LatentDirections()._age_mode == "svm_fallback"
    assert sorted(u.rsplit("/", 1)[1] for u in requested) == sorted(originals)
    assert all(downloads.INTERFACEGAN_COMMIT in u for u in requested)
    assert all(open(str(tmp_cache / n), "rb").read() == data for n, data in originals.items())
