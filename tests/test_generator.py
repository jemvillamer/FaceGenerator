"""generator.py: output numbering, the age-check retry loop and batch runs."""
import os

import numpy as np
import PIL.Image
import pytest

import downloads
import generator
from conftest import GENDER_FILE, DummyContext, FakeEstimator, FakeGs
from generator import generate_batch, generate_one, next_output_index
from latents import AGE_BUCKETS, DEFAULT_TRUNCATION_PSI, LatentDirections

FMT = {"marker": "output transform"}


# ── output numbering ───────────────────────────────────────────────────────
def touch(folder, *names):
    for n in names:
        open(os.path.join(str(folder), n), "wb").close()


def test_numbering_starts_at_zero(tmp_path):
    assert next_output_index(str(tmp_path / "missing")) == 0
    assert next_output_index(str(tmp_path)) == 0


def test_numbering_continues_after_the_highest_matching_file(tmp_path):
    touch(tmp_path, "generated_0000.png", "generated_0002.png", "GENERATED_0007.PNG", "photo.png",
          "generated_abc.png", "generated_0099.jpg", "my_generated_0500.png")
    assert next_output_index(str(tmp_path)) == 8


def test_numbering_handles_five_digits(tmp_path):
    touch(tmp_path, "generated_12345.png")
    assert next_output_index(str(tmp_path)) == 12346


# ── generate_one: the retry loop ───────────────────────────────────────────
def run_one(age, estimator=None, use_estimator=None, should_stop=lambda: False):
    if estimator is not None:
        use_estimator(estimator)
    gs = FakeGs()
    img, latents = generate_one(gs, LatentDirections(), age, "Any", FMT, should_stop)
    return gs, img, latents


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_tolerance_edges(regression_cache, use_estimator, bucket):
    lowest, highest = bucket.lo - bucket.tolerance[0], bucket.hi + bucket.tolerance[1]
    for age, accepted in [(lowest, True), (highest, True), ((lowest + highest) / 2, True),
                          (lowest - 0.01, False), (highest + 0.01, False)]:
        gs, img, _ = run_one(bucket.key, FakeEstimator(age), use_estimator)
        assert len(gs.calls) == (1 if accepted else bucket.max_tries), (age, accepted)
        assert img is not None


@pytest.mark.parametrize("bucket", AGE_BUCKETS, ids=lambda b: b.key)
def test_every_attempt_uses_the_bucket_truncation(regression_cache, use_estimator, bucket):
    gs, _, _ = run_one(bucket.key, FakeEstimator(-100.0), use_estimator)  # always rejected
    assert [c["truncation_psi"] for c in gs.calls] == [bucket.truncation_psi] * bucket.max_tries
    assert all(c["output_transform"] is FMT and c["randomize_noise"] for c in gs.calls)


def test_rejected_attempts_return_the_last_image(regression_cache, use_estimator):
    bucket = AGE_BUCKETS[2]
    gs, img, _ = run_one(bucket.key, FakeEstimator(-100.0), use_estimator)
    assert img[0, 0, 0] == len(gs.calls) == bucket.max_tries


def test_no_face_detected_counts_as_a_retry(regression_cache, use_estimator):
    gs, img, _ = run_one("26-35 years old", FakeEstimator([None, None, 30.0]), use_estimator)
    assert len(gs.calls) == 3 and img[0, 0, 0] == 3


def test_any_age_makes_one_attempt_without_age_check(regression_cache, use_estimator):
    est = use_estimator(FakeEstimator(99.0))
    gs, img, _ = run_one("Any")
    assert len(gs.calls) == 1
    assert gs.calls[0]["truncation_psi"] == DEFAULT_TRUNCATION_PSI
    assert est.calls == 0


def test_without_an_estimator_the_first_image_is_used(regression_cache):
    gs, img, _ = run_one("1-11 years old")  # default fixture: estimator unavailable
    assert len(gs.calls) == 1


def test_stop_during_retries_returns_nothing(regression_cache, use_estimator):
    use_estimator(FakeEstimator(-100.0))
    gs = FakeGs()
    stop_after_first = lambda: len(gs.calls) >= 1
    img, latents = generate_one(gs, LatentDirections(), "26-35 years old", "Any", FMT, stop_after_first)
    assert (img, latents) == (None, None)
    assert len(gs.calls) == 1


def test_latents_passed_to_the_network_are_the_edited_ones(regression_cache, use_estimator):
    gs, _, latents = run_one("26-35 years old", FakeEstimator(30.0), use_estimator)
    assert np.array_equal(gs.calls[-1]["latents"], latents)


# ── generate_batch ─────────────────────────────────────────────────────────
def batch(out_dir, n, age="Any", should_stop=lambda: False, on_saved=None, gs=None):
    gs = gs or FakeGs()
    events = []
    saved = generate_batch(gs, DummyContext(), DummyContext(), n, age, "Female", str(out_dir), FMT,
                           should_stop=should_stop,
                           on_saved=on_saved or (lambda c, img, p: events.append((c, img.copy(), p))))
    return saved, events, gs


def test_batch_saves_numbered_files_matching_the_images(regression_cache, tmp_path):
    out = tmp_path / "out"
    saved, events, _ = batch(out, 3)
    assert saved == 3
    assert [(c, os.path.basename(p)) for c, _, p in events] == [
        (1, "generated_0000.png"), (2, "generated_0001.png"), (3, "generated_0002.png")]
    for _, img, path in events:
        assert np.array_equal(np.array(PIL.Image.open(path)), img)


def test_batch_continues_numbering_and_never_overwrites(regression_cache, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    PIL.Image.fromarray(np.full((4, 4, 3), 200, np.uint8)).save(str(out / "generated_0007.png"))
    before = open(str(out / "generated_0007.png"), "rb").read()
    saved, events, _ = batch(out, 2)
    assert [os.path.basename(p) for _, _, p in events] == ["generated_0008.png", "generated_0009.png"]
    assert open(str(out / "generated_0007.png"), "rb").read() == before


def test_batch_skips_a_name_that_appears_mid_run(regression_cache, tmp_path):
    out = tmp_path / "out"
    seen = []

    def on_saved(count, img, path):
        seen.append(os.path.basename(path))
        if count == 1:  # someone else creates the next name
            open(str(out / "generated_0001.png"), "wb").write(b"not ours")
    batch(out, 2, on_saved=on_saved)
    assert seen == ["generated_0000.png", "generated_0002.png"]
    assert open(str(out / "generated_0001.png"), "rb").read() == b"not ours"


def test_batch_stops_when_asked(regression_cache, tmp_path):
    out = tmp_path / "out"
    gs = FakeGs()
    saved, events, _ = batch(out, 5, should_stop=lambda: len(gs.calls) >= 2, gs=gs)
    assert saved == 2 and len(os.listdir(str(out))) == 2


def test_batch_propagates_checksum_errors(regression_cache, tmp_path):
    out = tmp_path / "out"
    with open(str(regression_cache / GENDER_FILE), "r+b") as f:
        f.seek(300)
        f.write(b"\xff\xff")
    with pytest.raises(downloads.ChecksumError):
        batch(out, 1)
    assert not out.exists() or os.listdir(str(out)) == []
