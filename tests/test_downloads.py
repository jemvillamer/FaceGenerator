"""downloads.py: files are verified before use, and a failed or wrong download
never ends up under the real file name."""
import hashlib
import os

import pytest

import downloads
from downloads import ChecksumError, fetch_verified, sha256_of, verify_existing

GOOD = b"the expected file contents" * 100
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
HTML = b"<!DOCTYPE html><html><title>Google Drive - Virus scan warning</title></html>"


def writer(data):
    return lambda tmp: open(tmp, "wb").write(data)


@pytest.fixture
def dest(tmp_path):
    return str(tmp_path / "file.bin")


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "abc"
    p.write_bytes(b"abc")
    assert sha256_of(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_good_download_is_put_in_place(dest):
    fetch_verified(writer(GOOD), dest, GOOD_SHA)
    assert open(dest, "rb").read() == GOOD
    assert not os.path.exists(dest + ".part")


def test_interrupted_download_leaves_nothing(dest):
    def interrupted(tmp):
        open(tmp, "wb").write(GOOD[:100])
        raise ConnectionError("connection reset")
    with pytest.raises(ConnectionError):
        fetch_verified(interrupted, dest, GOOD_SHA)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_wrong_content_is_refused_and_removed(dest):
    with pytest.raises(ChecksumError, match="doesn't match the expected checksum"):
        fetch_verified(writer(HTML), dest, GOOD_SHA)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_downloader_that_writes_nothing_gives_clear_error(dest):
    with pytest.raises(RuntimeError, match="no file was written"):
        fetch_verified(lambda tmp: None, dest, GOOD_SHA)


def test_stale_part_file_is_replaced(dest):
    open(dest + ".part", "wb").write(b"left over from an earlier attempt")
    fetch_verified(writer(GOOD), dest, GOOD_SHA)
    assert open(dest, "rb").read() == GOOD
    assert not os.path.exists(dest + ".part")


def test_verify_existing_accepts_the_right_file(dest):
    open(dest, "wb").write(GOOD)
    verify_existing(dest, GOOD_SHA)  # no exception


def test_verify_existing_rejects_a_modified_file_with_instructions(dest):
    open(dest, "wb").write(GOOD[:-1])
    with pytest.raises(ChecksumError) as err:
        verify_existing(dest, GOOD_SHA)
    assert dest in str(err.value)
    assert "Delete it and restart" in str(err.value)


def test_boundary_urls_are_pinned_to_a_commit():
    for name in downloads.BOUNDARY_SHA256:
        url = downloads.boundary_url(name)
        assert downloads.INTERFACEGAN_COMMIT in url
        assert "/master/" not in url
        assert url.endswith("/boundaries/" + name)
