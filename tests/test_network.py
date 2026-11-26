"""Internet tests. Skipped by default; run with:  pytest -m network"""
import urllib.request

import pytest

import downloads

pytestmark = pytest.mark.network


@pytest.mark.parametrize("name", sorted(downloads.BOUNDARY_SHA256))
def test_pinned_boundary_download_matches_its_checksum(name, tmp_path):
    dest = str(tmp_path / name)
    downloads.fetch_verified(
        lambda tmp: urllib.request.urlretrieve(downloads.boundary_url(name), tmp),
        dest, downloads.BOUNDARY_SHA256[name])
    assert downloads.sha256_of(dest) == downloads.BOUNDARY_SHA256[name]
