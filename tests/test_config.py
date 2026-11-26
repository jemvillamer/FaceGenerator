"""Where data files and logs live: the project folder when running from source,
the user's local app data when running as the packaged app."""
import importlib
import os
import subprocess
import sys

import pytest

import config

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cache_dir_is_the_project_cache_folder():
    assert os.path.isabs(config.cache_dir)
    assert os.path.normcase(config.cache_dir) == os.path.normcase(os.path.join(PROJECT, "cache"))


def test_cache_dir_does_not_depend_on_the_launch_folder(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                               "import config; print(config.cache_dir)", PROJECT],
        cwd=str(tmp_path), capture_output=True, text=True, check=True)
    assert os.path.normcase(out.stdout.strip()) == os.path.normcase(os.path.join(PROJECT, "cache"))


def test_data_file_reads_from_the_cache_when_running_from_source():
    assert config.bundled_data_dir is None
    assert config.data_file("x.npy") == os.path.join(config.cache_dir, "x.npy")


@pytest.fixture
def frozen_config(tmp_path, monkeypatch):
    """config as the packaged app sees it: sys.frozen set, bundle in tmp_path."""
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    yield importlib.reload(config), bundle
    monkeypatch.undo()
    importlib.reload(config)  # back to the from-source paths for other tests


def test_packaged_app_writes_to_local_app_data(frozen_config, tmp_path):
    cfg, _ = frozen_config
    base = tmp_path / "appdata" / "FaceGenerator"
    assert cfg.frozen
    assert cfg.cache_dir == str(base / "cache")
    assert cfg.log_dir == str(base / "logs")


def test_packaged_app_prefers_shipped_data_files(frozen_config):
    cfg, bundle = frozen_config
    (bundle / "data" / "shipped.npz").write_bytes(b"x")
    assert cfg.data_file("shipped.npz") == str(bundle / "data" / "shipped.npz")
    # Not shipped: read from (and downloaded to) the writable cache.
    assert cfg.data_file("other.npy") == os.path.join(cfg.cache_dir, "other.npy")
