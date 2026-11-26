"""The cache folder is anchored to the project, not to the launch folder."""
import os
import subprocess
import sys

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
