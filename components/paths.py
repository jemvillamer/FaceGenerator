"""Filesystem locations."""
import os

# Folder that holds main.py, engine.py and style.qss
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_output_dir():
    path = os.path.join(os.path.expanduser("~"), "FaceGenerator", "output")
    os.makedirs(path, exist_ok=True)
    return path
