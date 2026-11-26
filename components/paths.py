"""Filesystem locations."""
import os
import sys

# Folder that holds app.py, engine.py and style.qss. In the packaged app
# (PyInstaller) the bundled files are unpacked to sys._MEIPASS instead.
PROJECT_DIR = getattr(sys, "_MEIPASS", None) or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

# App logo: window/taskbar icon and the sidebar header. (The .exe icon,
# packaging/facegen.ico, is made from it at build time.)
APP_LOGO = os.path.join(PROJECT_DIR, "packaging", "logo.png")


def default_output_dir():
    path = os.path.join(os.path.expanduser("~"), "FaceGenerator", "output")
    os.makedirs(path, exist_ok=True)
    return path
