"""Stylesheet loading (style.qss) and the generated dropdown-chevron icon files."""
import os
import tempfile

from .icons import svg_pixmap
from .paths import PROJECT_DIR


def prepare_icon_files():
    """QSS can only reference icons by file path, so write the dropdown
    chevron (1x + @2x) into a temp folder and return its path."""
    folder = os.path.join(tempfile.gettempdir(), "facegen_icons")
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, "chevron_down.png")
    svg_pixmap("chevron_down", "#9aa3b5", 16, stroke=2.0, scale=1).save(base, "PNG")
    svg_pixmap("chevron_down", "#9aa3b5", 16, stroke=2.0, scale=2).save(
        os.path.join(folder, "chevron_down@2x.png"), "PNG"
    )
    return base.replace("\\", "/")


def load_stylesheet():
    qss_path = os.path.join(PROJECT_DIR, "style.qss")
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            qss = f.read()
    except FileNotFoundError:
        print(f"[Warning] style.qss not found at {qss_path}. Running without stylesheet.")
        return ""
    return qss.replace("@CHEVRON_DOWN@", prepare_icon_files())
