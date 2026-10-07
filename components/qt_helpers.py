"""Small Qt/numpy helpers: re-polish, array -> QImage/QPixmap, rounded pixmaps, thumbnail icons."""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QPainterPath

from .constants import THUMB_SIZE


def _repolish(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def array_to_qimage(img_array):
    arr = np.asarray(img_array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr)
    h, w, ch = arr.shape
    fmt = QImage.Format_RGBA8888 if ch == 4 else QImage.Format_RGB888
    return QImage(arr.data, w, h, ch * w, fmt).copy()


def array_to_pixmap(img_array, max_size=None):
    pixmap = QPixmap.fromImage(array_to_qimage(img_array))
    if max_size:
        pixmap = pixmap.scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


def _rounded(src, size, radius, scale=2.0):
    px = int(size * scale)
    out = QPixmap(px, px)
    out.fill(Qt.transparent)
    fitted = src.scaled(px, px, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    path = QPainterPath()
    path.addRoundedRect(0, 0, px, px, radius * scale, radius * scale)
    p.setClipPath(path)
    p.drawPixmap((px - fitted.width()) // 2, (px - fitted.height()) // 2, fitted)
    p.end()
    out.setDevicePixelRatio(scale)
    return out


def make_thumb_icon(src):
    """Thumbnail icon: full-bleed when idle, inset (ring gap) when selected."""
    inner = THUMB_SIZE - 4  # 2px border on each side
    off = _rounded(src, inner, 10)
    selected = QPixmap(int(inner * 2), int(inner * 2))
    selected.fill(Qt.transparent)
    selected.setDevicePixelRatio(2.0)
    inset = _rounded(src, inner - 6, 8)
    p = QPainter(selected)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawPixmap(3, 3, inset)  # logical px: centered inside the ring gap
    p.end()
    icon = QIcon()
    icon.addPixmap(off, QIcon.Normal, QIcon.Off)
    icon.addPixmap(selected, QIcon.Normal, QIcon.On)
    return icon
