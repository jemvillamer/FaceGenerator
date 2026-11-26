"""Draw the app icon (packaging/facegen.ico): the app's accent blue with the
"user" glyph used in the sidebar. Run by build.ps1 before PyInstaller; needs
PyQt5 and Pillow (both in requirements.txt).

    python packaging/make_icon.py
"""
import io
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window needed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import PIL.Image  # noqa: E402
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PyQt5.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from components.icons import svg_pixmap  # noqa: E402

ACCENT = "#2f6fdb"   # style.qss accent
SIZE = 256
OUT = os.path.join(HERE, "facegen.ico")


def render():
    img = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ACCENT))
    p.drawRoundedRect(QRectF(8, 8, SIZE - 16, SIZE - 16), 48, 48)
    glyph = svg_pixmap("user", "#ffffff", SIZE * 5 // 8, stroke=2.0, scale=1)
    p.drawPixmap((SIZE - glyph.width()) // 2, (SIZE - glyph.height()) // 2, glyph)
    p.end()

    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return PIL.Image.open(io.BytesIO(bytes(data)))


def main():
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841 (needed for QPixmap)
    render().save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                              (128, 128), (256, 256)])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
