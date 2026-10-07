"""Reusable custom widgets: ElidedLabel, StepperEdit, PreviewCanvas."""
from PyQt5.QtWidgets import QLabel, QSizePolicy, QLineEdit, QFrame
from PyQt5.QtCore import Qt, QRectF, QByteArray
from PyQt5.QtGui import QPainter, QPainterPath, QColor, QFont
from PyQt5.QtSvg import QSvgRenderer

from .icons import _svg_markup
from .qt_helpers import _repolish


class ElidedLabel(QLabel):
    """Single-line label that shows 'C:\\Users\\name\\...' style truncation."""

    def __init__(self, text="", pad=28):
        super().__init__()
        self._pad = pad
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setText(text)

    def setText(self, text):
        self._full = text
        self.setToolTip(text)
        self._apply()

    def text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self):
        avail = max(self.width() - self._pad, 20)
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.ElideRight, avail))


class StepperEdit(QLineEdit):
    """Number field that lights up its parent box while focused."""

    def _set_focus_flag(self, on):
        parent = self.parentWidget()
        if parent is not None:
            parent.setProperty("focused", on)
            _repolish(parent)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._set_focus_flag(True)
        self.selectAll()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._set_focus_flag(False)


class PreviewCanvas(QFrame):
    """Dark canvas with a centered square card: placeholder or the image."""

    MARGIN = 50
    CARD_COLOR = QColor("#242833")
    CARD_RADIUS = 20

    def __init__(self):
        super().__init__()
        self.setObjectName("preview_canvas")
        self.setMinimumSize(260, 260)
        self._pixmap = None
        self._scaled = None
        self._icon_renderer = QSvgRenderer(
            QByteArray(_svg_markup("user", "#4a5266", 0.95).encode("utf-8"))
        )

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._scaled = None
        self.update()

    def clear_image(self):
        self.set_image(None)

    def _card_rect(self):
        side = max(min(self.width(), self.height()) - 2 * self.MARGIN, 140)
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self._card_rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(rect, self.CARD_RADIUS, self.CARD_RADIUS)
        p.fillPath(path, self.CARD_COLOR)

        if self._pixmap is not None and not self._pixmap.isNull():
            target = rect.size().toSize()
            if self._scaled is None or self._scaled.size() != target:
                self._scaled = self._pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            p.setClipPath(path)
            x = rect.x() + (rect.width() - self._scaled.width()) / 2
            y = rect.y() + (rect.height() - self._scaled.height()) / 2
            p.drawPixmap(int(x), int(y), self._scaled)
            return

        # placeholder: person glyph + hint text
        box = 150.0
        cx, cy = rect.center().x(), rect.center().y()
        self._icon_renderer.render(
            p, QRectF(cx - box / 2, cy - 13 - box / 2, box, box)
        )
        font = QFont(self.font())
        font.setPixelSize(14)
        p.setFont(font)
        p.setPen(QColor("#9aa3b5"))
        p.drawText(
            QRectF(rect.x(), cy + 62, rect.width(), 28),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "Generated face appears here, scaled to fit",
        )
