"""Toolbar actions on the current image: Save as, Copy, Delete."""
import os

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QToolTip
from PyQt5.QtCore import QPoint, QRect, QByteArray, QBuffer, QIODevice, QMimeData
from PyQt5.QtGui import QImage

from .qt_helpers import array_to_pixmap, array_to_qimage


class ImageActionsMixin:
    def save_current_as(self):
        entry = self._current_entry()
        if not entry:
            return
        default_name = os.path.basename(entry["path"])
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", default_name, "PNG Images (*.png);;All Files (*)"
        )
        if path:
            array_to_pixmap(entry["array"]).save(path)

    def copy_current_image(self):
        entry = self._current_entry()
        if not entry:
            return
        try:
            image = array_to_qimage(entry["array"]).convertToFormat(QImage.Format_RGB32)
            png = QByteArray()
            buf = QBuffer(png)
            buf.open(QIODevice.WriteOnly)
            image.save(buf, "PNG")
            buf.close()
            mime = QMimeData()
            mime.setImageData(image)
            mime.setData("image/png", png)
            QApplication.clipboard().setMimeData(mime)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Copy Failed", str(e))
            return
        QToolTip.showText(
            self.btn_copy.mapToGlobal(QPoint(0, self.btn_copy.height() + 4)),
            "Copied to clipboard",
            self.btn_copy,
            QRect(),
            1500,
        )

    def delete_current_image(self):
        entry = self._current_entry()
        if not entry:
            return
        reply = QMessageBox.question(
            self,
            "Delete Image",
            "Remove this image from the gallery and delete the file on disk?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        path = entry["path"]
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Failed", str(e))
                return
        del self._images[self._current_index]
        if not self._images:
            self._current_index = 0
            self.preview_canvas.clear_image()
            self.open_folder_btn.setEnabled(False)
        else:
            self._current_index = min(self._current_index, len(self._images) - 1)
            self._show_image_at(self._current_index)
        self._refresh_gallery()
        self._update_nav_and_toolbar()
        if self._images and not self._generating:
            self._set_status(
                "complete",
                f"Generated {len(self._images)} image{'s' if len(self._images) != 1 else ''}",
                "",
                show_check=True,
                progress=len(self._images),
            )
