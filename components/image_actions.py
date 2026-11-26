"""Toolbar actions on the current image: Save as, Copy, Delete."""
import os

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QToolTip
from PyQt5.QtCore import QPoint, QRect, QByteArray, QBuffer, QIODevice, QMimeData
from PyQt5.QtGui import QImage

from .qt_helpers import load_image


class ImageActionsMixin:
    def _load_current_image(self, entry, action):
        """Read the current image from its file; warn and return None if it's gone."""
        image = load_image(entry["path"])
        if image is None:
            QMessageBox.warning(
                self, f"{action} Failed",
                f"The image file could not be read:\n{entry['path']}\n\n"
                "It may have been moved or deleted.",
            )
        return image

    def save_current_as(self):
        entry = self._current_entry()
        if not entry:
            return
        default_name = os.path.basename(entry["path"])
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", default_name, "PNG Images (*.png);;All Files (*)"
        )
        if not path:
            return
        image = self._load_current_image(entry, "Save")
        if image is None:
            return
        if not image.save(path):  # format comes from the extension
            QMessageBox.warning(
                self, "Save Failed",
                f"Could not save the image as:\n{path}\n\n"
                "Use a supported extension such as .png or .jpg, and check "
                "that the folder is writable.",
            )

    def copy_current_image(self):
        entry = self._current_entry()
        if not entry:
            return
        image = self._load_current_image(entry, "Copy")
        if image is None:
            return
        try:
            image = image.convertToFormat(QImage.Format_RGB32)
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
