"""Preview navigation and the thumbnail strip."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from .constants import THUMB_VISIBLE
from .qt_helpers import array_to_pixmap


class GalleryMixin:
    def _show_image_at(self, index):
        if not self._images:
            self.preview_canvas.clear_image()
            return
        index = max(0, min(index, len(self._images) - 1))
        self._current_index = index
        self.preview_canvas.set_image(array_to_pixmap(self._images[index]["array"]))
        self._update_nav_and_toolbar()
        self._refresh_gallery()

    def show_previous_image(self):
        if not self._images:
            return
        self._follow_latest = False
        self._show_image_at((self._current_index - 1) % len(self._images))

    def show_next_image(self):
        if not self._images:
            return
        self._follow_latest = False
        self._show_image_at((self._current_index + 1) % len(self._images))

    def _update_nav_and_toolbar(self):
        n = len(self._images)
        if n:
            self.nav_label.setText(f"Image {self._current_index + 1} of {n}")
        else:
            self.nav_label.setText("Image 0 of 0")
        enabled = n > 0
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_save_as.setEnabled(enabled)
        self.btn_copy.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)

    def _gallery_total(self):
        n = len(self._images)
        return max(n, self._total_count) if self._generating else n

    def _refresh_gallery(self):
        n = len(self._images)
        cur = self._current_index
        start = 0 if cur < THUMB_VISIBLE else cur - THUMB_VISIBLE + 1
        self._thumb_start = start
        for slot, btn in enumerate(self.thumb_buttons):
            idx = start + slot
            has = idx < n
            btn.setEnabled(has)
            btn.setCursor(Qt.PointingHandCursor if has else Qt.ArrowCursor)
            btn.setIcon(self._images[idx]["icon"] if has else QIcon())
            btn.setChecked(has and idx == cur)
        remaining = self._gallery_total() - (start + THUMB_VISIBLE)
        if remaining > 0:
            self.thumb_overflow.setText(f"+{remaining}")
            self.thumb_overflow.show()
        else:
            self.thumb_overflow.hide()

    def _on_thumb_clicked(self, slot):
        self._follow_latest = False
        idx = self._thumb_start + slot
        if idx < len(self._images):
            self._show_image_at(idx)
        else:
            self._refresh_gallery()

    def _on_overflow_clicked(self):
        self._follow_latest = False
        target = self._thumb_start + THUMB_VISIBLE
        if target < len(self._images):
            self._show_image_at(target)

    def _current_entry(self):
        if not self._images:
            return None
        return self._images[self._current_index]
