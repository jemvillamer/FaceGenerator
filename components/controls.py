"""Sidebar control logic: number-of-images stepper (buttons or typed), gender, reset."""
import numpy as np

from .constants import MAX_IMAGES


class ControlsMixin:

    def _read_image_count(self):
        txt = self.stepper_input.text().strip()
        try:
            val = int(txt)
        except ValueError:
            return self._image_count
        return int(np.clip(val, 1, MAX_IMAGES))

    def _set_image_count(self, n):
        self._image_count = int(np.clip(n, 1, MAX_IMAGES))
        self.stepper_input.setText(str(self._image_count))

    def _commit_image_count(self):
        self._set_image_count(self._read_image_count())

    def _image_count_value(self):
        self._commit_image_count()
        return self._image_count

    def _adjust_image_count(self, delta):
        self._set_image_count(self._read_image_count() + delta)

    def _selected_gender(self):
        checked = self.gender_group.checkedButton()
        return checked.text() if checked else "Any"

    def reset_filters(self):
        self.age_combo.setCurrentIndex(0)
        self.gender_group.button(0).setChecked(True)
        self._set_image_count(20)
