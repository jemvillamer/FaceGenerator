"""Sidebar control logic: number-of-images stepper (buttons or typed), gender, reset."""
import numpy as np

from .constants import (
    DEFAULT_AGE, DEFAULT_GENDER, DEFAULT_IMAGE_COUNT, GENDERS, MAX_IMAGES,
)


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
        # by button id, not label text, so relabelling a button can't change
        # what the engine receives
        gid = self.gender_group.checkedId()
        return GENDERS[gid] if gid >= 0 else "Any"

    def _apply_filter_defaults(self):
        """Set age, gender and image count to the defaults (startup and Reset)."""
        self.age_combo.setCurrentIndex(self.age_combo.findData(DEFAULT_AGE))
        self.gender_group.button(GENDERS.index(DEFAULT_GENDER)).setChecked(True)
        self._set_image_count(DEFAULT_IMAGE_COUNT)

    def reset_filters(self):
        self._apply_filter_defaults()
