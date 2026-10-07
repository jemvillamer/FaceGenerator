"""Status card behaviour (icon, text, progress colouring)."""
from .icons import svg_pixmap
from .qt_helpers import _repolish


class StatusMixin:
    STATUS_ICONS = {
        "complete": ("check", "#7fe0a6"),
        "error": ("alert", "#ff9a9a"),
        "loading": ("clock", "#fbbf24"),
        "generating": ("clock", "#8ab8ff"),
        "idle": ("dot", "#9aa3b5"),
    }

    def _set_status(self, phase, title, subtitle="", show_check=True, progress=None):
        self._status_phase = phase
        for w in (self.status_card, self.status_icon, self.status_text):
            w.setProperty("status", phase)
            _repolish(w)
        name, color = self.STATUS_ICONS.get(phase, self.STATUS_ICONS["idle"])
        if phase == "complete" and not show_check:
            name, color = self.STATUS_ICONS["idle"]
        self.status_icon.setPixmap(svg_pixmap(name, color, 18, stroke=2.0))
        self.status_text.setText(title)
        self.status_subtext.setText(subtitle)
        self.status_subtext.setVisible(bool(subtitle) and phase in ("loading", "generating", "idle", "error"))
        self.open_folder_btn.setVisible(phase == "complete")
        if progress is not None:
            self.progress_bar.setValue(progress)
        self.progress_bar.setProperty("phase", "generating" if phase == "generating" else "idle")
        _repolish(self.progress_bar)
