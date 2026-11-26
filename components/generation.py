"""Model loading, output folder handling and the image-generation workflow."""
import os
import sys
import subprocess

from PyQt5.QtWidgets import QMessageBox, QFileDialog

import log_setup
from engine import ModelLoaderThread, GeneratorThread
from .icons import make_icon
from .qt_helpers import _repolish, array_to_pixmap, make_thumb_icon

# Generate button: (label, icon) for each mode
GEN_BUTTON_MODES = {
    "generate": ("  Generate images", "sparkle"),
    "cancel": ("  Cancel", "x"),
}


class GenerationMixin:
    def _set_gen_button_mode(self, mode):
        """Switch the Generate button between starting and cancelling a run."""
        label, icon = GEN_BUTTON_MODES[mode]
        self.gen_btn.setText(label)
        self.gen_btn.setIcon(make_icon(icon, "#ffffff" if mode == "generate" else "#ff9a9a",
                                       "#4a5266", size=20))
        self.gen_btn.setProperty("mode", mode)
        _repolish(self.gen_btn)

    def on_gen_btn_clicked(self):
        if self._generating:
            self.cancel_generation()
        else:
            self.generate_images()

    def cancel_generation(self):
        thread = self.generator_thread
        if not self._generating or thread is None or self._cancelling:
            return
        self._cancelling = True
        thread.requestInterruption()  # worker stops after the current render
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("  Stopping...")
        self._set_status(
            "generating",
            f"Generating {self._total_count} image{'s' if self._total_count > 1 else ''}...",
            "Stopping after the current image",
            show_check=False,
        )

    def load_model_async(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_load_error)
        self.loader_thread.start()

    def on_model_loaded(self, Gs, session, graph):
        self.Gs = Gs
        self.tf_session = session
        self.tf_graph = graph
        self.gen_btn.setEnabled(True)
        self._set_status("idle", "Ready to generate", "Choose filters and click Generate images")

    def on_model_load_error(self, err):
        self._set_status("error", "Failed to load model",
                         "Details in logs\\facegen.log" if log_setup.active_log_file()
                         else "See details in the dialog")
        QMessageBox.critical(self, "Model Load Error",
                             f"Failed to load the model:\n\n{err}{log_setup.details_hint()}")

    def browse_output(self):
        start = self.path_input.text() or self._output_dir
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start)
        if dir_path:
            self._output_dir = dir_path
            self.path_input.setText(dir_path)

    def open_output_folder(self):
        path = self.path_input.text().strip() or self._output_dir
        if not os.path.isdir(path):
            return
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def generate_images(self):
        if self.Gs is None:
            QMessageBox.warning(self, "Not Ready", "Model is still loading. Please wait.")
            return

        num_images = self._image_count_value()
        age = self.age_combo.currentData() or self.age_combo.currentText()
        gender = self._selected_gender()
        out_dir = self.path_input.text().strip() or self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        self._output_dir = out_dir

        self._images.clear()
        self._current_index = 0
        self._total_count = num_images
        self._generating = True
        self._cancelling = False
        self._follow_latest = True
        self.progress_bar.setMaximum(num_images)
        self.progress_bar.setValue(0)
        self._set_gen_button_mode("cancel")
        self.btn_browse.setEnabled(False)  # output folder is fixed for the run
        self.open_folder_btn.setEnabled(False)
        self.preview_canvas.clear_image()
        self._refresh_gallery()
        self._update_nav_and_toolbar()
        self._set_status(
            "generating",
            f"Generating {num_images} image{'s' if num_images > 1 else ''}...",
            "This may take a while on CPU",
            show_check=False,
            progress=0,
        )

        self.generator_thread = GeneratorThread(
            num_images, age, gender, out_dir, self.Gs, self.tf_session, self.tf_graph
        )
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.preview.connect(self.on_image_generated)
        self.generator_thread.completed.connect(self.generation_finished)
        self.generator_thread.error.connect(self.generation_error)
        self.generator_thread.start()

    def on_image_generated(self, img_array, path):
        # `path` is where the worker actually saved the file; don't rebuild it
        # from the gallery index, which drifts once an image is deleted mid-run.
        # Only the path and thumbnail are kept; the full image is read back from
        # its file when needed (keeping arrays cost ~3 MB per image).
        idx = len(self._images)
        pixmap = array_to_pixmap(img_array)
        self._images.append({"path": path, "icon": make_thumb_icon(pixmap)})
        if self._follow_latest:
            # show the newest image in the big preview right away, from memory
            self._show_image_at(idx, pixmap=pixmap)  # also refreshes the thumbnails
            self.preview_canvas.repaint()   # force a paint so every image is seen
        else:
            self._refresh_gallery()         # user is browsing: just fill the slot
            self._update_nav_and_toolbar()

    def update_progress(self, val):
        self.progress_bar.setValue(val)
        self._set_status(
            "generating",
            f"Generating {self._total_count} image{'s' if self._total_count > 1 else ''}...",
            "Stopping after the current image" if self._cancelling
            else f"{val} of {self._total_count} complete",
            show_check=False,
            progress=val,
        )

    def _end_run(self):
        """Restore the controls once the worker has stopped (any outcome)."""
        self._generating = False
        self._cancelling = False
        self._set_gen_button_mode("generate")
        self.gen_btn.setEnabled(True)
        self.btn_browse.setEnabled(True)

    def generation_finished(self, saved, cancelled):
        self._end_run()
        count = len(self._images)
        self.open_folder_btn.setEnabled(count > 0)
        if count:
            self._current_index = min(self._current_index, count - 1)
            self._show_image_at(self._current_index)
        self._refresh_gallery()
        if cancelled:
            title = f"Stopped at {saved} of {self._total_count}"  # short: shares a row with "Open folder"
        else:
            title = f"Generated {count} image{'s' if count != 1 else ''}"
        self._set_status(
            "complete",
            title,
            "",
            show_check=True,
            progress=saved if cancelled else self.progress_bar.maximum(),
        )

    def generation_error(self, err):
        self._end_run()
        self._refresh_gallery()
        self._set_status("error", "Error during generation", "See details in the dialog")
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{err}{log_setup.details_hint()}")
