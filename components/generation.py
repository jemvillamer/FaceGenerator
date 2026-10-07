"""Model loading, output folder handling and the image-generation workflow."""
import os
import sys
import subprocess

from PyQt5.QtWidgets import QMessageBox, QFileDialog

from engine import ModelLoaderThread, GeneratorThread
from .qt_helpers import array_to_pixmap, make_thumb_icon


class GenerationMixin:
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
        self._set_status("error", "Failed to load model", "Check console for details")
        QMessageBox.critical(self, "Model Load Error", f"Failed to load the model:\n\n{err}")

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
        self._follow_latest = True
        self.progress_bar.setMaximum(num_images)
        self.progress_bar.setValue(0)
        self.gen_btn.setEnabled(False)
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

        self.thread = GeneratorThread(
            num_images, age, gender, out_dir, self.Gs, self.tf_session, self.tf_graph
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.preview.connect(self.on_image_generated)
        self.thread.finished.connect(self.generation_finished)
        self.thread.error.connect(self.generation_error)
        self.thread.start()

    def on_image_generated(self, img_array):
        idx = len(self._images)
        path = os.path.join(self._output_dir, f"generated_{idx:04d}.png")
        arr = img_array.copy()
        icon = make_thumb_icon(array_to_pixmap(arr))
        self._images.append({"path": path, "array": arr, "icon": icon})
        if self._follow_latest:
            # show the newest image in the big preview right away
            self._show_image_at(idx)        # also refreshes the thumbnails
            self.preview_canvas.repaint()   # force a paint so every image is seen
        else:
            self._refresh_gallery()         # user is browsing: just fill the slot
            self._update_nav_and_toolbar()

    def update_progress(self, val):
        self.progress_bar.setValue(val)
        self._set_status(
            "generating",
            f"Generating {self._total_count} image{'s' if self._total_count > 1 else ''}...",
            f"{val} of {self._total_count} complete",
            show_check=False,
            progress=val,
        )

    def generation_finished(self, _msg):
        self._generating = False
        self.gen_btn.setEnabled(True)
        count = len(self._images)
        self.open_folder_btn.setEnabled(count > 0)
        if count:
            self._current_index = min(self._current_index, count - 1)
            self._show_image_at(self._current_index)
        self._refresh_gallery()
        self._set_status(
            "complete",
            f"Generated {count} image{'s' if count != 1 else ''}",
            "",
            show_check=True,
            progress=self.progress_bar.maximum(),
        )

    def generation_error(self, err):
        self._generating = False
        self.gen_btn.setEnabled(True)
        self._refresh_gallery()
        self._set_status("error", "Error during generation", "See details in the dialog")
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{err}")
