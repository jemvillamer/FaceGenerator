import logging
import os
import sys

# The packaged app has no console window, so sys.stdout/sys.stderr are None.
# Give them a sink: logging and gdown's progress bar write to them.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

import log_setup  # noqa: E402
# Before anything imports TensorFlow (components.window -> engine -> model),
# so its console noise can still be turned down.
log_file = log_setup.configure()

from PyQt5.QtGui import QIcon  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from components.paths import APP_ICON  # noqa: E402
from components.window import FaceGeneratorApp  # noqa: E402


def self_test(out_dir):
    """Check the app works without opening a window: load the model (downloading
    it if needed) and generate one face with an age filter, which also runs the
    InsightFace age check if it's available. Writes self_test.json to out_dir
    and returns the exit code. packaging/build.ps1 runs this on the built app:
    FaceGenerator.exe --self-test <folder>"""
    import json
    import time
    import traceback

    log = logging.getLogger("app")
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False, "log_file": log_file}
    started = time.monotonic()
    try:
        import generator
        from model import OUTPUT_TRANSFORM, load_model
        Gs, session, graph = load_model()
        result["age_check"] = generator.AgeEstimator.get()._available
        saved = []
        generator.generate_batch(
            Gs, session, graph, 1, "26-35 years old", "Female", out_dir,
            OUTPUT_TRANSFORM, on_saved=lambda count, img, path: saved.append(path))
        result.update(ok=len(saved) == 1, images=saved)
    except Exception:
        log.exception("Self-test failed")
        result["error"] = traceback.format_exc()
    result["seconds"] = round(time.monotonic() - started, 1)
    with open(os.path.join(out_dir, "self_test.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Self-test %s: %s", "passed" if result["ok"] else "FAILED", result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    logging.getLogger("app").info("Starting (Python %s); log file: %s",
                                  sys.version.split()[0], log_file)
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        sys.exit(self_test(sys.argv[2]))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    if os.path.isfile(APP_ICON):
        app.setWindowIcon(QIcon(APP_ICON))
    ex = FaceGeneratorApp()
    log_setup.set_error_reporter(ex.report_unexpected_error)
    ex.show()
    sys.exit(app.exec_())
