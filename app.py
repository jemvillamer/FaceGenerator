import logging
import sys

import log_setup
# Before anything imports TensorFlow (components.window -> engine -> model),
# so its console noise can still be turned down.
log_file = log_setup.configure()

from PyQt5.QtWidgets import QApplication  # noqa: E402

from components.window import FaceGeneratorApp  # noqa: E402


if __name__ == "__main__":
    logging.getLogger("app").info("Starting (Python %s); log file: %s",
                                  sys.version.split()[0], log_file)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = FaceGeneratorApp()
    log_setup.set_error_reporter(ex.report_unexpected_error)
    ex.show()
    sys.exit(app.exec_())
