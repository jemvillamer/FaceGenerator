import sys

from PyQt5.QtWidgets import QApplication

from components.window import FaceGeneratorApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = FaceGeneratorApp()
    ex.show()
    sys.exit(app.exec_())
