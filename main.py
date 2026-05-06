#!/usr/bin/env python3
"""RF Fingerprint CSI Monitor — application entry point."""

import sys
from PyQt5.QtWidgets import QApplication
from ui.style import make_palette
from ui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(make_palette())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
