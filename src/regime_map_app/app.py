from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from regime_map_app.approx.ui import ApproxModuleWindow


def create_application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("RegimeMapApp")
    return app


def main() -> int:
    app = create_application()
    window = ApproxModuleWindow()
    window.show()
    return app.exec()
