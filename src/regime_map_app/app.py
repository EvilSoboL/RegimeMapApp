from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in (None, ""):
    project_src = Path(__file__).resolve().parents[1]
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    ApproxModuleWindow = import_module("regime_map_app.approx.ui").ApproxModuleWindow
else:
    from .approx.ui import ApproxModuleWindow


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
