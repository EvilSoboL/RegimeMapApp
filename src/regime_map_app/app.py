from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

if __package__ in (None, ""):
    project_src = Path(__file__).resolve().parents[1]
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    RegimeMapMainWindow = import_module("regime_map_app.main_window").RegimeMapMainWindow
    resolve_app_icon_path = import_module("regime_map_app.resources").resolve_app_icon_path
else:
    from .main_window import RegimeMapMainWindow
    from .resources import resolve_app_icon_path


def create_application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("RegimeMapApp")
        icon_path = resolve_app_icon_path()
        if icon_path is not None:
            app.setWindowIcon(QIcon(str(icon_path)))
    return app


def main() -> int:
    app = create_application()
    window = RegimeMapMainWindow()
    window.show()
    return app.exec()
