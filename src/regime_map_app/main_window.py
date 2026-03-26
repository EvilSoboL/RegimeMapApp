from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from .approx.ui import ApproxModuleWidget
from .diff_surface.ui import DiffSurfaceModuleWidget


class RegimeMapMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RegimeMapApp")
        self.resize(1320, 820)

        self.tabs = QTabWidget(self)
        self.approx_widget = ApproxModuleWidget(parent=self.tabs)
        self.diff_surface_widget = DiffSurfaceModuleWidget(parent=self.tabs)

        self.tabs.addTab(self.approx_widget, "Аппроксимация")
        self.tabs.addTab(self.diff_surface_widget, "Дифференциальная поверхность")
        self.setCentralWidget(self.tabs)

        self.approx_widget.output_ready.connect(self.diff_surface_widget.apply_suggested_input_path)
