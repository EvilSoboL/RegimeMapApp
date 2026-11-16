from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QTabWidget

from .approx.ui import ApproxModuleWidget
from .diff_surface.ui import DiffSurfaceModuleWidget
from .experimental_ui import ExperimentalDataWidget
from .regime_map.ui import RegimeMapModuleWidget


class RegimeMapMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RegimeMapApp")
        self.resize(1320, 820)

        self.tabs = QTabWidget(self)
        self.experimental_widget = ExperimentalDataWidget(parent=self.tabs)
        self.approx_widget = ApproxModuleWidget(parent=self.tabs)
        self.diff_surface_widget = DiffSurfaceModuleWidget(parent=self.tabs)
        self.regime_map_widget = RegimeMapModuleWidget(parent=self.tabs)

        self.tabs.addTab(self.experimental_widget, "Экспериментальные данные")
        self.tabs.addTab(self.approx_widget, "Аппроксимация")
        self.tabs.addTab(self.diff_surface_widget, "Дифференциальная поверхность")
        self.tabs.addTab(self.regime_map_widget, "Режимная карта")
        self.setCentralWidget(self.tabs)

        self.experimental_widget.csv_ready.connect(self._open_prepared_csv)
        self.approx_widget.output_ready.connect(self.diff_surface_widget.apply_suggested_input_path)
        self.approx_widget.output_ready.connect(self.regime_map_widget.apply_suggested_input_path)

    def _open_prepared_csv(self, filename: str) -> None:
        if self.approx_widget.load_prepared_csv(Path(filename)):
            self.tabs.setCurrentWidget(self.approx_widget)
        else:
            self.experimental_widget.status_label.setText(
                f"CSV сохранён: {filename}. Аппроксимация занята, выберите файл после завершения."
            )
