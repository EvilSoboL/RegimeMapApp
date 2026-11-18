from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .diff_surface.exceptions import DiffSurfaceError
from .diff_surface.pipeline import DiffSurfacePipeline
from .fuel_correlation import FUEL_PROPERTIES, FuelSurface, find_fuel_surfaces, laplace_number, optimal_slope


class FuelPropertiesWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, int] = {}
        self._entries: dict[str, tuple[str, ...]] = {}
        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels([
            "Топливо", "Плотность, кг/м³", "Поверхностное\nнатяжение, мН/м",
            "Динамическая\nвязкость, мПа·с", "Диаметр, мм", "La", "Состав и примечания",
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.cellChanged.connect(self._update_row)

        layout = QVBoxLayout(self)
        self.summary = QLabel("Выберите папку во вкладке «Корреляция».")
        layout.addWidget(self.summary)
        layout.addWidget(self.table)
        formula = QLabel(
            "La = ρσD/μ². Начальный диаметр капли 2 мм можно изменить для каждого топлива. "
            "Источники дополнительных справочных значений указаны в примечаниях."
        )
        formula.setWordWrap(True)
        layout.addWidget(formula)

    def set_surfaces(self, surfaces: list[FuelSurface]) -> None:
        fuels: dict[str, list[FuelSurface]] = {}
        for surface in surfaces:
            fuels.setdefault(surface.key, []).append(surface)
        self._rows = {key: row for row, key in enumerate(fuels)}
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(fuels))
            for key, files in fuels.items():
                row = self._rows[key]
                properties = FUEL_PROPERTIES.get(key)
                label = properties.label if properties else files[0].fuel_name
                defaults = (
                    str(properties.density), str(properties.surface_tension), str(properties.viscosity),
                    "2", properties.composition,
                ) if properties else ("", "", "", "2", "")
                entries = self._entries.get(key, defaults)
                self._set_read_only(row, 0, label).setToolTip("\n".join(file.path.name for file in files))
                for column, entry in zip((1, 2, 3, 4, 6), entries):
                    item = QTableWidgetItem(entry)
                    item.setToolTip(entry)
                    self.table.setItem(row, column, item)
                self._set_read_only(row, 5, "—")
                self._update_row(row, 1)
        finally:
            self.table.blockSignals(False)
        self.summary.setText(f"Топлив в выбранной папке: {len(fuels)}. Свойства можно редактировать в таблице.")

    def laplace(self, key: str) -> float:
        row = self._rows[key]
        try:
            values = [float(self.table.item(row, column).text().replace(",", ".")) for column in range(1, 5)]
        except ValueError as exc:
            raise ValueError("Заполните плотность, поверхностное натяжение, вязкость и диаметр.") from exc
        return laplace_number(*values)

    def label(self, key: str) -> str:
        return self.table.item(self._rows[key], 0).text()

    def _set_read_only(self, row: int, column: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, column, item)
        return item

    def _update_row(self, row: int, column: int) -> None:
        if column not in (1, 2, 3, 4, 6):
            return
        key = next(key for key, index in self._rows.items() if index == row)
        item = self.table.item(row, column)
        entries = tuple(self.table.item(row, index).text() for index in (1, 2, 3, 4, 6))
        self._entries[key] = entries
        try:
            display = f"{self.laplace(key):.7g}"
        except ValueError:
            display = "—"
        previously_blocked = self.table.blockSignals(True)
        try:
            item.setToolTip(item.text())
            self.table.item(row, 5).setText(display)
        finally:
            self.table.blockSignals(previously_blocked)


class FuelCorrelationWidget(QWidget):
    def __init__(self, properties: FuelPropertiesWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.properties = properties
        self.pipeline = DiffSurfacePipeline()
        self._folder: Path | None = None
        self._surfaces: list[FuelSurface] = []
        self._has_plot = False

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Папка с поверхностями approx_*.csv")
        browse = QPushButton("Выбрать папку…")
        browse.clicked.connect(self._browse)
        self.folder.editingFinished.connect(self._load_folder)
        self.folder.textChanged.connect(self._invalidate)
        self.plot_button = QPushButton("Построить график")
        self.plot_button.clicked.connect(self.build_plot)
        self.steam_units = QComboBox()
        self.steam_units.addItem("г/ч", 0.001)
        self.steam_units.addItem("кг/ч", 1.0)
        self.steam_units.currentIndexChanged.connect(self._invalidate)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        row.addWidget(QLabel("Пар в CSV"))
        row.addWidget(self.steam_units)
        row.addWidget(self.plot_button)
        layout.addLayout(row)

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("Выберите папку с поверхностями CO для пара.")
        layout.addWidget(self.status)
        self.issues = QPlainTextEdit(readOnly=True)
        self.issues.setMaximumHeight(100)
        self.issues.hide()
        layout.addWidget(self.issues)
        self.properties.table.cellChanged.connect(self._property_changed)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка с поверхностями CSV", self.folder.text())
        if folder:
            self.folder.setText(folder)
            self._load_folder()

    def _load_folder(self, *, refresh: bool = False) -> bool:
        text = self.folder.text().strip()
        folder = Path(text) if text else None
        if folder is None or not folder.is_dir():
            self._folder = None
            self._surfaces = []
            self.properties.set_surfaces([])
            self.status.setText("Укажите существующую папку с CSV.")
            self._show_issues([])
            return False
        if folder == self._folder and not refresh:
            return True
        self._folder = folder
        try:
            self._surfaces, self._skipped = find_fuel_surfaces(folder)
        except OSError as exc:
            self._surfaces, self._skipped = [], [str(exc)]
        self.properties.set_surfaces(self._surfaces)
        self.status.setText(
            f"Найдено поверхностей: {len(self._surfaces)}. Свойства доступны во вкладке «Свойства топлива»."
        )
        self._show_issues(self._skipped)
        return True

    def _show_issues(self, issues: list[str]) -> None:
        self.issues.setPlainText("\n".join(issues))
        self.issues.setVisible(bool(issues))

    def _property_changed(self, _row: int, column: int) -> None:
        if column in range(1, 5):
            self._invalidate()

    def _invalidate(self) -> None:
        if self._has_plot:
            self._has_plot = False
            self.figure.clear()
            self.canvas.draw_idle()
            self.status.setText("Входные данные изменены. Постройте график заново.")

    def build_plot(self) -> None:
        self._invalidate()
        if not self._load_folder(refresh=True):
            return
        issues = list(self._skipped)
        points = []
        counts = Counter(surface.key for surface in self._surfaces)
        for surface in self._surfaces:
            try:
                laplace = self.properties.laplace(surface.key)
                slope = optimal_slope(surface, self.steam_units.currentData(), self.pipeline)
            except (ValueError, DiffSurfaceError) as exc:
                issues.append(f"{surface.path.name}: {exc}")
                continue
            label = self.properties.label(surface.key)
            if counts[surface.key] > 1:
                label = f"{label} ({surface.path.stem[-10:]})"
            points.append((laplace, slope, label))
        self._show_issues(issues)
        if not points:
            self.status.setText("Нет поверхностей с данными, достаточными для построения графика.")
            return

        axis = self.figure.subplots()
        for laplace, slope, label in points:
            axis.scatter(laplace, slope, label=label)
        x, y = np.asarray([(point[0], point[1]) for point in points]).T
        title = "Угловой коэффициент"
        message = f"Нанесено точек: {len(points)} из {len(self._surfaces)}."
        if np.unique(x).size >= 2:
            fit = np.polyfit(x, y, 1)
            endpoints = np.asarray([x.min(), x.max()])
            axis.plot(endpoints, np.polyval(fit, endpoints), color="black", label="Аппроксимация")
            if np.ptp(y) > 0:
                r_squared = 1 - np.sum((y - np.polyval(fit, x)) ** 2) / np.sum((y - y.mean()) ** 2)
                title += f" (R² = {r_squared:.3f})"
        else:
            message += " Для линейной аппроксимации нужны два различных числа Лапласа."
        axis.set(
            title=title, xlabel="Число Лапласа",
            ylabel="Угловой коэффициент оптимального режима",
        )
        axis.grid(True, alpha=0.65)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        self.status.setText(message)
        self._has_plot = True
        self.canvas.draw_idle()
