from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .diff_surface.exceptions import DiffSurfaceError
from .diff_surface.pipeline import DiffSurfacePipeline
from .fuel_correlation import (
    DEFAULT_FUEL_PROFILE,
    FuelProfile,
    FuelSurface,
    find_fuel_surfaces,
    laplace_number,
    load_fuel_profile,
    optimal_slope,
)


class FuelPropertiesWidget(QWidget):
    profile_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = DEFAULT_FUEL_PROFILE
        self._surfaces: list[FuelSurface] = []
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
        profile_row = QHBoxLayout()
        self.profile_label = QLabel(self.profile.name)
        self.profile_label.setTextFormat(Qt.TextFormat.PlainText)
        open_profile = QPushButton("Открыть конфиг…")
        open_profile.clicked.connect(self._open_profile)
        default_profile = QPushButton("Конфиг по умолчанию")
        default_profile.clicked.connect(lambda: self._apply_profile(DEFAULT_FUEL_PROFILE))
        profile_row.addWidget(QLabel("Профиль топлива"))
        profile_row.addWidget(self.profile_label, 1)
        profile_row.addWidget(open_profile)
        profile_row.addWidget(default_profile)
        layout.addLayout(profile_row)
        self.summary = QLabel("Выберите папку во вкладке «Корреляция».")
        layout.addWidget(self.summary)
        layout.addWidget(self.table)
        formula = QLabel(
            "La = ρσD/μ². Источники справочных значений указаны в примечаниях. "
            "Ручные правки действуют до смены профиля или закрытия приложения."
        )
        formula.setWordWrap(True)
        layout.addWidget(formula)

    def _open_profile(self) -> None:
        desktop = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        filename, _ = QFileDialog.getOpenFileName(self, "Конфиг параметров топлива", desktop, "Конфиг JSON (*.json)")
        if not filename:
            return
        try:
            self.load_profile(Path(filename))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка чтения конфига", str(exc))

    def load_profile(self, path: Path) -> None:
        self._apply_profile(load_fuel_profile(path), path)

    def _apply_profile(self, profile: FuelProfile, path: Path | None = None) -> None:
        self.profile = profile
        self.profile_label.setText(profile.name)
        self.profile_label.setToolTip(str(path) if path else "Встроенный профиль")
        self._entries.clear()
        if self._surfaces:
            self.set_surfaces(self._surfaces)
        self.profile_changed.emit()

    def set_surfaces(self, surfaces: list[FuelSurface]) -> None:
        self._surfaces = surfaces
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
                properties = self.profile.fuels.get(key)
                label = properties.label if properties else files[0].fuel_name
                defaults = (
                    str(properties.density), str(properties.surface_tension), str(properties.viscosity),
                    str(properties.diameter), properties.composition,
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


@dataclass(frozen=True)
class CorrelationPoint:
    surface: FuelSurface
    laplace: float
    slope: float
    label: str


class FuelCorrelationWidget(QWidget):
    def __init__(self, properties: FuelPropertiesWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.properties = properties
        self.pipeline = DiffSurfacePipeline()
        self._folder: Path | None = None
        self._surfaces: list[FuelSurface] = []
        self._points: list[CorrelationPoint] = []
        self._excluded_fuels: set[str] = set()
        self._fuel_colors: dict[str, str] = {}

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
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.filter_button = QToolButton(self.toolbar)
        self.filter_button.setText("Отбор топлива")
        self.filter_button.setCheckable(True)
        self.filter_button.setChecked(True)
        self.filter_action = self.toolbar.addWidget(self.filter_button)
        self.filter_action.setVisible(False)
        layout.addWidget(self.toolbar)

        chart_row = QHBoxLayout()
        chart_row.addWidget(self.canvas, 1)
        self.fuel_panel = QGroupBox("Топливо на графике")
        self.fuel_choices = QListWidget()
        self.fuel_choices.setMinimumWidth(240)
        self.fuel_choices.setMaximumWidth(320)
        self.fuel_choices.itemChanged.connect(self._change_fuel_visibility)
        self.filter_button.toggled.connect(self.fuel_panel.setVisible)
        panel_layout = QVBoxLayout(self.fuel_panel)
        panel_layout.addWidget(self.fuel_choices)
        self.fuel_panel.hide()
        chart_row.addWidget(self.fuel_panel)
        layout.addLayout(chart_row, 1)

        self.status = QLabel("Выберите папку с поверхностями CO для пара.")
        layout.addWidget(self.status)
        self.issues = QPlainTextEdit(readOnly=True)
        self.issues.setMaximumHeight(100)
        self.issues.hide()
        layout.addWidget(self.issues)
        self.properties.table.cellChanged.connect(self._property_changed)
        self.properties.profile_changed.connect(self._invalidate)

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
        if folder != self._folder:
            self._excluded_fuels.clear()
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
        if self._points:
            self._points.clear()
            self.fuel_panel.hide()
            self.filter_action.setVisible(False)
            self.figure.clear()
            self.toolbar.update()
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
            points.append(CorrelationPoint(surface, laplace, slope, label))
        self._show_issues(issues)
        if not points:
            self.status.setText("Нет поверхностей с данными, достаточными для построения графика.")
            return

        self._points = points
        self._populate_fuel_choices()
        self._render_plot()

    def _populate_fuel_choices(self) -> None:
        files_by_fuel: dict[str, list[str]] = {}
        for point in self._points:
            files_by_fuel.setdefault(point.surface.key, []).append(point.surface.path.name)

        colors = colormaps["tab20"].colors
        palette = colors[::2] + colors[1::2]
        self._fuel_colors.clear()
        with QSignalBlocker(self.fuel_choices):
            self.fuel_choices.clear()
            for index, (key, files) in enumerate(files_by_fuel.items()):
                color = to_hex(palette[index % len(palette)])
                self._fuel_colors[key] = color
                swatch = QPixmap(12, 12)
                swatch.fill(QColor(color))
                label = self.properties.label(key)
                item = QListWidgetItem(QIcon(swatch), label)
                item.setToolTip("\n".join((label, *files)))
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked if key in self._excluded_fuels else Qt.CheckState.Checked)
                self.fuel_choices.addItem(item)
        self.filter_action.setVisible(True)
        self.fuel_panel.setVisible(self.filter_button.isChecked())

    def _change_fuel_visibility(self, item: QListWidgetItem) -> None:
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._excluded_fuels.discard(key)
        else:
            self._excluded_fuels.add(key)
        self._render_plot()

    def _render_plot(self) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        points = [point for point in self._points if point.surface.key not in self._excluded_fuels]
        for point in points:
            axis.scatter(point.laplace, point.slope, color=self._fuel_colors[point.surface.key], label=point.label)
        title = "Угловой коэффициент"
        message = f"На графике {len(points)} из {len(self._points)} точек."
        x = np.asarray([point.laplace for point in points])
        y = np.asarray([point.slope for point in points])
        if not points:
            message = "Выберите хотя бы одно топливо."
        elif np.unique(x).size >= 2:
            fit = np.polyfit(x, y, 1)
            endpoints = np.asarray([x.min(), x.max()])
            axis.plot(endpoints, np.polyval(fit, endpoints), color="black", label="Аппроксимация")
            if np.ptp(y) > 0:
                r = np.corrcoef(x, y)[0, 1]
                title += f" (r = {r:.3f}, R² = {r**2:.3f})"
            else:
                message += " Коэффициент корреляции не определён при одинаковых угловых коэффициентах."
        else:
            message += " Для линейной аппроксимации нужны два различных числа Лапласа."
        axis.set(
            title=title, xlabel="Число Лапласа",
            ylabel="Угловой коэффициент оптимального режима",
        )
        axis.grid(True, alpha=0.65)
        if points:
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        self.status.setText(message)
        self.toolbar.update()
        self.canvas.draw_idle()
