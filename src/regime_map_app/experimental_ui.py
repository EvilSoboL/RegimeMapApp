from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from re import fullmatch

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .approx.exceptions import ValidationError
from .approx.validation import parse_file_metadata
from .experimental import (
    COMPONENTS,
    PLC_ANALOG,
    PLC_FUEL,
    PLC_WATER,
    GasLog,
    PlcLog,
    approximation_csv,
    read_gas_log,
    read_number,
    read_plc_log,
)
from .experimental_comparison import FuelComparisonDialog, FuelComparisonRegime


@dataclass
class ExperimentPoint:
    fuel: float
    frequency: float
    additive: float
    averages: dict[str, tuple[float, int]]
    overrides: dict[str, float]
    source: str
    interval: str
    record_count: int
    plc_summary: str = "—"
    plc_details: str = ""
    nozzle_p: float | None = None
    extrapolated: bool = False
    comparison_regime: FuelComparisonRegime | None = None
    calibration_equation: str = ""

    def value(self, component: str) -> float | None:
        if component in self.overrides:
            return self.overrides[component]
        result = self.averages.get(component)
        return result[0] if result else None


class ExperimentalDataWidget(QWidget):
    csv_ready = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.logs: dict[Path, GasLog] = {}
        self.plc_log: PlcLog | None = None
        self.points: list[ExperimentPoint] = []
        self.comparison_only: list[FuelComparisonRegime] = []
        self.comparison_dialog = FuelComparisonDialog(self)
        self.comparison_dialog.remove_requested.connect(self._remove_comparison_regimes)

        layout = QVBoxLayout(self)
        self.add_comparison_button = QPushButton("Добавить для сверки")
        self.add_comparison_button.setToolTip(
            "Сохранить окно, f, P и разбавитель для сверки с ПЛК без добавления точки на карту."
        )
        self.compare_button = QPushButton("Сверить с ПЛК…")
        self.compare_button.setToolTip("Сравнить замеры и получить коэффициенты для точек без времени.")
        form_row = QHBoxLayout()
        form_row.addWidget(self._build_regime_form(), 1)
        form_row.addWidget(self._build_export_form(), 1)
        layout.addLayout(form_row)

        self.points_table = QTableWidget(0, 9)
        self.points_table.setHorizontalHeaderLabels(
            ["Топливо (fuel)", "Разбавитель (additive)", "Компонент", "Записей", "Файл", "Интервал", "Данные ПЛК", "f", "P"]
        )
        self.points_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.points_table, 1)

        actions = QHBoxLayout()
        self.remove_button = QPushButton("Удалить точку")
        self.override_button = QPushButton("Задать значение выбранной точки")
        self.save_button = QPushButton("Сохранить CSV и открыть аппроксимацию")
        actions.addWidget(self.remove_button)
        actions.addWidget(self.override_button)
        actions.addStretch()
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.component_combo.currentIndexChanged.connect(self._refresh_points)
        self.points_table.itemSelectionChanged.connect(self._refresh_actions)
        self.remove_button.clicked.connect(self._remove_point)
        self.override_button.clicked.connect(self._override_point)
        self.compare_button.clicked.connect(self._show_comparison)
        self.save_button.clicked.connect(self._save_csv)
        self.comparison_dialog.calibrations_changed.connect(self._refresh_calibration_status)
        self._refresh_calibration_status()
        self._refresh_actions()

    def _build_regime_form(self) -> QGroupBox:
        group = QGroupBox("Результаты газового анализа")
        form = QFormLayout(group)
        self.source_combo = QComboBox()
        self.source_combo.setPlaceholderText("Выберите файл")
        self.source_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source_combo.setMinimumContentsLength(18)
        load_button = QPushButton("Добавить XLSX…")
        sources = QHBoxLayout()
        sources.addWidget(self.source_combo, 1)
        sources.addWidget(load_button)
        form.addRow("Файл режима", sources)

        self.coverage_label = QLabel()
        form.addRow("Запись", self.coverage_label)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDisplayFormat("dd.MM.yyyy")
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setAccessibleName("Дата начала режима")
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setAccessibleName("Время начала режима")
        self.start_time_edit.setPlaceholderText("ЧЧ:ММ или ЧЧ:ММ:СС")
        self.start_time_edit.setToolTip("Например, 16:09 или 16:09:13. Если секунды пропущены, используются 00.")
        start_fields = QHBoxLayout()
        start_fields.addWidget(self.start_date_edit)
        start_fields.addWidget(self.start_time_edit, 1)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 86400)
        self.duration_spin.setValue(180)
        self.duration_spin.setSuffix(" с")
        self.average_mode = QComboBox()
        self.average_mode.addItems(["Последние секунды", "Последние записи"])
        self.tail_spin = QSpinBox()
        self.tail_spin.setRange(1, 86400)
        self.tail_spin.setValue(60)
        averaging = QHBoxLayout()
        averaging.addWidget(self.average_mode, 1)
        averaging.addWidget(self.tail_spin)
        form.addRow("Начало режима", start_fields)
        form.addRow("Длительность", self.duration_spin)
        form.addRow("Усреднение", averaging)

        self.frequency_edit = QLineEdit()
        self.frequency_edit.setPlaceholderText("Например, f32 или 32")
        self.nozzle_p_edit = QLineEdit()
        self.nozzle_p_edit.setPlaceholderText("Например, P3000")
        self.nozzle_p_edit.setToolTip(
            "Для калибровки введите фактическую уставку режима. Для точки без времени нужны коэффициенты при этом P."
        )
        self.fuel_edit = QLineEdit()
        self.additive_edit = QLineEdit()
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("Оставьте пустым для усреднения")
        form.addRow("Частота форсунки (f)", self.frequency_edit)
        form.addRow("Уставка форсунки (P)", self.nozzle_p_edit)
        form.addRow("Топливо для режима (fuel)", self.fuel_edit)
        form.addRow("Разбавитель (additive)", self.additive_edit)
        form.addRow("Значение вручную", self.value_edit)

        buttons = QHBoxLayout()
        self.add_regime_button = QPushButton("Добавить режим")
        self.add_manual_button = QPushButton("Добавить точку без времени")
        buttons.addWidget(self.add_regime_button)
        buttons.addWidget(self.add_manual_button)
        form.addRow(buttons)

        load_button.clicked.connect(self._choose_files)
        self.source_combo.currentIndexChanged.connect(self._select_source)
        self.add_regime_button.clicked.connect(self._add_regime)
        self.add_comparison_button.clicked.connect(self._add_comparison_regime)
        self.add_manual_button.clicked.connect(self._add_manual)
        self._select_source()
        return group

    def _build_export_form(self) -> QGroupBox:
        group = QGroupBox("Данные для аппроксимации")
        form = QFormLayout(group)
        self.fuel_name_edit = QLineEdit()
        self.diluent_name_edit = QLineEdit()
        self.component_combo = QComboBox()
        for label, component in COMPONENTS.items():
            self.component_combo.addItem(label, component)
        self.component_combo.setCurrentIndex(self.component_combo.findData("CO"))
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd-MM-yyyy")
        self.filename_label = QLabel()
        self.filename_label.setWordWrap(True)
        form.addRow("Название топлива", self.fuel_name_edit)
        form.addRow("Название разбавителя", self.diluent_name_edit)
        form.addRow("Компонент", self.component_combo)
        form.addRow("Дата для имени файла", self.date_edit)
        form.addRow("Имя CSV", self.filename_label)
        plc_buttons = QHBoxLayout()
        self.plc_file_edit = QLineEdit()
        self.plc_file_edit.setReadOnly(True)
        self.plc_file_edit.setPlaceholderText("CSV с записями ПЛК")
        plc_button = QPushButton("Выбрать CSV…")
        clear_plc_button = QPushButton("Убрать")
        plc_buttons.addWidget(self.plc_file_edit, 1)
        plc_buttons.addWidget(plc_button)
        plc_buttons.addWidget(clear_plc_button)
        form.addRow("Данные с ПЛК", plc_buttons)
        self.plc_coverage_label = QLabel()
        self.plc_coverage_label.setWordWrap(True)
        form.addRow("", self.plc_coverage_label)
        comparison = QHBoxLayout()
        comparison.addWidget(self.add_comparison_button)
        comparison.addWidget(self.compare_button)
        comparison.addStretch()
        form.addRow(comparison)
        self.calibration_status_label = QLabel()
        self.calibration_status_label.setWordWrap(True)
        form.addRow(self.calibration_status_label)
        plc_button.clicked.connect(self._choose_plc)
        clear_plc_button.clicked.connect(self._clear_plc)
        self.fuel_name_edit.textChanged.connect(self._refresh_filename)
        self.diluent_name_edit.textChanged.connect(self._refresh_filename)
        self.component_combo.currentIndexChanged.connect(self._refresh_filename)
        self.date_edit.dateChanged.connect(self._refresh_filename)
        self._refresh_filename()
        return group

    def _refresh_filename(self) -> None:
        self.filename_label.setText(self._filename(placeholders=True))

    def _filename(self, *, placeholders: bool = False) -> str:
        fuel = self.fuel_name_edit.text().strip() or ("топливо" if placeholders else "")
        diluent = self.diluent_name_edit.text().strip() or ("разбавитель" if placeholders else "")
        date = self.date_edit.date().toString("dd-MM-yyyy")
        return f"{fuel}-{diluent}-{self.component_combo.currentData()}-{date}.csv"

    def _choose_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(self, "Добавить результаты газового анализа", "", "Excel (*.xlsx)")
        if filenames:
            self.load_files([Path(filename) for filename in filenames])

    def _choose_plc(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Выбрать данные с ПЛК", "", "CSV (*.csv)")
        if filename:
            self.load_plc_file(Path(filename))

    def load_plc_file(self, path: Path) -> None:
        try:
            log = read_plc_log(path)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.plc_log = log
        self._refresh_comparison()
        self.plc_file_edit.setText(path.name)
        self.plc_file_edit.setToolTip(str(path))
        self.plc_coverage_label.setText(
            f"{log.name} · {log.times[0]:%d.%m.%Y %H:%M:%S} — {log.times[-1]:%d.%m.%Y %H:%M:%S}"
        )
        self.fuel_edit.setPlaceholderText("Пусто — взять среднее из ПЛК")
        self.status_label.setText(f"Загружено записей ПЛК: {len(log.times)}. Файл применяется к новым режимам.")

    def _clear_plc(self) -> None:
        self.plc_log = None
        self._refresh_comparison()
        self.plc_file_edit.clear()
        self.plc_file_edit.setToolTip("")
        self.plc_coverage_label.clear()
        self.fuel_edit.setPlaceholderText("")
        self.status_label.setText("Источник ПЛК для новых режимов снят.")

    def load_files(self, paths: list[Path]) -> None:
        errors = []
        for path in paths:
            path = path.resolve()
            if path in self.logs:
                continue
            try:
                log = read_gas_log(path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not self.logs:
                self.date_edit.setDate(QDate(log.times[0].date()))
            self.logs[path] = log
            self.source_combo.addItem(path.name, path)
            self.source_combo.setItemData(
                self.source_combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole
            )
        if self.source_combo.currentIndex() < 0 and self.source_combo.count():
            self.source_combo.setCurrentIndex(0)
        self.status_label.setText(f"Загружено файлов: {len(self.logs)}.")
        if errors:
            QMessageBox.warning(self, "Загрузка XLSX", "\n".join(errors))

    def _select_source(self) -> None:
        log = self.logs.get(self.source_combo.currentData())
        controls = (
            self.start_time_edit, self.duration_spin, self.average_mode, self.tail_spin,
            self.add_regime_button, self.add_comparison_button,
        )
        for control in controls:
            control.setEnabled(log is not None)
        if log is None:
            self.start_date_edit.setEnabled(False)
            self.coverage_label.setText("Добавьте один или несколько файлов XLSX")
            return
        first, last = log.times[0], log.times[-1]
        self.coverage_label.setText(f"{first:%d.%m.%Y %H:%M:%S} — {last:%d.%m.%Y %H:%M:%S}")
        self.start_date_edit.setDateRange(QDate(first.date()), QDate(last.date()))
        self.start_date_edit.setDate(QDate(first.date()))
        self.start_date_edit.setEnabled(first.date() != last.date())
        self.start_time_edit.setText(f"{first:%H:%M:%S}")

    def _read_regime_start(self) -> datetime:
        value = self.start_time_edit.text().strip()
        if fullmatch(r"[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?", value):
            try:
                start_time = datetime.strptime(value, "%H:%M:%S" if value.count(":") == 2 else "%H:%M").time()
                return datetime.combine(self.start_date_edit.date().toPython(), start_time)
            except ValueError:
                pass
        self.start_time_edit.setFocus()
        self.start_time_edit.selectAll()
        raise ValueError("Введите время начала в формате ЧЧ:ММ или ЧЧ:ММ:СС, например 16:09 или 16:09:13.")

    def _add_regime(self) -> None:
        self._add_point(manual=False)

    def _add_manual(self) -> None:
        self._add_point(manual=True)

    def _read_regime_settings(self, *, require_p: bool) -> tuple[float, float | None, float]:
        frequency_text = self.frequency_edit.text().strip().lower().removeprefix("f")
        frequency = read_number(frequency_text, "Частота форсунки (f)")
        if frequency <= 0:
            raise ValueError("Частота форсунки (f) должна быть больше нуля.")
        nozzle_p = None
        p_text = self.nozzle_p_edit.text().strip().lower()
        if require_p or p_text:
            nozzle_p = read_number(p_text.removeprefix("p"), "Уставка форсунки (P)")
            if nozzle_p <= 0:
                raise ValueError("Уставка форсунки (P) должна быть больше нуля.")
        additive = read_number(self.additive_edit.text(), "Разбавитель (additive)")
        return frequency, nozzle_p, additive

    def _averaging_settings(self) -> tuple[int, int, int | None]:
        tail = self.tail_spin.value()
        records = tail if self.average_mode.currentIndex() == 1 else None
        return self.duration_spin.value(), tail, records

    def _comparison_regimes(self) -> list[FuelComparisonRegime]:
        return self.comparison_only + [
            point.comparison_regime for point in self.points if point.comparison_regime is not None
        ]

    def _show_comparison(self) -> None:
        self._refresh_comparison()
        self.comparison_dialog.open()

    def _refresh_comparison(self) -> None:
        self.comparison_dialog.set_data(self.plc_log, self._comparison_regimes(), self.comparison_only)

    def _refresh_calibration_status(self) -> None:
        calibrations = self.comparison_dialog.calibrations
        if calibrations:
            pressures = ", ".join(f"{pressure:g}" for pressure in sorted(calibrations))
            self.calibration_status_label.setText(f"Коэффициенты готовы для P {pressures}.")
            self.add_manual_button.setToolTip("Расход в кг/ч берётся из калибровки ПЛК для указанного P.")
        else:
            self.calibration_status_label.setText("Для точек без времени получите коэффициенты в сверке с ПЛК.")
            self.add_manual_button.setToolTip(self.calibration_status_label.text())

    def _add_comparison_regime(self) -> None:
        try:
            frequency, nozzle_p, additive = self._read_regime_settings(require_p=True)
            log = self.logs[self.source_combo.currentData()]
            start = self._read_regime_start()
            _, _, first, end = log.averaging_window(start, *self._averaging_settings())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        regime = FuelComparisonRegime(log.path, first, end, frequency, nozzle_p, additive)
        regimes = self._comparison_regimes()
        if regime in regimes:
            self.status_label.setText("Этот режим уже есть в сверке.")
            return
        self.comparison_only.append(regime)
        self.status_label.setText(f"Режимов для сверки с ПЛК: {len(set(regimes)) + 1}.")

    def _remove_comparison_regimes(self, regimes: list[FuelComparisonRegime]) -> None:
        selected = set(regimes)
        self.comparison_only = [regime for regime in self.comparison_only if regime not in selected]
        self._refresh_comparison()

    def _add_point(self, *, manual: bool) -> None:
        try:
            frequency, nozzle_p, additive = self._read_regime_settings(require_p=manual)
            calibration = self.comparison_dialog.calibrations.get(nozzle_p) if manual else None
            if manual and calibration is None:
                raise ValueError(f"Для P {nozzle_p:g} нажмите «Получить коэффициенты» в окне «Сверить с ПЛК».")
            component = self.component_combo.currentData()
            overrides = {}
            if manual or self.value_edit.text().strip():
                overrides[component] = read_number(self.value_edit.text(), "Значение вручную")
            if calibration is not None:
                fuel, extrapolated = calibration.estimate(frequency)
                point = ExperimentPoint(
                    fuel, frequency, additive, {}, overrides, "", "", 0,
                    nozzle_p=nozzle_p, extrapolated=extrapolated,
                    calibration_equation=calibration.equation,
                )
            else:
                log = self.logs[self.source_combo.currentData()]
                start = self._read_regime_start()
                duration, tail, records = self._averaging_settings()
                averages, count = log.summarize(start, duration, tail, records)
                if component not in averages and component not in overrides:
                    raise ValueError(
                        f"В окне усреднения нет числовых показаний {component}. "
                        "Заполните поле «Значение вручную», если известно значение."
                    )
                plc_values = {}
                plc_summary = "—"
                plc_details = ""
                _, _, first, end = log.averaging_window(start, duration, tail, records)
                if self.plc_log is not None:
                    plc_values, plc_count = self.plc_log.summarize(first, end)
                    if not plc_count:
                        raise ValueError("Нет записей ПЛК в окне усреднения. Проверьте дату и время обеих выгрузок.")
                    plc_summary = f"{plc_count} зап."
                    partial = first < self.plc_log.times[0] or end > self.plc_log.times[-1]
                    if partial:
                        plc_summary += " (неполное окно)"
                    lines = [
                        str(self.plc_log.path),
                        f"Окно: {first:%d.%m.%Y %H:%M:%S} — {end:%d.%m.%Y %H:%M:%S}",
                        "Средние в единицах выгрузки",
                    ]
                    for channel in (PLC_FUEL, PLC_WATER, PLC_ANALOG):
                        value = plc_values.get(channel)
                        lines.append(f"{channel}: {value:.8g}" if value is not None else f"{channel}: нет показаний")
                    plc_details = "\n".join(lines)
                if not self.fuel_edit.text().strip() and self.plc_log is not None:
                    if PLC_FUEL not in plc_values:
                        raise ValueError("В этом окне нет показаний расхода топлива ПЛК. Введите fuel вручную.")
                    fuel = plc_values[PLC_FUEL]
                else:
                    fuel = read_number(self.fuel_edit.text(), "Топливо (fuel)")
                point = ExperimentPoint(
                    fuel, frequency, additive, averages, overrides, log.path.name,
                    f"{start:%d.%m %H:%M:%S}, {duration} с", count,
                    plc_summary, plc_details, nozzle_p=nozzle_p,
                    comparison_regime=FuelComparisonRegime(log.path, first, end, frequency, nozzle_p, additive),
                )
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self.points.append(point)
        self.value_edit.clear()
        self._refresh_points()
        self.points_table.selectRow(len(self.points) - 1)
        status = f"Добавлено точек: {len(self.points)}."
        if point.extrapolated:
            status += " Расход рассчитан вне диапазона калибровки."
        self.status_label.setText(status)

    def _refresh_points(self) -> None:
        component = self.component_combo.currentData()
        self.points_table.setHorizontalHeaderItem(2, QTableWidgetItem(self.component_combo.currentText()))
        self.points_table.setRowCount(len(self.points))
        for index, point in enumerate(self.points):
            value = point.value(component)
            count = point.averages.get(component, (0, 0))[1]
            value_label = "—" if value is None else f"{value:.8g}"
            if component in point.overrides:
                value_label += " (вручную)"
            source_label = point.source or (
                "Калибровка ПЛК (экстраполяция)" if point.extrapolated else "Калибровка ПЛК"
            )
            cells = (
                f"{point.fuel:.8g}", f"{point.additive:.8g}", value_label,
                f"{count} из {point.record_count}" if point.record_count else "—",
                source_label, point.interval, point.plc_summary, f"{point.frequency:.8g}",
                "—" if point.nozzle_p is None else f"{point.nozzle_p:.8g}",
            )
            for column, label in enumerate(cells):
                item = QTableWidgetItem(label)
                if column == 4 and point.calibration_equation:
                    item.setToolTip(f"{label}\n{point.calibration_equation}")
                elif column == 6 and point.plc_details:
                    item.setToolTip(point.plc_details)
                else:
                    item.setToolTip(label)
                self.points_table.setItem(index, column, item)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        selected = bool(self.points_table.selectionModel().hasSelection())
        self.remove_button.setEnabled(selected)
        self.override_button.setEnabled(selected)
        self.save_button.setEnabled(bool(self.points))

    def _remove_point(self) -> None:
        index = self.points_table.currentRow()
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self._refresh_points()
            self.status_label.setText(f"Точек в таблице: {len(self.points)}.")

    def _override_point(self) -> None:
        index = self.points_table.currentRow()
        if not 0 <= index < len(self.points):
            return
        try:
            value = read_number(self.value_edit.text(), "Значение вручную")
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.points[index].overrides[self.component_combo.currentData()] = value
        self.value_edit.clear()
        self._refresh_points()
        self.status_label.setText("Значение выбранной точки обновлено.")

    def _save_csv(self) -> None:
        component = self.component_combo.currentData()
        filename = self._filename()
        try:
            if any(char in filename for char in '<>:"/\\|?*'):
                raise ValueError("В названиях топлива и разбавителя есть недопустимые символы имени файла.")
            parse_file_metadata(Path(filename))
            points = []
            for index, point in enumerate(self.points, 1):
                value = point.value(component)
                if value is None:
                    raise ValueError(f"В точке {index} нет значения для компонента {component}.")
                points.append((point.fuel, point.additive, value))
            contents = approximation_csv(points)
        except (ValueError, ValidationError) as exc:
            self.status_label.setText(str(exc))
            return

        directory = next(iter(self.logs)).parent if self.logs else Path.home()
        selected, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", str(directory / filename), "CSV (*.csv)")
        if not selected:
            return
        path = Path(selected)
        try:
            metadata = parse_file_metadata(path)
            if metadata.component_name.casefold() != component.casefold():
                raise ValueError(f"В имени файла укажите выбранный компонент {component}.")
            path.write_text(contents, encoding="utf-8", newline="")
        except (OSError, ValueError, ValidationError) as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(f"Сохранено точек: {len(points)}. {path}")
        self.csv_ready.emit(str(path))
