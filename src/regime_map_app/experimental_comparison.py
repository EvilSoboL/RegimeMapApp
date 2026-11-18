from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
from statistics import mean

from PySide6.QtCore import QLocale, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .experimental import PLC_FUEL, FuelCalibration, PlcLog


@dataclass(frozen=True)
class FuelComparisonRegime:
    source: Path
    start: datetime
    end: datetime
    frequency: float
    nozzle_p: float | None
    additive: float


class FuelComparisonDialog(QDialog):
    remove_requested = Signal(list)
    calibrations_changed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Сверка и калибровка расхода по ПЛК")
        self.resize(1180, 540)
        self.log: PlcLog | None = None
        self.regimes: list[FuelComparisonRegime] = []
        self.removable: set[FuelComparisonRegime] = set()
        self.calibrations: dict[float, FuelCalibration] = {}
        self.fit_samples: list[float | None] = []

        layout = QVBoxLayout(self)
        self.coverage_label = QLabel()
        self.coverage_label.setWordWrap(True)
        layout.addWidget(self.coverage_label)
        form = QFormLayout()
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-86400, 86400)
        self.offset_spin.setSuffix(" с")
        self.offset_spin.setToolTip("Положительное число, если часы ПЛК опережают часы газоанализатора.")
        self.unit_combo = QComboBox()
        for label, factor in (
            ("Выберите по выгрузке ПЛК", None),
            ("кг/ч", 1.0), ("г/с", 3.6), ("г/мин", 0.06), ("кг/с", 3600.0), ("кг/мин", 60.0),
        ):
            self.unit_combo.addItem(label, factor)
        form.addRow("Время ПЛК минус время газоанализа", self.offset_spin)
        form.addRow("Единицы расхода топлива в ПЛК", self.unit_combo)
        layout.addLayout(form)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)
        self.calibration_label = QLabel()
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.calibration_label)
        self.fit_status = QLabel()
        self.fit_status.setWordWrap(True)
        layout.addWidget(self.fit_status)

        self.table = QTableWidget(0, 10)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.selection_label = QLabel()
        self.remove_button = QPushButton("Убрать из сверки")
        self.remove_button.setToolTip("Удаляет только режимы, добавленные кнопкой «Добавить для сверки».")
        self.copy_button = QPushButton("Скопировать выбранные для Excel")
        self.fit_button = QPushButton("Получить коэффициенты")
        close_button = QPushButton("Закрыть")
        actions.addWidget(self.selection_label)
        actions.addStretch()
        for button in (self.fit_button, self.remove_button, self.copy_button, close_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.offset_spin.valueChanged.connect(self._settings_changed)
        self.unit_combo.currentIndexChanged.connect(self._settings_changed)
        self.table.itemSelectionChanged.connect(self._update_selection)
        self.copy_button.clicked.connect(self._copy_selection)
        self.remove_button.clicked.connect(self._remove_selection)
        self.fit_button.clicked.connect(self._fit_calibrations)
        close_button.clicked.connect(self.close)

    def set_data(
        self, log: PlcLog | None, regimes: list[FuelComparisonRegime], removable: list[FuelComparisonRegime]
    ) -> None:
        self.table.clearSelection()
        if self.log is not log:
            with QSignalBlocker(self.offset_spin), QSignalBlocker(self.unit_combo):
                self.offset_spin.setValue(0)
                self.unit_combo.setCurrentIndex(0)
            self._clear_calibrations()
            self.fit_status.clear()
        self.log = log
        self.regimes = sorted(dict.fromkeys(regimes), key=lambda regime: (regime.frequency, regime.start))
        self.removable = set(removable)
        if log is None:
            self.coverage_label.setText("Загрузите CSV ПЛК во вкладке эксперимента.")
            self.coverage_label.setToolTip("")
        else:
            self.coverage_label.setText(
                f"ПЛК {log.name} · {log.path.name} · "
                f"{log.times[0]:%d.%m.%Y %H:%M:%S} — {log.times[-1]:%d.%m.%Y %H:%M:%S}"
            )
            self.coverage_label.setToolTip(str(log.path))
        self.hint_label.setText(
            "Для каждого P нужно минимум 2 разные f, лучше 4–6 по рабочему диапазону. "
            "Выделите устойчивые режимы через Ctrl или Shift. Без выделения используются все подходящие строки."
            if self.regimes else
            "Задайте интервал режима во вкладке эксперимента и нажмите «Добавить режим» или «Добавить для сверки»."
        )
        self._refresh()

    def _clear_calibrations(self) -> None:
        if self.calibrations:
            self.calibrations.clear()
            self.calibration_label.clear()
            self.calibrations_changed.emit()

    def _settings_changed(self) -> None:
        self.fit_status.setText("Настройки изменились. Получите коэффициенты заново." if self.calibrations else "")
        self._clear_calibrations()
        self._refresh()

    def _fit_calibrations(self) -> None:
        if self.log is None or self.unit_combo.currentData() is None:
            self.fit_status.setText("Загрузите CSV ПЛК и укажите единицы расхода топлива.")
            return
        selected = sorted(index.row() for index in self.table.selectionModel().selectedRows())
        rows = selected or range(len(self.regimes))
        groups: dict[float, list[tuple[float, float]]] = {}
        skipped = 0
        for row in rows:
            regime = self.regimes[row]
            measured = self.fit_samples[row]
            if regime.nozzle_p is None or measured is None:
                skipped += 1
                continue
            groups.setdefault(regime.nozzle_p, []).append((regime.frequency, measured))

        fitted = {}
        messages = []
        for nozzle_p, samples in sorted(groups.items()):
            try:
                fitted[nozzle_p] = FuelCalibration.fit(samples)
            except ValueError as exc:
                messages.append(f"P {nozzle_p:g}. {exc}")
        if not groups:
            messages.append("Нужны режимы с P и полным окном из двух или более замеров расхода ПЛК.")
        if skipped:
            messages.append(f"Пропущено неподходящих строк: {skipped}.")
        if fitted:
            self.calibrations.update(fitted)
            self.calibration_label.setText("\n".join(
                f"P {nozzle_p:g} · {calibration.equation} · "
                f"f {calibration.minimum_frequency:g}–{calibration.maximum_frequency:g} · "
                f"частот {calibration.frequency_count}, режимов {calibration.regime_count}"
                for nozzle_p, calibration in sorted(self.calibrations.items())
            ))
            messages.insert(0, "Коэффициенты применяются при добавлении точки без времени.")
            self.calibrations_changed.emit()
            self._refresh()
        self.fit_status.setText(" ".join(messages))

    def _refresh(self) -> None:
        factor = self.unit_combo.currentData()
        offset = timedelta(seconds=self.offset_spin.value())
        plc_unit = "кг/ч" if factor is not None else "исходные ед."
        self.table.setHorizontalHeaderLabels([
            "Окно газоанализа", "f", "P", "Разбавитель", "Расчёт, кг/ч",
            f"Среднее ПЛК, {plc_unit}", "ΔQ, кг/ч", "Замеров", f"Мин…макс, {plc_unit}", "Проверка",
        ])
        self.table.setRowCount(len(self.regimes))
        self.fit_samples = []
        for row, regime in enumerate(self.regimes):
            calculated = None
            messages = []
            if regime.nozzle_p is None:
                messages.append("Нужно P")
            elif regime.nozzle_p in self.calibrations:
                try:
                    calculated, extrapolated = self.calibrations[regime.nozzle_p].estimate(regime.frequency)
                    if extrapolated:
                        messages.append("Экстраполяция")
                except ValueError:
                    messages.append("Расчёт недоступен")

            plc_start, plc_end = regime.start + offset, regime.end + offset
            readings = []
            partial = False
            if self.log is None:
                messages.append("Нет CSV ПЛК")
            else:
                readings = [
                    sample[PLC_FUEL] for sample in self.log.window(plc_start, plc_end) if PLC_FUEL in sample
                ]
                if not readings:
                    messages.append("Нет замеров топлива в окне")
                else:
                    partial = plc_start < self.log.times[0] or plc_end > self.log.times[-1]
                    if partial:
                        messages.append("Неполное окно ПЛК")
                    if len(readings) == 1:
                        messages.append("Один замер")
                    if factor is None:
                        messages.append("Укажите единицы ПЛК")

            average = delta = spread = "—"
            fit_sample = None
            if readings:
                scale = 1.0 if factor is None else factor
                measured = mean(readings) * scale
                average = f"{measured:.8g}"
                spread = f"{min(readings) * scale:.8g} … {max(readings) * scale:.8g}"
                if factor is not None and calculated is not None:
                    delta = f"{measured - calculated:+.8g}"
                if measured <= 0 or not isfinite(measured):
                    messages.append("Неположительный или некорректный расход")
                elif factor is not None and len(readings) >= 2 and not partial:
                    fit_sample = measured
            self.fit_samples.append(fit_sample)
            end_format = "%H:%M:%S" if regime.start.date() == regime.end.date() else "%d.%m %H:%M:%S"
            cells = (
                f"{regime.start:%d.%m.%Y %H:%M:%S} — {regime.end:{end_format}}",
                f"{regime.frequency:g}", "—" if regime.nozzle_p is None else f"{regime.nozzle_p:g}",
                f"{regime.additive:g}", "—" if calculated is None else f"{calculated:.8g}",
                average, delta, str(len(readings)), spread, " · ".join(messages) or "Есть замеры",
            )
            details = f"{regime.source}\nОкно в ПЛК: {plc_start:%d.%m.%Y %H:%M:%S} — {plc_end:%d.%m.%Y %H:%M:%S}"
            for column, label in enumerate(cells):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                item.setText(label)
                item.setToolTip(f"{label}\n{details}")
        self._update_selection()

    def _update_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        count = len(rows)
        self.selection_label.setText(f"Выбрано режимов: {count} из {len(self.regimes)}")
        self.copy_button.setEnabled(count > 0)
        self.remove_button.setEnabled(bool(rows) and all(self.regimes[index.row()] in self.removable for index in rows))

    def _remove_selection(self) -> None:
        self.remove_requested.emit([self.regimes[index.row()] for index in self.table.selectionModel().selectedRows()])

    def _copy_selection(self) -> None:
        columns = range(self.table.columnCount())
        header = "\t".join(self.table.horizontalHeaderItem(column).text() for column in columns)
        rows = sorted(index.row() for index in self.table.selectionModel().selectedRows())
        lines = [header]
        decimal = QLocale().decimalPoint()
        for row in rows:
            cells = [self.table.item(row, column).text() for column in columns]
            cells[1:7] = [value.replace(".", decimal) for value in cells[1:7]]
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self.selection_label.setText(f"Скопировано режимов: {len(rows)}")
