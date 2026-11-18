import csv
import json
from datetime import datetime, timedelta

import pytest
from openpyxl import Workbook
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QFileDialog

from regime_map_app.approx.models import InputMode
from regime_map_app.experimental import COMPONENTS, PLC_FUEL, TIME_HEADER, FuelCalibration, read_gas_log
from regime_map_app.experimental_ui import ExperimentalDataWidget
from regime_map_app.main_window import RegimeMapMainWindow


def make_log(path, start, values):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append([TIME_HEADER, *COMPONENTS])
    for offset, co in values:
        sheet.append([start + timedelta(seconds=offset), 5, co, 3, 2, 10, 0, 0, 0])
    workbook.save(path)
    return path


def test_average_tail_by_time_or_count_stays_inside_regime(tmp_path):
    start = datetime(2026, 11, 9, 16, 46)
    path = make_log(tmp_path / "gas.xlsx", start, [(0, 900), (35, 700), (80, 10), (95, "-"), (119, 30), (120, 500)])
    log = read_gas_log(path)
    averages, count = log.summarize(start, 120, 60, None)
    assert averages["CO"] == (20, 2)
    assert count == 3
    averages, count = log.summarize(start, 120, 60, 2)
    assert averages["CO"] == (30, 1)
    assert count == 2
    with pytest.raises(ValueError, match="выходит за пределы"):
        log.summarize(start, 122, 60, None)


def test_import_and_average_xlsx_with_missing_gas_channels(qtbot, tmp_path):
    start = datetime(2026, 11, 16, 14, 23, 9)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append([TIME_HEADER, "% O2", "ппм СO", "ппм NO"])
    sheet.append([start, 21.5, 4, 0])
    sheet.append([start + timedelta(seconds=1), 21.4, 6, 1])
    path = tmp_path / "diesel.xlsx"
    workbook.save(path)

    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.load_files([path])
    tab.duration_spin.setValue(2)
    tab.frequency_edit.setText("31")
    tab.fuel_edit.setText("1.2")
    tab.additive_edit.setText("800")
    tab.add_regime_button.click()

    assert len(tab.points) == 1
    assert tab.points[0].averages["CO"] == (5, 2)
    assert "NOx" not in tab.points[0].averages
    assert "CO2" not in tab.points[0].averages

    tab.component_combo.setCurrentIndex(tab.component_combo.findData("NOx"))
    tab.add_regime_button.click()
    assert len(tab.points) == 1
    assert "нет числовых показаний NOx" in tab.status_label.text()

    tab.value_edit.setText("50")
    tab.add_regime_button.click()
    assert tab.points[1].value("NOx") == 50


def test_multi_file_export_manual_point_and_approximation_handoff(qtbot, monkeypatch, tmp_path, qapp):
    start = datetime(2026, 11, 9, 16, 46)
    first = make_log(tmp_path / "first.xlsx", start, [(0, 10), (59, 30)])
    second = make_log(tmp_path / "second.xlsx", start, [(0, 50), (59, 70)])
    window = RegimeMapMainWindow()
    qtbot.addWidget(window)
    tab = window.experimental_widget
    tab.load_files([first, second, first])
    assert tab.source_combo.count() == 2

    tab.duration_spin.setValue(60)
    tab.frequency_edit.setText("f22")
    tab.fuel_edit.setText("0,78")
    tab.additive_edit.setText("400")
    tab.add_regime_button.click()
    tab.source_combo.setCurrentIndex(1)
    tab.additive_edit.setText("600")
    tab.add_regime_button.click()
    tab.value_edit.setText("500")
    tab.frequency_edit.clear()
    tab.add_manual_button.click()
    assert len(tab.points) == 2
    assert "Частота форсунки" in tab.status_label.text()
    tab.frequency_edit.setText("F32")
    for invalid_p in ("", "P0", "-10", "nan"):
        tab.nozzle_p_edit.setText(invalid_p)
        tab.add_manual_button.click()
        assert len(tab.points) == 2
        assert "Уставка форсунки" in tab.status_label.text()

    tab.nozzle_p_edit.setText("P2500")
    tab.comparison_dialog.calibrations[2500] = FuelCalibration.fit([(30, 1.0), (34, 1.25)])
    tab.add_manual_button.click()
    assert tab.points[2].fuel == pytest.approx(1.125)
    assert tab.points_table.item(2, 7).text() == "32"
    assert tab.points_table.item(2, 8).text() == "2500"

    tab.nozzle_p_edit.setText("3000")
    tab.comparison_dialog.calibrations[3000] = FuelCalibration.fit([(30, 1.25), (34, 1.5)])
    tab.value_edit.setText("450")
    tab.add_manual_button.click()
    assert tab.points[2].nozzle_p == 2500
    assert tab.points[3].fuel == pytest.approx(1.375)
    assert tab.points_table.item(3, 8).text() == "3000"

    tab.fuel_name_edit.setText("diesel")
    tab.diluent_name_edit.setText("steam")
    output = tmp_path / "diesel-steam-CO-09-11-2026.csv"
    assert tab.filename_label.text() == output.name
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "CSV (*.csv)"))
    window.approx_widget.mode_combo.setCurrentIndex(1)
    tab.save_button.click()

    assert output.read_text(encoding="utf-8").splitlines() == [
        "fuel;additive;component", "0.78;400.0;20.0", "0.78;600.0;60.0",
        "1.125;600.0;500.0", "1.375;600.0;450.0",
    ]
    assert window.tabs.currentWidget() is window.approx_widget
    config = window.approx_widget.collect_config()
    assert config.input_mode is InputMode.SINGLE_FILE
    assert window.approx_widget.pipeline.validate_inputs(config).is_valid
    window.regime_map_widget.apply_suggested_input_path(str(tmp_path / "approx_diesel-steam-O2-09-11-2026.csv"))
    assert window.regime_map_widget.colorbar_label_edit.text() == "O2, %"
    assert not window.regime_map_widget.collect_config().is_co_component
    qapp.processEvents()


def test_fitted_plc_coefficients_are_used_for_manual_points_at_the_same_p(qtbot, tmp_path):
    start = datetime(2026, 11, 16, 14, 23)
    xlsx = make_log(tmp_path / "gas.xlsx", start, [(second, 20) for second in range(32)])
    plc = tmp_path / "plc.csv"
    with plc.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["recorded_at", "plc_name", "payload"])
        for second, value in [(0, 1), (1, 1), (10, 1.2), (11, 1.2), (20, 1.08), (21, 1.08), (30, 9), (31, 9), (32, 9)]:
            writer.writerow([
                start + timedelta(seconds=second), "PLC210", json.dumps([{"path": PLC_FUEL, "value": value}]),
            ])

    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.load_files([xlsx])
    tab.load_plc_file(plc)
    tab.duration_spin.setValue(2)
    tab.additive_edit.setText("1000")
    tab.nozzle_p_edit.setText("3000")
    tab.frequency_edit.setText("22")
    tab.value_edit.setText("500")
    tab.add_manual_button.click()
    assert len(tab.points) == 0
    assert "Получить коэффициенты" in tab.status_label.text()

    tab.add_regime_button.click()
    for second, frequency, pressure in [(10, 26, 3000), (20, 22, 3000), (30, 30, 2500)]:
        tab.start_time_edit.setText(f"14:23:{second:02d}")
        tab.frequency_edit.setText(str(frequency))
        tab.nozzle_p_edit.setText(str(pressure))
        tab.add_comparison_button.click()
    tab.compare_button.click()
    dialog = tab.comparison_dialog
    assert dialog.table.item(0, 4).text() == "—"
    dialog.fit_button.click()
    assert "укажите единицы" in dialog.fit_status.text()
    dialog.unit_combo.setCurrentIndex(dialog.unit_combo.findText("кг/ч"))
    dialog.table.selectRow(0)
    dialog.fit_button.click()
    assert "минимум 2 разные f" in dialog.fit_status.text()
    assert not dialog.calibrations
    dialog.table.clearSelection()
    dialog.fit_button.click()
    assert list(dialog.calibrations) == [3000]
    assert "P 2500" in dialog.fit_status.text()
    assert "Q = 0.16 + 0.04 × f" in dialog.calibration_label.text()
    assert "частот 2, режимов 3" in dialog.calibration_label.text()
    assert tab.calibration_status_label.text() == "Коэффициенты готовы для P 3000."
    assert dialog.table.item(0, 6).text() == "-0.04"
    dialog.close()

    tab.frequency_edit.setText("24")
    tab.nozzle_p_edit.setText("3000")
    tab.value_edit.setText("500")
    tab.add_manual_button.click()
    assert tab.points[1].fuel == pytest.approx(1.12)
    assert not tab.points[1].extrapolated
    assert "Q = " in tab.points_table.item(1, 4).toolTip()

    tab.frequency_edit.setText("28")
    tab.value_edit.setText("400")
    tab.add_manual_button.click()
    assert tab.points[2].extrapolated
    assert "вне диапазона" in tab.status_label.text()
    assert "экстраполяция" in tab.points_table.item(2, 4).text()

    tab.nozzle_p_edit.setText("2750")
    tab.add_manual_button.click()
    assert len(tab.points) == 3
    assert "Для P 2750" in tab.status_label.text()
    tab.compare_button.click()
    dialog.offset_spin.setValue(1)
    assert not dialog.calibrations
    assert "заново" in dialog.fit_status.text()


def test_type_and_paste_regime_start_near_end_of_recording(qtbot, tmp_path, qapp):
    start = datetime(2026, 11, 9, 15, 14, 43)
    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.show()
    tab.load_files([make_log(tmp_path / "gas.xlsx", start, [(0, 5), (16, 10), (3257, 20), (3270, 30)])])
    tab.frequency_edit.setText("30")
    tab.fuel_edit.setText("1")
    tab.additive_edit.setText("400")
    tab.duration_spin.setValue(14)
    assert tab.start_time_edit.text() == "15:14:43"
    assert not tab.start_date_edit.isEnabled()

    tab.start_time_edit.setText("15:14:59")
    tab.start_time_edit.setFocus()
    qtbot.keyClick(tab.start_time_edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(tab.start_time_edit, "16:09")
    qtbot.mouseClick(tab.add_regime_button, Qt.MouseButton.LeftButton)
    assert len(tab.points) == 1
    assert tab.points[0].interval == "09.11 16:09:00, 14 с"
    assert tab.points[0].value("CO") == 25

    tab.duration_spin.setValue(1)
    qapp.clipboard().setText("16:09:13")
    tab.start_time_edit.selectAll()
    qtbot.keyClick(tab.start_time_edit, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    tab.add_regime_button.click()
    assert len(tab.points) == 2
    assert tab.points[1].value("CO") == 30

    for value in ("16:09:14", "24:00", "16:"):
        tab.start_time_edit.setText(value)
        tab.add_regime_button.click()
        assert len(tab.points) == 2
        assert tab.start_time_edit.text() == value
        expected = "выходит за пределы" if value == "16:09:14" else "Введите время начала"
        assert expected in tab.status_label.text()


def test_regime_start_date_can_change_when_recording_crosses_midnight(qtbot, tmp_path):
    start = datetime(2026, 11, 9, 23, 59, 59)
    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.load_files([make_log(tmp_path / "gas.xlsx", start, [(0, 10), (1, 20)])])
    assert tab.start_date_edit.isEnabled()
    tab.start_date_edit.setDate(QDate(2026, 11, 10))
    tab.start_time_edit.setText("0:00")
    tab.frequency_edit.setText("30")
    tab.fuel_edit.setText("1")
    tab.additive_edit.setText("400")
    tab.duration_spin.setValue(1)
    tab.add_regime_button.click()
    assert len(tab.points) == 1
    assert tab.points[0].interval == "10.11 00:00:00, 1 с"
    assert tab.points[0].value("CO") == 20


def test_plc_uses_same_window_as_last_gas_records(qtbot, tmp_path):
    start = datetime(2026, 11, 9, 16, 46)
    xlsx = make_log(tmp_path / "gas.xlsx", start, [(0, 999), (30, 40), (50, 60), (60, 999)])
    plc = tmp_path / "plc.csv"
    with plc.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["id", "recorded_at", "plc_name", "payload"])
        for index, (time, fuel) in enumerate([(60, 999), (50.5, 2), (30, 1), (29.9, 999)]):
            timestamp = start + timedelta(seconds=time)
            writer.writerow([index, timestamp, "PLC210", json.dumps([{"path": PLC_FUEL, "value": fuel}])])

    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.load_files([xlsx])
    tab.load_plc_file(plc)
    tab.duration_spin.setValue(60)
    tab.average_mode.setCurrentIndex(1)
    tab.tail_spin.setValue(2)
    tab.frequency_edit.setText("f32")
    tab.nozzle_p_edit.setText("3000")
    tab.additive_edit.setText("400")
    tab.add_regime_button.click()

    assert len(tab.points) == 1
    assert tab.points[0].value("CO") == 50
    assert tab.points[0].fuel == 1.5
    assert tab.points[0].frequency == 32
    assert tab.points[0].nozzle_p == 3000
    assert tab.points_table.item(0, 6).text() == "2 зап."

    tab.duration_spin.setValue(10)
    tab.add_regime_button.click()
    assert len(tab.points) == 1
    assert "Нет записей ПЛК" in tab.status_label.text()


def test_fuel_comparison_reuses_regime_windows_with_clock_offset_and_units(qtbot, tmp_path, qapp):
    start = datetime(2026, 11, 9, 16, 46)
    xlsx = make_log(
        tmp_path / "gas.xlsx", start, [(0, 99), (20, 20), (29, 30), (30, 99), (50, 50), (59, 60), (60, 99)]
    )
    plc = tmp_path / "plc.csv"
    with plc.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["recorded_at", "plc_name", "payload"])
        for offset, channel, value in [
            (24, PLC_FUEL, 900), (25, PLC_FUEL, 0.25), (29, "other", 900),
            (34, PLC_FUEL, 0.75), (35, PLC_FUEL, 900),
        ]:
            writer.writerow([
                start + timedelta(seconds=offset), "PLC210", json.dumps([{"path": channel, "value": value}]),
            ])

    tab = ExperimentalDataWidget()
    qtbot.addWidget(tab)
    tab.load_files([xlsx])
    tab.duration_spin.setValue(30)
    tab.average_mode.setCurrentIndex(1)
    tab.tail_spin.setValue(2)
    tab.frequency_edit.setText("32")
    tab.nozzle_p_edit.setText("P3000")
    tab.fuel_edit.setText("1.2")
    tab.additive_edit.setText("800")
    tab.add_regime_button.click()
    tab.start_time_edit.setText("16:46:30")
    tab.frequency_edit.setText("21")
    tab.additive_edit.setText("400")
    tab.fuel_edit.clear()
    tab.add_comparison_button.click()
    tab.add_comparison_button.click()
    assert len(tab.points) == 1
    assert "уже есть" in tab.status_label.text()
    tab.load_plc_file(plc)
    tab.compare_button.click()

    dialog = tab.comparison_dialog
    table = dialog.table
    assert table.rowCount() == 2
    assert table.item(0, 1).text() == "21"
    assert table.item(1, 0).text() == "09.11.2026 16:46:20 — 16:46:30"
    assert table.item(1, 6).text() == "—"
    dialog.offset_spin.setValue(5)
    assert table.item(1, 5).text() == "0.5"
    assert table.item(1, 6).text() == "—"
    assert table.item(1, 7).text() == "2"
    assert "16:46:25" in table.item(1, 5).toolTip()
    dialog.unit_combo.setCurrentIndex(dialog.unit_combo.findText("г/с"))
    assert table.item(1, 4).text() == "—"
    assert table.item(1, 5).text() == "1.8"
    assert table.item(1, 6).text() == "—"
    assert table.item(1, 8).text() == "0.9 … 2.7"
    assert "Нет замеров топлива" in table.item(0, 9).text()
    assert tab.points[0].fuel == 1.2
    assert tab.points[0].value("CO") == 25
    dialog.offset_spin.setValue(15)
    assert table.item(1, 7).text() == "1"
    assert "Неполное окно ПЛК · Один замер" in table.item(1, 9).text()
    dialog.offset_spin.setValue(5)

    table.selectRow(1)
    assert not dialog.remove_button.isEnabled()
    dialog.copy_button.click()
    header, data = qapp.clipboard().text().splitlines()
    assert "Среднее ПЛК, кг/ч" in header
    assert data.split("\t")[1:4] == ["32", "3000", "800"]
    assert "скопировано" in dialog.selection_label.text().lower()
    table.clearSelection()
    table.selectRow(0)
    dialog.remove_button.click()
    assert table.rowCount() == 1
    dialog.close()
    tab.compare_button.click()
    assert table.rowCount() == 1
    assert table.item(0, 5).text() == "1.8"
