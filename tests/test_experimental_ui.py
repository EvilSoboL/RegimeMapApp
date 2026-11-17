import csv
import json
from datetime import datetime, timedelta

import pytest
from openpyxl import Workbook
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QFileDialog

from regime_map_app.approx.models import InputMode
from regime_map_app.experimental import COMPONENTS, PLC_FUEL, TIME_HEADER, read_gas_log
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
    tab._add_manual()
    assert len(tab.points) == 2
    assert "Частота форсунки" in tab.status_label.text()
    tab.frequency_edit.setText("F32")
    tab._add_manual()
    assert tab.points[2].fuel == pytest.approx(1.1144)
    assert tab.points_table.item(2, 7).text() == "32"

    tab.fuel_name_edit.setText("diesel")
    tab.diluent_name_edit.setText("steam")
    output = tmp_path / "diesel-steam-CO-09-11-2026.csv"
    assert tab.filename_label.text() == output.name
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "CSV (*.csv)"))
    window.approx_widget.mode_combo.setCurrentIndex(1)
    tab.save_button.click()

    assert output.read_text(encoding="utf-8").splitlines() == [
        "fuel;additive;component", "0.78;400.0;20.0", "0.78;600.0;60.0", "1.1144;600.0;500.0"
    ]
    assert window.tabs.currentWidget() is window.approx_widget
    config = window.approx_widget.collect_config()
    assert config.input_mode is InputMode.SINGLE_FILE
    assert window.approx_widget.pipeline.validate_inputs(config).is_valid
    window.regime_map_widget.apply_suggested_input_path(str(tmp_path / "approx_diesel-steam-O2-09-11-2026.csv"))
    assert window.regime_map_widget.colorbar_label_edit.text() == "O2, %"
    assert not window.regime_map_widget.collect_config().is_co_component
    qapp.processEvents()


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
    tab.additive_edit.setText("400")
    tab.add_regime_button.click()

    assert len(tab.points) == 1
    assert tab.points[0].value("CO") == 50
    assert tab.points[0].fuel == 1.5
    assert tab.points[0].frequency == 32
    assert tab.points_table.item(0, 6).text() == "2 зап."

    tab.duration_spin.setValue(10)
    tab.add_regime_button.click()
    assert len(tab.points) == 1
    assert "Нет записей ПЛК" in tab.status_label.text()
