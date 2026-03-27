from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt

regime_ui_module = import_module("regime_map_app.regime_map.ui")

RegimeMapModuleWidget = regime_ui_module.RegimeMapModuleWidget


def _write_success_surface_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "fuel;additive;component",
                "0;0;0",
                "1;0;5",
                "2;0;6",
                "3;0;7",
                "0;1;4",
                "1;1;0",
                "2;1;10",
                "3;1;11",
                "0;2;8",
                "1;2;7",
                "2;2;0",
                "3;2;5",
                "0;3;9",
                "1;3;8",
                "2;3;7",
                "3;3;0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_regime_map_defaults_use_auto_ranges_and_enabled_line_options(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    assert widget.co_checkbox.isChecked()
    assert widget.show_min_line_checkbox.isChecked()
    assert widget.show_right_line_checkbox.isChecked()
    assert widget.show_mean_line_checkbox.isChecked()
    assert not widget.use_custom_x_limits_checkbox.isChecked()
    assert not widget.use_custom_y_limits_checkbox.isChecked()
    assert not widget.use_custom_ppm_scale_checkbox.isChecked()
    assert not widget.x_min_spin.isEnabled()
    assert not widget.y_min_spin.isEnabled()
    assert not widget.ppm_min_spin.isEnabled()


def test_unchecking_co_disables_right_line_overlay(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    widget.co_checkbox.setChecked(False)

    assert not widget.show_right_line_checkbox.isEnabled()
    assert not widget.show_right_line_checkbox.isChecked()
    assert widget.show_mean_line_checkbox.isEnabled()


def test_invalid_manual_x_limits_disable_run_button(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)

    widget.use_custom_x_limits_checkbox.setChecked(True)
    widget.x_min_spin.setValue(1.5)
    widget.x_max_spin.setValue(1.0)

    assert not widget.run_button.isEnabled()


def test_regime_map_background_build_enables_save(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)

    qtbot.mouseClick(widget.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)

    assert widget.progress_bar.value() == 100
    assert widget.save_button.isEnabled()
    assert "Готово" in widget.status_label.text()
