from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

diff_models = import_module("regime_map_app.diff_surface.models")
regime_cmaps_module = import_module("regime_map_app.regime_map.cmaps")
regime_ui_module = import_module("regime_map_app.regime_map.ui")

DEFAULT_ANALYSIS_CONTOUR_LEVELS = diff_models.DEFAULT_ANALYSIS_CONTOUR_LEVELS
DEFAULT_CMAP_NAME = regime_cmaps_module.DEFAULT_CMAP_NAME
MaximaDetectionMethod = diff_models.MaximaDetectionMethod
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


def test_regime_map_defaults_keep_diff_analysis_controls_inactive(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    assert not widget.show_min_line_checkbox.isChecked()
    assert not widget.show_right_line_checkbox.isChecked()
    assert not widget.show_mean_line_checkbox.isChecked()
    assert widget.current_maxima_detection_method() is MaximaDetectionMethod.ROW_PEAKS
    assert widget.contour_levels_edit.text() == DEFAULT_ANALYSIS_CONTOUR_LEVELS
    assert not widget.maxima_detection_combo.isEnabled()
    assert not widget.contour_levels_edit.isEnabled()
    assert not widget.decrease_contour_levels_button.isEnabled()
    assert not widget.increase_contour_levels_button.isEnabled()
    assert widget.x_axis_label_edit.text() == "Расход топлива, кг/ч"
    assert widget.y_axis_label_edit.text() == "Расход пара, кг/ч"
    assert widget.colorbar_label_edit.text() == "CO, ppm"
    assert widget.cmap_combo.currentText() == DEFAULT_CMAP_NAME
    assert widget.cmap_combo.isEditable()
    assert widget.font_size_spin.value() == 12
    assert not widget.use_custom_x_limits_checkbox.isChecked()
    assert not widget.use_custom_y_limits_checkbox.isChecked()
    assert not widget.use_custom_ppm_scale_checkbox.isChecked()
    assert not widget.x_min_spin.isEnabled()
    assert not widget.y_min_spin.isEnabled()
    assert not widget.ppm_min_spin.isEnabled()


def test_diff_analysis_controls_follow_selected_line_method(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    widget.show_right_line_checkbox.setChecked(True)

    assert widget.maxima_detection_combo.isEnabled()
    assert not widget.contour_levels_edit.isEnabled()

    widget.maxima_detection_combo.setCurrentIndex(
        widget.maxima_detection_combo.findData(MaximaDetectionMethod.CONTOUR_LEVELS)
    )

    assert widget.current_maxima_detection_method() is MaximaDetectionMethod.CONTOUR_LEVELS
    assert widget.contour_levels_edit.isEnabled()
    assert widget.decrease_contour_levels_button.isEnabled()
    assert widget.increase_contour_levels_button.isEnabled()

    widget.show_right_line_checkbox.setChecked(False)

    assert not widget.maxima_detection_combo.isEnabled()
    assert not widget.contour_levels_edit.isEnabled()


def test_custom_labels_font_size_and_diff_analysis_settings_are_collected_into_config(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    widget.show_mean_line_checkbox.setChecked(True)
    widget.maxima_detection_combo.setCurrentIndex(
        widget.maxima_detection_combo.findData(MaximaDetectionMethod.CONTOUR_LEVELS)
    )
    widget.contour_levels_edit.setText("3, 5")
    widget.x_axis_label_edit.setText("Fuel flow")
    widget.y_axis_label_edit.setText("Steam flow")
    widget.colorbar_label_edit.setText("CO concentration")
    widget.cmap_combo.setCurrentText("plasma")
    widget.font_size_spin.setValue(15)

    config = widget.collect_config()

    assert config.is_co_component is True
    assert config.show_mean_line is True
    assert config.x_axis_label == "Fuel flow"
    assert config.y_axis_label == "Steam flow"
    assert config.colorbar_label == "CO concentration"
    assert config.cmap_name == "plasma"
    assert config.maxima_detection_method is MaximaDetectionMethod.CONTOUR_LEVELS
    assert config.contour_levels_text == "3, 5"
    assert config.font_size == 15


def test_invalid_cmap_disables_run_button(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)
    widget.cmap_combo.setCurrentText("not-a-real-cmap")

    assert not widget.run_button.isEnabled()


def test_invalid_contour_levels_disable_run_button_when_right_line_uses_contours(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)
    widget.show_right_line_checkbox.setChecked(True)
    widget.maxima_detection_combo.setCurrentIndex(
        widget.maxima_detection_combo.findData(MaximaDetectionMethod.CONTOUR_LEVELS)
    )
    widget.contour_levels_edit.setText("0, bad")

    assert not widget.run_button.isEnabled()


def test_contour_level_buttons_shift_values_by_one(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    widget.show_right_line_checkbox.setChecked(True)
    widget.maxima_detection_combo.setCurrentIndex(
        widget.maxima_detection_combo.findData(MaximaDetectionMethod.CONTOUR_LEVELS)
    )
    widget.contour_levels_edit.setText("3, 4")

    qtbot.mouseClick(widget.increase_contour_levels_button, Qt.LeftButton)
    assert widget.contour_levels_edit.text() == "4, 5"

    qtbot.mouseClick(widget.decrease_contour_levels_button, Qt.LeftButton)
    qtbot.mouseClick(widget.decrease_contour_levels_button, Qt.LeftButton)
    qtbot.mouseClick(widget.decrease_contour_levels_button, Qt.LeftButton)
    assert widget.contour_levels_edit.text() == "1, 2"


def test_plot_canvas_is_top_aligned_in_gui(qtbot) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    plot_group = widget.layout().itemAt(1).widget()
    plot_layout = plot_group.layout()
    canvas_item = plot_layout.itemAt(0)

    assert canvas_item.widget() is widget.canvas
    assert canvas_item.alignment() & Qt.AlignTop
    assert widget.canvas.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum


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
