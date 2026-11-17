from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QPushButton

from regime_map_app.approx.boundary_dialog import BoundaryPointsDialog
ApproxModuleWidget = import_module("regime_map_app.approx.ui").ApproxModuleWidget
InputMode = import_module("regime_map_app.approx.models").InputMode
CSV_SEPARATOR = import_module("regime_map_app.approx.models").CSV_SEPARATOR
REQUIRED_COLUMNS = import_module("regime_map_app.approx.models").REQUIRED_COLUMNS


def _write_valid_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                CSV_SEPARATOR.join(REQUIRED_COLUMNS),
                CSV_SEPARATOR.join(("0", "0", "1")),
                CSV_SEPARATOR.join(("0", "1", "2")),
                CSV_SEPARATOR.join(("1", "0", "3")),
                CSV_SEPARATOR.join(("1", "1", "4")),
                CSV_SEPARATOR.join(("0.5", "0.5", "2.5")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_button_is_disabled_until_form_is_valid(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    assert not widget.run_button.isEnabled()

    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)
    widget.set_input_paths([input_file])
    widget.set_output_dir(tmp_path / "out")

    assert widget.run_button.isEnabled()


def test_selecting_file_sets_output_dir_to_file_parent(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)

    widget.set_input_paths([input_file])

    assert widget.output_dir_edit.text() == str(tmp_path)


def test_source_preview_uses_raw_points_and_clears_for_invalid_csv(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)
    input_file = tmp_path / "diesel-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)

    assert widget.load_prepared_csv(input_file)
    axis, colorbar_axis = widget.preview_figure.axes
    points, = axis.collections
    assert points.get_offsets().tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]
    assert points.get_array().tolist() == [1, 2, 3, 4, 2.5]
    assert colorbar_axis.get_ylabel() == "CO"

    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData(InputMode.FOLDER_BATCH))
    assert widget.preview_group.isHidden()
    assert not widget.preview_figure.axes

    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("fuel;additive;component\n0;0;1\n", encoding="utf-8")
    assert widget.load_prepared_csv(invalid_file)
    assert widget.preview_canvas.isHidden()
    assert "недостаточно точек" in widget.preview_label.text()


def test_selecting_folder_sets_output_dir_to_same_folder(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    input_folder = tmp_path / "batch"
    input_folder.mkdir()
    widget.mode_combo.setCurrentIndex(1)

    widget.set_input_paths([input_folder])

    assert widget.output_dir_edit.text() == str(input_folder)


def test_co_boundary_editor_adds_only_new_points_to_preview_and_approximation(qtbot, monkeypatch, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)
    input_file = tmp_path / "diesel-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)
    contents = input_file.read_text(encoding="utf-8").replace("1;1;4", "1;1;500")
    input_file.write_text(contents, encoding="utf-8")
    widget.load_prepared_csv(input_file)
    assert not widget.boundary_button.isHidden()

    def fill_boundaries(dialog: BoundaryPointsDialog) -> int:
        fields = {field.accessibleName(): field for field in dialog.findChildren(QLineEdit)}
        fields["Правая граница, пар 0"].setText("f40")
        fields["Правая граница, пар 1"].setText("1")
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(BoundaryPointsDialog, "exec", fill_boundaries)
    widget.boundary_button.click()

    config = widget.collect_config()
    assert config.boundary_points == ((1.3812, 0.0, 500.0),)
    points, source = widget.preview_figure.axes[0].collections
    assert points.get_offsets().tolist() == [[1.3812, 0]]
    assert len(source.get_offsets()) == 5

    summary = widget.pipeline.process_job(config)
    assert summary.succeeded == 1
    grid = summary.successful_results[0].preview_surface
    assert grid.fuel_grid[0, -1] == pytest.approx(1.3812)
    assert grid.component_grid[0, -1] == pytest.approx(500)
    assert input_file.read_text(encoding="utf-8") == contents


def test_input_mode_has_only_file_and_folder_with_single_select_button(qtbot) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    modes = [widget.mode_combo.itemData(index) for index in range(widget.mode_combo.count())]

    assert modes == [
        InputMode.SINGLE_FILE,
        InputMode.FOLDER_BATCH,
    ]
    assert widget.select_input_button.text() == "Выбрать"
    assert "Открыть результат" not in [button.text() for button in widget.findChildren(QPushButton)]


def test_median_filter_checkbox_controls_median_size_in_config(qtbot) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    widget.median_size_spin.setValue(25)

    assert not widget.use_median_filter_checkbox.isChecked()
    assert not widget.median_size_spin.isEnabled()
    assert widget.collect_config().median_size == 0

    widget.use_median_filter_checkbox.setChecked(True)

    assert widget.median_size_spin.isEnabled()
    assert widget.collect_config().median_size == 25


def test_validate_button_shows_russian_error_in_log(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    bad_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    bad_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")
    widget.set_input_paths([bad_file])
    widget.set_output_dir(tmp_path / "out")

    qtbot.mouseClick(widget.validate_button, Qt.LeftButton)

    assert widget.log_edit.toPlainText() == (
        "Файл waste_oil-steam-CO-13-03-2026.csv использует неверный разделитель ','. "
        "Ожидается разделитель ';'. "
        f"Заголовок должен выглядеть так: {CSV_SEPARATOR.join(REQUIRED_COLUMNS)}"
    )


def test_background_processing_updates_progress_and_actions(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)
    widget.set_input_paths([input_file])
    widget.set_output_dir(tmp_path / "out")
    widget.resolution_x_spin.setValue(7)
    widget.resolution_y_spin.setValue(5)
    widget.use_median_filter_checkbox.setChecked(True)
    widget.median_size_spin.setValue(3)

    qtbot.mouseClick(widget.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)

    assert widget.progress_bar.value() == 100
    assert widget.open_folder_button.isEnabled()
    assert "Готово" in widget.status_label.text()
    source_axis, result_axis, colorbar_axis = widget.preview_figure.axes
    heatmap, = result_axis.images
    assert heatmap.norm is source_axis.collections[0].norm
    assert source_axis.get_xlim() == result_axis.get_xlim()
    assert source_axis.get_ylim() == result_axis.get_ylim()
    exported = np.loadtxt(tmp_path / "out" / f"approx_{input_file.name}", delimiter=";", skiprows=1)
    np.testing.assert_allclose(heatmap.get_array(), exported[:, 2].reshape(5, 7))
    assert colorbar_axis.get_ylabel() == "CO"

    widget.set_input_paths([input_file])
    assert len(widget.preview_figure.axes) == 2
