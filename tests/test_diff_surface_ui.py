from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt

app_main_window_module = import_module("regime_map_app.main_window")
approx_models = import_module("regime_map_app.approx.models")
diff_ui_module = import_module("regime_map_app.diff_surface.ui")

RegimeMapMainWindow = app_main_window_module.RegimeMapMainWindow
BatchProcessSummary = approx_models.BatchProcessSummary
FileProcessResult = approx_models.FileProcessResult
DiffSurfaceModuleWidget = diff_ui_module.DiffSurfaceModuleWidget


def _write_success_surface_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "fuel;additive;component",
                "0;0;0",
                "1;0;-1",
                "2;0;-3",
                "3;0;-6",
                "0;1;0",
                "1;1;1",
                "2;1;3",
                "3;1;3.2",
                "0;2;0",
                "1;2;0",
                "2;2;2",
                "3;2;3",
                "0;3;0",
                "1;3;0.5",
                "2;3;2",
                "3;3;5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_diff_surface_run_button_is_disabled_until_file_is_selected(qtbot, tmp_path: Path) -> None:
    widget = DiffSurfaceModuleWidget()
    qtbot.addWidget(widget)

    assert not widget.run_button.isEnabled()
    assert not widget.validate_button.isEnabled()
    assert not widget.save_button.isEnabled()

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)

    assert widget.run_button.isEnabled()
    assert widget.validate_button.isEnabled()
    assert not widget.save_button.isEnabled()


def test_diff_surface_run_button_stays_disabled_for_invalid_csv(qtbot, tmp_path: Path) -> None:
    widget = DiffSurfaceModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")
    widget.set_input_path(input_file, user_selected=True)

    assert widget.validate_button.isEnabled()
    assert not widget.run_button.isEnabled()


def test_diff_surface_background_build_enables_save(qtbot, tmp_path: Path) -> None:
    widget = DiffSurfaceModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)

    qtbot.mouseClick(widget.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)

    assert widget.progress_bar.value() == 100
    assert widget.save_button.isEnabled()
    assert "Готово" in widget.status_label.text()


def test_main_window_has_two_tabs_and_autofills_second_tab(qtbot, tmp_path: Path) -> None:
    window = RegimeMapMainWindow()
    qtbot.addWidget(window)

    assert window.tabs.count() == 2
    assert window.tabs.tabText(0) == "Аппроксимация"
    assert window.tabs.tabText(1) == "Дифференциальная поверхность"

    output_path = tmp_path / "approx_surface.csv"
    summary = BatchProcessSummary(
        total_files=1,
        succeeded=1,
        failed=0,
        output_dir=tmp_path,
        results=(
            FileProcessResult(
                input_path=tmp_path / "source.csv",
                success=True,
                messages=("ok",),
                output_path=output_path,
            ),
        ),
    )

    window.approx_widget._on_completed(summary)

    assert window.diff_surface_widget.input_path_edit.text() == str(output_path)
