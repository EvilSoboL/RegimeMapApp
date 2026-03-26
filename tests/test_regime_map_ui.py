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


def test_regime_map_run_button_is_disabled_until_file_is_selected(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    assert widget.component_label.text() == "CO, ppm"
    assert not widget.run_button.isEnabled()
    assert not widget.validate_button.isEnabled()
    assert not widget.save_button.isEnabled()

    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    widget.set_input_path(input_file, user_selected=True)

    assert widget.run_button.isEnabled()
    assert widget.validate_button.isEnabled()
    assert not widget.save_button.isEnabled()


def test_regime_map_run_button_stays_disabled_for_invalid_csv(qtbot, tmp_path: Path) -> None:
    widget = RegimeMapModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")
    widget.set_input_path(input_file, user_selected=True)

    assert widget.validate_button.isEnabled()
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
