from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt

ApproxModuleWidget = import_module("regime_map_app.approx.ui").ApproxModuleWidget


def _write_valid_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "fuel;;additive;;component",
                "0;;0;;1",
                "0;;1;;2",
                "1;;0;;3",
                "1;;1;;4",
                "0.5;;0.5;;2.5",
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


def test_validate_button_shows_russian_error_in_log(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    bad_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    bad_file.write_text("fuel;additive;component\n0;0;1\n", encoding="utf-8")
    widget.set_input_paths([bad_file])
    widget.set_output_dir(tmp_path / "out")

    qtbot.mouseClick(widget.validate_button, Qt.LeftButton)

    assert "разделитель" in widget.log_edit.toPlainText()


def test_background_processing_updates_progress_and_actions(qtbot, tmp_path: Path) -> None:
    widget = ApproxModuleWidget()
    qtbot.addWidget(widget)

    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    _write_valid_csv(input_file)
    widget.set_input_paths([input_file])
    widget.set_output_dir(tmp_path / "out")
    widget.median_size_spin.setValue(0)

    qtbot.mouseClick(widget.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)

    assert widget.progress_bar.value() == 100
    assert widget.open_result_button.isEnabled()
    assert "Готово" in widget.status_label.text()
