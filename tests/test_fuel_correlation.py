import json

import pytest
from PySide6.QtCore import Qt

from regime_map_app import fuel_correlation_ui
from regime_map_app.fuel_correlation import DEFAULT_FUEL_PROFILE, laplace_number, load_fuel_profile


def test_laplace_uses_diameter_and_converts_to_si() -> None:
    diesel = DEFAULT_FUEL_PROFILE.fuels["diesel"]
    expected = 839.89 * 0.0308 * 0.002 / 0.00344**2

    assert laplace_number(diesel.density, diesel.surface_tension, diesel.viscosity, 2) == pytest.approx(expected)
    assert laplace_number(diesel.density, diesel.surface_tension, diesel.viscosity, 4) == pytest.approx(2 * expected)


def test_json_profile_uses_csv_fuel_identifier_and_validates_properties(tmp_path) -> None:
    path = tmp_path / "profile.json"
    properties = {"density": 890, "surface_tension": 30, "viscosity": 50, "diameter": 4}
    data = {"name": "Проба", "fuels": {"МАЗУТ": properties}}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")

    fuel = load_fuel_profile(path).fuels["heavy_oil"]
    assert fuel.viscosity == 50
    assert fuel.diameter == 4

    properties["viscosity"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="МАЗУТ"):
        load_fuel_profile(path)


def test_correlation_recalculates_after_toggling_fuels(qtbot, qapp, tmp_path, monkeypatch) -> None:
    slopes = {"diesel": 3, "crude_oil": 1, "heavy_oil": 2}
    calls = []

    def read_slope(surface, _steam_scale, _pipeline):
        calls.append(surface.key)
        return slopes[surface.key]

    monkeypatch.setattr(fuel_correlation_ui, "optimal_slope", read_slope)
    for key in slopes:
        (tmp_path / f"approx_{key}-steam-CO-18-11-2026.csv").touch()
    properties = fuel_correlation_ui.FuelPropertiesWidget()
    widget = fuel_correlation_ui.FuelCorrelationWidget(properties)
    qtbot.addWidget(widget)
    qtbot.addWidget(properties)
    widget.folder.setText(str(tmp_path))
    widget.build_plot()
    original_title = widget.figure.axes[0].get_title()
    assert len(widget.figure.axes[0].collections) == 3

    diesel = widget.fuel_choices.item(0)
    diesel.setCheckState(Qt.CheckState.Unchecked)
    axis = widget.figure.axes[0]
    assert len(axis.collections) == 2
    assert "Дизельное топливо" not in [text.get_text() for text in axis.get_legend().get_texts()]
    assert "r = -1.000, R² = 1.000" in axis.get_title()

    widget.fuel_choices.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert len(widget.figure.axes[0].collections) == 1
    assert not widget.figure.axes[0].lines
    widget.fuel_choices.item(2).setCheckState(Qt.CheckState.Unchecked)
    assert not widget.figure.axes[0].collections

    for index in range(widget.fuel_choices.count()):
        widget.fuel_choices.item(index).setCheckState(Qt.CheckState.Checked)
    assert widget.figure.axes[0].get_title() == original_title
    assert calls == list(slopes)
    qapp.processEvents()
