import json

import pytest

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
