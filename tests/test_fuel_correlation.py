import pytest

from regime_map_app.fuel_correlation import FUEL_PROPERTIES, laplace_number


def test_laplace_uses_diameter_and_converts_to_si() -> None:
    diesel = FUEL_PROPERTIES["diesel"]
    expected = 839.89 * 0.0308 * 0.002 / 0.00344**2

    assert laplace_number(diesel.density, diesel.surface_tension, diesel.viscosity, 2) == pytest.approx(expected)
    assert laplace_number(diesel.density, diesel.surface_tension, diesel.viscosity, 4) == pytest.approx(2 * expected)
