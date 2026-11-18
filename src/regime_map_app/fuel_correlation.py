from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .approx.exceptions import ValidationError as MetadataError
from .approx.validation import parse_file_metadata
from .diff_surface.pipeline import DiffSurfacePipeline


@dataclass(frozen=True)
class FuelProperties:
    label: str
    density: float
    surface_tension: float
    viscosity: float
    composition: str = ""


# Temperature fits for the automotive diesel sample studied by Esteban et al.
# Density and kinematic viscosity: https://doi.org/10.1016/j.biombioe.2012.03.007
# Tables 4 and 6: ρ = 844.2 − 0.67009 t, ν = exp(−1.2202 − 499.62/T + 373853.59/T²).
# Surface tension: https://doi.org/10.1016/j.fuel.2012.07.042
# σ = 29.6168 − 0.0789 t. t is in °C, T is in K, and μ [mPa·s] = ρ ν / 1000.
DIESEL_TEMPERATURE_NOTE = (
    "Справочная оценка по измерениям образца Esteban и др. (2012). "
    "Плотность и вязкость — Biomass and Bioenergy 42, табл. 4 и 6. "
    "Поверхностное натяжение — Fuel 102, с. 231–238. Состав партии опытов неизвестен."
)

# Other rows use the values from the biofuel correlation notebook.
FUEL_PROPERTIES = {
    "diesel": FuelProperties("Дизельное топливо", 839.89, 30.8, 3.44),
    "дизель27": FuelProperties("Дизельное топливо, 27 °C", 826.1, 27.49, 2.927, DIESEL_TEMPERATURE_NOTE),
    "дизель45": FuelProperties("Дизельное топливо, 45 °C", 814.0, 26.07, 2.008, DIESEL_TEMPERATURE_NOTE),
    "дизель75": FuelProperties("Дизельное топливо, 75 °C", 793.9, 23.70, 1.219, DIESEL_TEMPERATURE_NOTE),
    "crude_oil": FuelProperties("Сырая нефть", 815.61, 28, 5.14),
    "heavy_oil": FuelProperties("Мазут", 890, 30, 50),
    "kerosene": FuelProperties("Керосин", 788.89, 26.6, 1.49),
    "waste_oil": FuelProperties("Отработанное масло", 920, 26.6, 125.82),
    "waste_oil_40_2024": FuelProperties("Отработанное масло, 40 °C", 899.9, 26.6, 49.34),
    "waste_oil_60_2024": FuelProperties("Отработанное масло, 60 °C", 879.2, 26.6, 23.54),
    "diesel_90_mefa_rapeseed_10": FuelProperties(
        "Дизель + МЭЖК (rapeseed)", 751, 27.5, 2.65854,
        "90% дизельного топлива, 10% МЭЖК. В ноутбуке есть подписи «рапсовое» и «рыжиковое».",
    ),
    "diesel_90_mefa_tall_10": FuelProperties(
        "Дизель + талловые МЭЖК", 735, 26.8, 2.793, "90% дизельного топлива, 10% талловых МЭЖК.",
    ),
}
FUEL_ALIASES = {"дизель": "diesel", "керосин": "kerosene", "мазут": "heavy_oil", "сырая нефть": "crude_oil"}
MINIMUM_ADJUSTMENTS = {"waste_oil": (5, 0.025), "waste_oil_40_2024": (17, 0), "waste_oil_60_2024": (17, 0.1)}
BOUNDARY_ADJUSTMENTS = {
    "diesel": (8, -0.05),
    "heavy_oil": (5, -0.1),
    "waste_oil_40_2024": (0, 0.05),
    "waste_oil_60_2024": (15, -0.3),
}


def laplace_number(density: float, surface_tension: float, viscosity: float, diameter: float) -> float:
    """Use kg/m³, mN/m, mPa·s and mm, respectively."""
    if not all(isfinite(value) and value > 0 for value in (density, surface_tension, viscosity, diameter)):
        raise ValueError("Для расчёта La нужны четыре положительных конечных значения.")
    # mm and mN/m each contribute 10⁻³, cancelling (10⁻³)² from mPa·s.
    result = (density / viscosity) * (diameter / viscosity) * surface_tension
    if not isfinite(result) or result == 0:
        raise ValueError("Число Лапласа выходит за пределы численного диапазона.")
    return result


@dataclass(frozen=True)
class FuelSurface:
    path: Path
    fuel_name: str
    key: str


def find_fuel_surfaces(folder: Path) -> tuple[list[FuelSurface], list[str]]:
    surfaces = []
    skipped = []
    for path in sorted(folder.iterdir(), key=lambda file: file.name.casefold()):
        if not path.is_file() or path.suffix.casefold() != ".csv" or not path.name.casefold().startswith("approx_"):
            continue
        try:
            metadata = parse_file_metadata(Path(path.name[len("approx_"):]))
        except MetadataError:
            skipped.append(f"{path.name}: ожидается имя approx_топливо-разбавитель-компонент-дд-мм-гггг.csv.")
            continue
        if metadata.component_name.casefold() != "co" or metadata.diluent.casefold() not in {"steam", "пар"}:
            skipped.append(f"{path.name}: для графика нужна поверхность CO с подачей пара.")
            continue
        key = metadata.fuel_name.casefold()
        surfaces.append(FuelSurface(path, metadata.fuel_name, FUEL_ALIASES.get(key, key)))
    order = {key: index for index, key in enumerate(FUEL_PROPERTIES)}
    surfaces.sort(key=lambda surface: order.get(surface.key, len(order)))
    return surfaces, skipped


def optimal_slope(surface: FuelSurface, steam_to_kg: float, pipeline: DiffSurfacePipeline) -> float:
    """Reproduce the notebook's CO contour method in kg/h on both axes."""
    frame = pipeline.read_dataset(surface.path)
    fuel, steam, grid = pipeline.build_regular_grid(frame)
    # The notebook stores steam rows in descending order, which affects contour path order.
    steam = steam[::-1] * steam_to_kg
    grid = grid[::-1]
    if not np.isfinite(grid).all() or not np.isfinite(fuel).all() or not np.isfinite(steam).all():
        raise ValueError("В сетке есть бесконечные значения.")

    minimum_fuel = fuel[np.argmin(grid, axis=1)]
    minimum = pipeline.fit_line(np.column_stack((minimum_fuel, steam)), "линии минимумов CO")
    min_slope, min_intercept = _adjust_line(
        minimum.slope, minimum.intercept, (minimum_fuel[0] + minimum_fuel[-1]) / 2,
        MINIMUM_ADJUSTMENTS.get(surface.key, (0, 0)),
    )
    figure = Figure()
    try:
        contours = figure.subplots().contourf(fuel, steam, grid)
        # The fourth filled band is a compound path, including disconnected polygons.
        bands = contours.get_paths()
        if len(bands) <= 3 or not len(bands[3].vertices):
            raise ValueError("В поверхности CO отсутствует четвёртая контурная полоса.")
        points = bands[3].vertices
        points = points[points[:, 1] <= min_slope * points[:, 0] + min_intercept]
    finally:
        figure.clear()
    boundary = pipeline.fit_line(points, "границы оптимального режима")
    boundary_slope, _ = _adjust_line(
        boundary.slope, boundary.intercept, (fuel[0] + fuel[-1]) / 2,
        BOUNDARY_ADJUSTMENTS.get(surface.key, (0, 0)),
    )
    return (min_slope + boundary_slope) / 2


def _adjust_line(
    slope: float, intercept: float, pivot: float, adjustment: tuple[float, float],
) -> tuple[float, float]:
    angle, offset = adjustment
    theta = np.deg2rad(angle)
    new_slope = slope * np.cos(theta) - slope * pivot * np.sin(theta)
    return new_slope, intercept + (slope - new_slope) * pivot - offset
