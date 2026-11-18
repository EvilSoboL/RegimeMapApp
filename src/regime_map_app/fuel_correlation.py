from __future__ import annotations

import json
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
    diameter: float = 2.0


@dataclass(frozen=True)
class FuelProfile:
    name: str
    fuels: dict[str, FuelProperties]


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


def _fuel_key(name: str) -> str:
    key = name.strip().casefold()
    return FUEL_ALIASES.get(key, key)


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Ключ «{name}» повторяется в конфиге.")
        result[name] = value
    return result


def load_fuel_profile(path: Path) -> FuelProfile:
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream, object_pairs_hook=_unique_json_fields)
    if not isinstance(data, dict) or not isinstance(data.get("fuels"), dict) or not data["fuels"]:
        raise ValueError("В конфиге нужен непустой объект «fuels».")
    name = data.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Название конфига «name» должно быть непустой строкой.")

    fuels = {}
    for identifier, entry in data["fuels"].items():
        key = _fuel_key(identifier)
        if not key or key in fuels or not isinstance(entry, dict):
            raise ValueError(f"Недопустимая или повторная запись топлива «{identifier}».")
        label = entry.get("label", identifier)
        composition = entry.get("composition", "")
        if not isinstance(label, str) or not label.strip() or not isinstance(composition, str):
            raise ValueError(f"«{identifier}». Поля label и composition должны быть строками.")
        fields = ("density", "surface_tension", "viscosity", "diameter")
        values = [entry.get(field) for field in fields[:3]] + [entry.get("diameter", 2.0)]
        for field, value in zip(fields, values):
            if type(value) not in (int, float):
                raise ValueError(f"«{identifier}». Параметр {field} должен быть числом.")
        try:
            laplace_number(*values)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"«{identifier}». Некорректные свойства топлива. {exc}") from exc
        density, surface_tension, viscosity, diameter = values
        fuels[key] = FuelProperties(label, density, surface_tension, viscosity, composition, diameter)
    return FuelProfile(name.strip(), fuels)


DEFAULT_FUEL_PROFILE = load_fuel_profile(Path(__file__).with_name("assets") / "fuel_properties.json")


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
        surfaces.append(FuelSurface(path, metadata.fuel_name, _fuel_key(metadata.fuel_name)))
    order = {key: index for index, key in enumerate(DEFAULT_FUEL_PROFILE.fuels)}
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
