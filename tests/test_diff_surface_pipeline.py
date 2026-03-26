from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

models = import_module("regime_map_app.diff_surface.models")
pipeline_module = import_module("regime_map_app.diff_surface.pipeline")

CSV_SEPARATOR = models.CSV_SEPARATOR
REQUIRED_COLUMNS = models.REQUIRED_COLUMNS
DiffSurfaceJobConfig = models.DiffSurfaceJobConfig
SurfaceMode = models.SurfaceMode
DiffSurfacePipeline = pipeline_module.DiffSurfacePipeline


def _write_csv(path: Path, rows: list[tuple[object, object, object]]) -> None:
    lines = [CSV_SEPARATOR.join(REQUIRED_COLUMNS)]
    lines.extend(CSV_SEPARATOR.join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_success_surface_csv(path: Path) -> None:
    rows = [
        (0, 0, 0),
        (1, 0, 5),
        (2, 0, 6),
        (3, 0, 7),
        (0, 1, 4),
        (1, 1, 0),
        (2, 1, 10),
        (3, 1, 11),
        (0, 2, 8),
        (1, 2, 7),
        (2, 2, 0),
        (3, 2, 5),
        (0, 3, 9),
        (1, 3, 8),
        (2, 3, 7),
        (3, 3, 0),
    ]
    _write_csv(path, rows)


def test_build_regular_grid_returns_sorted_axes_and_matrix() -> None:
    frame = pd.DataFrame(
        [
            {"fuel": 2, "additive": 1, "component": 21},
            {"fuel": 1, "additive": 0, "component": 10},
            {"fuel": 2, "additive": 0, "component": 20},
            {"fuel": 1, "additive": 1, "component": 11},
        ]
    )

    fuel_axis, additive_axis, component_grid = DiffSurfacePipeline().build_regular_grid(frame)

    assert np.array_equal(fuel_axis, np.array([1.0, 2.0]))
    assert np.array_equal(additive_axis, np.array([0.0, 1.0]))
    assert np.array_equal(component_grid, np.array([[10.0, 20.0], [11.0, 21.0]]))


def test_compute_derivatives_returns_expected_values_for_plane() -> None:
    fuel_axis = np.array([0.0, 1.0, 2.0])
    additive_axis = np.array([0.0, 1.0, 2.0])
    fuel_grid, additive_grid = np.meshgrid(fuel_axis, additive_axis)
    component_grid = 2.0 * fuel_grid + 3.0 * additive_grid
    pipeline = DiffSurfacePipeline()

    dz_dx, dz_dy = pipeline.compute_derivatives(component_grid, additive_axis, fuel_axis)
    gradient_surface = pipeline.build_selected_surface(dz_dx, dz_dy, SurfaceMode.GRADIENT_MAGNITUDE)

    assert np.allclose(dz_dx, 2.0)
    assert np.allclose(dz_dy, 3.0)
    assert np.allclose(gradient_surface, np.sqrt(13.0))


def test_find_minima_points_and_fit_line() -> None:
    pipeline = DiffSurfacePipeline()
    component_grid = np.array(
        [
            [0.0, 5.0, 6.0, 7.0],
            [4.0, 0.0, 10.0, 11.0],
            [8.0, 7.0, 0.0, 5.0],
            [9.0, 8.0, 7.0, 0.0],
        ]
    )
    fuel_axis = np.array([0.0, 1.0, 2.0, 3.0])
    additive_axis = np.array([0.0, 1.0, 2.0, 3.0])

    minima_points = pipeline.find_minima_points(
        component_grid,
        fuel_axis,
        additive_axis,
    )
    minima_line_fit = pipeline.fit_line(minima_points, "минимумов концентрации")

    assert np.array_equal(minima_points, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert np.isclose(minima_line_fit.slope, 1.0)
    assert np.isclose(minima_line_fit.intercept, 0.0)


def test_find_minima_points_tracks_minimum_for_each_fuel_value() -> None:
    pipeline = DiffSurfacePipeline()
    component_grid = np.array(
        [
            [5.0, 0.0, 5.0],
            [0.0, 5.0, 0.5],
            [6.0, 1.0, 2.0],
        ]
    )
    fuel_axis = np.array([0.0, 1.0, 2.0])
    additive_axis = np.array([0.0, 1.0, 2.0])

    minima_points = pipeline.find_minima_points(
        component_grid,
        fuel_axis,
        additive_axis,
    )

    assert np.array_equal(minima_points, np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]))


def test_process_job_builds_surface_and_finds_minimum_line(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    pipeline = DiffSurfacePipeline()

    result = pipeline.process_job(
        DiffSurfaceJobConfig(
            input_path=input_file,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        )
    )

    assert result.surface_mode is SurfaceMode.GRADIENT_MAGNITUDE
    assert result.selected_surface.shape == (4, 4)
    assert np.array_equal(result.minima_points, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert np.isclose(result.minima_line_fit.slope, 1.0)
    assert np.isclose(result.minima_line_fit.intercept, 0.0)
