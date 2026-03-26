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


def test_find_minima_points_tracks_minimum_concentration_line() -> None:
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

    minima_points = pipeline.find_minima_points(component_grid, fuel_axis, additive_axis)
    minima_line_fit = pipeline.fit_line(minima_points, "линии минимумов концентрации")

    assert np.array_equal(minima_points, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert np.isclose(minima_line_fit.slope, 1.0)
    assert np.isclose(minima_line_fit.intercept, 0.0)


def test_find_maxima_points_uses_minimum_line_for_grouping() -> None:
    pipeline = DiffSurfacePipeline()
    surface = np.array(
        [
            [8.0, 2.0, 1.0, 7.0, 1.0, 0.0],
            [1.0, 7.0, 1.0, 9.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 9.0, 1.0, 7.0],
            [0.0, 1.0, 2.0, 8.0, 1.0, 7.0],
        ]
    )
    fuel_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    additive_axis = np.array([0.0, 1.0, 2.0, 3.0])
    minimum_line_fit = models.LineFit(slope=1.0, intercept=0.0)

    left_points, right_points = pipeline.find_maxima_points(
        surface,
        fuel_axis,
        additive_axis,
        minimum_line_fit,
    )

    assert np.array_equal(left_points, np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]))
    assert np.array_equal(right_points, np.array([[3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [5.0, 2.0], [5.0, 3.0]]))


def test_process_job_builds_surface_and_restores_three_lines(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    pipeline = DiffSurfacePipeline()
    synthetic_surface = np.array(
        [
            [8.0, 2.0, 7.0, 0.0],
            [1.0, 7.0, 9.0, 0.0],
            [0.0, 1.0, 9.0, 7.0],
            [0.0, 1.0, 8.0, 7.0],
        ]
    )

    def fake_compute_derivatives(
        component_grid: np.ndarray,
        additive_axis: np.ndarray,
        fuel_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros_like(component_grid, dtype=float)
        return zeros, zeros

    def fake_build_selected_surface(
        dz_dx: np.ndarray,
        dz_dy: np.ndarray,
        surface_mode: SurfaceMode,
    ) -> np.ndarray:
        return synthetic_surface

    pipeline.compute_derivatives = fake_compute_derivatives  # type: ignore[method-assign]
    pipeline.build_selected_surface = fake_build_selected_surface  # type: ignore[method-assign]

    result = pipeline.process_job(
        DiffSurfaceJobConfig(
            input_path=input_file,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        )
    )

    assert result.surface_mode is SurfaceMode.GRADIENT_MAGNITUDE
    assert np.array_equal(result.minima_points, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert np.array_equal(result.left_maxima_points, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0], [3.0, 3.0]]))
    assert np.array_equal(result.right_maxima_points, np.array([[2.0, 0.0], [2.0, 1.0], [3.0, 2.0]]))
    assert np.isfinite(result.minima_line_fit.slope)
    assert np.isfinite(result.left_line_fit.slope)
    assert np.isfinite(result.right_line_fit.slope)
