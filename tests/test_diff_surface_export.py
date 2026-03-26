from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

models = import_module("regime_map_app.diff_surface.models")
pipeline_module = import_module("regime_map_app.diff_surface.pipeline")
validation_module = import_module("regime_map_app.diff_surface.validation")
visualization_module = import_module("regime_map_app.diff_surface.visualization")

DifferentialSurfaceResult = models.DifferentialSurfaceResult
LineFit = models.LineFit
SurfaceMode = models.SurfaceMode
DiffSurfacePipeline = pipeline_module.DiffSurfacePipeline
create_figure = visualization_module.create_figure
render_result = visualization_module.render_result
resolve_export_paths = validation_module.resolve_export_paths
save_plot = visualization_module.save_plot


def _build_result(input_path: Path) -> DifferentialSurfaceResult:
    return DifferentialSurfaceResult(
        input_path=input_path,
        surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        fuel_axis=np.array([0.0, 1.0, 2.0, 3.0]),
        additive_axis=np.array([0.0, 1.0, 2.0, 3.0]),
        component_grid=np.array(
            [
                [0.0, 5.0, 6.0, 7.0],
                [4.0, 0.0, 10.0, 11.0],
                [8.0, 7.0, 0.0, 5.0],
                [9.0, 8.0, 7.0, 0.0],
            ]
        ),
        dz_dx=np.ones((4, 4), dtype=float),
        dz_dy=np.ones((4, 4), dtype=float),
        selected_surface=np.array(
            [
                [8.0, 2.0, 1.0, 7.0],
                [1.0, 7.0, 1.0, 9.0],
                [0.0, 1.0, 9.0, 7.0],
                [0.0, 1.0, 8.0, 7.0],
            ]
        ),
        minima_points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        minima_line_fit=LineFit(slope=1.0, intercept=0.0),
        left_maxima_points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]),
        right_maxima_points=np.array([[3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [3.0, 3.0]]),
        left_line_fit=LineFit(slope=1.2, intercept=0.1),
        right_line_fit=LineFit(slope=0.9, intercept=0.2),
    )


def test_export_creates_png_and_json(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    output_dir = tmp_path / "out"
    input_file.write_text("fuel;additive;component\n0;0;0\n1;0;1\n0;1;1\n1;1;0\n", encoding="utf-8")
    result = _build_result(input_file)
    pipeline = DiffSurfacePipeline()
    png_path, json_path = resolve_export_paths(output_dir, input_file, SurfaceMode.GRADIENT_MAGNITUDE)

    save_plot(result, png_path)
    pipeline.export_line_parameters(result, json_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["input_file"] == str(input_file)
    assert payload["surface_mode"] == "grad"
    assert payload["min_concentration_line"]["points_count"] == 4
    assert payload["min_concentration_line"]["a"] == pytest.approx(1.0)
    assert payload["min_concentration_line"]["b"] == pytest.approx(0.0)
    assert payload["left_max_gradient_line"]["points_count"] == 4
    assert payload["left_max_gradient_line"]["a"] == pytest.approx(1.2)
    assert payload["left_max_gradient_line"]["b"] == pytest.approx(0.1)
    assert payload["right_max_gradient_line"]["points_count"] == 4
    assert payload["right_max_gradient_line"]["a"] == pytest.approx(0.9)
    assert payload["right_max_gradient_line"]["b"] == pytest.approx(0.2)


def test_render_result_clips_three_lines_to_surface_bounds() -> None:
    figure = create_figure()
    result = DifferentialSurfaceResult(
        input_path=Path("surface.csv"),
        surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        fuel_axis=np.array([0.0, 1.0]),
        additive_axis=np.array([0.0, 1.0]),
        component_grid=np.array([[0.0, 1.0], [1.0, 0.0]]),
        dz_dx=np.array([[1.0, 1.0], [1.0, 1.0]]),
        dz_dy=np.array([[1.0, 1.0], [1.0, 1.0]]),
        selected_surface=np.array([[1.0, 1.0], [1.0, 1.0]]),
        minima_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        minima_line_fit=LineFit(slope=1.0, intercept=0.0),
        left_maxima_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        right_maxima_points=np.array([[0.0, 1.0], [1.0, 0.0]]),
        left_line_fit=LineFit(slope=2.0, intercept=-1.0),
        right_line_fit=LineFit(slope=-2.0, intercept=2.0),
    )

    render_result(figure, result)

    main_axis = figure.axes[0]
    min_line = main_axis.lines[0]
    left_line = main_axis.lines[1]
    right_line = main_axis.lines[2]

    assert np.allclose(min_line.get_xdata(), np.array([0.0, 1.0]))
    assert np.allclose(min_line.get_ydata(), np.array([0.0, 1.0]))
    assert np.allclose(left_line.get_xdata(), np.array([0.5, 1.0]))
    assert np.allclose(left_line.get_ydata(), np.array([0.0, 1.0]))
    assert np.allclose(right_line.get_xdata(), np.array([0.5, 1.0]))
    assert np.allclose(right_line.get_ydata(), np.array([1.0, 0.0]))
    assert main_axis.get_xlim() == pytest.approx((0.0, 1.0))
    assert main_axis.get_ylim() == pytest.approx((0.0, 1.0))
