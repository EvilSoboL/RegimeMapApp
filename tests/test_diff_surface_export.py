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

DiffSurfaceJobConfig = models.DiffSurfaceJobConfig
DifferentialSurfaceResult = models.DifferentialSurfaceResult
LineFit = models.LineFit
SurfaceMode = models.SurfaceMode
DiffSurfacePipeline = pipeline_module.DiffSurfacePipeline
create_figure = visualization_module.create_figure
render_result = visualization_module.render_result
resolve_export_paths = validation_module.resolve_export_paths
save_plot = visualization_module.save_plot


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


def test_export_creates_png_and_json(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    output_dir = tmp_path / "out"
    _write_success_surface_csv(input_file)
    pipeline = DiffSurfacePipeline()
    result = pipeline.process_job(
        DiffSurfaceJobConfig(
            input_path=input_file,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        )
    )
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


def test_render_result_clips_line_to_surface_bounds() -> None:
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
        minima_line_fit=LineFit(slope=2.0, intercept=-1.0),
    )

    render_result(figure, result)

    main_axis = figure.axes[0]
    line = main_axis.lines[0]

    assert np.allclose(line.get_xdata(), np.array([0.5, 1.0]))
    assert np.allclose(line.get_ydata(), np.array([0.0, 1.0]))
    assert main_axis.get_xlim() == pytest.approx((0.0, 1.0))
    assert main_axis.get_ylim() == pytest.approx((0.0, 1.0))
