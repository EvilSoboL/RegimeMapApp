from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

models = import_module("regime_map_app.diff_surface.models")
pipeline_module = import_module("regime_map_app.diff_surface.pipeline")
validation_module = import_module("regime_map_app.diff_surface.validation")
visualization_module = import_module("regime_map_app.diff_surface.visualization")

DiffSurfaceJobConfig = models.DiffSurfaceJobConfig
SplitMethod = models.SplitMethod
SurfaceMode = models.SurfaceMode
DiffSurfacePipeline = pipeline_module.DiffSurfacePipeline
resolve_export_paths = validation_module.resolve_export_paths
save_plot = visualization_module.save_plot


def _write_success_surface_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "fuel;additive;component",
                "0;0;0",
                "1;0;-1",
                "2;0;-3",
                "3;0;-6",
                "0;1;0",
                "1;1;1",
                "2;1;3",
                "3;1;3.2",
                "0;2;0",
                "1;2;0",
                "2;2;2",
                "3;2;3",
                "0;3;0",
                "1;3;0.5",
                "2;3;2",
                "3;3;5",
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
            split_method=SplitMethod.FUEL_MIDPOINT,
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
    assert payload["split_method"] == "fuel_midpoint"
    assert payload["line_1"]["points_count"] == 2
    assert payload["line_2"]["points_count"] == 2
