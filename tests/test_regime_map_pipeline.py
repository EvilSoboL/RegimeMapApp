from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np

regime_models = import_module("regime_map_app.regime_map.models")
regime_pipeline_module = import_module("regime_map_app.regime_map.pipeline")
diff_models = import_module("regime_map_app.diff_surface.models")

RegimeMapJobConfig = regime_models.RegimeMapJobConfig
RegimeMapPipeline = regime_pipeline_module.RegimeMapPipeline
CO_LEVELS = regime_models.CO_LEVELS
LineFit = diff_models.LineFit
DifferentialSurfaceResult = diff_models.DifferentialSurfaceResult
DiffSurfaceValidationResult = diff_models.ValidationResult
SurfaceMode = diff_models.SurfaceMode


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


class StubDiffPipeline:
    def __init__(self, result: DifferentialSurfaceResult) -> None:
        self.result = result
        self.validated_config = None
        self.processed_config = None

    def validate_inputs(self, config):
        self.validated_config = config
        return DiffSurfaceValidationResult(is_valid=True, errors=(), checked_points=16)

    def process_job(self, config, **_kwargs):
        self.processed_config = config
        return self.result


def test_validate_inputs_reuses_diff_surface_csv_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")

    validation = RegimeMapPipeline().validate_inputs(RegimeMapJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert any("разделитель" in error.lower() for error in validation.errors)


def test_process_job_uses_diff_surface_result_and_builds_mean_line(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    diff_result = DifferentialSurfaceResult(
        input_path=input_file,
        surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        fuel_axis=np.array([0.0, 1.0, 2.0]),
        additive_axis=np.array([0.0, 1.0, 2.0]),
        component_grid=np.array([[0.0, 10.0, 20.0], [5.0, 0.0, 25.0], [10.0, 5.0, 0.0]]),
        dz_dx=np.zeros((3, 3), dtype=float),
        dz_dy=np.zeros((3, 3), dtype=float),
        selected_surface=np.ones((3, 3), dtype=float),
        minima_points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        minima_line_fit=LineFit(slope=1.0, intercept=0.1),
        left_maxima_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        right_maxima_points=np.array([[1.0, 0.8], [2.0, 0.6]]),
        left_line_fit=LineFit(slope=1.1, intercept=0.0),
        right_line_fit=LineFit(slope=-0.2, intercept=1.0),
    )
    stub_diff_pipeline = StubDiffPipeline(diff_result)
    pipeline = RegimeMapPipeline(diff_pipeline=stub_diff_pipeline)

    result = pipeline.process_job(RegimeMapJobConfig(input_path=input_file))

    assert stub_diff_pipeline.processed_config.input_path == input_file
    assert stub_diff_pipeline.processed_config.surface_mode is SurfaceMode.GRADIENT_MAGNITUDE
    assert np.array_equal(result.component_grid, diff_result.component_grid)
    assert result.minima_line_fit == diff_result.minima_line_fit
    assert result.right_line_fit == diff_result.right_line_fit
    assert result.mean_line_fit.slope == 0.4
    assert result.mean_line_fit.intercept == 0.55
    assert np.array_equal(result.co_levels, CO_LEVELS)
