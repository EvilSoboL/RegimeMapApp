from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

regime_models = import_module("regime_map_app.regime_map.models")
regime_pipeline_module = import_module("regime_map_app.regime_map.pipeline")
diff_models = import_module("regime_map_app.diff_surface.models")

AUTO_CONTOUR_LEVELS = regime_models.AUTO_CONTOUR_LEVELS
CO_COMPONENT_LABEL = regime_models.CO_COMPONENT_LABEL
GENERIC_COMPONENT_LABEL = regime_models.GENERIC_COMPONENT_LABEL
RegimeMapJobConfig = regime_models.RegimeMapJobConfig
RegimeMapPipeline = regime_pipeline_module.RegimeMapPipeline
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


def _build_diff_result(input_path: Path) -> DifferentialSurfaceResult:
    return DifferentialSurfaceResult(
        input_path=input_path,
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


def test_validate_inputs_reuses_diff_surface_csv_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")

    validation = RegimeMapPipeline().validate_inputs(RegimeMapJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert any("разделитель" in error.lower() for error in validation.errors)


def test_process_job_uses_diff_surface_result_with_auto_ranges_and_non_co_overlay_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    stub_diff_pipeline = StubDiffPipeline(_build_diff_result(input_file))
    pipeline = RegimeMapPipeline(diff_pipeline=stub_diff_pipeline)

    result = pipeline.process_job(
        RegimeMapJobConfig(
            input_path=input_file,
            is_co_component=False,
            show_min_line=True,
            show_right_line=True,
            show_mean_line=True,
        )
    )

    assert stub_diff_pipeline.processed_config.input_path == input_file
    assert stub_diff_pipeline.processed_config.surface_mode is SurfaceMode.GRADIENT_MAGNITUDE
    assert result.component_label == GENERIC_COMPONENT_LABEL
    assert result.co_levels == AUTO_CONTOUR_LEVELS
    assert result.x_limits is None
    assert result.y_limits is None
    assert result.show_min_line is True
    assert result.show_right_line is False
    assert result.show_mean_line is True
    assert result.mean_line_fit.slope == pytest.approx(0.4)
    assert result.mean_line_fit.intercept == pytest.approx(0.55)


def test_process_job_supports_custom_ranges_and_ppm_scale(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    stub_diff_pipeline = StubDiffPipeline(_build_diff_result(input_file))
    pipeline = RegimeMapPipeline(diff_pipeline=stub_diff_pipeline)

    result = pipeline.process_job(
        RegimeMapJobConfig(
            input_path=input_file,
            is_co_component=True,
            show_min_line=True,
            show_right_line=True,
            show_mean_line=False,
            use_custom_x_limits=True,
            x_min=0.7,
            x_max=1.3,
            use_custom_y_limits=True,
            y_min=0.6,
            y_max=1.0,
            use_custom_ppm_scale=True,
            ppm_min=0.0,
            ppm_max=200.0,
            ppm_step=25.0,
        )
    )

    assert result.component_label == CO_COMPONENT_LABEL
    assert np.array_equal(result.co_levels, np.arange(0.0, 201.0, 25.0))
    assert result.x_limits == pytest.approx((0.7, 1.3))
    assert result.y_limits == pytest.approx((0.6, 1.0))
    assert result.show_right_line is True
    assert result.show_mean_line is False
