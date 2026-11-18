from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

regime_models = import_module("regime_map_app.regime_map.models")
regime_cmaps_module = import_module("regime_map_app.regime_map.cmaps")
regime_pipeline_module = import_module("regime_map_app.regime_map.pipeline")
diff_models = import_module("regime_map_app.diff_surface.models")

AUTO_CONTOUR_LEVELS = regime_models.AUTO_CONTOUR_LEVELS
CO_COMPONENT_LABEL = regime_models.CO_COMPONENT_LABEL
DEFAULT_CMAP_NAME = regime_cmaps_module.DEFAULT_CMAP_NAME
GENERIC_COMPONENT_LABEL = regime_models.GENERIC_COMPONENT_LABEL
RegimeMapJobConfig = regime_models.RegimeMapJobConfig
RegimeMapPipeline = regime_pipeline_module.RegimeMapPipeline
DEFAULT_ANALYSIS_CONTOUR_LEVELS = diff_models.DEFAULT_ANALYSIS_CONTOUR_LEVELS
DiffSurfaceValidationResult = diff_models.ValidationResult
DiffSurfacePipeline = regime_pipeline_module.DiffSurfacePipeline
MaximaDetectionMethod = diff_models.MaximaDetectionMethod


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


class StubDiffPipeline(DiffSurfacePipeline):
    def resolve_contour_level_values(self, _surface, indices):
        assert indices == (3, 4)
        return 0.25, 0.5

    def extract_contour_points(self, _surface, _fuel, _additive, values):
        assert values == (0.25, 0.5)
        return np.array([[2.0, 0.0], [2.5, 1.0], [3.0, 2.0]])


class NoLineStubDiffPipeline:
    def __init__(self) -> None:
        self.validated_config = None
        self.process_job_called = False

    def validate_inputs(self, config):
        self.validated_config = config
        return DiffSurfaceValidationResult(is_valid=True, errors=(), checked_points=16)

    def read_dataset(self, _path):
        return object()

    def build_regular_grid(self, _frame):
        return (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([[0.0, 10.0, 20.0], [5.0, 0.0, 25.0], [10.0, 5.0, 0.0]], dtype=float),
        )

    def process_job(self, config, **_kwargs):
        self.process_job_called = True
        raise AssertionError(f"process_job must not be called when all line checkboxes are disabled: {config}")


def test_validate_inputs_reuses_diff_surface_csv_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n", encoding="utf-8")

    validation = RegimeMapPipeline().validate_inputs(RegimeMapJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert any("разделитель" in error.lower() for error in validation.errors)


def test_minimum_path_follows_bent_valley_and_breaks_at_separate_valley(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    grid = [
        [80, 10, 15, 80, 80],
        [80, 15, 10, 80, 80],
        [80, 14, 10, 80, 80],
        [80, 10, 15, 80, 80],
        [80, 80, 80, 80, 10],
    ]
    rows = ["fuel;additive;component"]
    rows.extend(f"{x};{y};{value}" for y, row in enumerate(grid) for x, value in enumerate(row))
    input_file.write_text("\n".join(rows), encoding="utf-8")

    result = RegimeMapPipeline().process_job(RegimeMapJobConfig(input_path=input_file, show_min_line=True))

    assert np.isnan(result.minima_points[4]).all()
    points = result.minima_points[np.isfinite(result.minima_points[:, 0])]
    assert points[:, 1] == pytest.approx(np.arange(5))
    for x, y in points:
        assert grid[int(y)][int(x)] <= min(grid[int(y)]) + 10
    assert points[-1, 0] == 4


def test_validate_inputs_ignores_contour_settings_without_right_line_analysis(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    stub_diff_pipeline = NoLineStubDiffPipeline()
    pipeline = RegimeMapPipeline(diff_pipeline=stub_diff_pipeline)

    validation = pipeline.validate_inputs(
        RegimeMapJobConfig(
            input_path=input_file,
            maxima_detection_method=MaximaDetectionMethod.CONTOUR_LEVELS,
            contour_levels_text="0, bad",
        )
    )

    assert validation.is_valid
    assert stub_diff_pipeline.validated_config.maxima_detection_method is MaximaDetectionMethod.ROW_PEAKS
    assert stub_diff_pipeline.validated_config.contour_levels_text == ""


def test_process_job_uses_slice_paths_with_auto_ranges_and_non_co_overlay_rules(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    pipeline = RegimeMapPipeline()

    result = pipeline.process_job(
        RegimeMapJobConfig(
            input_path=input_file,
            is_co_component=False,
            show_min_line=True,
            show_right_line=True,
            show_mean_line=True,
        )
    )

    assert result.component_label == GENERIC_COMPONENT_LABEL
    assert result.co_levels == AUTO_CONTOUR_LEVELS
    assert result.x_limits is None
    assert result.y_limits is None
    assert result.show_min_line is True
    assert result.show_right_line is False
    assert result.show_mean_line is True
    valid = np.isfinite(result.right_points[:, 0])
    assert valid.any()
    assert result.minima_points[np.isfinite(result.minima_points[:, 0]), 0] == pytest.approx([0, 1, 2, 3])
    assert result.mean_points[valid, 0] == pytest.approx(
        (result.minima_points[valid, 0] + result.right_points[valid, 0]) / 2
    )


def test_process_job_builds_map_without_line_calculations_when_checkboxes_are_disabled(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    stub_diff_pipeline = NoLineStubDiffPipeline()
    pipeline = RegimeMapPipeline(diff_pipeline=stub_diff_pipeline)

    result = pipeline.process_job(
        RegimeMapJobConfig(
            input_path=input_file,
            show_min_line=False,
            show_right_line=False,
            show_mean_line=False,
            maxima_detection_method=MaximaDetectionMethod.CONTOUR_LEVELS,
            contour_levels_text="3, 4",
        )
    )

    assert stub_diff_pipeline.process_job_called is False
    assert result.show_min_line is False
    assert result.show_right_line is False
    assert result.show_mean_line is False
    assert np.array_equal(result.fuel_axis, np.array([0.0, 1.0, 2.0]))
    assert result.analysis_contour_indices == ()


def test_process_job_supports_custom_ranges_ppm_scale_and_contour_based_right_line(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    pipeline = RegimeMapPipeline(diff_pipeline=StubDiffPipeline())
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
            x_axis_label="Fuel flow",
            y_axis_label="Steam flow",
            colorbar_label="CO concentration",
            cmap_name="Plasma",
            maxima_detection_method=MaximaDetectionMethod.CONTOUR_LEVELS,
            contour_levels_text="3, 4",
            font_size=16,
        )
    )

    assert result.component_label == CO_COMPONENT_LABEL
    assert np.array_equal(result.co_levels, np.arange(0.0, 201.0, 25.0))
    assert result.x_limits == pytest.approx((0.7, 1.3))
    assert result.y_limits == pytest.approx((0.6, 1.0))
    assert result.show_right_line is True
    assert result.show_mean_line is False
    assert result.x_axis_label == "Fuel flow"
    assert result.y_axis_label == "Steam flow"
    assert result.colorbar_label == "CO concentration"
    assert result.cmap_name == "plasma"
    assert result.maxima_detection_method is MaximaDetectionMethod.CONTOUR_LEVELS
    assert result.analysis_contour_indices == (3, 4)
    assert result.analysis_contour_values == pytest.approx((0.25, 0.5))
    assert result.font_size == 16


def test_process_job_uses_default_cmap_when_value_is_blank(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_success_surface_csv(input_file)
    pipeline = RegimeMapPipeline()

    result = pipeline.process_job(
        RegimeMapJobConfig(
            input_path=input_file,
            cmap_name="   ",
        )
    )

    assert result.cmap_name == DEFAULT_CMAP_NAME
