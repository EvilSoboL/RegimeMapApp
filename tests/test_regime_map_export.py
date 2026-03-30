from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

regime_models = import_module("regime_map_app.regime_map.models")
regime_validation = import_module("regime_map_app.regime_map.validation")
regime_visualization = import_module("regime_map_app.regime_map.visualization")
diff_models = import_module("regime_map_app.diff_surface.models")

CO_COMPONENT_LABEL = regime_models.CO_COMPONENT_LABEL
DEFAULT_FONT_FAMILY = regime_models.DEFAULT_FONT_FAMILY
GENERIC_COMPONENT_LABEL = regime_models.GENERIC_COMPONENT_LABEL
RegimeMapResult = regime_models.RegimeMapResult
LineFit = diff_models.LineFit
create_figure = regime_visualization.create_figure
render_result = regime_visualization.render_result
resolve_export_path = regime_validation.resolve_export_path
save_plot = regime_visualization.save_plot


def _build_custom_result(input_path: Path) -> RegimeMapResult:
    return RegimeMapResult(
        input_path=input_path,
        fuel_axis=np.array([0.5, 1.0, 1.5]),
        additive_axis=np.array([0.4, 0.8, 1.2]),
        component_grid=np.array(
            [
                [20.0, 40.0, 60.0],
                [50.0, 80.0, 120.0],
                [90.0, 140.0, 220.0],
            ]
        ),
        component_label=CO_COMPONENT_LABEL,
        co_levels=np.arange(0.0, 201.0, 25.0),
        x_limits=(0.7, 1.3),
        y_limits=(0.6, 1.0),
        is_co_component=True,
        show_min_line=True,
        show_right_line=False,
        show_mean_line=True,
        minima_line_fit=LineFit(slope=1.0, intercept=0.0),
        right_line_fit=LineFit(slope=-0.5, intercept=1.25),
        mean_line_fit=LineFit(slope=0.25, intercept=0.625),
    )


def _build_auto_result(input_path: Path) -> RegimeMapResult:
    return RegimeMapResult(
        input_path=input_path,
        fuel_axis=np.array([0.5, 1.0, 1.5]),
        additive_axis=np.array([0.4, 0.8, 1.2]),
        component_grid=np.array(
            [
                [20.0, 40.0, 60.0],
                [50.0, 80.0, 120.0],
                [90.0, 140.0, 220.0],
            ]
        ),
        component_label=GENERIC_COMPONENT_LABEL,
        co_levels=10,
        x_limits=None,
        y_limits=None,
        is_co_component=False,
        show_min_line=False,
        show_right_line=True,
        show_mean_line=False,
        minima_line_fit=LineFit(slope=1.0, intercept=0.0),
        right_line_fit=LineFit(slope=-0.5, intercept=1.25),
        mean_line_fit=LineFit(slope=0.25, intercept=0.625),
    )


def test_export_creates_png_only(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel;additive;component\n0;0;0\n1;0;1\n0;1;1\n1;1;0\n", encoding="utf-8")
    png_path = resolve_export_path(tmp_path / "out", input_file)

    save_plot(_build_custom_result(input_file), png_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert png_path.name == "regime_map_surface.png"


def test_render_result_draws_only_requested_lines_and_clips_to_custom_bounds() -> None:
    figure = create_figure()
    render_result(figure, _build_custom_result(Path("surface.csv")))

    main_axis = figure.axes[0]
    colorbar_axis = figure.axes[1]
    minima_line = main_axis.lines[0]
    mean_line = main_axis.lines[1]
    legend = main_axis.get_legend()

    assert len(main_axis.lines) == 2
    assert np.allclose(minima_line.get_xdata(), np.array([0.7, 1.0]))
    assert np.allclose(minima_line.get_ydata(), np.array([0.7, 1.0]))
    assert np.allclose(mean_line.get_xdata(), np.array([0.7, 1.3]))
    assert np.allclose(mean_line.get_ydata(), np.array([0.8, 0.95]))
    assert main_axis.get_xlim() == pytest.approx((0.7, 1.3))
    assert main_axis.get_ylim() == pytest.approx((0.6, 1.0))
    assert main_axis.get_xlabel() == "Расход топлива, кг/ч"
    assert main_axis.get_ylabel() == "Расход пара, кг/ч"
    assert colorbar_axis.get_ylabel() == "CO, ppm"
    assert main_axis.xaxis.label.get_fontfamily() == [DEFAULT_FONT_FAMILY]
    assert main_axis.yaxis.label.get_fontfamily() == [DEFAULT_FONT_FAMILY]
    assert main_axis.title.get_fontfamily() == [DEFAULT_FONT_FAMILY]
    assert colorbar_axis.yaxis.label.get_fontfamily() == [DEFAULT_FONT_FAMILY]
    assert legend is not None
    bbox = legend.get_bbox_to_anchor().transformed(main_axis.transAxes.inverted())
    assert bbox.y0 >= 1.0


def test_render_result_keeps_automatic_bounds_when_custom_limits_are_disabled() -> None:
    figure = create_figure()
    render_result(figure, _build_auto_result(Path("surface.csv")))

    main_axis = figure.axes[0]
    right_line = main_axis.lines[0]

    assert len(main_axis.lines) == 1
    assert np.allclose(right_line.get_xdata(), np.array([0.5, 1.5]))
    assert np.allclose(right_line.get_ydata(), np.array([1.0, 0.5]))
    assert main_axis.get_xlim() == pytest.approx((0.5, 1.5))
    assert main_axis.get_ylim() == pytest.approx((0.4, 1.2))
