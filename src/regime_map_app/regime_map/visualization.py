from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from .exceptions import SaveError
from .models import RegimeMapResult

BASE_FIGURE_WIDTH = 6.8
BASE_FIGURE_HEIGHT = 5.6
HEIGHT_SCALE_FACTOR = 4 / 3


def create_figure() -> Figure:
    return Figure(figsize=(BASE_FIGURE_WIDTH, BASE_FIGURE_HEIGHT * HEIGHT_SCALE_FACTOR))


def render_placeholder(figure: Figure, message: str) -> None:
    figure.clear()
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Times New Roman",
    )
    _draw_if_possible(figure)


def render_result(figure: Figure, result: RegimeMapResult) -> None:
    figure.clear()
    axis = figure.add_subplot(111)

    fuel_grid, additive_grid = np.meshgrid(result.fuel_axis, result.additive_axis)
    cmap = colormaps["viridis"].copy()
    cmap.set_over("darkred")

    contour_kwargs: dict[str, object] = {
        "cmap": cmap,
    }
    if isinstance(result.co_levels, int):
        contour_kwargs["levels"] = result.co_levels
        contour_extend = "neither"
    else:
        contour_kwargs["levels"] = result.co_levels
        contour_kwargs["extend"] = "max"
        contour_kwargs["vmin"] = float(result.co_levels[0])
        contour_kwargs["vmax"] = float(result.co_levels[-1])
        contour_extend = "max"

    contour = axis.contourf(
        fuel_grid,
        additive_grid,
        result.component_grid,
        **contour_kwargs,
    )
    colorbar = figure.colorbar(contour, ax=axis, extend=contour_extend)
    colorbar.set_label(
        result.colorbar_label,
        fontfamily=result.font_family,
        fontsize=result.font_size,
    )
    if not isinstance(result.co_levels, int):
        colorbar.set_ticks(result.co_levels)
    _apply_axis_tick_font(colorbar.ax, result.font_family, result.font_size)

    plot_x_limits = result.x_limits or (float(np.min(result.fuel_axis)), float(np.max(result.fuel_axis)))
    plot_y_limits = result.y_limits or (float(np.min(result.additive_axis)), float(np.max(result.additive_axis)))

    if result.show_min_line:
        _plot_clipped_line(
            axis,
            result.minima_line_fit.slope,
            result.minima_line_fit.intercept,
            plot_x_limits,
            plot_y_limits,
            color="#d62728",
            linestyle="--",
            label=_line_label("Линия минимальной концентрации"),
        )
    if result.show_right_line:
        _plot_clipped_line(
            axis,
            result.right_line_fit.slope,
            result.right_line_fit.intercept,
            plot_x_limits,
            plot_y_limits,
            color="white",
            linestyle=":",
            label=_line_label("Правая линия максимумов"),
        )
    if result.show_mean_line:
        _plot_clipped_line(
            axis,
            result.mean_line_fit.slope,
            result.mean_line_fit.intercept,
            plot_x_limits,
            plot_y_limits,
            color="#ff7f0e",
            linestyle="-.",
            label=_line_label("Средняя линия"),
        )

    axis.set_xlabel(result.x_axis_label, fontfamily=result.font_family, fontsize=result.font_size)
    axis.set_ylabel(result.y_axis_label, fontfamily=result.font_family, fontsize=result.font_size)
    if result.x_limits is not None:
        axis.set_xlim(*result.x_limits)
    if result.y_limits is not None:
        axis.set_ylim(*result.y_limits)
    axis.set_title(result.input_path.name, fontfamily=result.font_family, fontsize=result.font_size)
    _apply_axis_tick_font(axis, result.font_family, result.font_size)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.14, 1.0, 0.3),
            borderaxespad=0.0,
            ncol=1,
            prop={"family": result.font_family, "size": result.font_size},
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.75))
    _draw_if_possible(figure)


def save_plot(result: RegimeMapResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    render_result(figure, result)
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    except Exception as exc:
        raise SaveError(f"Не удалось сохранить график в {output_path}: {exc}") from exc


def _line_label(prefix: str) -> str:
    return prefix


def _plot_clipped_line(
    axis,
    slope: float,
    intercept: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    clipped_segment = _clip_line_to_bounds(slope, intercept, x_limits, y_limits)
    if clipped_segment is None:
        return

    line_x, line_y = clipped_segment
    axis.plot(
        line_x,
        line_y,
        color=color,
        linestyle=linestyle,
        linewidth=2.5,
        label=label,
    )


def _clip_line_to_bounds(
    slope: float,
    intercept: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    tolerance = 1e-9

    if np.isclose(slope, 0.0):
        y_value = float(intercept)
        if y_value < y_min - tolerance or y_value > y_max + tolerance:
            return None
        clipped_y = float(np.clip(y_value, y_min, y_max))
        return np.array([x_min, x_max], dtype=float), np.array([clipped_y, clipped_y], dtype=float)

    candidates: list[tuple[float, float]] = []

    for x_value in (x_min, x_max):
        y_value = slope * x_value + intercept
        if y_min - tolerance <= y_value <= y_max + tolerance:
            candidates.append((x_value, float(np.clip(y_value, y_min, y_max))))

    for y_value in (y_min, y_max):
        x_value = (y_value - intercept) / slope
        if x_min - tolerance <= x_value <= x_max + tolerance:
            candidates.append((float(np.clip(x_value, x_min, x_max)), y_value))

    unique_points: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.allclose(point, saved_point, atol=tolerance, rtol=0.0) for saved_point in unique_points):
            unique_points.append(point)

    if len(unique_points) < 2:
        return None

    start_point, end_point = max(
        combinations(unique_points, 2),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    if start_point[0] > end_point[0] or (np.isclose(start_point[0], end_point[0]) and start_point[1] > end_point[1]):
        start_point, end_point = end_point, start_point

    return (
        np.array([start_point[0], end_point[0]], dtype=float),
        np.array([start_point[1], end_point[1]], dtype=float),
    )


def _draw_if_possible(figure: Figure) -> None:
    if figure.canvas is not None:
        figure.canvas.draw_idle()


def _apply_axis_tick_font(axis, font_family: str, font_size: int) -> None:
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(font_family)
        tick_label.set_fontsize(font_size)
