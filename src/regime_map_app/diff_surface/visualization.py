from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .exceptions import SaveError
from .models import DifferentialSurfaceResult, LineFit


def create_figure() -> Figure:
    return Figure(figsize=(6.8, 5.6))


def render_placeholder(figure: Figure, message: str) -> None:
    figure.clear()
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    _draw_if_possible(figure)


def render_result(figure: Figure, result: DifferentialSurfaceResult) -> None:
    figure.clear()
    axis = figure.add_subplot(111)

    fuel_grid, additive_grid = np.meshgrid(result.fuel_axis, result.additive_axis)
    levels = _build_levels(result.selected_surface)
    contour = axis.contourf(fuel_grid, additive_grid, result.selected_surface, levels=levels, cmap="viridis")
    figure.colorbar(contour, ax=axis, label=result.surface_mode.label)

    if result.analysis_contour_values:
        axis.contour(
            fuel_grid,
            additive_grid,
            result.selected_surface,
            levels=np.asarray(result.analysis_contour_values, dtype=float),
            colors="white",
            linestyles=":",
            linewidths=1.3,
        )

    axis.scatter(
        result.left_maxima_points[:, 0],
        result.left_maxima_points[:, 1],
        s=26,
        c="white",
        edgecolors="#d62728",
        linewidths=0.9,
        label="Точки максимумов слева",
    )
    axis.scatter(
        result.right_maxima_points[:, 0],
        result.right_maxima_points[:, 1],
        s=26,
        c="white",
        edgecolors="#1f77b4",
        linewidths=0.9,
        label="Точки максимумов справа",
    )

    _plot_clipped_line(
        axis,
        result.fuel_axis,
        result.additive_axis,
        result.minima_line_fit,
        color="#ff7f0e",
        linestyle="--",
        label=_line_label("Линия минимальной концентрации", result.minima_line_fit.slope, result.minima_line_fit.intercept),
    )
    _plot_clipped_line(
        axis,
        result.fuel_axis,
        result.additive_axis,
        result.left_line_fit,
        color="#d62728",
        linestyle="-",
        label=_line_label("Левая линия максимумов", result.left_line_fit.slope, result.left_line_fit.intercept),
    )
    _plot_clipped_line(
        axis,
        result.fuel_axis,
        result.additive_axis,
        result.right_line_fit,
        color="#1f77b4",
        linestyle="-",
        label=_line_label("Правая линия максимумов", result.right_line_fit.slope, result.right_line_fit.intercept),
    )

    axis.set_xlabel("fuel")
    axis.set_ylabel("additive")
    axis.set_xlim(float(result.fuel_axis.min()), float(result.fuel_axis.max()))
    axis.set_ylim(float(result.additive_axis.min()), float(result.additive_axis.max()))
    axis.set_title(f"Дифференциальная поверхность: {result.surface_mode.label}")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best")
    figure.tight_layout()
    _draw_if_possible(figure)


def save_plot(result: DifferentialSurfaceResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    render_result(figure, result)
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    except Exception as exc:
        raise SaveError(f"Не удалось сохранить график в {output_path}: {exc}") from exc


def _build_levels(surface: np.ndarray) -> int | np.ndarray:
    surface_min = float(np.min(surface))
    surface_max = float(np.max(surface))
    if np.isclose(surface_min, surface_max):
        padding = max(abs(surface_min) * 0.05, 1e-9)
        return np.linspace(surface_min - padding, surface_max + padding, 3)
    return 20


def _line_label(prefix: str, slope: float, intercept: float) -> str:
    return f"{prefix}: y = {slope:.3g}x + {intercept:.3g}"


def _plot_clipped_line(
    axis,
    fuel_axis: np.ndarray,
    additive_axis: np.ndarray,
    line_fit: LineFit,
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    clipped_segment = _clip_line_to_surface_bounds(
        fuel_axis,
        additive_axis,
        line_fit.slope,
        line_fit.intercept,
    )
    if clipped_segment is None:
        return

    line_x, line_y = clipped_segment
    axis.plot(
        line_x,
        line_y,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )


def _clip_line_to_surface_bounds(
    fuel_axis: np.ndarray,
    additive_axis: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_min = float(np.min(fuel_axis))
    x_max = float(np.max(fuel_axis))
    y_min = float(np.min(additive_axis))
    y_max = float(np.max(additive_axis))
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
