from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .exceptions import SaveError
from .models import DifferentialSurfaceResult


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

    axis.scatter(
        result.maxima_points[:, 0],
        result.maxima_points[:, 1],
        s=28,
        c="white",
        edgecolors="black",
        linewidths=0.6,
        label="Точки максимумов",
    )

    line_x = np.linspace(float(result.fuel_axis.min()), float(result.fuel_axis.max()), 200)
    axis.plot(
        line_x,
        result.line_1_fit.slope * line_x + result.line_1_fit.intercept,
        color="#d62728",
        linewidth=2.0,
        label=_line_label("Линия 1", result.line_1_fit.slope, result.line_1_fit.intercept),
    )
    axis.plot(
        line_x,
        result.line_2_fit.slope * line_x + result.line_2_fit.intercept,
        color="#1f77b4",
        linewidth=2.0,
        label=_line_label("Линия 2", result.line_2_fit.slope, result.line_2_fit.intercept),
    )

    axis.set_xlabel("fuel")
    axis.set_ylabel("additive")
    axis.set_title(f"Дифференциальная поверхность: {result.surface_mode.label}")
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


def _draw_if_possible(figure: Figure) -> None:
    if figure.canvas is not None:
        figure.canvas.draw_idle()
