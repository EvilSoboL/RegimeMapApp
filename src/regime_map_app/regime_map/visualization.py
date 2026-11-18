from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from .cmaps import DEFAULT_CMAP_NAME, resolve_cmap_name
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
    cmap_name = resolve_cmap_name(result.cmap_name) or DEFAULT_CMAP_NAME
    cmap = colormaps[cmap_name].copy()
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

    overlays = (
        (result.show_min_line, result.minima_points, "#d62728", "--", "Линия минимальной концентрации"),
        (result.show_right_line, result.right_points, "white", ":", "Правая линия максимумов"),
        (result.show_mean_line, result.mean_points, "#ff7f0e", "-.", "Средняя линия"),
    )
    for visible, points, color, linestyle, label in overlays:
        valid = np.isfinite(points[:, 0])
        if visible and valid.any():
            previous = np.r_[False, valid[:-1]]
            following = np.r_[valid[1:], False]
            isolated = np.flatnonzero(valid & ~previous & ~following).tolist()
            axis.plot(
                *points.T, color=color, linestyle=linestyle, linewidth=2.5, label=label,
                marker=".", markevery=isolated,
            )

    axis.set_xlabel(result.x_axis_label, fontfamily=result.font_family, fontsize=result.font_size)
    axis.set_ylabel(result.y_axis_label, fontfamily=result.font_family, fontsize=result.font_size)
    axis.set_xlim(*(result.x_limits or (result.fuel_axis[0], result.fuel_axis[-1])))
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


def _draw_if_possible(figure: Figure) -> None:
    if figure.canvas is not None:
        figure.canvas.draw_idle()


def _apply_axis_tick_font(axis, font_family: str, font_size: int) -> None:
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(font_family)
        tick_label.set_fontsize(font_size)
