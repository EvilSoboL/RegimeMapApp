from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..diff_surface.models import DEFAULT_ANALYSIS_CONTOUR_LEVELS, LineFit, MaximaDetectionMethod
from .cmaps import DEFAULT_CMAP_NAME

DEFAULT_CO_LEVELS = np.arange(0.0, 201.0, 25.0, dtype=float)
DEFAULT_X_LIMITS = (0.7, 1.3)
DEFAULT_Y_LIMITS = (0.6, 1.0)
AUTO_CONTOUR_LEVELS = 10
CO_COMPONENT_LABEL = "CO, ppm"
GENERIC_COMPONENT_LABEL = "component"
DEFAULT_X_AXIS_LABEL = "Расход топлива, кг/ч"
DEFAULT_Y_AXIS_LABEL = "Расход пара, кг/ч"
DEFAULT_FONT_FAMILY = "Times New Roman"
DEFAULT_FONT_SIZE = 12


@dataclass(frozen=True)
class RegimeMapJobConfig:
    input_path: Path | None
    is_co_component: bool = True
    show_min_line: bool = False
    show_right_line: bool = False
    show_mean_line: bool = False
    use_custom_x_limits: bool = False
    x_min: float = DEFAULT_X_LIMITS[0]
    x_max: float = DEFAULT_X_LIMITS[1]
    use_custom_y_limits: bool = False
    y_min: float = DEFAULT_Y_LIMITS[0]
    y_max: float = DEFAULT_Y_LIMITS[1]
    use_custom_ppm_scale: bool = False
    ppm_min: float = DEFAULT_CO_LEVELS[0]
    ppm_max: float = DEFAULT_CO_LEVELS[-1]
    ppm_step: float = float(DEFAULT_CO_LEVELS[1] - DEFAULT_CO_LEVELS[0])
    x_axis_label: str = DEFAULT_X_AXIS_LABEL
    y_axis_label: str = DEFAULT_Y_AXIS_LABEL
    colorbar_label: str = CO_COMPONENT_LABEL
    cmap_name: str = DEFAULT_CMAP_NAME
    maxima_detection_method: MaximaDetectionMethod = MaximaDetectionMethod.ROW_PEAKS
    contour_levels_text: str = DEFAULT_ANALYSIS_CONTOUR_LEVELS
    font_size: int = DEFAULT_FONT_SIZE


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = ()
    checked_points: int = 0


@dataclass(frozen=True)
class RegimeMapResult:
    input_path: Path
    fuel_axis: NDArray
    additive_axis: NDArray
    component_grid: NDArray
    component_label: str
    co_levels: int | NDArray
    x_limits: tuple[float, float] | None
    y_limits: tuple[float, float] | None
    is_co_component: bool
    show_min_line: bool
    show_right_line: bool
    show_mean_line: bool
    minima_line_fit: LineFit
    right_line_fit: LineFit
    mean_line_fit: LineFit
    x_axis_label: str = DEFAULT_X_AXIS_LABEL
    y_axis_label: str = DEFAULT_Y_AXIS_LABEL
    colorbar_label: str = CO_COMPONENT_LABEL
    cmap_name: str = DEFAULT_CMAP_NAME
    maxima_detection_method: MaximaDetectionMethod = MaximaDetectionMethod.ROW_PEAKS
    analysis_contour_indices: tuple[int, ...] = ()
    analysis_contour_values: tuple[float, ...] = ()
    font_family: str = DEFAULT_FONT_FAMILY
    font_size: int = DEFAULT_FONT_SIZE
