from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..diff_surface.models import LineFit

DEFAULT_CO_LEVELS = np.arange(0.0, 201.0, 25.0, dtype=float)
DEFAULT_X_LIMITS = (0.7, 1.3)
DEFAULT_Y_LIMITS = (0.6, 1.0)
AUTO_CONTOUR_LEVELS = 10
CO_COMPONENT_LABEL = "CO, ppm"
GENERIC_COMPONENT_LABEL = "component"


@dataclass(frozen=True)
class RegimeMapJobConfig:
    input_path: Path | None
    is_co_component: bool = True
    show_min_line: bool = True
    show_right_line: bool = True
    show_mean_line: bool = True
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
