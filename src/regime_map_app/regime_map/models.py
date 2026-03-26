from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..diff_surface.models import LineFit

CO_LEVELS = np.arange(0.0, 201.0, 25.0, dtype=float)
REGIME_MAP_X_LIMITS = (0.7, 1.3)
REGIME_MAP_Y_LIMITS = (0.6, 1.0)
REGIME_MAP_COMPONENT_LABEL = "CO, ppm"


@dataclass(frozen=True)
class RegimeMapJobConfig:
    input_path: Path | None


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
    co_levels: NDArray
    minima_line_fit: LineFit
    right_line_fit: LineFit
    mean_line_fit: LineFit
