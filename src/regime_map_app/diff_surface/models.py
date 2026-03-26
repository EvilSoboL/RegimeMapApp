from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from numpy.typing import NDArray

CSV_SEPARATOR = ";"
REQUIRED_COLUMNS = ("fuel", "additive", "component")


class SurfaceMode(str, Enum):
    GRADIENT_MAGNITUDE = "grad"

    @property
    def label(self) -> str:
        return "|grad|"


@dataclass(frozen=True)
class DiffSurfaceJobConfig:
    input_path: Path | None
    surface_mode: SurfaceMode = SurfaceMode.GRADIENT_MAGNITUDE


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = ()
    checked_points: int = 0


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept: float


@dataclass(frozen=True)
class DifferentialSurfaceResult:
    input_path: Path
    surface_mode: SurfaceMode
    fuel_axis: NDArray
    additive_axis: NDArray
    component_grid: NDArray
    dz_dx: NDArray
    dz_dy: NDArray
    selected_surface: NDArray
    left_maxima_points: NDArray
    right_maxima_points: NDArray
    left_line_fit: LineFit
    right_line_fit: LineFit
