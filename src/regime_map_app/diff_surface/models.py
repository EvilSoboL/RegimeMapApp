from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from numpy.typing import NDArray

CSV_SEPARATOR = ";"
REQUIRED_COLUMNS = ("fuel", "additive", "component")


class SurfaceMode(str, Enum):
    DZ_DX = "dz_dx"
    DZ_DY = "dz_dy"
    GRADIENT_MAGNITUDE = "grad"

    @property
    def label(self) -> str:
        if self is SurfaceMode.DZ_DX:
            return "dz/dx"
        if self is SurfaceMode.DZ_DY:
            return "dz/dy"
        return "|grad|"


class SplitMethod(str, Enum):
    FUEL_MIDPOINT = "fuel_midpoint"

    @property
    def label(self) -> str:
        return "по середине диапазона fuel"


@dataclass(frozen=True)
class DiffSurfaceJobConfig:
    input_path: Path | None
    surface_mode: SurfaceMode = SurfaceMode.DZ_DX
    split_method: SplitMethod = SplitMethod.FUEL_MIDPOINT


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
    split_method: SplitMethod
    fuel_axis: NDArray
    additive_axis: NDArray
    component_grid: NDArray
    dz_dx: NDArray
    dz_dy: NDArray
    selected_surface: NDArray
    maxima_points: NDArray
    line_1_points: NDArray
    line_2_points: NDArray
    line_1_fit: LineFit
    line_2_fit: LineFit
