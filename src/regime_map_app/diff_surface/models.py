from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from numpy.typing import NDArray

CSV_SEPARATOR = ";"
REQUIRED_COLUMNS = ("fuel", "additive", "component")
DEFAULT_ANALYSIS_CONTOUR_LEVELS = "3"
DEFAULT_CONTOUR_LEVEL_COUNT = 10


class SurfaceMode(str, Enum):
    GRADIENT_MAGNITUDE = "grad"

    @property
    def label(self) -> str:
        return "|grad|"


class MaximaDetectionMethod(str, Enum):
    ROW_PEAKS = "row_peaks"
    CONTOUR_LEVELS = "contour_levels"

    @property
    def label(self) -> str:
        if self is MaximaDetectionMethod.ROW_PEAKS:
            return "По локальным пикам строк"
        return "По линиям уровня"

    @property
    def uses_contour_levels(self) -> bool:
        return self is MaximaDetectionMethod.CONTOUR_LEVELS


@dataclass(frozen=True)
class DiffSurfaceJobConfig:
    input_path: Path | None
    surface_mode: SurfaceMode = SurfaceMode.GRADIENT_MAGNITUDE
    maxima_detection_method: MaximaDetectionMethod = MaximaDetectionMethod.ROW_PEAKS
    contour_levels_text: str = DEFAULT_ANALYSIS_CONTOUR_LEVELS


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
    minima_points: NDArray
    minima_line_fit: LineFit
    left_maxima_points: NDArray
    right_maxima_points: NDArray
    left_line_fit: LineFit
    right_line_fit: LineFit
    maxima_detection_method: MaximaDetectionMethod = MaximaDetectionMethod.ROW_PEAKS
    analysis_contour_indices: tuple[int, ...] = ()
    analysis_contour_values: tuple[float, ...] = ()
