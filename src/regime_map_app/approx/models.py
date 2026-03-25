from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

from numpy.typing import NDArray

CSV_SEPARATOR = ";;"
REQUIRED_COLUMNS = ("fuel", "additive", "component")
DEFAULT_RESOLUTION_X = 100
DEFAULT_RESOLUTION_Y = 100
DEFAULT_MEDIAN_SIZE = 20
DEFAULT_KERNEL = "linear"
ALLOWED_KERNELS = (
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
)


class InputMode(str, Enum):
    SINGLE_FILE = "single_file"
    FOLDER_BATCH = "folder_batch"

    @property
    def is_single(self) -> bool:
        return self is InputMode.SINGLE_FILE


@dataclass(frozen=True)
class FileMetadata:
    fuel_name: str
    diluent: str
    component_name: str
    measured_on: date
    original_name: str


@dataclass(frozen=True)
class SurfaceGrid:
    fuel_grid: NDArray
    additive_grid: NDArray
    component_grid: NDArray


@dataclass(frozen=True)
class ApproxJobConfig:
    input_mode: InputMode
    input_paths: tuple[Path, ...]
    output_dir: Path | None
    output_filename: str | None = None
    resolution_x: int = DEFAULT_RESOLUTION_X
    resolution_y: int = DEFAULT_RESOLUTION_Y
    kernel: str = DEFAULT_KERNEL
    median_size: int = DEFAULT_MEDIAN_SIZE
    clamp_zero: bool = True
    auto_output_name: bool = True

    @property
    def resolution(self) -> tuple[int, int]:
        return self.resolution_x, self.resolution_y


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_files: int = 0


@dataclass(frozen=True)
class FileProcessResult:
    input_path: Path
    success: bool
    messages: tuple[str, ...]
    output_path: Path | None = None
    output_rows: int = 0
    metadata: FileMetadata | None = None


@dataclass(frozen=True)
class BatchProcessSummary:
    total_files: int
    succeeded: int
    failed: int
    output_dir: Path
    results: tuple[FileProcessResult, ...]

    @property
    def successful_results(self) -> tuple[FileProcessResult, ...]:
        return tuple(result for result in self.results if result.success)

    @property
    def last_output_path(self) -> Path | None:
        for result in reversed(self.results):
            if result.output_path is not None:
                return result.output_path
        return None
