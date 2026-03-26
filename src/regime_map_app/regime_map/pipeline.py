from __future__ import annotations

from typing import Callable

import numpy as np

from ..diff_surface.exceptions import CancellationError as DiffSurfaceCancellationError
from ..diff_surface.exceptions import DiffSurfaceError
from ..diff_surface.models import DiffSurfaceJobConfig, LineFit, SurfaceMode
from ..diff_surface.pipeline import DiffSurfacePipeline
from .exceptions import CancellationError, ProcessingError, ValidationError
from .models import CO_LEVELS, RegimeMapJobConfig, RegimeMapResult, ValidationResult
from .validation import validate_job_config

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]


class RegimeMapPipeline:
    def __init__(self, diff_pipeline: DiffSurfacePipeline | None = None) -> None:
        self.diff_pipeline = diff_pipeline or DiffSurfacePipeline()

    def validate_inputs(self, config: RegimeMapJobConfig) -> ValidationResult:
        base_validation = validate_job_config(config)
        if not base_validation.is_valid:
            return base_validation

        diff_validation = self.diff_pipeline.validate_inputs(self._to_diff_config(config))
        return ValidationResult(
            is_valid=diff_validation.is_valid,
            errors=tuple(diff_validation.errors),
            checked_points=diff_validation.checked_points,
        )

    def process_job(
        self,
        config: RegimeMapJobConfig,
        *,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> RegimeMapResult:
        validation = validate_job_config(config)
        if not validation.is_valid:
            raise ValidationError("\n".join(validation.errors))

        try:
            diff_result = self.diff_pipeline.process_job(
                self._to_diff_config(config),
                on_log=on_log,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
        except DiffSurfaceCancellationError as exc:
            raise CancellationError(str(exc)) from exc
        except DiffSurfaceError as exc:
            raise ProcessingError(str(exc)) from exc

        mean_line_fit = self.compute_mean_line(diff_result.minima_line_fit, diff_result.right_line_fit)
        self._emit_log(
            on_log,
            f"Средняя линия: additive = {mean_line_fit.slope:.6g} * fuel + {mean_line_fit.intercept:.6g}",
        )

        return RegimeMapResult(
            input_path=diff_result.input_path,
            fuel_axis=np.asarray(diff_result.fuel_axis, dtype=float),
            additive_axis=np.asarray(diff_result.additive_axis, dtype=float),
            component_grid=np.asarray(diff_result.component_grid, dtype=float),
            co_levels=np.asarray(CO_LEVELS, dtype=float),
            minima_line_fit=diff_result.minima_line_fit,
            right_line_fit=diff_result.right_line_fit,
            mean_line_fit=mean_line_fit,
        )

    def compute_mean_line(self, minima_line_fit: LineFit, right_line_fit: LineFit) -> LineFit:
        slope = (minima_line_fit.slope + right_line_fit.slope) / 2.0
        intercept = (minima_line_fit.intercept + right_line_fit.intercept) / 2.0
        if not np.isfinite((slope, intercept)).all():
            raise ProcessingError("Не удалось вычислить среднюю линию между линией минимума и правой линией максимумов.")
        return LineFit(slope=float(slope), intercept=float(intercept))

    def _to_diff_config(self, config: RegimeMapJobConfig) -> DiffSurfaceJobConfig:
        return DiffSurfaceJobConfig(
            input_path=config.input_path,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        )

    def _emit_log(self, callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)
