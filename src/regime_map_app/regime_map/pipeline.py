from __future__ import annotations

from typing import Callable

import numpy as np

from ..diff_surface.exceptions import CancellationError as DiffSurfaceCancellationError
from ..diff_surface.exceptions import DiffSurfaceError
from ..diff_surface.models import DiffSurfaceJobConfig, LineFit, SurfaceMode
from ..diff_surface.pipeline import DiffSurfacePipeline
from .exceptions import CancellationError, ProcessingError, ValidationError
from .models import (
    AUTO_CONTOUR_LEVELS,
    CO_COMPONENT_LABEL,
    GENERIC_COMPONENT_LABEL,
    RegimeMapJobConfig,
    RegimeMapResult,
    ValidationResult,
)
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
        component_label = CO_COMPONENT_LABEL if config.is_co_component else GENERIC_COMPONENT_LABEL
        show_right_line = config.is_co_component and config.show_right_line
        x_limits = (float(config.x_min), float(config.x_max)) if config.use_custom_x_limits else None
        y_limits = (float(config.y_min), float(config.y_max)) if config.use_custom_y_limits else None
        co_levels = self.resolve_co_levels(config)

        if config.show_right_line and not config.is_co_component:
            self._emit_log(on_log, "Правая линия максимумов скрыта, потому что флажок CO снят.")

        self._emit_log(
            on_log,
            f"Средняя линия: additive = {mean_line_fit.slope:.6g} * fuel + {mean_line_fit.intercept:.6g}",
        )

        return RegimeMapResult(
            input_path=diff_result.input_path,
            fuel_axis=np.asarray(diff_result.fuel_axis, dtype=float),
            additive_axis=np.asarray(diff_result.additive_axis, dtype=float),
            component_grid=np.asarray(diff_result.component_grid, dtype=float),
            component_label=component_label,
            co_levels=co_levels,
            x_limits=x_limits,
            y_limits=y_limits,
            is_co_component=config.is_co_component,
            show_min_line=config.show_min_line,
            show_right_line=show_right_line,
            show_mean_line=config.show_mean_line,
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

    def resolve_co_levels(self, config: RegimeMapJobConfig) -> int | np.ndarray:
        if not config.use_custom_ppm_scale:
            return AUTO_CONTOUR_LEVELS

        values = np.arange(config.ppm_min, config.ppm_max, config.ppm_step, dtype=float)
        values = np.append(values, float(config.ppm_max))
        unique_values = np.unique(values)
        if unique_values.size < 2:
            raise ProcessingError("Не удалось сформировать пользовательскую шкалу ppm.")
        return unique_values

    def _to_diff_config(self, config: RegimeMapJobConfig) -> DiffSurfaceJobConfig:
        return DiffSurfaceJobConfig(
            input_path=config.input_path,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
        )

    def _emit_log(self, callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)
