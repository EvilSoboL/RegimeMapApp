from __future__ import annotations

from typing import Callable

import numpy as np

from ..diff_surface.exceptions import CancellationError as DiffSurfaceCancellationError
from ..diff_surface.exceptions import DiffSurfaceError
from ..diff_surface.models import DiffSurfaceJobConfig, MaximaDetectionMethod, SurfaceMode
from ..diff_surface.pipeline import DiffSurfacePipeline
from ..diff_surface.validation import parse_contour_level_indices
from .cmaps import DEFAULT_CMAP_NAME, resolve_cmap_name
from .exceptions import CancellationError, ProcessingError, ValidationError
from .models import (
    AUTO_CONTOUR_LEVELS,
    CO_COMPONENT_LABEL,
    DEFAULT_FONT_FAMILY,
    DEFAULT_X_AXIS_LABEL,
    DEFAULT_Y_AXIS_LABEL,
    GENERIC_COMPONENT_LABEL,
    RegimeMapJobConfig,
    RegimeMapResult,
    ValidationResult,
)
from .validation import validate_job_config

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]
CO_MINIMUM_TOLERANCE = 10.0


class RegimeMapPipeline:
    def __init__(self, diff_pipeline: DiffSurfacePipeline | None = None) -> None:
        self.diff_pipeline = diff_pipeline or DiffSurfacePipeline()

    def validate_inputs(self, config: RegimeMapJobConfig) -> ValidationResult:
        base_validation = validate_job_config(config)
        if not base_validation.is_valid:
            return base_validation

        diff_validation = self.diff_pipeline.validate_inputs(
            self._to_diff_config(config, include_analysis=self._needs_right_line_analysis(config))
        )
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

        default_component_label = CO_COMPONENT_LABEL if config.is_co_component else GENERIC_COMPONENT_LABEL
        colorbar_label = config.colorbar_label.strip() or default_component_label
        show_right_line = config.is_co_component and config.show_right_line
        show_mean_line = config.show_mean_line
        needs_right_line = self._needs_right_line_analysis(config)
        needs_min_line = config.show_min_line or needs_right_line
        x_limits = (float(config.x_min), float(config.x_max)) if config.use_custom_x_limits else None
        y_limits = (float(config.y_min), float(config.y_max)) if config.use_custom_y_limits else None
        co_levels = self.resolve_co_levels(config)
        cmap_name = resolve_cmap_name(config.cmap_name) or DEFAULT_CMAP_NAME
        x_axis_label = config.x_axis_label.strip() or DEFAULT_X_AXIS_LABEL
        y_axis_label = config.y_axis_label.strip() or DEFAULT_Y_AXIS_LABEL
        minima_points = right_points = mean_points = np.empty((0, 2), dtype=float)
        maxima_detection_method = MaximaDetectionMethod.ROW_PEAKS
        analysis_contour_indices: tuple[int, ...] = ()
        analysis_contour_values: tuple[float, ...] = ()

        if config.show_right_line and not config.is_co_component:
            self._emit_log(on_log, "Правая линия максимумов скрыта, потому что флаг CO снят.")

        try:
            input_path, fuel_axis, additive_axis, component_grid = self._build_base_map(
                config, on_progress=on_progress, should_cancel=should_cancel,
            )
            if needs_min_line:
                tolerance = CO_MINIMUM_TOLERANCE if config.is_co_component else 0.0
                valley_fuel = self.find_minimum_path(
                    component_grid, fuel_axis, tolerance, should_cancel,
                )
                fit = self.diff_pipeline.fit_line(
                    np.column_stack((additive_axis, valley_fuel)), "линии минимальной концентрации",
                )
                minimum_fuel = fit.slope * additive_axis + fit.intercept
                minima_points = np.column_stack((minimum_fuel, additive_axis))
                if config.show_min_line:
                    self._emit_log(
                        on_log, f"Линия минимальной концентрации: fuel = {fit.slope:.6g} × steam {fit.intercept:+.6g}.",
                    )
                if needs_right_line:
                    maxima_detection_method = config.maxima_detection_method
                    right_fuel, analysis_contour_indices, analysis_contour_values = self.find_right_path(
                        config, component_grid, fuel_axis, additive_axis, minimum_fuel,
                    )
                    right_points = np.column_stack((right_fuel, additive_axis))
                    if show_mean_line:
                        mean_points = np.column_stack(((minimum_fuel + right_fuel) / 2, additive_axis))
            self._emit_progress(on_progress, 100)
        except DiffSurfaceCancellationError as exc:
            raise CancellationError(str(exc)) from exc
        except DiffSurfaceError as exc:
            raise ProcessingError(str(exc)) from exc

        return RegimeMapResult(
            input_path=input_path,
            fuel_axis=fuel_axis,
            additive_axis=additive_axis,
            component_grid=component_grid,
            component_label=default_component_label,
            co_levels=co_levels,
            x_limits=x_limits,
            y_limits=y_limits,
            is_co_component=config.is_co_component,
            show_min_line=config.show_min_line,
            show_right_line=show_right_line,
            show_mean_line=show_mean_line,
            minima_points=minima_points,
            right_points=right_points,
            mean_points=mean_points,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            colorbar_label=colorbar_label,
            cmap_name=cmap_name,
            maxima_detection_method=maxima_detection_method,
            analysis_contour_indices=analysis_contour_indices,
            analysis_contour_values=analysis_contour_values,
            font_family=DEFAULT_FONT_FAMILY,
            font_size=int(config.font_size),
        )

    def find_minimum_path(
        self, grid: np.ndarray, fuel: np.ndarray, tolerance: float,
        should_cancel: CancelCallback | None = None,
    ) -> np.ndarray:
        row_minima = grid.min(axis=1)
        candidates = [np.flatnonzero(row <= minimum + tolerance) for row, minimum in zip(grid, row_minima)]
        centers = np.array([fuel[indices].mean() for indices in candidates])
        indices = np.empty(len(grid), dtype=int)
        previous_fuel = centers[0]
        for row_index, choices in enumerate(candidates):
            self._check_cancel(should_cancel)
            # Use neighboring slice centers to anticipate the direction of the valley.
            center = centers[max(0, row_index - 1):row_index + 2].mean()
            target = (center + previous_fuel) / 2 if row_index else center
            nearest = choices[np.argmin(abs(fuel[choices] - target))]
            if row_index:
                previous_index = indices[row_index - 1]
                lower = grid[row_index - 1] - row_minima[row_index - 1]
                upper = grid[row_index] - row_minima[row_index]
                # Test the preferred continuation first, then the closest feasible one.
                ordered = choices[np.argsort(abs(fuel[choices] - target), kind="stable")]
                for choice in ordered:
                    if self._segment_stays_in_band(fuel, lower, upper, previous_index, choice, tolerance):
                        nearest = choice
                        break
            indices[row_index] = nearest
            previous_fuel = fuel[nearest]
        return fuel[indices]

    @staticmethod
    def _segment_stays_in_band(
        fuel: np.ndarray, lower: np.ndarray, upper: np.ndarray, start: int, end: int, tolerance: float,
    ) -> bool:
        if start == end:
            return True
        left, right = sorted((start, end))
        columns = slice(left, right + 1)
        fraction = (fuel[columns] - fuel[start]) / (fuel[end] - fuel[start])
        low, high = lower[columns], upper[columns]
        values = low + fraction * (high - low)
        limit = tolerance + 1e-9
        if np.any(values > limit):
            return False
        # Within each crossed cell the bilinear interpolation along the segment is quadratic.
        quadratic = np.diff(fraction) * np.diff(high - low)
        linear = np.diff(values) - quadratic
        concave = quadratic < 0
        vertex = np.clip(-linear[concave] / (2 * quadratic[concave]), 0, 1)
        peaks = values[:-1][concave] + vertex * (linear[concave] + vertex * quadratic[concave])
        return bool(np.all(peaks <= limit))

    def find_right_path(
        self, config: RegimeMapJobConfig, grid: np.ndarray, fuel: np.ndarray,
        additive: np.ndarray, minimum_fuel: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
        dz_dx, dz_dy = self.diff_pipeline.compute_derivatives(grid, additive, fuel)
        gradient = self.diff_pipeline.build_selected_surface(dz_dx, dz_dy, SurfaceMode.GRADIENT_MAGNITUDE)
        if config.maxima_detection_method is MaximaDetectionMethod.ROW_PEAKS:
            right_side = fuel > minimum_fuel[:, None]
            maxima = np.argmax(np.where(right_side, gradient, -np.inf), axis=1)
            return np.where(right_side.any(axis=1), fuel[maxima], np.nan), (), ()

        indices = parse_contour_level_indices(config.contour_levels_text)
        values = self.diff_pipeline.resolve_contour_level_values(gradient, indices)
        points = self.diff_pipeline.extract_contour_points(gradient, fuel, additive, values)
        right = points[points[:, 0] > np.interp(points[:, 1], additive, minimum_fuel)]
        right_fuel = np.full_like(additive, np.nan)
        if np.unique(right[:, 1]).size >= 2:
            slope, intercept = np.polyfit(right[:, 1], right[:, 0], deg=1)
            estimate = slope * additive + intercept
            supported = (
                (additive >= right[:, 1].min()) & (additive <= right[:, 1].max())
                & (estimate > minimum_fuel) & (estimate <= fuel[-1])
            )
            right_fuel[supported] = estimate[supported]
        return right_fuel, indices, values

    def resolve_co_levels(self, config: RegimeMapJobConfig) -> int | np.ndarray:
        if not config.use_custom_ppm_scale:
            return AUTO_CONTOUR_LEVELS

        values = np.arange(config.ppm_min, config.ppm_max, config.ppm_step, dtype=float)
        values = np.append(values, float(config.ppm_max))
        unique_values = np.unique(values)
        if unique_values.size < 2:
            raise ProcessingError("Не удалось сформировать пользовательскую шкалу ppm.")
        return unique_values

    def _build_base_map(
        self,
        config: RegimeMapJobConfig,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
        input_path = config.input_path
        assert input_path is not None

        self._check_cancel(should_cancel)
        frame = self.diff_pipeline.read_dataset(input_path)
        self._emit_progress(on_progress, 40)

        self._check_cancel(should_cancel)
        fuel_axis, additive_axis, component_grid = self.diff_pipeline.build_regular_grid(frame)
        self._emit_progress(on_progress, 80)

        return (
            input_path,
            np.asarray(fuel_axis, dtype=float),
            np.asarray(additive_axis, dtype=float),
            np.asarray(component_grid, dtype=float),
        )

    def _needs_right_line_analysis(self, config: RegimeMapJobConfig) -> bool:
        return (config.is_co_component and config.show_right_line) or config.show_mean_line

    def _to_diff_config(
        self,
        config: RegimeMapJobConfig,
        *,
        include_analysis: bool,
    ) -> DiffSurfaceJobConfig:
        maxima_detection_method = config.maxima_detection_method if include_analysis else MaximaDetectionMethod.ROW_PEAKS
        contour_levels_text = config.contour_levels_text if include_analysis else ""
        return DiffSurfaceJobConfig(
            input_path=config.input_path,
            surface_mode=SurfaceMode.GRADIENT_MAGNITUDE,
            maxima_detection_method=maxima_detection_method,
            contour_levels_text=contour_levels_text,
        )

    def _emit_log(self, callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def _emit_progress(self, callback: ProgressCallback | None, value: int) -> None:
        if callback is not None:
            callback(value)

    def _check_cancel(self, should_cancel: CancelCallback | None) -> None:
        if should_cancel is not None and should_cancel():
            raise CancellationError("Построение режимной карты остановлено пользователем.")
