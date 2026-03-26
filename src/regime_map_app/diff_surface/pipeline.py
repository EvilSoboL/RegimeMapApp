from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .exceptions import CancellationError, CsvValidationError, DiffSurfaceError, ProcessingError, ValidationError
from .models import (
    CSV_SEPARATOR,
    REQUIRED_COLUMNS,
    DiffSurfaceJobConfig,
    DifferentialSurfaceResult,
    LineFit,
    SurfaceMode,
    ValidationResult,
)
from .validation import validate_job_config

ALTERNATIVE_CSV_DELIMITERS = (",", "\t", "|")

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]


class DiffSurfacePipeline:
    def validate_inputs(self, config: DiffSurfaceJobConfig) -> ValidationResult:
        base_validation = validate_job_config(config)
        errors = list(base_validation.errors)
        checked_points = 0

        if errors:
            return ValidationResult(is_valid=False, errors=tuple(errors))

        assert config.input_path is not None
        try:
            frame = self.read_dataset(config.input_path)
            self.build_regular_grid(frame)
            checked_points = len(frame.index)
        except DiffSurfaceError as exc:
            errors.append(str(exc))

        return ValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
            checked_points=checked_points,
        )

    def read_dataset(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise CsvValidationError(f"Файл {path.name} не существует.")
        if not path.is_file():
            raise CsvValidationError(f"Путь {path} не является файлом.")
        if path.suffix.lower() != ".csv":
            raise CsvValidationError(f"Файл {path.name} должен иметь расширение .csv.")

        try:
            frame = pd.read_csv(path, sep=CSV_SEPARATOR, engine="python", dtype=str)
        except pd.errors.EmptyDataError as exc:
            raise CsvValidationError(f"CSV-файл {path.name} пуст.") from exc
        except UnicodeDecodeError as exc:
            raise CsvValidationError(f"Не удалось прочитать CSV-файл {path.name} в кодировке UTF-8.") from exc
        except Exception as exc:
            raise CsvValidationError(f"Не удалось прочитать CSV-файл {path.name}: {exc}") from exc

        if frame.empty:
            raise CsvValidationError(f"CSV-файл {path.name} не содержит строк данных.")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            if len(frame.columns) == 1:
                detected_separator = _detect_wrong_separator(path)
                if detected_separator is not None:
                    raise CsvValidationError(
                        f"Файл {path.name} использует неверный разделитель {detected_separator!r}. "
                        f"Ожидается разделитель {CSV_SEPARATOR!r}."
                    )
            raise CsvValidationError(
                f"В файле {path.name} отсутствуют обязательные столбцы: {', '.join(missing_columns)}."
            )

        numeric_frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        for column in REQUIRED_COLUMNS:
            try:
                numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="raise")
            except Exception as exc:
                raise CsvValidationError(
                    f"В файле {path.name} столбец {column} содержит нечисловые значения."
                ) from exc

        if numeric_frame.isna().any().any():
            raise CsvValidationError(f"В файле {path.name} обнаружены пропущенные значения.")

        if numeric_frame["fuel"].nunique() < 2 or numeric_frame["additive"].nunique() < 2:
            raise CsvValidationError(
                f"В файле {path.name} недостаточно уникальных значений fuel или additive."
            )

        if numeric_frame.duplicated(subset=["fuel", "additive"]).any():
            raise CsvValidationError(
                f"В файле {path.name} найдены дублирующиеся точки с одинаковыми fuel и additive."
            )

        expected_points = numeric_frame["fuel"].nunique() * numeric_frame["additive"].nunique()
        if len(numeric_frame.index) != expected_points:
            raise CsvValidationError(f"В файле {path.name} данные не образуют полную регулярную сетку.")

        fuel_axis, additive_axis, component_grid = self.build_regular_grid(numeric_frame)
        if component_grid.shape != (len(additive_axis), len(fuel_axis)):
            raise CsvValidationError(f"В файле {path.name} не удалось построить корректную регулярную сетку.")

        return numeric_frame

    def build_regular_grid(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fuel_axis = np.sort(frame["fuel"].unique().astype(float))
        additive_axis = np.sort(frame["additive"].unique().astype(float))
        pivot = frame.pivot(index="additive", columns="fuel", values="component")
        pivot = pivot.sort_index().sort_index(axis=1)
        pivot = pivot.reindex(index=additive_axis, columns=fuel_axis)
        if pivot.isna().any().any():
            raise ProcessingError("Не удалось построить регулярную сетку без пропусков.")
        component_grid = pivot.to_numpy(dtype=float)
        return fuel_axis, additive_axis, component_grid

    def compute_derivatives(
        self,
        component_grid: np.ndarray,
        additive_axis: np.ndarray,
        fuel_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(fuel_axis) < 2 or len(additive_axis) < 2:
            raise ProcessingError("Для вычисления градиента нужны минимум два значения по каждой оси.")
        try:
            dz_dy, dz_dx = np.gradient(component_grid, additive_axis, fuel_axis)
        except Exception as exc:
            raise ProcessingError(f"Не удалось вычислить градиент поверхности: {exc}") from exc
        return np.asarray(dz_dx, dtype=float), np.asarray(dz_dy, dtype=float)

    def build_selected_surface(
        self,
        dz_dx: np.ndarray,
        dz_dy: np.ndarray,
        surface_mode: SurfaceMode,
    ) -> np.ndarray:
        if surface_mode is not SurfaceMode.GRADIENT_MAGNITUDE:
            raise ProcessingError("Поддерживается только режим поверхности |grad|.")
        return np.sqrt(np.square(dz_dx) + np.square(dz_dy))

    def find_minima_points(
        self,
        component_grid: np.ndarray,
        fuel_axis: np.ndarray,
        additive_axis: np.ndarray,
    ) -> np.ndarray:
        min_indices = np.argmin(component_grid, axis=0)
        return np.column_stack((fuel_axis, additive_axis[min_indices])).astype(float)

    def find_peak_pair_points(
        self,
        surface: np.ndarray,
        fuel_axis: np.ndarray,
        additive_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_points: list[tuple[float, float]] = []
        right_points: list[tuple[float, float]] = []

        for row_index, row in enumerate(surface):
            left_index, right_index = self.find_two_peak_indices(row)
            additive_value = float(additive_axis[row_index])
            left_points.append((float(fuel_axis[left_index]), additive_value))
            right_points.append((float(fuel_axis[right_index]), additive_value))

        return np.asarray(left_points, dtype=float), np.asarray(right_points, dtype=float)

    def split_maxima_points_by_minimum_line(
        self,
        first_peak_points: np.ndarray,
        second_peak_points: np.ndarray,
        minimum_line_fit: LineFit,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points = np.vstack((first_peak_points, second_peak_points)).astype(float)
        minimum_line_values = minimum_line_fit.slope * all_points[:, 0] + minimum_line_fit.intercept
        tolerance = 1e-9
        left_points = all_points[all_points[:, 1] >= minimum_line_values - tolerance]
        right_points = all_points[all_points[:, 1] < minimum_line_values - tolerance]

        if len(left_points) == 0 or len(right_points) == 0:
            raise ProcessingError(
                "Не удалось разделить точки максимумов относительно линии минимальной концентрации."
            )

        return self._sort_points(left_points), self._sort_points(right_points)

    def find_maxima_points(
        self,
        surface: np.ndarray,
        fuel_axis: np.ndarray,
        additive_axis: np.ndarray,
        minimum_line_fit: LineFit,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_peak_points, second_peak_points = self.find_peak_pair_points(surface, fuel_axis, additive_axis)
        return self.split_maxima_points_by_minimum_line(first_peak_points, second_peak_points, minimum_line_fit)

    def find_two_peak_indices(self, row: np.ndarray) -> tuple[int, int]:
        if row.ndim != 1 or row.size < 2:
            raise ProcessingError("Для поиска двух максимумов строка поверхности должна содержать минимум две точки.")

        candidate_indices: list[int] = []
        last_index = row.size - 1

        for index in range(row.size):
            left_value = row[index - 1] if index > 0 else -np.inf
            right_value = row[index + 1] if index < last_index else -np.inf
            is_local_peak = row[index] >= left_value and row[index] >= right_value and (
                row[index] > left_value or row[index] > right_value
            )
            if is_local_peak:
                candidate_indices.append(index)

        if len(candidate_indices) < 2:
            candidate_indices = np.argsort(row)[-2:].tolist()

        top_indices = sorted(
            candidate_indices,
            key=lambda index: (float(row[index]), -index),
            reverse=True,
        )[:2]

        if len(top_indices) < 2:
            raise ProcessingError("Не удалось найти две точки максимумов на строке дифференциальной поверхности.")

        left_index, right_index = sorted(top_indices)
        if left_index == right_index:
            raise ProcessingError("Не удалось выделить две разные точки максимумов на строке дифференциальной поверхности.")
        return left_index, right_index

    def fit_line(self, points: np.ndarray, line_label: str) -> LineFit:
        if len(points) < 2:
            raise ProcessingError(f"Для аппроксимации {line_label} нужны минимум две точки.")
        if np.unique(points[:, 0]).size < 2:
            raise ProcessingError(f"Для аппроксимации {line_label} нужны точки с разными значениями fuel.")
        try:
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], deg=1)
        except Exception as exc:
            raise ProcessingError(f"Не удалось аппроксимировать {line_label}: {exc}") from exc
        if not np.isfinite((slope, intercept)).all():
            raise ProcessingError(f"Коэффициенты {line_label} получились некорректными.")
        return LineFit(slope=float(slope), intercept=float(intercept))

    def _sort_points(self, points: np.ndarray) -> np.ndarray:
        order = np.lexsort((points[:, 0], points[:, 1]))
        return np.asarray(points[order], dtype=float)

    def process_job(
        self,
        config: DiffSurfaceJobConfig,
        *,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> DifferentialSurfaceResult:
        validation = validate_job_config(config)
        if not validation.is_valid:
            raise ValidationError("\n".join(validation.errors))

        assert config.input_path is not None
        self._check_cancel(should_cancel)
        self._emit_log(on_log, f"Чтение файла {config.input_path.name}.")
        frame = self.read_dataset(config.input_path)
        self._emit_progress(on_progress, 20)

        self._check_cancel(should_cancel)
        fuel_axis, additive_axis, component_grid = self.build_regular_grid(frame)
        self._emit_log(on_log, f"Построена регулярная сетка {len(fuel_axis)}x{len(additive_axis)}.")
        self._emit_progress(on_progress, 40)

        self._check_cancel(should_cancel)
        dz_dx, dz_dy = self.compute_derivatives(component_grid, additive_axis, fuel_axis)
        selected_surface = self.build_selected_surface(dz_dx, dz_dy, config.surface_mode)
        self._emit_log(on_log, f"Рассчитана поверхность в режиме {config.surface_mode.label}.")
        self._emit_progress(on_progress, 60)

        self._check_cancel(should_cancel)
        minima_points = self.find_minima_points(component_grid, fuel_axis, additive_axis)
        minima_line_fit = self.fit_line(minima_points, "линии минимумов концентрации")
        left_maxima_points, right_maxima_points = self.find_maxima_points(
            selected_surface,
            fuel_axis,
            additive_axis,
            minima_line_fit,
        )
        self._emit_log(on_log, f"Найдена линия минимумов концентрации по {len(minima_points)} точкам.")
        self._emit_log(on_log, f"Найдено точек максимумов слева выше линии минимума: {len(left_maxima_points)}.")
        self._emit_log(on_log, f"Найдено точек максимумов справа ниже линии минимума: {len(right_maxima_points)}.")
        self._emit_progress(on_progress, 80)

        self._check_cancel(should_cancel)
        left_line_fit = self.fit_line(left_maxima_points, "левой линии максимумов")
        right_line_fit = self.fit_line(right_maxima_points, "правой линии максимумов")
        self._emit_log(
            on_log,
            f"Линия минимумов концентрации: additive = {minima_line_fit.slope:.6g} * fuel + {minima_line_fit.intercept:.6g}",
        )
        self._emit_log(
            on_log,
            f"Левая линия максимумов: additive = {left_line_fit.slope:.6g} * fuel + {left_line_fit.intercept:.6g}",
        )
        self._emit_log(
            on_log,
            f"Правая линия максимумов: additive = {right_line_fit.slope:.6g} * fuel + {right_line_fit.intercept:.6g}",
        )
        self._emit_progress(on_progress, 100)

        return DifferentialSurfaceResult(
            input_path=config.input_path,
            surface_mode=config.surface_mode,
            fuel_axis=fuel_axis,
            additive_axis=additive_axis,
            component_grid=component_grid,
            dz_dx=dz_dx,
            dz_dy=dz_dy,
            selected_surface=np.asarray(selected_surface, dtype=float),
            minima_points=minima_points,
            minima_line_fit=minima_line_fit,
            left_maxima_points=left_maxima_points,
            right_maxima_points=right_maxima_points,
            left_line_fit=left_line_fit,
            right_line_fit=right_line_fit,
        )

    def _emit_log(self, callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def _emit_progress(self, callback: ProgressCallback | None, value: int) -> None:
        if callback is not None:
            callback(value)

    def _check_cancel(self, should_cancel: CancelCallback | None) -> None:
        if should_cancel is not None and should_cancel():
            raise CancellationError("Построение дифференциальной поверхности остановлено пользователем.")


def _detect_wrong_separator(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as source:
            header_line = source.readline().strip()
    except OSError:
        return None
    except UnicodeDecodeError:
        return None

    if not header_line:
        return None

    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=CSV_SEPARATOR + "".join(ALTERNATIVE_CSV_DELIMITERS))
    except csv.Error:
        dialect = None

    if dialect is not None and dialect.delimiter != CSV_SEPARATOR:
        return dialect.delimiter

    for delimiter in ALTERNATIVE_CSV_DELIMITERS:
        if delimiter in header_line:
            return delimiter

    return None
