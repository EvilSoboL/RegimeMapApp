from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import median_filter

from regime_map_app.approx.exceptions import CsvValidationError, ProcessingError
from regime_map_app.approx.models import ApproxJobConfig, CSV_SEPARATOR, REQUIRED_COLUMNS, SurfaceGrid


class ApproximationBackend(Protocol):
    def read_dataset(self, path: Path) -> pd.DataFrame:
        ...

    def approximate_surface(self, frame: pd.DataFrame, config: ApproxJobConfig) -> SurfaceGrid:
        ...

    def filter_surface(self, surface: SurfaceGrid, config: ApproxJobConfig) -> SurfaceGrid:
        ...

    def export_surface(self, surface: SurfaceGrid, output_path: Path) -> pd.DataFrame:
        ...


class ScipyApproximationBackend:
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
                raise CsvValidationError(
                    f"Файл {path.name} не содержит ожидаемых столбцов. "
                    "Проверьте заголовок и разделитель ';;'."
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

        if numeric_frame[["fuel", "additive"]].drop_duplicates().shape[0] < 3:
            raise CsvValidationError(
                f"В файле {path.name} недостаточно точек для аппроксимации. Нужны минимум 3 уникальные точки."
            )

        if numeric_frame["fuel"].nunique() < 2 or numeric_frame["additive"].nunique() < 2:
            raise CsvValidationError(
                f"В файле {path.name} недостаточно уникальных значений fuel или additive."
            )

        return numeric_frame

    def approximate_surface(self, frame: pd.DataFrame, config: ApproxJobConfig) -> SurfaceGrid:
        try:
            points = frame[["fuel", "additive"]].to_numpy(dtype=float)
            values = frame["component"].to_numpy(dtype=float)
            fuel_axis = np.linspace(frame["fuel"].min(), frame["fuel"].max(), config.resolution_x)
            additive_axis = np.linspace(
                frame["additive"].min(),
                frame["additive"].max(),
                config.resolution_y,
            )
            fuel_grid, additive_grid = np.meshgrid(fuel_axis, additive_axis)
            grid_points = np.column_stack((fuel_grid.ravel(), additive_grid.ravel()))
            interpolator = RBFInterpolator(points, values, kernel=config.kernel)
            component_grid = interpolator(grid_points).reshape(additive_grid.shape)
        except Exception as exc:
            raise ProcessingError(f"Не удалось построить RBF-поверхность: {exc}") from exc

        return SurfaceGrid(
            fuel_grid=fuel_grid,
            additive_grid=additive_grid,
            component_grid=np.asarray(component_grid, dtype=float),
        )

    def filter_surface(self, surface: SurfaceGrid, config: ApproxJobConfig) -> SurfaceGrid:
        component_grid = np.array(surface.component_grid, copy=True)
        try:
            if config.clamp_zero:
                component_grid = np.maximum(component_grid, 0.0)
            if config.median_size > 0:
                component_grid = median_filter(component_grid, size=config.median_size)
        except Exception as exc:
            raise ProcessingError(f"Не удалось применить фильтрацию поверхности: {exc}") from exc

        return SurfaceGrid(
            fuel_grid=surface.fuel_grid,
            additive_grid=surface.additive_grid,
            component_grid=component_grid,
        )

    def export_surface(self, surface: SurfaceGrid, output_path: Path) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "fuel": surface.fuel_grid.ravel(),
                "additive": surface.additive_grid.ravel(),
                "component": surface.component_grid.ravel(),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lines = [CSV_SEPARATOR.join(REQUIRED_COLUMNS)]
            for row in frame.itertuples(index=False, name=None):
                lines.append(CSV_SEPARATOR.join(_format_value(value) for value in row))
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ProcessingError(f"Не удалось сохранить результат в {output_path}: {exc}") from exc
        return frame


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)
