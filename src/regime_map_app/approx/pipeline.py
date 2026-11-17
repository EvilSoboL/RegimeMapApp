from __future__ import annotations

from math import isclose, isfinite
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .backend import ApproximationBackend, ScipyApproximationBackend
from .exceptions import ApproximationError, CancellationError, ProcessingError, ValidationError
from .models import ApproxJobConfig, BatchProcessSummary, FileProcessResult, REQUIRED_COLUMNS, ValidationResult
from .validation import (
    normalize_input_paths,
    parse_file_metadata,
    resolve_output_path,
    validate_job_config,
)

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CurrentFileCallback = Callable[[Path], None]
CancelCallback = Callable[[], bool]


class ApproxPipeline:
    def __init__(self, backend: ApproximationBackend | None = None) -> None:
        self.backend = backend or ScipyApproximationBackend()

    def validate_inputs(self, config: ApproxJobConfig) -> ValidationResult:
        base_validation = validate_job_config(config, require_output_dir=False)
        errors = list(base_validation.errors)

        if errors:
            return ValidationResult(is_valid=False, errors=tuple(errors))

        try:
            input_paths = normalize_input_paths(config)
        except ValidationError as exc:
            errors.append(str(exc))
            return ValidationResult(is_valid=False, errors=tuple(errors))

        checked_files = 0
        for path in input_paths:
            try:
                parse_file_metadata(path)
                self._read_input(path, config)
                checked_files += 1
            except ApproximationError as exc:
                errors.append(str(exc))

        return ValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
            checked_files=checked_files,
        )

    def process_one(self, path: Path, config: ApproxJobConfig) -> FileProcessResult:
        metadata = None
        try:
            metadata = parse_file_metadata(path)
            frame = self._read_input(path, config)
            surface = self.backend.approximate_surface(frame, config)
            filtered_surface = self.backend.filter_surface(surface, config)
            output_path = resolve_output_path(config, path)
            exported = self.backend.export_surface(filtered_surface, output_path)
        except ApproximationError as exc:
            return FileProcessResult(
                input_path=path,
                success=False,
                messages=(str(exc),),
                metadata=metadata,
            )
        except Exception as exc:
            return FileProcessResult(
                input_path=path,
                success=False,
                messages=(f"Непредвиденная ошибка при обработке файла {path.name}: {exc}",),
                metadata=metadata,
            )

        return FileProcessResult(
            input_path=path,
            success=True,
            messages=(f"Файл {path.name} успешно обработан.",),
            output_path=output_path,
            output_rows=len(exported.index),
            metadata=metadata,
            preview_surface=filtered_surface if config.input_mode.is_single else None,
        )

    def _read_input(self, path: Path, config: ApproxJobConfig) -> pd.DataFrame:
        frame = self.backend.read_dataset(path)
        if not config.boundary_points:
            return frame
        if not config.input_mode.is_single or parse_file_metadata(path).component_name.casefold() != "co":
            raise ValidationError("Граничные точки доступны для одиночного CSV с данными CO.")

        added = []
        for fuel, additive, co in config.boundary_points:
            if not all(isfinite(value) for value in (fuel, additive, co)) or fuel <= 0 or co < 0:
                raise ValidationError("Проверьте числовые значения граничных точек CO.")
            row = frame.loc[frame["additive"] == additive]
            if row.empty:
                raise ValidationError(f"В исходном CSV нет точек при расходе пара {additive:g}.")
            same_fuel = row["fuel"].map(lambda value: isclose(value, fuel, rel_tol=0, abs_tol=1e-9))
            if same_fuel.any():
                if (row.loc[same_fuel, "component"] == co).all():
                    continue
                raise ValidationError(f"При расходе топлива {fuel:g} и пара {additive:g} уже есть другое значение CO.")
            if row["fuel"].min() < fuel < row["fuel"].max():
                raise ValidationError(f"При расходе пара {additive:g} граница должна быть за крайними исходными точками.")
            added.append((fuel, additive, co))

        if not added:
            return frame
        boundaries = pd.DataFrame(added, columns=REQUIRED_COLUMNS).drop_duplicates()
        if boundaries.duplicated(subset=["fuel", "additive"]).any():
            raise ValidationError("Заданы разные значения CO для одной граничной точки.")
        return pd.concat([frame, boundaries], ignore_index=True)

    def process_many(
        self,
        paths: Sequence[Path],
        config: ApproxJobConfig,
        *,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_current_file: CurrentFileCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> BatchProcessSummary:
        results: list[FileProcessResult] = []
        succeeded = 0
        failed = 0
        total = len(paths)

        for index, path in enumerate(paths, start=1):
            if should_cancel and should_cancel():
                raise CancellationError("Обработка остановлена пользователем.")

            if on_current_file:
                on_current_file(path)
            if on_log:
                on_log(f"Обработка файла {index}/{total}: {path.name}")

            result = self.process_one(path, config)
            results.append(result)

            if result.success:
                succeeded += 1
            else:
                failed += 1

            if on_log:
                for message in result.messages:
                    on_log(message)
            if on_progress:
                on_progress(index, total)

        if config.output_dir is None:
            raise ProcessingError("Не выбрана выходная директория.")

        return BatchProcessSummary(
            total_files=total,
            succeeded=succeeded,
            failed=failed,
            output_dir=config.output_dir,
            results=tuple(results),
        )

    def process_job(
        self,
        config: ApproxJobConfig,
        *,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_current_file: CurrentFileCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> BatchProcessSummary:
        run_validation = validate_job_config(config, require_output_dir=True)
        if not run_validation.is_valid:
            raise ValidationError("\n".join(run_validation.errors))

        input_paths = normalize_input_paths(config)
        if config.input_mode.is_single:
            validation = self.validate_inputs(config)
            if not validation.is_valid:
                raise ValidationError("\n".join(validation.errors))
        try:
            config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProcessingError(
                f"Не удалось создать выходную директорию {config.output_dir}: {exc}"
            ) from exc
        return self.process_many(
            input_paths,
            config,
            on_log=on_log,
            on_progress=on_progress,
            on_current_file=on_current_file,
            should_cancel=should_cancel,
        )
