from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from regime_map_app.approx.backend import ApproximationBackend, ScipyApproximationBackend
from regime_map_app.approx.exceptions import ApproximationError, CancellationError, ProcessingError, ValidationError
from regime_map_app.approx.models import ApproxJobConfig, BatchProcessSummary, FileProcessResult, ValidationResult
from regime_map_app.approx.validation import (
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
                self.backend.read_dataset(path)
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
            frame = self.backend.read_dataset(path)
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
        )

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

        validation = self.validate_inputs(config)
        if not validation.is_valid:
            raise ValidationError("\n".join(validation.errors))

        input_paths = normalize_input_paths(config)
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
