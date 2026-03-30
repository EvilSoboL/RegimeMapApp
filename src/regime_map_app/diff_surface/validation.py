from __future__ import annotations

from pathlib import Path

from .models import DiffSurfaceJobConfig, MaximaDetectionMethod, SurfaceMode, ValidationResult


def parse_contour_level_indices(levels_text: str) -> tuple[int, ...]:
    raw_value = levels_text.strip()
    if not raw_value:
        raise ValueError("Укажите хотя бы один номер линии уровня.")

    values: list[int] = []
    seen: set[int] = set()
    for chunk in raw_value.split(","):
        token = chunk.strip()
        if not token:
            raise ValueError("Номера линий уровня должны быть перечислены через запятую без пустых значений.")
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError("Номера линий уровня должны быть целыми числами.") from exc
        if index <= 0:
            raise ValueError("Номера линий уровня должны быть больше 0.")
        if index not in seen:
            values.append(index)
            seen.add(index)
    return tuple(values)


def validate_job_config(config: DiffSurfaceJobConfig) -> ValidationResult:
    errors: list[str] = []

    if config.input_path is None:
        errors.append("Не выбран входной CSV-файл.")
        return ValidationResult(is_valid=False, errors=tuple(errors))

    input_path = config.input_path
    if not input_path.exists():
        errors.append(f"Файл {input_path} не существует.")
    elif not input_path.is_file():
        errors.append(f"Путь {input_path} должен указывать на файл.")
    elif input_path.suffix.lower() != ".csv":
        errors.append(f"Файл {input_path.name} должен иметь расширение .csv.")

    if config.maxima_detection_method is MaximaDetectionMethod.CONTOUR_LEVELS:
        try:
            parse_contour_level_indices(config.contour_levels_text)
        except ValueError as exc:
            errors.append(str(exc))

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def generate_export_basename(input_path: Path, surface_mode: SurfaceMode) -> str:
    return f"diff_surface_{input_path.stem}_{surface_mode.value}"


def resolve_export_path(output_dir: Path, input_path: Path, surface_mode: SurfaceMode) -> Path:
    basename = generate_export_basename(input_path, surface_mode)
    return output_dir / f"{basename}.png"
