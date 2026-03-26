from __future__ import annotations

from pathlib import Path

from .models import DiffSurfaceJobConfig, SurfaceMode, ValidationResult


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

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def generate_export_basename(input_path: Path, surface_mode: SurfaceMode) -> str:
    return f"diff_surface_{input_path.stem}_{surface_mode.value}"


def resolve_export_paths(output_dir: Path, input_path: Path, surface_mode: SurfaceMode) -> tuple[Path, Path]:
    basename = generate_export_basename(input_path, surface_mode)
    return output_dir / f"{basename}.png", output_dir / f"{basename}_lines.json"
