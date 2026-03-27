from __future__ import annotations

from pathlib import Path

from .models import RegimeMapJobConfig, ValidationResult


def validate_job_config(config: RegimeMapJobConfig) -> ValidationResult:
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

    if config.use_custom_x_limits and config.x_min >= config.x_max:
        errors.append("Нижняя граница X должна быть меньше верхней.")

    if config.use_custom_y_limits and config.y_min >= config.y_max:
        errors.append("Нижняя граница Y должна быть меньше верхней.")

    if config.use_custom_ppm_scale:
        if config.ppm_min >= config.ppm_max:
            errors.append("Нижняя граница шкалы ppm должна быть меньше верхней.")
        if config.ppm_step <= 0:
            errors.append("Шаг шкалы ppm должен быть положительным.")

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def generate_export_basename(input_path: Path) -> str:
    return f"regime_map_{input_path.stem}"


def resolve_export_path(output_dir: Path, input_path: Path) -> Path:
    basename = generate_export_basename(input_path)
    return output_dir / f"{basename}.png"
