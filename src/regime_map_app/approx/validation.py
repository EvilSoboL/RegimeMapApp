from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .exceptions import ValidationError
from .models import (
    ALLOWED_KERNELS,
    ApproxJobConfig,
    FileMetadata,
    InputMode,
    ValidationResult,
)

FILENAME_PATTERN = re.compile(
    r"^(?P<fuel>[^-]+)-(?P<diluent>[^-]+)-(?P<component>[^-]+)-"
    r"(?P<day>\d{2})-(?P<month>\d{2})-(?P<year>\d{4})\.csv$",
    re.IGNORECASE,
)


def parse_file_metadata(path: Path) -> FileMetadata:
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValidationError(
            f"Файл {path.name} не соответствует шаблону "
            "<топливо>-<разбавитель>-<компонент>-дд-мм-гггг.csv."
        )
    try:
        measured_on = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError as exc:
        raise ValidationError(f"Файл {path.name} содержит некорректную дату в имени.") from exc
    return FileMetadata(
        fuel_name=match.group("fuel"),
        diluent=match.group("diluent"),
        component_name=match.group("component"),
        measured_on=measured_on,
        original_name=path.name,
    )


def generate_output_filename(input_path: Path) -> str:
    return f"approx_{input_path.name}"


def resolve_output_path(config: ApproxJobConfig, input_path: Path) -> Path:
    if config.output_dir is None:
        raise ValidationError("Не выбрана выходная директория.")
    if config.input_mode.is_single and not config.auto_output_name and config.output_filename:
        filename = config.output_filename
    else:
        filename = generate_output_filename(input_path)
    return config.output_dir / filename


def normalize_input_paths(config: ApproxJobConfig) -> tuple[Path, ...]:
    if config.input_mode is InputMode.FOLDER_BATCH:
        if len(config.input_paths) != 1:
            raise ValidationError("Для пакетной обработки нужно выбрать одну папку.")
        folder = config.input_paths[0]
        if not folder.exists():
            raise ValidationError(f"Папка {folder} не существует.")
        if not folder.is_dir():
            raise ValidationError(f"Путь {folder} не является папкой.")
        csv_files = tuple(sorted(path for path in folder.iterdir() if path.suffix.lower() == ".csv"))
        if not csv_files:
            raise ValidationError(f"В папке {folder} не найдено CSV-файлов.")
        return csv_files

    unique_paths = []
    seen = set()
    for path in config.input_paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return tuple(unique_paths)


def validate_job_config(config: ApproxJobConfig, *, require_output_dir: bool = True) -> ValidationResult:
    errors: list[str] = []

    if not config.input_paths:
        errors.append("Не выбран входной файл, набор файлов или папка.")

    if require_output_dir and config.output_dir is None:
        errors.append("Не выбрана выходная директория.")
    elif config.output_dir is not None and config.output_dir.exists() and not config.output_dir.is_dir():
        errors.append("Выходной путь должен указывать на директорию.")

    if config.resolution_x <= 0 or config.resolution_y <= 0:
        errors.append("Параметры resolution_x и resolution_y должны быть больше 0.")

    if config.median_size < 0:
        errors.append("Параметр median_size не может быть отрицательным.")

    if config.kernel not in ALLOWED_KERNELS:
        errors.append(f"Ядро {config.kernel!r} не поддерживается.")

    if config.input_mode is InputMode.SINGLE_FILE and len(config.input_paths) != 1:
        errors.append("В режиме одиночного файла нужно выбрать ровно один CSV.")

    if config.input_mode is InputMode.MULTI_FILES and len(config.input_paths) < 1:
        errors.append("В режиме нескольких файлов нужно выбрать хотя бы один CSV.")

    if config.input_mode is InputMode.FOLDER_BATCH and len(config.input_paths) != 1:
        errors.append("В режиме пакетной обработки нужно выбрать одну папку.")

    if config.input_mode.is_single and not config.auto_output_name:
        if not config.output_filename:
            errors.append("Для одиночного режима укажите имя выходного файла.")
        elif not config.output_filename.lower().endswith(".csv"):
            errors.append("Имя выходного файла должно оканчиваться на .csv.")

    for path in config.input_paths:
        if config.input_mode is InputMode.FOLDER_BATCH:
            if not path.exists():
                errors.append(f"Папка {path} не существует.")
            elif not path.is_dir():
                errors.append(f"Путь {path} должен быть папкой.")
            continue

        if not path.exists():
            errors.append(f"Файл {path} не существует.")
            continue
        if not path.is_file():
            errors.append(f"Путь {path} должен указывать на файл.")
            continue
        if path.suffix.lower() != ".csv":
            errors.append(f"Файл {path.name} должен иметь расширение .csv.")

    return ValidationResult(is_valid=not errors, errors=tuple(errors))
