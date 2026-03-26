from __future__ import annotations

from pathlib import Path

from importlib import import_module

models = import_module("regime_map_app.diff_surface.models")
pipeline_module = import_module("regime_map_app.diff_surface.pipeline")

CSV_SEPARATOR = models.CSV_SEPARATOR
REQUIRED_COLUMNS = models.REQUIRED_COLUMNS
DiffSurfaceJobConfig = models.DiffSurfaceJobConfig
DiffSurfacePipeline = pipeline_module.DiffSurfacePipeline


def _write_csv(path: Path, rows: list[tuple[object, object, object]], header: str | None = None) -> None:
    csv_header = header or CSV_SEPARATOR.join(REQUIRED_COLUMNS)
    lines = [csv_header]
    lines.extend(CSV_SEPARATOR.join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_inputs_reports_wrong_delimiter(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    input_file.write_text("fuel,additive,component\n0,0,1\n0,1,2\n1,0,3\n1,1,4\n", encoding="utf-8")

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == ("Файл surface.csv использует неверный разделитель ','. Ожидается разделитель ';'.",)


def test_validate_inputs_reports_missing_columns(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_csv(input_file, [(0, 0), (0, 1), (1, 0)], header="fuel;additive")

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == ("В файле surface.csv отсутствуют обязательные столбцы: component.",)


def test_validate_inputs_reports_non_numeric_values(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_csv(input_file, [(0, 0, 1), ("bad", 1, 2), (1, 0, 3), (1, 1, 4)])

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == ("В файле surface.csv столбец fuel содержит нечисловые значения.",)


def test_validate_inputs_reports_duplicate_points(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_csv(input_file, [(0, 0, 1), (0, 0, 2), (0, 1, 3), (1, 0, 4), (1, 1, 5)])

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == (
        "В файле surface.csv найдены дублирующиеся точки с одинаковыми fuel и additive.",
    )


def test_validate_inputs_reports_incomplete_regular_grid(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_csv(input_file, [(0, 0, 1), (0, 1, 2), (1, 0, 3)])

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == ("В файле surface.csv данные не образуют полную регулярную сетку.",)


def test_validate_inputs_reports_insufficient_unique_axis_values(tmp_path: Path) -> None:
    input_file = tmp_path / "surface.csv"
    _write_csv(input_file, [(0, 0, 1), (0, 1, 2), (0, 2, 3)])

    validation = DiffSurfacePipeline().validate_inputs(DiffSurfaceJobConfig(input_path=input_file))

    assert not validation.is_valid
    assert validation.errors == ("В файле surface.csv недостаточно уникальных значений fuel или additive.",)
