from __future__ import annotations

from pathlib import Path

from regime_map_app.approx.models import ApproxJobConfig, InputMode
from regime_map_app.approx.validation import (
    generate_output_filename,
    normalize_input_paths,
    parse_file_metadata,
    validate_job_config,
)


def test_parse_file_metadata_reads_expected_fields() -> None:
    metadata = parse_file_metadata(Path("waste_oil-steam-CO-13-03-2026.csv"))

    assert metadata.fuel_name == "waste_oil"
    assert metadata.diluent == "steam"
    assert metadata.component_name == "CO"
    assert metadata.measured_on.year == 2026


def test_generate_output_filename_adds_prefix() -> None:
    result = generate_output_filename(Path("waste_oil-steam-CO-13-03-2026.csv"))

    assert result == "approx_waste_oil-steam-CO-13-03-2026.csv"


def test_validate_job_config_rejects_negative_median_size(tmp_path: Path) -> None:
    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    input_file.write_text("", encoding="utf-8")
    config = ApproxJobConfig(
        input_mode=InputMode.SINGLE_FILE,
        input_paths=(input_file,),
        output_dir=tmp_path / "out",
        median_size=-1,
    )

    validation = validate_job_config(config)

    assert not validation.is_valid
    assert any("median_size" in error for error in validation.errors)


def test_validate_job_config_rejects_unknown_kernel(tmp_path: Path) -> None:
    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    input_file.write_text("", encoding="utf-8")
    config = ApproxJobConfig(
        input_mode=InputMode.SINGLE_FILE,
        input_paths=(input_file,),
        output_dir=tmp_path / "out",
        kernel="unknown",
    )

    validation = validate_job_config(config)

    assert not validation.is_valid
    assert any("не поддерживается" in error for error in validation.errors)


def test_normalize_input_paths_returns_sorted_csvs_from_folder(tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "b-steam-CO-13-03-2026.csv").write_text("", encoding="utf-8")
    (folder / "a-steam-CO-13-03-2026.csv").write_text("", encoding="utf-8")
    (folder / "ignore.txt").write_text("", encoding="utf-8")

    config = ApproxJobConfig(
        input_mode=InputMode.FOLDER_BATCH,
        input_paths=(folder,),
        output_dir=tmp_path / "out",
    )

    normalized = normalize_input_paths(config)

    assert [path.name for path in normalized] == [
        "a-steam-CO-13-03-2026.csv",
        "b-steam-CO-13-03-2026.csv",
    ]
