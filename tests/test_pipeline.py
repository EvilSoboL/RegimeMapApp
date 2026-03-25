from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_map_app.approx.models import ApproxJobConfig, InputMode
from regime_map_app.approx.pipeline import ApproxPipeline


def _write_valid_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "fuel;;additive;;component",
                "0;;0;;1",
                "0;;1;;2",
                "1;;0;;3",
                "1;;1;;4",
                "0.5;;0.5;;2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_process_single_file_creates_output_csv(tmp_path: Path) -> None:
    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    output_dir = tmp_path / "out"
    _write_valid_csv(input_file)
    pipeline = ApproxPipeline()
    config = ApproxJobConfig(
        input_mode=InputMode.SINGLE_FILE,
        input_paths=(input_file,),
        output_dir=output_dir,
        resolution_x=5,
        resolution_y=4,
        median_size=0,
        clamp_zero=False,
    )

    summary = pipeline.process_job(config)

    assert summary.succeeded == 1
    assert summary.failed == 0
    exported_path = output_dir / "approx_waste_oil-steam-CO-13-03-2026.csv"
    assert exported_path.exists()
    frame = pd.read_csv(exported_path, sep=";;", engine="python")
    assert list(frame.columns) == ["fuel", "additive", "component"]
    assert len(frame.index) == 20


def test_validate_inputs_reports_wrong_delimiter(tmp_path: Path) -> None:
    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    input_file.write_text(
        "\n".join(
            [
                "fuel;additive;component",
                "0;0;1",
                "0;1;2",
                "1;0;3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = ApproxJobConfig(
        input_mode=InputMode.SINGLE_FILE,
        input_paths=(input_file,),
        output_dir=tmp_path / "out",
    )

    validation = ApproxPipeline().validate_inputs(config)

    assert not validation.is_valid
    assert any("разделитель" in error for error in validation.errors)


def test_batch_processing_continues_after_bad_file(tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    good_file = folder / "waste_oil-steam-CO-13-03-2026.csv"
    bad_file = folder / "waste_oil-steam-CO2-13-03-2026.csv"
    _write_valid_csv(good_file)
    bad_file.write_text(
        "\n".join(["fuel;;additive", "0;;0", "0;;1", "1;;0"]) + "\n",
        encoding="utf-8",
    )
    config = ApproxJobConfig(
        input_mode=InputMode.FOLDER_BATCH,
        input_paths=(folder,),
        output_dir=tmp_path / "out",
        resolution_x=4,
        resolution_y=4,
        median_size=0,
    )

    summary = ApproxPipeline().process_job(config)

    assert summary.total_files == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert (tmp_path / "out" / "approx_waste_oil-steam-CO-13-03-2026.csv").exists()


def test_validate_inputs_reports_non_numeric_values(tmp_path: Path) -> None:
    input_file = tmp_path / "waste_oil-steam-CO-13-03-2026.csv"
    input_file.write_text(
        "\n".join(
            [
                "fuel;;additive;;component",
                "0;;0;;1",
                "bad;;1;;2",
                "1;;0;;3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = ApproxJobConfig(
        input_mode=InputMode.SINGLE_FILE,
        input_paths=(input_file,),
        output_dir=tmp_path / "out",
    )

    validation = ApproxPipeline().validate_inputs(config)

    assert not validation.is_valid
    assert any("нечисловые" in error for error in validation.errors)
