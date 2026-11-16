from __future__ import annotations

import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from math import isfinite
from pathlib import Path
from statistics import mean
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from .approx.models import CSV_SEPARATOR, REQUIRED_COLUMNS


COMPONENTS = {
    "% O2": "O2",
    "ппм СO": "CO",
    "ппм NOx": "NOx",
    "ппм NO": "NO",
    "% CO2": "CO2",
    "ппм SO2": "SO2",
    "ппм СН": "CH",
    "ппм H2S": "H2S",
}
TIME_HEADER = "Дата / время"
HEADER_TRANSLATION = str.maketrans("СОН", "COH")
PLC_FUEL = "Fuel_mass_scale_1/avMass"
PLC_WATER = "Water_mass_scale_1/avMass"
PLC_ANALOG = "PLC_PRG/rAIN4"
PLC_CHANNELS = (PLC_FUEL, PLC_WATER, PLC_ANALOG)


def _normalize_header(value: object) -> str:
    return "".join(str(value).translate(HEADER_TRANSLATION).split()).casefold()


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except ValueError:
        return None
    return number if isfinite(number) else None


@dataclass(frozen=True)
class GasLog:
    path: Path
    times: tuple[datetime, ...]
    readings: tuple[dict[str, float | None], ...]

    def summarize(
        self, start: datetime, duration: int, tail_seconds: int, tail_records: int | None
    ) -> tuple[dict[str, tuple[float, int]], int]:
        first, stop, _, _ = self.averaging_window(start, duration, tail_seconds, tail_records)
        window = self.readings[first:stop]

        averages = {}
        for component in COMPONENTS.values():
            values = [row[component] for row in window if row[component] is not None]
            if values:
                averages[component] = (mean(values), len(values))
        return averages, len(window)

    def averaging_window(
        self, start: datetime, duration: int, tail_seconds: int, tail_records: int | None
    ) -> tuple[int, int, datetime, datetime]:
        end = start + timedelta(seconds=duration)
        if start < self.times[0] or end > self.times[-1] + timedelta(seconds=1):
            raise ValueError("Интервал режима выходит за пределы выбранного файла.")
        stop = bisect_left(self.times, end)
        if tail_records is None:
            cutoff = max(start, end - timedelta(seconds=tail_seconds))
            first = bisect_left(self.times, cutoff, hi=stop)
        else:
            first = max(bisect_left(self.times, start, hi=stop), stop - tail_records)
        if first == stop:
            raise ValueError("В выбранном окне усреднения нет записей.")
        if tail_records is not None:
            cutoff = self.times[first]
        return first, stop, cutoff, end


@dataclass(frozen=True)
class PlcLog:
    path: Path
    name: str
    times: tuple[datetime, ...]
    readings: tuple[dict[str, float], ...]

    def summarize(self, start: datetime, end: datetime) -> tuple[dict[str, float], int]:
        first = bisect_left(self.times, start)
        stop = bisect_left(self.times, end, lo=first)
        window = self.readings[first:stop]
        averages = {}
        for channel in PLC_CHANNELS:
            values = [row[channel] for row in window if channel in row]
            if values:
                averages[channel] = mean(values)
        return averages, len(window)


def read_plc_log(path: Path) -> PlcLog:
    samples = {}
    names = set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file, delimiter=CSV_SEPARATOR)
            if not {"recorded_at", "plc_name", "payload"}.issubset(rows.fieldnames or ()):
                raise ValueError("Ожидаются столбцы recorded_at, plc_name и payload.")
            for line, row in enumerate(rows, 2):
                try:
                    timestamp = datetime.fromisoformat(row["recorded_at"])
                    if timestamp.tzinfo is not None:
                        raise ValueError("Ожидается местное время без часового пояса.")
                    names.add(row["plc_name"])
                    readings = {}
                    for item in json.loads(row["payload"]):
                        channel = item["path"]
                        if channel in PLC_CHANNELS:
                            value = _number(item["value"])
                            if value is not None:
                                readings[channel] = value
                    if timestamp in samples and samples[timestamp] != readings:
                        raise ValueError("Разные показания с одинаковым временем.")
                    samples[timestamp] = readings
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"Строка {line}: {exc}") from exc
    except (OSError, csv.Error, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать CSV ПЛК «{path.name}»: {exc}") from exc

    if not samples:
        raise ValueError(f"В файле «{path.name}» нет записей ПЛК.")
    if len(names) != 1 or not next(iter(names)):
        raise ValueError("Выберите выгрузку одного ПЛК.")
    times = tuple(sorted(samples))
    return PlcLog(path, next(iter(names)), times, tuple(samples[time] for time in times))


def read_gas_log(path: Path) -> GasLog:
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (OSError, BadZipFile, InvalidFileException, ValueError, KeyError) as exc:
        raise ValueError(f"Не удалось открыть XLSX «{path.name}»: {exc}") from exc

    try:
        rows = workbook.active.values
        header = next(rows, ())
        columns = {_normalize_header(value): index for index, value in enumerate(header)}
        required = (TIME_HEADER, *COMPONENTS)
        missing = [label for label in required if _normalize_header(label) not in columns]
        if missing:
            raise ValueError(f"В файле «{path.name}» отсутствуют столбцы: {', '.join(missing)}.")
        time_index = columns[_normalize_header(TIME_HEADER)]
        component_indices = {key: columns[_normalize_header(label)] for label, key in COMPONENTS.items()}
        samples = []
        for row_number, row in enumerate(rows, 2):
            if all(value is None for value in row):
                continue
            timestamp = row[time_index]
            if not isinstance(timestamp, datetime):
                raise ValueError(f"В файле «{path.name}» некорректное время в строке {row_number}.")
            readings = {key: _number(row[index]) for key, index in component_indices.items()}
            samples.append((timestamp, readings))
    finally:
        workbook.close()

    if not samples:
        raise ValueError(f"В файле «{path.name}» нет записей газового анализа.")
    samples.sort(key=lambda sample: sample[0])
    times, readings = zip(*samples)
    return GasLog(path, times, readings)


def approximation_csv(points: list[tuple[float, float, float]]) -> str:
    coordinates = {(fuel, additive) for fuel, additive, _ in points}
    if len(coordinates) != len(points):
        raise ValueError("В таблице есть точки с одинаковыми значениями fuel и additive.")
    if len(coordinates) < 3:
        raise ValueError("Для аппроксимации нужны минимум три разные пары fuel и additive.")
    if len({fuel for fuel, _ in coordinates}) < 2 or len({additive for _, additive in coordinates}) < 2:
        raise ValueError("Нужны минимум два разных значения fuel и два разных значения additive.")

    output = StringIO()
    writer = csv.writer(output, delimiter=CSV_SEPARATOR, lineterminator="\n")
    writer.writerow(REQUIRED_COLUMNS)
    writer.writerows(points)
    return output.getvalue()
