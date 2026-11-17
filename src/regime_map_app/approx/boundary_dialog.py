from __future__ import annotations

from math import isclose

import numpy as np
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..experimental import fuel_flow_from_frequency, read_number


class BoundaryPointsDialog(QDialog):
    def __init__(
        self,
        data: np.ndarray,
        boundary_points: tuple[tuple[float, float, float], ...],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Граничные точки CO")
        self.resize(950, 360)
        self.points = boundary_points
        self._rows = []
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "В столбцах «Крайняя» показаны точки из CSV. "
            "Для новой границы введите расход топлива или частоту с буквой f (например, f32). "
            "Оставьте поле пустым, если эту сторону добавлять не нужно."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        levels = np.unique(data[:, 1])
        table = QTableWidget(len(levels), 6)
        table.setHorizontalHeaderLabels([
            "Пар", "Крайняя слева", "Крайняя справа", "Левая граница", "Правая граница", "CO, ppm",
        ])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        layout.addWidget(table)

        for index, steam in enumerate(levels):
            source = data[data[:, 1] == steam]
            left = source[source[:, 0].argmin()]
            right = source[source[:, 0].argmax()]
            for column, text in enumerate((
                f"{steam:g}", f"{left[0]:.8g} (CO {left[2]:g})", f"{right[0]:.8g} (CO {right[2]:g})",
            )):
                table.setItem(index, column, QTableWidgetItem(text))

            left_edit, right_edit, co_edit = QLineEdit(), QLineEdit(), QLineEdit("500")
            for field in (left_edit, right_edit):
                field.setPlaceholderText("Расход или f")
            for fuel, additive, co in boundary_points:
                if additive == steam:
                    field = left_edit if fuel < left[0] else right_edit
                    field.setText(str(fuel))
                    co_edit.setText(str(co))
            for column, field in enumerate((left_edit, right_edit, co_edit), 3):
                field.setAccessibleName(f"{table.horizontalHeaderItem(column).text()}, пар {steam:g}")
                table.setCellWidget(index, column, field)
            self._rows.append((left, right, left_edit, right_edit, co_edit))

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Применить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        points = []
        for left, right, left_edit, right_edit, co_edit in self._rows:
            steam = left[1]
            if not (left_edit.text().strip() or right_edit.text().strip()):
                continue
            try:
                co = read_number(co_edit.text(), "CO")
                if co < 0:
                    raise ValueError("CO должен быть неотрицательным.")
                for edge, field, direction, side in (
                    (left, left_edit, -1, "Левая"), (right, right_edit, 1, "Правая"),
                ):
                    text = field.text().strip().lower()
                    if not text:
                        continue
                    frequency = text.startswith("f")
                    value = read_number(text.removeprefix("f"), side + " граница")
                    if value <= 0:
                        raise ValueError(f"{side} граница должна быть больше нуля.")
                    fuel = fuel_flow_from_frequency(value) if frequency else value
                    if isclose(fuel, edge[0], rel_tol=0, abs_tol=1e-9) and co == edge[2]:
                        continue
                    if direction * (fuel - edge[0]) <= 1e-9:
                        raise ValueError(f"{side} граница должна лежать за крайней исходной точкой.")
                    points.append((fuel, float(steam), co))
            except ValueError as exc:
                self.error_label.setText(f"Пар {steam:g}. {exc}")
                return

        self.points = tuple(points)
        super().accept()
