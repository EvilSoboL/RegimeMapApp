from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QStyle


CSV_FORMAT_HELP = (
    "Формат CSV\n"
    "Кодировка UTF-8, разделитель — точка с запятой (;).\n"
    "Первая строка: fuel;additive;component\n"
    "fuel — параметр топлива, additive — параметр разбавителя,\n"
    "component — концентрация исследуемого компонента.\n"
    "Значения числовые, дробная часть отделяется точкой.\n"
    "Пример строки: 1.5;0.2;25"
)

APPROX_CSV_HELP = (
    f"{CSV_FORMAT_HELP}\n\n"
    "Шаблон имени: топливо-разбавитель-компонент-дд-мм-гггг.csv\n"
    "Пример: waste_oil-steam-CO-13-03-2026.csv\n"
    "Нужны минимум три разные пары fuel и additive\n"
    "и минимум два разных значения в каждом из этих столбцов."
)

GRID_CSV_HELP = (
    f"{CSV_FORMAT_HELP}\n\n"
    "Подойдёт любое имя с расширением .csv, например\n"
    "approx_waste_oil-steam-CO-13-03-2026.csv.\n"
    "Нужны минимум два разных значения fuel и additive.\n"
    "Каждая комбинация fuel и additive должна встречаться\n"
    "ровно один раз, без пропусков в данных."
)


def csv_source_row(form_layout: QFormLayout, tooltip: str) -> QHBoxLayout:
    icon = QLabel()
    icon.setPixmap(icon.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion).pixmap(16))
    icon.setAccessibleName("Справка о формате CSV")
    icon.setToolTip(tooltip)

    layout = QHBoxLayout()
    layout.addLayout(form_layout, 1)
    layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
    return layout
