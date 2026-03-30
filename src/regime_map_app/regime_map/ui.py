from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .cmaps import AVAILABLE_CMAP_NAMES, CMAP_REFERENCE_URL, DEFAULT_CMAP_NAME, resolve_cmap_name
from .exceptions import RegimeMapError
from .models import (
    CO_COMPONENT_LABEL,
    DEFAULT_CO_LEVELS,
    DEFAULT_FONT_SIZE,
    DEFAULT_X_AXIS_LABEL,
    DEFAULT_X_LIMITS,
    DEFAULT_Y_AXIS_LABEL,
    DEFAULT_Y_LIMITS,
    RegimeMapJobConfig,
    RegimeMapResult,
)
from .pipeline import RegimeMapPipeline
from .validation import resolve_export_path, validate_job_config
from .visualization import create_figure, render_placeholder, render_result, save_plot
from .worker import RegimeMapWorker


class RegimeMapModuleWidget(QWidget):
    def __init__(self, pipeline: RegimeMapPipeline | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline or RegimeMapPipeline()
        self._selected_input_path: Path | None = None
        self._auto_selected_input_path: Path | None = None
        self._manual_selection = False
        self._thread: QThread | None = None
        self._worker: RegimeMapWorker | None = None
        self._last_result: RegimeMapResult | None = None
        self._busy = False

        self._build_ui()
        self._wire_signals()
        self._refresh_parameter_controls()
        self.refresh_form_state()
        render_placeholder(self.figure, "Результат еще не построен.")

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)

        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.addWidget(self._build_input_group())
        controls_layout.addWidget(self._build_parameters_group())
        controls_layout.addWidget(self._build_actions_group())
        controls_layout.addWidget(self._build_info_group())
        controls_layout.addStretch(1)

        plot_group = QGroupBox("График")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = create_figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        plot_layout.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignTop)
        plot_layout.addStretch(1)

        root_layout.addWidget(controls_widget, 0)
        root_layout.addWidget(plot_group, 1)

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Источник данных")
        layout = QVBoxLayout(group)

        form_layout = QFormLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        form_layout.addRow("CSV-файл:", self.input_path_edit)
        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.select_input_button = QPushButton("Выбрать файл")
        buttons_layout.addWidget(self.select_input_button)
        layout.addLayout(buttons_layout)
        return group

    def _build_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Параметры")
        layout = QFormLayout(group)

        self.show_min_line_checkbox = QCheckBox("Наносить")
        self.show_min_line_checkbox.setChecked(True)
        self.show_right_line_checkbox = QCheckBox("Наносить")
        self.show_right_line_checkbox.setChecked(True)
        self.show_mean_line_checkbox = QCheckBox("Наносить")
        self.show_mean_line_checkbox.setChecked(True)

        self.x_axis_label_edit = QLineEdit(DEFAULT_X_AXIS_LABEL)
        self.y_axis_label_edit = QLineEdit(DEFAULT_Y_AXIS_LABEL)
        self.colorbar_label_edit = QLineEdit(CO_COMPONENT_LABEL)
        self.cmap_combo = self._build_cmap_combo()
        self.cmap_help_button = self._build_cmap_help_button()
        self.cmap_widget = self._build_cmap_widget()

        self.font_size_spin = self._build_int_spin(8, 36, DEFAULT_FONT_SIZE)
        self.font_size_spin.setSingleStep(1)
        self.font_size_spin.setSuffix(" pt")

        self.use_custom_x_limits_checkbox = QCheckBox("Задать")
        self.x_min_spin = self._build_float_spin(*DEFAULT_X_LIMITS)
        self.x_max_spin = self._build_float_spin(*DEFAULT_X_LIMITS, value=DEFAULT_X_LIMITS[1])
        self.x_limits_widget = self._build_range_widget(self.use_custom_x_limits_checkbox, self.x_min_spin, self.x_max_spin)

        self.use_custom_y_limits_checkbox = QCheckBox("Задать")
        self.y_min_spin = self._build_float_spin(*DEFAULT_Y_LIMITS)
        self.y_max_spin = self._build_float_spin(*DEFAULT_Y_LIMITS, value=DEFAULT_Y_LIMITS[1])
        self.y_limits_widget = self._build_range_widget(self.use_custom_y_limits_checkbox, self.y_min_spin, self.y_max_spin)

        self.use_custom_ppm_scale_checkbox = QCheckBox("Задать")
        self.ppm_min_spin = self._build_int_spin(int(DEFAULT_CO_LEVELS[0]), int(DEFAULT_CO_LEVELS[-1]), int(DEFAULT_CO_LEVELS[0]))
        self.ppm_max_spin = self._build_int_spin(int(DEFAULT_CO_LEVELS[0]), 10_000, int(DEFAULT_CO_LEVELS[-1]))
        self.ppm_step_spin = self._build_int_spin(1, 10_000, int(DEFAULT_CO_LEVELS[1] - DEFAULT_CO_LEVELS[0]))
        self.ppm_scale_widget = self._build_ppm_widget()

        layout.addRow("Линия минимума:", self.show_min_line_checkbox)
        layout.addRow("Правая линия максимумов:", self.show_right_line_checkbox)
        layout.addRow("Средняя линия:", self.show_mean_line_checkbox)
        layout.addRow("Подпись оси X:", self.x_axis_label_edit)
        layout.addRow("Подпись оси Y:", self.y_axis_label_edit)
        layout.addRow("Подпись шкалы:", self.colorbar_label_edit)
        layout.addRow("Cmap:", self.cmap_widget)
        layout.addRow("Размер шрифта:", self.font_size_spin)
        layout.addRow("Границы X:", self.x_limits_widget)
        layout.addRow("Границы Y:", self.y_limits_widget)
        layout.addRow("Шкала Z:", self.ppm_scale_widget)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Управление")
        layout = QHBoxLayout(group)

        self.validate_button = QPushButton("Проверить")
        self.run_button = QPushButton("Построить")
        self.save_button = QPushButton("Сохранить")
        self.save_button.setEnabled(False)

        layout.addWidget(self.validate_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.save_button)
        return group

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("Ход выполнения")
        layout = QVBoxLayout(group)

        self.status_label = QLabel("Статус: ожидание")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_edit)
        return group

    def _build_float_spin(self, minimum: float, maximum: float, value: float | None = None) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setValue(minimum if value is None else value)
        return spin

    def _build_int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _build_cmap_combo(self) -> QComboBox:
        combo = QComboBox(self)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.addItems(AVAILABLE_CMAP_NAMES)
        combo.setCurrentText(DEFAULT_CMAP_NAME)
        combo.setMaxVisibleItems(20)
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
        if combo.lineEdit() is not None:
            combo.lineEdit().setPlaceholderText(DEFAULT_CMAP_NAME)
        return combo

    def _build_cmap_help_button(self) -> QToolButton:
        button = QToolButton(self)
        button.setText("?")
        button.setToolTip("Показать справку по доступным color maps")
        return button

    def _build_cmap_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.cmap_combo)
        layout.addWidget(self.cmap_help_button)
        return widget

    def _build_range_widget(self, checkbox: QCheckBox, lower_spin: QDoubleSpinBox, upper_spin: QDoubleSpinBox) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(checkbox)
        layout.addWidget(QLabel("от"))
        layout.addWidget(lower_spin)
        layout.addWidget(QLabel("до"))
        layout.addWidget(upper_spin)
        return widget

    def _build_ppm_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.use_custom_ppm_scale_checkbox)
        layout.addWidget(QLabel("от"))
        layout.addWidget(self.ppm_min_spin)
        layout.addWidget(QLabel("до"))
        layout.addWidget(self.ppm_max_spin)
        layout.addWidget(QLabel("шаг"))
        layout.addWidget(self.ppm_step_spin)
        return widget

    def _wire_signals(self) -> None:
        self.select_input_button.clicked.connect(self._choose_input)
        self.validate_button.clicked.connect(self.validate_data)
        self.run_button.clicked.connect(self.start_processing)
        self.save_button.clicked.connect(self.save_results)
        self.cmap_help_button.clicked.connect(self._show_cmap_help)

        self.show_min_line_checkbox.toggled.connect(self._on_parameters_changed)
        self.show_right_line_checkbox.toggled.connect(self._on_parameters_changed)
        self.show_mean_line_checkbox.toggled.connect(self._on_parameters_changed)
        self.use_custom_x_limits_checkbox.toggled.connect(self._on_parameters_changed)
        self.use_custom_y_limits_checkbox.toggled.connect(self._on_parameters_changed)
        self.use_custom_ppm_scale_checkbox.toggled.connect(self._on_parameters_changed)

        self.x_axis_label_edit.textChanged.connect(self._on_parameters_changed)
        self.y_axis_label_edit.textChanged.connect(self._on_parameters_changed)
        self.colorbar_label_edit.textChanged.connect(self._on_parameters_changed)
        self.cmap_combo.currentTextChanged.connect(self._on_parameters_changed)
        self.font_size_spin.valueChanged.connect(self._on_parameters_changed)
        self.x_min_spin.valueChanged.connect(self._on_parameters_changed)
        self.x_max_spin.valueChanged.connect(self._on_parameters_changed)
        self.y_min_spin.valueChanged.connect(self._on_parameters_changed)
        self.y_max_spin.valueChanged.connect(self._on_parameters_changed)
        self.ppm_min_spin.valueChanged.connect(self._on_parameters_changed)
        self.ppm_max_spin.valueChanged.connect(self._on_parameters_changed)
        self.ppm_step_spin.valueChanged.connect(self._on_parameters_changed)
        if self.cmap_combo.lineEdit() is not None:
            self.cmap_combo.lineEdit().editingFinished.connect(self._normalize_cmap_text)

    def collect_config(self) -> RegimeMapJobConfig:
        return RegimeMapJobConfig(
            input_path=self._selected_input_path,
            is_co_component=True,
            show_min_line=self.show_min_line_checkbox.isChecked(),
            show_right_line=self.show_right_line_checkbox.isChecked(),
            show_mean_line=self.show_mean_line_checkbox.isChecked(),
            use_custom_x_limits=self.use_custom_x_limits_checkbox.isChecked(),
            x_min=self.x_min_spin.value(),
            x_max=self.x_max_spin.value(),
            use_custom_y_limits=self.use_custom_y_limits_checkbox.isChecked(),
            y_min=self.y_min_spin.value(),
            y_max=self.y_max_spin.value(),
            use_custom_ppm_scale=self.use_custom_ppm_scale_checkbox.isChecked(),
            ppm_min=float(self.ppm_min_spin.value()),
            ppm_max=float(self.ppm_max_spin.value()),
            ppm_step=float(self.ppm_step_spin.value()),
            x_axis_label=self.x_axis_label_edit.text(),
            y_axis_label=self.y_axis_label_edit.text(),
            colorbar_label=self.colorbar_label_edit.text(),
            cmap_name=self.cmap_combo.currentText(),
            font_size=self.font_size_spin.value(),
        )

    def refresh_form_state(self) -> None:
        config = self.collect_config()
        validate_ready = validate_job_config(config).is_valid and not self._busy
        run_ready = validate_ready and self.pipeline.validate_inputs(config).is_valid
        self.validate_button.setEnabled(validate_ready)
        self.run_button.setEnabled(run_ready)
        self.save_button.setEnabled(self._last_result is not None and not self._busy)
        self.select_input_button.setEnabled(not self._busy)
        self._refresh_parameter_controls()

    def set_input_path(self, path: Path, *, user_selected: bool) -> None:
        self._selected_input_path = path
        self.input_path_edit.setText(str(path))
        if user_selected:
            self._manual_selection = True
        else:
            self._auto_selected_input_path = path
            self._manual_selection = False
        self._invalidate_result()
        self.refresh_form_state()

    def apply_suggested_input_path(self, path_value: str) -> None:
        if self._busy:
            return
        path = Path(path_value)
        if self._manual_selection and self._selected_input_path != self._auto_selected_input_path:
            return
        self.set_input_path(path, user_selected=False)
        self.append_log(f"Подставлен результат аппроксимации: {path.name}")

    def validate_data(self) -> None:
        config = self.collect_config()
        validation = self.pipeline.validate_inputs(config)
        self.log_edit.clear()
        if validation.is_valid:
            self.append_log(f"Проверка завершена успешно. Подготовлено точек: {validation.checked_points}.")
            self.status_label.setText("Статус: данные готовы к построению")
            return

        for error in validation.errors:
            self.append_log(error)
        self.status_label.setText("Статус: найдены ошибки валидации")

    def start_processing(self) -> None:
        config = self.collect_config()
        validation = validate_job_config(config)
        if not validation.is_valid:
            self.status_label.setText("Статус: запуск отменен")
            for error in validation.errors:
                self.append_log(error)
            return

        input_path = config.input_path
        assert input_path is not None

        self._last_result = None
        self.log_edit.clear()
        self.append_log(f"Файл: {input_path.name}")
        self.append_log(f"Линии: {self._format_line_list(config)}")
        self.append_log(f"Подпись X: {config.x_axis_label.strip() or DEFAULT_X_AXIS_LABEL}")
        self.append_log(f"Подпись Y: {config.y_axis_label.strip() or DEFAULT_Y_AXIS_LABEL}")
        self.append_log(f"Подпись шкалы: {config.colorbar_label.strip() or CO_COMPONENT_LABEL}")
        self.append_log(f"Cmap: {resolve_cmap_name(config.cmap_name) or config.cmap_name.strip() or DEFAULT_CMAP_NAME}")
        self.append_log(f"Размер шрифта: {config.font_size} pt")
        self.append_log(f"Границы X: {self._format_range(config.use_custom_x_limits, config.x_min, config.x_max)}")
        self.append_log(f"Границы Y: {self._format_range(config.use_custom_y_limits, config.y_min, config.y_max)}")
        self.append_log(f"Шкала ppm: {self._format_ppm_scale(config)}")
        self.progress_bar.setValue(0)
        self.status_label.setText("Статус: запуск фонового построения")
        self._set_busy(True)

        self._thread = QThread(self)
        self._worker = RegimeMapWorker(self.pipeline, config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self.append_log)
        self._worker.progress_changed.connect(self.progress_bar.setValue)
        self._worker.status_changed.connect(self._on_status_changed)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def save_results(self) -> None:
        if self._last_result is None:
            return

        directory = QFileDialog.getExistingDirectory(self, "Выбрать директорию для сохранения")
        if not directory:
            return

        output_dir = Path(directory)
        png_path = resolve_export_path(output_dir, self._last_result.input_path)

        try:
            save_plot(self._last_result, png_path)
        except RegimeMapError as exc:
            self.append_log(str(exc))
            self.status_label.setText("Статус: ошибка сохранения")
            return

        self.append_log(f"Сохранен график: {png_path.name}")
        self.status_label.setText("Статус: результаты сохранены")

    def append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    def _choose_input(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV", filter="CSV Files (*.csv)")
        if file_name:
            self.set_input_path(Path(file_name), user_selected=True)

    def _invalidate_result(self) -> None:
        self._last_result = None
        render_placeholder(self.figure, "Результат еще не построен.")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.refresh_form_state()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._thread = None
        self._set_busy(False)

    def _on_status_changed(self, status: str) -> None:
        self.status_label.setText(f"Статус: {status}")

    def _on_completed(self, result: RegimeMapResult) -> None:
        self._last_result = result
        render_result(self.figure, result)
        if result.show_min_line:
            self.append_log(
                f"Построение завершено. Линия минимальной концентрации: a={result.minima_line_fit.slope:.6g}, b={result.minima_line_fit.intercept:.6g}."
            )
        if result.show_right_line:
            self.append_log(
                f"Правая линия максимумов: a={result.right_line_fit.slope:.6g}, b={result.right_line_fit.intercept:.6g}."
            )
        if result.show_mean_line:
            self.append_log(
                f"Средняя линия: a={result.mean_line_fit.slope:.6g}, b={result.mean_line_fit.intercept:.6g}."
            )
        self.save_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.append_log(message)

    def _on_cancelled(self) -> None:
        self.append_log("Построение остановлено пользователем.")

    def _on_parameters_changed(self, *_args) -> None:
        self._invalidate_result()
        self._refresh_parameter_controls()
        self.refresh_form_state()

    def _normalize_cmap_text(self) -> None:
        resolved_name = resolve_cmap_name(self.cmap_combo.currentText())
        if resolved_name is not None and resolved_name != self.cmap_combo.currentText():
            self.cmap_combo.setCurrentText(resolved_name)

    def _refresh_parameter_controls(self) -> None:
        x_enabled = self.use_custom_x_limits_checkbox.isChecked() and not self._busy
        y_enabled = self.use_custom_y_limits_checkbox.isChecked() and not self._busy
        ppm_enabled = self.use_custom_ppm_scale_checkbox.isChecked() and not self._busy

        self.x_min_spin.setEnabled(x_enabled)
        self.x_max_spin.setEnabled(x_enabled)
        self.y_min_spin.setEnabled(y_enabled)
        self.y_max_spin.setEnabled(y_enabled)
        self.ppm_min_spin.setEnabled(ppm_enabled)
        self.ppm_max_spin.setEnabled(ppm_enabled)
        self.ppm_step_spin.setEnabled(ppm_enabled)
        self.show_right_line_checkbox.setEnabled(not self._busy)
        self.show_min_line_checkbox.setEnabled(not self._busy)
        self.show_mean_line_checkbox.setEnabled(not self._busy)
        self.use_custom_x_limits_checkbox.setEnabled(not self._busy)
        self.use_custom_y_limits_checkbox.setEnabled(not self._busy)
        self.use_custom_ppm_scale_checkbox.setEnabled(not self._busy)
        self.x_axis_label_edit.setEnabled(not self._busy)
        self.y_axis_label_edit.setEnabled(not self._busy)
        self.colorbar_label_edit.setEnabled(not self._busy)
        self.cmap_combo.setEnabled(not self._busy)
        self.font_size_spin.setEnabled(not self._busy)

    def _show_cmap_help(self) -> None:
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setWindowTitle("Справка по cmap")
        message_box.setText("Описание доступных color maps есть в справочнике Matplotlib.")
        message_box.setInformativeText(CMAP_REFERENCE_URL)
        open_button = message_box.addButton("Открыть ссылку", QMessageBox.ButtonRole.ActionRole)
        message_box.addButton(QMessageBox.StandardButton.Ok)
        message_box.exec()
        if message_box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl(CMAP_REFERENCE_URL))

    def _format_line_list(self, config: RegimeMapJobConfig) -> str:
        names: list[str] = []
        if config.show_min_line:
            names.append("минимум")
        if config.show_right_line:
            names.append("правая линия максимумов")
        if config.show_mean_line:
            names.append("средняя")
        return ", ".join(names) if names else "не наносить"

    def _format_range(self, use_custom: bool, lower: float, upper: float) -> str:
        if not use_custom:
            return "авто"
        return f"{lower:.4g}..{upper:.4g}"

    def _format_ppm_scale(self, config: RegimeMapJobConfig) -> str:
        if not config.use_custom_ppm_scale:
            return "авто"
        return f"{config.ppm_min:.0f}..{config.ppm_max:.0f}, шаг {config.ppm_step:.0f}"


class RegimeMapModuleWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RegimeMapApp - Модуль режимной карты")
        self.resize(1280, 760)
        self.setCentralWidget(RegimeMapModuleWidget(parent=self))
