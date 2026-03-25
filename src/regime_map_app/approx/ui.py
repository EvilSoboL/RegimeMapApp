from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .models import ALLOWED_KERNELS, ApproxJobConfig, BatchProcessSummary, InputMode
from .pipeline import ApproxPipeline
from .validation import generate_output_filename, validate_job_config
from .worker import ApproxWorker


class ApproxModuleWidget(QWidget):
    def __init__(self, pipeline: ApproxPipeline | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline or ApproxPipeline()
        self._selected_input_paths: tuple[Path, ...] = ()
        self._thread: QThread | None = None
        self._worker: ApproxWorker | None = None
        self._last_summary: BatchProcessSummary | None = None
        self._busy = False

        self._build_ui()
        self._wire_signals()
        self._update_mode_state()
        self.refresh_form_state()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self._build_input_group())
        root_layout.addWidget(self._build_output_group())
        root_layout.addWidget(self._build_parameters_group())
        root_layout.addWidget(self._build_actions_group())
        root_layout.addWidget(self._build_info_group())

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Источник данных")
        layout = QVBoxLayout(group)

        form_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Одиночный файл", InputMode.SINGLE_FILE)
        self.mode_combo.addItem("Папка", InputMode.FOLDER_BATCH)
        form_layout.addRow("Режим:", self.mode_combo)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        form_layout.addRow("Выбор:", self.input_path_edit)
        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.select_input_button = QPushButton("Выбрать")
        buttons_layout.addWidget(self.select_input_button)
        layout.addLayout(buttons_layout)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Сохранение результата")
        layout = QVBoxLayout(group)

        form_layout = QFormLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        form_layout.addRow("Директория:", self.output_dir_edit)

        self.auto_output_name_checkbox = QCheckBox("Автоимя выходного файла")
        self.auto_output_name_checkbox.setChecked(True)
        form_layout.addRow("", self.auto_output_name_checkbox)

        self.output_name_edit = QLineEdit()
        form_layout.addRow("Имя файла:", self.output_name_edit)
        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.select_output_dir_button = QPushButton("Выбрать директорию")
        buttons_layout.addWidget(self.select_output_dir_button)
        layout.addLayout(buttons_layout)
        return group

    def _build_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Параметры аппроксимации")
        layout = QFormLayout(group)

        self.resolution_x_spin = QSpinBox()
        self.resolution_x_spin.setRange(1, 10_000)
        self.resolution_x_spin.setValue(100)
        layout.addRow("Разрешение оси x:", self.resolution_x_spin)

        self.resolution_y_spin = QSpinBox()
        self.resolution_y_spin.setRange(1, 10_000)
        self.resolution_y_spin.setValue(100)
        layout.addRow("Разрешение оси y:", self.resolution_y_spin)

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(ALLOWED_KERNELS)
        self.kernel_combo.setCurrentText("linear")
        layout.addRow("kernel:", self.kernel_combo)

        self.median_size_spin = QSpinBox()
        self.median_size_spin.setRange(0, 999)
        self.median_size_spin.setValue(20)
        layout.addRow("median_size:", self.median_size_spin)

        self.clamp_zero_checkbox = QCheckBox("Обнулять отрицательные значения")
        self.clamp_zero_checkbox.setChecked(True)
        layout.addRow("", self.clamp_zero_checkbox)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Управление")
        layout = QHBoxLayout(group)

        self.validate_button = QPushButton("Проверить данные")
        self.run_button = QPushButton("Запустить обработку")
        self.open_result_button = QPushButton("Открыть результат")
        self.open_result_button.setEnabled(False)
        self.open_folder_button = QPushButton("Открыть папку результата")
        self.open_folder_button.setEnabled(False)

        layout.addWidget(self.validate_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.open_result_button)
        layout.addWidget(self.open_folder_button)
        return group

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("Ход выполнения")
        layout = QVBoxLayout(group)

        self.current_file_label = QLabel("Текущий файл: -")
        self.status_label = QLabel("Статус: ожидание")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        layout.addWidget(self.current_file_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_edit)
        return group

    def _wire_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._update_mode_state)
        self.mode_combo.currentIndexChanged.connect(self.refresh_form_state)
        self.auto_output_name_checkbox.toggled.connect(self._update_output_name_state)
        self.auto_output_name_checkbox.toggled.connect(self.refresh_form_state)
        self.select_input_button.clicked.connect(self._choose_input)
        self.select_output_dir_button.clicked.connect(self._choose_output_dir)
        self.output_name_edit.textChanged.connect(self.refresh_form_state)
        self.resolution_x_spin.valueChanged.connect(self.refresh_form_state)
        self.resolution_y_spin.valueChanged.connect(self.refresh_form_state)
        self.kernel_combo.currentTextChanged.connect(self.refresh_form_state)
        self.median_size_spin.valueChanged.connect(self.refresh_form_state)
        self.validate_button.clicked.connect(self.validate_data)
        self.run_button.clicked.connect(self.start_processing)
        self.open_result_button.clicked.connect(self.open_result)
        self.open_folder_button.clicked.connect(self.open_output_folder)

    def current_input_mode(self) -> InputMode:
        mode = self.mode_combo.currentData()
        if isinstance(mode, InputMode):
            return mode
        return InputMode(mode)

    def collect_config(self) -> ApproxJobConfig:
        output_dir_text = self.output_dir_edit.text().strip()
        output_dir = Path(output_dir_text) if output_dir_text else None
        output_filename = self.output_name_edit.text().strip() or None
        auto_output_name = self.auto_output_name_checkbox.isChecked() or not self.current_input_mode().is_single
        return ApproxJobConfig(
            input_mode=self.current_input_mode(),
            input_paths=self._selected_input_paths,
            output_dir=output_dir,
            output_filename=output_filename,
            resolution_x=self.resolution_x_spin.value(),
            resolution_y=self.resolution_y_spin.value(),
            kernel=self.kernel_combo.currentText(),
            median_size=self.median_size_spin.value(),
            clamp_zero=self.clamp_zero_checkbox.isChecked(),
            auto_output_name=auto_output_name,
        )

    def set_input_paths(self, paths: list[Path] | tuple[Path, ...]) -> None:
        self._selected_input_paths = tuple(paths)
        self._sync_output_dir_from_input()
        self._sync_input_summary()
        self._sync_output_name_preview()
        self.refresh_form_state()

    def set_output_dir(self, path: Path) -> None:
        self.output_dir_edit.setText(str(path))
        self.refresh_form_state()

    def refresh_form_state(self) -> None:
        config = self.collect_config()
        validate_ready = validate_job_config(config, require_output_dir=False).is_valid and not self._busy
        run_ready = validate_job_config(config, require_output_dir=True).is_valid and not self._busy
        self.validate_button.setEnabled(validate_ready)
        self.run_button.setEnabled(run_ready)
        self._update_output_name_state()

    def validate_data(self) -> None:
        config = self.collect_config()
        validation = self.pipeline.validate_inputs(config)
        self.log_edit.clear()
        if validation.is_valid:
            self.append_log(f"Проверка завершена успешно. Проверено файлов: {validation.checked_files}.")
            self.status_label.setText("Статус: данные готовы к обработке")
            return

        for error in validation.errors:
            self.append_log(error)
        self.status_label.setText("Статус: найдены ошибки валидации")

    def start_processing(self) -> None:
        config = self.collect_config()
        run_validation = validate_job_config(config, require_output_dir=True)
        if not run_validation.is_valid:
            self.status_label.setText("Статус: запуск отменён")
            for error in run_validation.errors:
                self.append_log(error)
            return

        validation = self.pipeline.validate_inputs(config) if config.input_mode.is_single else None
        if validation is not None and not validation.is_valid:
            self.status_label.setText("Статус: запуск отменён")
            for error in validation.errors:
                self.append_log(error)
            return

        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.current_file_label.setText("Текущий файл: -")
        self.status_label.setText("Статус: запуск фоновой обработки")
        self._set_busy(True)

        self._thread = QThread(self)
        self._worker = ApproxWorker(self.pipeline, config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self.append_log)
        self._worker.progress_changed.connect(self.progress_bar.setValue)
        self._worker.current_file_changed.connect(self._on_current_file_changed)
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

    def append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    def open_result(self) -> None:
        if self._last_summary is None:
            return
        target = self._last_summary.last_output_path
        if target is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def open_output_folder(self) -> None:
        if self._last_summary is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_summary.output_dir)))

    def _choose_input(self) -> None:
        mode = self.current_input_mode()
        if mode is InputMode.SINGLE_FILE:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV", filter="CSV Files (*.csv)")
            if file_name:
                self.set_input_paths([Path(file_name)])
            return

        directory = QFileDialog.getExistingDirectory(self, "Выбрать папку с CSV")
        if directory:
            self.set_input_paths([Path(directory)])

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Выбрать директорию для результата")
        if directory:
            self.set_output_dir(Path(directory))

    def _sync_input_summary(self) -> None:
        mode = self.current_input_mode()
        if not self._selected_input_paths:
            self.input_path_edit.clear()
            return
        if mode is InputMode.SINGLE_FILE:
            self.input_path_edit.setText(str(self._selected_input_paths[0]))
            return

        self.input_path_edit.setText(str(self._selected_input_paths[0]))

    def _sync_output_dir_from_input(self) -> None:
        if not self._selected_input_paths:
            return

        source_path = self._selected_input_paths[0]
        output_dir = source_path.parent if self.current_input_mode().is_single else source_path
        self.output_dir_edit.setText(str(output_dir))

    def _sync_output_name_preview(self) -> None:
        mode = self.current_input_mode()
        if mode.is_single and self._selected_input_paths:
            next_value = self.output_name_edit.text()
            if self.auto_output_name_checkbox.isChecked() or not next_value.strip():
                next_value = generate_output_filename(self._selected_input_paths[0])
            if self.output_name_edit.text() != next_value:
                self.output_name_edit.setText(next_value)
        elif not mode.is_single:
            next_value = "Имена формируются автоматически"
            if self.output_name_edit.text() != next_value:
                self.output_name_edit.setText(next_value)

    def _update_mode_state(self) -> None:
        mode = self.current_input_mode()
        self.select_input_button.setEnabled(not self._busy)
        self.auto_output_name_checkbox.setEnabled(mode.is_single and not self._busy)
        self._sync_input_summary()
        self._sync_output_name_preview()
        self._update_output_name_state()

    def _update_output_name_state(self) -> None:
        mode = self.current_input_mode()
        editable = mode.is_single and not self.auto_output_name_checkbox.isChecked() and not self._busy
        self.output_name_edit.setEnabled(editable)
        if mode.is_single:
            self._sync_output_name_preview()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.mode_combo.setEnabled(not busy)
        self.select_input_button.setEnabled(not busy)
        self.select_output_dir_button.setEnabled(not busy)
        self.auto_output_name_checkbox.setEnabled(not busy and self.current_input_mode().is_single)
        self.output_name_edit.setEnabled(
            not busy and self.current_input_mode().is_single and not self.auto_output_name_checkbox.isChecked()
        )
        self.validate_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._thread = None
        self._set_busy(False)
        self.refresh_form_state()

    def _on_current_file_changed(self, file_name: str) -> None:
        self.current_file_label.setText(f"Текущий файл: {file_name}")

    def _on_status_changed(self, status: str) -> None:
        self.status_label.setText(f"Статус: {status}")

    def _on_completed(self, summary: BatchProcessSummary) -> None:
        self._last_summary = summary
        self.open_folder_button.setEnabled(True)
        self.open_result_button.setEnabled(summary.last_output_path is not None)
        self.append_log(
            f"Итог: обработано {summary.total_files}, успешно {summary.succeeded}, с ошибками {summary.failed}."
        )

    def _on_failed(self, message: str) -> None:
        self.append_log(message)

    def _on_cancelled(self) -> None:
        self.append_log("Обработка остановлена пользователем.")


class ApproxModuleWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RegimeMapApp - Модуль аппроксимации")
        self.resize(1000, 720)
        self.setCentralWidget(ApproxModuleWidget(parent=self))
