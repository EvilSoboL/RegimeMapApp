from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)

from .exceptions import DiffSurfaceError
from .models import DiffSurfaceJobConfig, DifferentialSurfaceResult, SurfaceMode
from .pipeline import DiffSurfacePipeline
from .validation import resolve_export_path, validate_job_config
from .visualization import create_figure, render_placeholder, render_result, save_plot
from .worker import DiffSurfaceWorker


class DiffSurfaceModuleWidget(QWidget):
    def __init__(self, pipeline: DiffSurfacePipeline | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline or DiffSurfacePipeline()
        self._selected_input_path: Path | None = None
        self._auto_selected_input_path: Path | None = None
        self._manual_selection = False
        self._thread: QThread | None = None
        self._worker: DiffSurfaceWorker | None = None
        self._last_result: DifferentialSurfaceResult | None = None
        self._busy = False

        self._build_ui()
        self._wire_signals()
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
        plot_layout.addWidget(self.canvas)

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

        self.surface_mode_label = QLabel(SurfaceMode.GRADIENT_MAGNITUDE.label)
        layout.addRow("Тип поверхности:", self.surface_mode_label)
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

    def _wire_signals(self) -> None:
        self.select_input_button.clicked.connect(self._choose_input)
        self.validate_button.clicked.connect(self.validate_data)
        self.run_button.clicked.connect(self.start_processing)
        self.save_button.clicked.connect(self.save_results)

    def current_surface_mode(self) -> SurfaceMode:
        return SurfaceMode.GRADIENT_MAGNITUDE

    def collect_config(self) -> DiffSurfaceJobConfig:
        return DiffSurfaceJobConfig(
            input_path=self._selected_input_path,
            surface_mode=self.current_surface_mode(),
        )

    def refresh_form_state(self) -> None:
        config = self.collect_config()
        validate_ready = validate_job_config(config).is_valid and not self._busy
        run_ready = validate_ready and self.pipeline.validate_inputs(config).is_valid
        self.validate_button.setEnabled(validate_ready)
        self.run_button.setEnabled(run_ready)
        self.save_button.setEnabled(self._last_result is not None and not self._busy)
        self.select_input_button.setEnabled(not self._busy)

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
        self.append_log(f"Режим поверхности: {config.surface_mode.label}")
        self.progress_bar.setValue(0)
        self.status_label.setText("Статус: запуск фонового построения")
        self._set_busy(True)

        self._thread = QThread(self)
        self._worker = DiffSurfaceWorker(self.pipeline, config)
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
        png_path = resolve_export_path(
            output_dir,
            self._last_result.input_path,
            self._last_result.surface_mode,
        )

        try:
            save_plot(self._last_result, png_path)
        except DiffSurfaceError as exc:
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

    def _on_completed(self, result: DifferentialSurfaceResult) -> None:
        self._last_result = result
        render_result(self.figure, result)
        self.append_log(
            f"Построение завершено. Линия минимальной концентрации: a={result.minima_line_fit.slope:.6g}, b={result.minima_line_fit.intercept:.6g}."
        )
        self.append_log(
            f"Левая линия максимумов: a={result.left_line_fit.slope:.6g}, b={result.left_line_fit.intercept:.6g}."
        )
        self.append_log(
            f"Правая линия максимумов: a={result.right_line_fit.slope:.6g}, b={result.right_line_fit.intercept:.6g}."
        )
        self.save_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.append_log(message)

    def _on_cancelled(self) -> None:
        self.append_log("Построение остановлено пользователем.")


class DiffSurfaceModuleWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RegimeMapApp - Модуль дифференциальной поверхности")
        self.resize(1280, 760)
        self.setCentralWidget(DiffSurfaceModuleWidget(parent=self))
