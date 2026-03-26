from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .exceptions import CancellationError, DiffSurfaceError
from .models import DiffSurfaceJobConfig
from .pipeline import DiffSurfacePipeline


class DiffSurfaceWorker(QObject):
    log_message = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, pipeline: DiffSurfacePipeline, config: DiffSurfaceJobConfig) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._config = config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit("Подготовка к построению...")
            self.progress_changed.emit(0)
            result = self._pipeline.process_job(
                self._config,
                on_log=self.log_message.emit,
                on_progress=self.progress_changed.emit,
                should_cancel=self._should_cancel,
            )
        except CancellationError:
            self.status_changed.emit("Построение остановлено.")
            self.cancelled.emit()
        except DiffSurfaceError as exc:
            self.status_changed.emit("Построение завершилось с ошибкой.")
            self.failed.emit(str(exc))
        except Exception as exc:
            self.status_changed.emit("Построение завершилось с непредвиденной ошибкой.")
            self.failed.emit(f"Непредвиденная ошибка: {exc}")
        else:
            self.progress_changed.emit(100)
            self.status_changed.emit("Готово: дифференциальная поверхность построена.")
            self.completed.emit(result)

    def _should_cancel(self) -> bool:
        return self._cancel_requested
