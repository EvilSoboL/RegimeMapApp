from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .exceptions import ApproximationError, CancellationError
from .models import ApproxJobConfig
from .pipeline import ApproxPipeline


class ApproxWorker(QObject):
    log_message = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int)
    current_file_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, pipeline: ApproxPipeline, config: ApproxJobConfig) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._config = config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit("Подготовка к обработке...")
            self.progress_changed.emit(0)
            summary = self._pipeline.process_job(
                self._config,
                on_log=self.log_message.emit,
                on_progress=self._emit_progress,
                on_current_file=self._emit_current_file,
                should_cancel=self._should_cancel,
            )
        except CancellationError:
            self.status_changed.emit("Обработка остановлена.")
            self.cancelled.emit()
        except ApproximationError as exc:
            self.status_changed.emit("Обработка завершилась с ошибкой.")
            self.failed.emit(str(exc))
        except Exception as exc:
            self.status_changed.emit("Обработка завершилась с непредвиденной ошибкой.")
            self.failed.emit(f"Непредвиденная ошибка: {exc}")
        else:
            self.progress_changed.emit(100)
            self.status_changed.emit(
                f"Готово: успешно {summary.succeeded}, с ошибками {summary.failed}."
            )
            self.completed.emit(summary)

    def _emit_progress(self, current: int, total: int) -> None:
        percent = 0 if total == 0 else int(current / total * 100)
        self.progress_changed.emit(percent)

    def _emit_current_file(self, path) -> None:
        self.current_file_changed.emit(path.name)

    def _should_cancel(self) -> bool:
        return self._cancel_requested
