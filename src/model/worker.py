import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    # This survives even after the Worker is deleted
    finished = Signal(bool)
    error = Signal(str, str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  # Create the signaler

    @Slot()
    def run(self) -> None:
        complete = False
        try:
            self.fn(*self.args, **self.kwargs)
            complete = True
        except Exception as e:
            self.signals.error.emit(str(e), traceback.print_exc)
        finally:
            self.signals.finished.emit(complete)
