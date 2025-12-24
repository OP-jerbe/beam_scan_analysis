from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    # This survives even after the Worker is deleted
    finished = Signal()
    # You could also add error = Signal(str) or result = Signal(object)


class Worker(QRunnable, QObject):
    finished_sig = Signal()

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  # Create the signaler

    @Slot()
    def run(self) -> None:
        try:
            self.fn(*self.args, **self.kwargs)
        finally:
            self.signals.finished.emit()
