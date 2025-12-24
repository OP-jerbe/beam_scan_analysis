from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from .beam_scan import BeamScan
from .worker import Worker


class Model(QObject):
    worker_finished_sig = Signal()

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
        self.thread_pool = QThreadPool()

    @Slot()
    def worker_finished(self) -> None:
        self.worker_finished_sig.emit()

    def load_scan_data(self) -> None:
        self.worker = Worker(self.bs.load_scan_data)
        self.worker.signals.finished.connect(self.worker_finished)
        self.thread_pool.start(self.worker)

    def create_grid(self, *args, **kwargs) -> None:
        self.worker = Worker(self.bs.create_grid, *args, **kwargs)
        self.thread_pool.start(self.worker)
