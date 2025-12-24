from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from .beam_scan import BeamScan
from .worker import Worker


class Model(QObject):
    scan_data_loaded_sig = Signal()
    create_grid_finished_sig = Signal()

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
        self.thread_pool = QThreadPool()

    def load_scan_data(self) -> None:
        self.worker = Worker(self.bs.load_scan_data)
        self.worker.signals.finished.connect(self.load_scan_data_finished)
        self.thread_pool.start(self.worker)

    @Slot()
    def load_scan_data_finished(self) -> None:
        self.scan_data_loaded_sig.emit()

    def create_grid(self, *args, **kwargs) -> None:
        self.worker = Worker(self.bs.create_grid, *args, **kwargs)
        self.worker.signals.finished.connect(self.create_grid_finished)
        self.thread_pool.start(self.worker)

    @Slot()
    def create_grid_finished(self) -> None:
        self.create_grid_finished_sig.emit()
