from PySide6.QtCore import QObject, QThreadPool, Signal

from .beam_scan import BeamScan
from .worker import Worker


class Model(QObject):
    worker_finished_sig = Signal()

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
        self.thread_pool = QThreadPool()

    def load_scan_data(self) -> None:
        self.worker = Worker(self.bs.load_scan_data)
        self.thread_pool.start(self.worker)

    def create_grid(self, *args, **kwargs) -> None:
        self.worker = Worker(self.bs.create_grid, *args, **kwargs)
        self.thread_pool.start(self.worker)
