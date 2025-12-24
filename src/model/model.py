from PySide6.QtCore import QObject, Signal

from .beam_scan import BeamScan


class Model(QObject):
    worker_finished_sig = Signal()

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
