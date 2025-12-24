from PySide6.QtCore import QObject

from .beam_scan import BeamScan


class Model(QObject):
    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
