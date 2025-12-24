from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from .beam_scan import BeamScan
from .worker import Worker


class Model(QObject):
    scan_data_loaded_sig = Signal(dict)
    load_scan_data_failed_sig = Signal(str)
    create_grid_finished_sig = Signal()
    create_grid_failed_sig = Signal(str)

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
        self.thread_pool = QThreadPool()

    # --- Load Scan Data ---

    def load_scan_data(self) -> None:
        worker = Worker(self.bs.load_scan_data)
        worker.signals.finished.connect(self.load_scan_data_finished)
        worker.signals.error.connect(self.load_scan_data_failed)
        self.thread_pool.start(worker)

    @Slot()
    def load_scan_data_finished(self, completed: bool) -> None:
        if completed:
            stats = self.stats()
            self.scan_data_loaded_sig.emit(stats)

    @Slot()
    def load_scan_data_failed(self, error: str) -> None:
        self.load_scan_data_failed_sig.emit(error)

    # --- Create Grid ---

    def create_grid(self, *args, **kwargs) -> None:
        worker = Worker(self.bs.create_grid, *args, **kwargs)
        worker.signals.finished.connect(self.create_grid_finished)
        worker.signals.error.connect(self.create_grid_failed)
        self.thread_pool.start(worker)

    @Slot()
    def create_grid_finished(self, completed: bool) -> None:
        if completed:
            self.create_grid_finished_sig.emit()

    @Slot()
    def create_grid_failed(self, error: str) -> None:
        self.create_grid_failed_sig.emit(error)

    # --- Stats ---

    def stats(self) -> dict[str, str]:
        serial_number = self.bs.serial_number
        datetime = self.bs.scan_datetime
        test_stand = self.bs.test_stand
        resolution = self.bs.resolution
        step_size = self.bs.step_size
        polarity = self.bs.polarity
        power = self.bs.power
        pressure = self.bs.pressure
        beam_voltage = self.bs.beam_voltage
        ext_voltage = self.bs.extractor_voltage
        beam_supply_current = self.bs.beam_supply_current
        centroid = self.bs.weighted_centroid
        peak_location = self.bs.peak_location
        peak_cup_current = self.bs.peak_cup_current
        hm_contour_area = self.bs.hm_contour_area
        hm_contour_diams = self.bs.hm_contour_diams
        qm_contour_area = self.bs.qm_contour_area
        qm_contour_diams = self.bs.qm_contour_diams
        solenoid_current = self.bs.solenoid_current

        stats: dict[str, str] = {
            'serial_number': serial_number,
            'datetime': datetime,
            'test_stand': test_stand,
            'resolution': resolution,
            'step_size': str(step_size),
            'polarity': polarity,
            'power': str(power),
            'pressure': str(pressure),
            'beam_voltage': str(beam_voltage),
            'ext_voltage': str(ext_voltage),
            'beam_supply_current': str(beam_supply_current),
            'centroid_x': str(centroid[0]),
            'centroid_y': str(centroid[1]),
            'peak_location_x': str(peak_location[0]),
            'peak_location_y': str(peak_location[1]),
            'peak_cup_current': str(peak_cup_current),
            'hm_contour_area': str(hm_contour_area),
            'hm_min_diam': str(hm_contour_diams[0]),
            'hm_max_diam': str(hm_contour_diams[1]),
            'qm_contour_area': str(qm_contour_area),
            'qm_min_diam': str(qm_contour_diams[0]),
            'qm_max_diam': str(qm_contour_diams[1]),
            'solenoid_current': str(solenoid_current),
        }

        return stats
