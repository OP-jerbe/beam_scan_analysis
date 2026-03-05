import pandas as pd
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from .beam_scan import BeamScan
from .worker import Worker


class Model(QObject):
    scan_data_loaded_sig = Signal(dict)
    load_scan_data_failed_sig = Signal(str, str)
    create_grid_finished_sig = Signal()
    create_grid_failed_sig = Signal(str)
    centroid_coords_sig = Signal(list)

    def __init__(self, beam_scan: BeamScan) -> None:
        super().__init__()
        self.bs = beam_scan
        self.thread_pool = QThreadPool()

    # --- Load Scan Data ---

    def load_scan_data(self, filepath: str) -> None:
        worker = Worker(self.bs.load_scan_data, filepath=filepath)
        worker.signals.finished.connect(self.load_scan_data_finished)
        worker.signals.error.connect(self.load_scan_data_failed)
        self.thread_pool.start(worker)

    @Slot()
    def load_scan_data_finished(self, completed: bool) -> None:
        if completed:
            stats = self.stats()
            self.scan_data_loaded_sig.emit(stats)
            centroid_coords = [stats['centroid_x'], stats['centroid_y']]
            self.centroid_coords_sig.emit(centroid_coords)

    @Slot()
    def load_scan_data_failed(self, error: str, traceback: str) -> None:
        self.load_scan_data_failed_sig.emit(error, traceback)

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
        lens_voltage = self.bs.lens_voltage
        beam_supply_current = self.bs.beam_supply_current
        centroid = self.bs.weighted_centroid
        peak_location = self.bs.peak_location
        peak_cup_current = self.bs.peak_cup_current
        peak_total_current = self.bs.peak_total_current
        hm_contour_area = self.bs.hm_contour_area
        hm_contour_diams = self.bs.hm_contour_diams
        qm_contour_area = self.bs.qm_contour_area
        qm_contour_diams = self.bs.qm_contour_diams
        solenoid_current = self.bs.solenoid_current
        fcup_diam = self.bs.fcup_diameter
        fcup_dist = self.bs.fcup_distance

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
            'lens_voltage': str(lens_voltage),
            'beam_supply_current': str(beam_supply_current),
            'centroid_x': str(centroid[0]),
            'centroid_y': str(centroid[1]),
            'peak_location_x': str(peak_location[0]),
            'peak_location_y': str(peak_location[1]),
            'peak_cup_current': str(peak_cup_current),
            'peak_total_current': str(peak_total_current),
            'hm_contour_area': str(hm_contour_area),
            'hm_min_diam': str(hm_contour_diams[0]),
            'hm_max_diam': str(hm_contour_diams[1]),
            'qm_contour_area': str(qm_contour_area),
            'qm_min_diam': str(qm_contour_diams[0]),
            'qm_max_diam': str(qm_contour_diams[1]),
            'solenoid_current': str(solenoid_current),
            'fcup_diam': str(fcup_diam),
            'fcup_dist': str(fcup_dist),
        }

        return stats

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

    # --- CSV Export ---

    @Slot()
    def export_to_csv(self, filename: str, inputs: dict) -> None:
        scan_datetime: str = self.bs.scan_datetime
        # date_obj = datetime.strptime(scan_datetime, '%m/%d/%Y %I:%M %p')
        # scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
        step_size: float = self.bs.step_size
        serial_number: str = inputs['serial_number']
        beam_voltage: str = inputs['beam_voltage']
        ext_voltage: str = inputs['ext_voltage']
        lens_voltage: str = inputs['lens_voltage']
        solenoid_current: str = inputs['solenoid_current']
        power: str = inputs['power']
        test_stand: str = inputs['test_stand']
        beam_supply_current: float = self.bs.beam_supply_current
        pressure: float = self.bs.pressure
        fcup_diam: str = inputs['fcup_diam']
        fcup_dist: str = inputs['fcup_dist']
        data = pd.DataFrame(
            {
                'X': self.bs.x_location,
                'Y': self.bs.y_location,
                'cup_current': self.bs.cup_current * 1e-9,  # convert from nA to A
                'screen_current': self.bs.screen_current * 1e-6,  # convert from uA to A
                'total_current': self.bs.total_current * 1e-6,  # convert from uA to A
            }
        )

        with open(filename, 'w') as f:
            f.write('CSV export version,4\n')
            f.write(f'Serial Number,{serial_number}\n')
            f.write(f'Scan Datetime,{scan_datetime}\n')
            f.write(f'Step Size (mm),{step_size}\n')
            f.write(f'Beam Voltage (kV),{beam_voltage}\n')
            f.write(f'Extractor Voltage (kV),{ext_voltage}\n')
            f.write(f'Lens Voltage (kV),{lens_voltage}\n')
            f.write(f'Solenoid Current (A),{solenoid_current}\n')
            f.write(f'Test Stand,{test_stand}\n')
            f.write(f'Beam Supply Current (uA),{beam_supply_current}\n')
            f.write(f'Pressure (mBar),{pressure}\n')
            f.write(f'F-Cup Distance (mm),{fcup_dist}\n')
            f.write(f'F-Cup Diameter (mm),{fcup_diam}\n')
            f.write(f'Power (W),{power}\n')

        data.to_csv(filename, index=False, mode='a')  # append the data
