import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QEvent, QObject, QRegularExpression, Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QMouseEvent, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet

import src.helpers.helpers as h
from src.model.model import Model
from src.view.override_centroid_window import OverrideCentroidWindow


class MainWindow(QMainWindow):
    load_scan_data_sig = Signal(str)
    plot_3d_surface_sig = Signal(dict)
    plot_heatmap_sig = Signal(dict)
    plot_xy_cross_sections_sig = Signal(dict)
    plot_i_prime_sig = Signal(dict)
    export_to_csv_sig = Signal(str, dict)
    override_centroid_sig = Signal(tuple)
    enable_interp_sig = Signal(bool)
    save_html_figure_sig = Signal(str, dict)
    save_png_figure_sig = Signal(str, dict)
    open_quick_start_guide_sig = Signal()
    folder_path_sig = Signal(str)
    filename_sig = Signal(str)
    file_type_sig = Signal(str)
    titles_sig = Signal(list)

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self.installEventFilter(self)
        self.create_gui()
        self.centroid_coords: list[float] = []

        # Connect events to handlers.
        self.select_csv_button.clicked.connect(self.select_csv_handler)
        self.plot_button.clicked.connect(self.plot_beam_scan_handler)
        self.export_csv_option.triggered.connect(self.export_to_csv_handler)
        self.override_centroid_option.triggered.connect(self.override_centroid_handler)
        self.enable_interp_option.triggered.connect(self.enable_interp_handler)
        self.exit_option.triggered.connect(QApplication.quit)
        self.save_3D_surface_option.triggered.connect(self.save_3d_surface_html_handler)
        self.save_heatmap_option.triggered.connect(self.save_heatmap_html_handler)
        self.save_xy_profiles_option.triggered.connect(
            self.save_xy_cross_section_html_handler
        )
        self.save_i_prime_option.triggered.connect(self.save_i_prime_html_handler)
        self.save_all_html_option.triggered.connect(self.save_all_html_handler)
        self.save_all_png_option.triggered.connect(self.save_all_png_handler)
        self.open_quick_start_guide.triggered.connect(
            self.open_quick_start_guide_handler
        )

        # Connect external Signals to Slots
        self.model.load_scan_data_failed_sig.connect(self.csv_load_error_message)
        self.model.scan_data_loaded_sig.connect(self.update_ui)
        self.model.centroid_coords_sig.connect(self.receive_centroid_coords_sig)
        self.model.create_grid_failed_sig.connect(self.create_grid_failed_error_message)

    # --- Internal Helpers ---

    @property
    def _default_dir(self) -> Path:
        dir = Path(r'C:\\Teststand Data')
        return dir

    def _get_filename_info(self) -> str:
        scan_datetime: str = self.model.bs.scan_datetime
        date_obj = datetime.strptime(scan_datetime, '%m/%d/%Y %I:%M %p')
        scan_datetime = date_obj.strftime('%Y-%m-%d %H_%M')
        serial_number: str = self.serial_number_input.text().strip()
        beam_voltage: str = self.beam_voltage_input.text().strip()
        ext_voltage: str = self.ext_voltage_input.text().strip()
        lens_voltage: str = self.lens_voltage_input.text().strip()
        solenoid_current: str = self.solenoid_current_input.text().strip()
        power: str = self.power_input.text().strip()
        test_stand: str = self.test_stand_input.text().strip()
        filename = f'{scan_datetime} SN-{serial_number} on TS{test_stand} @ {power} W, {beam_voltage},{ext_voltage},{lens_voltage} kV, {solenoid_current} A'
        return filename

    def _make_titles(self, filetype: Literal['html', 'png']) -> list[str]:
        prefix: str = self._get_filename_info()
        titles: list[str] = [
            prefix + ' heatmap.' + filetype,
            prefix + ' ang_int_vs_div_ang.' + filetype,
            prefix + ' suface.' + filetype,
            prefix + ' xy_cross_section.' + filetype,
        ]
        return titles

    def _get_inputs(self) -> dict:
        lower_bound = None
        upper_bound = None
        if self.lower_bound_input.text().strip():
            lower_bound = float(self.lower_bound_input.text().strip())
        if self.upper_bound_input.text().strip():
            upper_bound = float(self.upper_bound_input.text().strip())
        inputs = {
            'serial_number': self.serial_number_input.text().strip(),
            'test_stand': self.test_stand_input.text().strip(),
            'beam_voltage': self.beam_voltage_input.text().strip(),
            'ext_voltage': self.ext_voltage_input.text().strip(),
            'lens_voltage': self.lens_voltage_input.text().strip(),
            'power': self.power_input.text().strip(),
            'solenoid_current': self.solenoid_current_input.text().strip(),
            'z_scale_low': lower_bound,
            'z_scale_high': upper_bound,
            'fcup_diam': float(self.fcup_diameter_input.text().strip()),
            'fcup_dist': float(self.fcup_distance_input.text().strip()),
            'centroid_x': self.centroid_coords[0],
            'centroid_y': self.centroid_coords[1],
        }
        return inputs

    # --- Event handlers ---

    def select_csv_handler(self) -> None:
        dir = str(self._default_dir)
        filepath = h.select_file(dir)
        if not filepath:
            return
        self.load_scan_data_sig.emit(filepath)

    def plot_beam_scan_handler(self) -> None:
        inputs = self._get_inputs()

        if self.surface_cb.isChecked():
            self.plot_3d_surface_sig.emit(inputs)
        if self.heatmap_cb.isChecked():
            self.plot_heatmap_sig.emit(inputs)
        if self.xy_profile_cb.isChecked():
            self.plot_xy_cross_sections_sig.emit(inputs)
        if self.i_prime_cb.isChecked():
            self.plot_i_prime_sig.emit(inputs)

    def export_to_csv_handler(self) -> None:
        dir: Path = self._default_dir
        filename_info = Path(self._get_filename_info() + '.csv')
        default_name = str(dir / filename_info)
        filename = h.get_csv_save_filename(default_name)
        if not filename:
            return
        inputs = {
            'serial_number': self.serial_number_input.text().strip(),
            'beam_voltage': self.beam_voltage_input.text().strip(),
            'ext_voltage': self.ext_voltage_input.text().strip(),
            'lens_voltage': self.lens_voltage_input.text().strip(),
            'solenoid_current': self.solenoid_current_input.text().strip(),
            'power': self.power_input.text().strip(),
            'test_stand': self.test_stand_input.text().strip(),
            'fcup_diam': self.fcup_diameter_input.text().strip(),
            'fcup_dist': self.fcup_distance_input.text().strip(),
        }
        self.export_to_csv_sig.emit(filename, inputs)

    def override_centroid_handler(self) -> None:
        # open the override window
        if self.override_centroid_option.isChecked():
            self.override_centroid_window = OverrideCentroidWindow(
                self, self.centroid_coords
            )
            self.override_centroid_window.centroid_coords_sig.connect(
                self.receive_centroid_coords_sig
            )
            self.override_centroid_window.window_closed_without_input_sig.connect(
                self.receive_window_closed_without_input_sig
            )
            self.override_centroid_window.show()
        else:
            self.centroid_coords = self.model.bs.weighted_centroid

    def enable_interp_handler(self) -> None:
        checked: bool = self.enable_interp_option.isChecked()
        self.enable_interp_sig.emit(checked)

    def save_3d_surface_html_handler(self) -> None:
        dir: Path = self._default_dir
        filename = Path(self._get_filename_info() + ' 3D_surface.html')
        default_name = str(dir / filename)
        filepath: str = h.get_html_save_filename(default_name)
        if not filepath:
            return
        self.filename_sig.emit(filepath)
        self.file_type_sig.emit('html')
        inputs = self._get_inputs()
        which = 'surface'
        self.save_html_figure_sig.emit(which, inputs)

    def save_heatmap_html_handler(self) -> None:
        dir: Path = self._default_dir
        filename = Path(self._get_filename_info() + ' heatmap.html')
        default_name = str(dir / filename)
        filepath: str = h.get_html_save_filename(default_name)
        if not filepath:
            return
        self.filename_sig.emit(filepath)
        self.file_type_sig.emit('html')
        inputs = self._get_inputs()
        which = 'heatmap'
        self.save_html_figure_sig.emit(which, inputs)

    def save_xy_cross_section_html_handler(self) -> None:
        dir: Path = self._default_dir
        filename = Path(self._get_filename_info() + ' xy_cross_sections.html')
        default_name = str(dir / filename)
        filepath: str = h.get_html_save_filename(default_name)
        if not filepath:
            return
        self.filename_sig.emit(filepath)
        self.file_type_sig.emit('html')
        inputs = self._get_inputs()
        which = 'xy_cross_section'
        self.save_html_figure_sig.emit(which, inputs)

    def save_i_prime_html_handler(self) -> None:
        dir: Path = self._default_dir
        filename = Path(self._get_filename_info() + ' ang_int_vs_div_ang.html')
        default_name = str(dir / filename)
        filepath: str = h.get_html_save_filename(default_name)
        if not filepath:
            return
        self.filename_sig.emit(filepath)
        self.file_type_sig.emit('html')
        inputs = self._get_inputs()
        which = 'i_prime'
        self.save_html_figure_sig.emit(which, inputs)

    def save_all_html_handler(self) -> None:
        dir = str(self._default_dir)
        folder_path: str = h.select_folder(dir)
        if not folder_path:
            return
        self.folder_path_sig.emit(folder_path)
        filetype = 'html'
        titles: list[str] = self._make_titles(filetype)
        self.file_type_sig.emit(filetype)
        self.titles_sig.emit(titles)
        inputs = self._get_inputs()
        which = 'all'
        self.save_html_figure_sig.emit(which, inputs)

    def save_all_png_handler(self) -> None:
        dir = str(self._default_dir)
        folder_path: str = h.select_folder(dir)
        if not folder_path:
            return
        self.folder_path_sig.emit(folder_path)
        filetype = 'png'
        titles: list[str] = self._make_titles(filetype)
        self.file_type_sig.emit(filetype)
        self.titles_sig.emit(titles)
        inputs = self._get_inputs()
        which = 'all'
        self.save_png_figure_sig.emit(which, inputs)

    def open_quick_start_guide_handler(self) -> None:
        root_dir = h.get_root_dir()
        filepath = root_dir / 'assets' / 'quick_start_guide.html'
        if not filepath.is_file():
            self.quick_start_guide_error_message(self)
            return
        webbrowser.open_new_tab(filepath.resolve().as_uri())

    # --- General GUI methods ---

    def create_gui(self) -> None:
        input_box_height = 28

        ver = h.get_app_version()
        root_dir: Path = h.get_root_dir()
        icon_path: str = str(root_dir / 'assets' / 'icon.ico')
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle(f'Beam Scan Analysis v{ver}')
        apply_stylesheet(self, theme='dark_lightgreen.xml', invert_secondary=True)
        self.setStyleSheet(
            self.styleSheet() + """QLineEdit, QTextEdit {color: lightgreen;}"""
        )
        self.resize(600, 0)

        # Create the validator for numerical inputs
        number_regex = QRegularExpression(r'^-?\d*\.?\d*$')
        validator = QRegularExpressionValidator(number_regex)
        # '^-?': Matches an optional - at the beginning
        # '\d*': Matches zero or more digits
        # '\.?': Matches an optional decimal point
        # '\d*$': Matches zero or more digits after the decimal point, until the end of the string

        # Create the menu bar
        self.menu_bar = self.menuBar()

        # Create the menu bar items
        self.file_menu = self.menu_bar.addMenu('File')
        self.options_menu = self.menu_bar.addMenu('Options')
        self.save_menu = self.menu_bar.addMenu('Save')
        self.help_menu = self.menu_bar.addMenu('Help')

        # Create the QAction objects for the menus
        self.export_csv_option = QAction('Export Scan Data', self)
        self.export_csv_option.setEnabled(False)
        self.exit_option = QAction('Exit', self)
        self.override_centroid_option = QAction('Override Centroid', self)
        self.override_centroid_option.setCheckable(True)
        self.override_centroid_option.setEnabled(False)
        self.enable_interp_option = QAction('Enable Interpolation', self)
        self.enable_interp_option.setCheckable(True)
        self.enable_interp_option.setEnabled(False)
        self.save_3D_surface_option = QAction('Save 3D Surface', self)
        self.save_3D_surface_option.setEnabled(False)
        self.save_heatmap_option = QAction('Save Heatmap', self)
        self.save_heatmap_option.setEnabled(False)
        self.save_xy_profiles_option = QAction('Save XY-Profiles', self)
        self.save_xy_profiles_option.setEnabled(False)
        self.save_i_prime_option = QAction("Save I' vs Divergence Angle", self)
        self.save_i_prime_option.setEnabled(False)
        self.save_all_html_option = QAction('Save all as HTML', self)
        self.save_all_html_option.setEnabled(False)
        self.save_all_png_option = QAction('Save all as png', self)
        self.save_all_png_option.setEnabled(False)
        self.open_quick_start_guide = QAction('Quick Start Guide', self)

        # Add the action objects to the menu bar items
        self.file_menu.addAction(self.export_csv_option)
        self.file_menu.addAction(self.exit_option)
        self.options_menu.addAction(self.override_centroid_option)
        self.options_menu.addAction(self.enable_interp_option)
        self.save_menu.addAction(self.save_3D_surface_option)
        self.save_menu.addAction(self.save_heatmap_option)
        self.save_menu.addAction(self.save_xy_profiles_option)
        self.save_menu.addAction(self.save_i_prime_option)
        self.save_menu.addAction(self.save_all_html_option)
        self.save_menu.addAction(self.save_all_png_option)
        self.help_menu.addAction(self.open_quick_start_guide)

        # Create data entry fields and labels
        self.serial_number_input = QLineEdit()
        self.serial_number_input.setFixedHeight(input_box_height)
        self.beam_voltage_input = QLineEdit()
        self.beam_voltage_input.setFixedHeight(input_box_height)
        self.beam_voltage_input.setValidator(validator)
        self.ext_voltage_input = QLineEdit()
        self.ext_voltage_input.setFixedHeight(input_box_height)
        self.ext_voltage_input.setValidator(validator)
        self.lens_voltage_input = QLineEdit()
        self.lens_voltage_input.setFixedHeight(input_box_height)
        self.lens_voltage_input.setValidator(validator)
        self.solenoid_current_input = QLineEdit()
        self.solenoid_current_input.setFixedHeight(input_box_height)
        self.solenoid_current_input.setValidator(validator)
        self.power_input = QLineEdit()
        self.power_input.setFixedHeight(input_box_height)
        self.power_input.setValidator(validator)
        self.test_stand_input = QLineEdit()
        self.test_stand_input.setFixedHeight(input_box_height)
        self.upper_bound_input = QLineEdit()
        self.upper_bound_input.setFixedHeight(input_box_height)
        self.upper_bound_input.setValidator(validator)
        self.lower_bound_input = QLineEdit()
        self.lower_bound_input.setFixedHeight(input_box_height)
        self.lower_bound_input.setValidator(validator)
        self.fcup_distance_label = QLabel('Dist. to Cup (mm)')
        self.fcup_distance_input = QLineEdit('205.0')
        self.fcup_distance_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fcup_distance_input.setFixedHeight(input_box_height)
        self.fcup_distance_input.setValidator(validator)
        self.fcup_diameter_label = QLabel('Cup Diam. (mm)')
        self.fcup_diameter_input = QLineEdit('2.5')
        self.fcup_diameter_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fcup_diameter_input.setFixedHeight(input_box_height)
        self.fcup_diameter_input.setValidator(validator)

        # Create buttons to select csv file and analyze beam scan
        self.select_csv_button = QPushButton('Select CSV File')
        self.select_csv_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plot_button = QPushButton('Plot Beam Scan')
        self.plot_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plot_button.setDisabled(True)

        # Create checkboxes to select which plot to show
        self.surface_cb = QCheckBox('3D Surface')
        self.surface_cb.setChecked(True)
        self.heatmap_cb = QCheckBox('Heatmap')
        self.heatmap_cb.setChecked(True)
        self.xy_profile_cb = QCheckBox('X/Y Slices')
        self.xy_profile_cb.setChecked(True)
        self.i_prime_cb = QCheckBox("I' vs Divergence")
        self.i_prime_cb.setChecked(True)

        # Create labels for the statistics display
        self.stat_serial_number = QLabel()
        self.stat_serial_number.setFixedWidth(100)
        self.stat_datetime = QLabel()
        self.stat_resolution = QLabel()
        self.stat_step_size = QLabel()
        self.stat_polarity = QLabel()
        self.stat_power = QLabel()
        self.stat_pressure = QLabel()
        self.stat_beam_voltage = QLabel()
        self.stat_ext_voltage = QLabel()
        self.stat_lens_voltage = QLabel()
        self.stat_beam_supply_current = QLabel()
        self.stat_centroid_location = QLabel()
        self.stat_peak_location = QLabel()
        self.stat_peak_cup_current = QLabel()
        self.stat_peak_total_current = QLabel()
        self.stat_fwhm_area = QLabel()
        self.stat_fwhm_max_diam = QLabel()
        self.stat_fwhm_min_diam = QLabel()
        self.stat_fwqm_area = QLabel()
        self.stat_fwqm_max_diam = QLabel()
        self.stat_fwqm_min_diam = QLabel()

        # Arrange widgets in window
        editables_layout = QFormLayout()
        editables_layout.addRow('Serial Number', self.serial_number_input)
        editables_layout.addRow('Beam Voltage (kV)', self.beam_voltage_input)
        editables_layout.addRow('Extractor Voltage (kV)', self.ext_voltage_input)
        editables_layout.addRow('Lens Voltage (kV)', self.lens_voltage_input)
        editables_layout.addRow('Solenoid Current (A)', self.solenoid_current_input)
        editables_layout.addRow('RF Power (W)', self.power_input)
        editables_layout.addRow('Test Stand', self.test_stand_input)
        editables_layout.addRow('Set Max Z (µA)', self.upper_bound_input)
        editables_layout.addRow('Set Min Z (µA)', self.lower_bound_input)

        g_checkboxes_layout = QGridLayout()
        g_checkboxes_layout.addWidget(self.surface_cb, 0, 0)
        g_checkboxes_layout.addWidget(self.heatmap_cb, 0, 1)
        g_checkboxes_layout.addWidget(self.xy_profile_cb, 1, 0)
        g_checkboxes_layout.addWidget(self.i_prime_cb, 1, 1)

        g_i_prime_layout = QGridLayout()
        g_i_prime_layout.addWidget(self.fcup_diameter_label, 0, 0)
        g_i_prime_layout.addWidget(self.fcup_distance_label, 0, 1)
        g_i_prime_layout.addWidget(self.fcup_diameter_input, 1, 0)
        g_i_prime_layout.addWidget(self.fcup_distance_input, 1, 1)

        v_sub1_main_layout = QVBoxLayout()
        v_sub1_main_layout.addWidget(self.select_csv_button)
        v_sub1_main_layout.addLayout(editables_layout)
        v_sub1_main_layout.addWidget(self.plot_button)
        v_sub1_main_layout.addLayout(g_checkboxes_layout)
        v_sub1_main_layout.addLayout(g_i_prime_layout)

        v_sub2_main_layout = QFormLayout()
        v_sub2_main_layout.addRow('Serial Number: ', self.stat_serial_number)
        v_sub2_main_layout.addRow('Scan Datetime: ', self.stat_datetime)
        v_sub2_main_layout.addRow('Resolution: ', self.stat_resolution)
        v_sub2_main_layout.addRow('Step Size (mm): ', self.stat_step_size)
        v_sub2_main_layout.addRow('Polarity: ', self.stat_polarity)
        v_sub2_main_layout.addRow('Power (W): ', self.stat_power)
        v_sub2_main_layout.addRow('Pressure (mBar): ', self.stat_pressure)
        v_sub2_main_layout.addRow('Beam Voltage (kV): ', self.stat_beam_voltage)
        v_sub2_main_layout.addRow('Ext Voltage (kV): ', self.stat_ext_voltage)
        v_sub2_main_layout.addRow('Lens Voltage (kV):', self.stat_lens_voltage)
        v_sub2_main_layout.addRow(
            'Beam Supply Current (µA): ', self.stat_beam_supply_current
        )
        v_sub2_main_layout.addRow('Centroid Location: ', self.stat_centroid_location)
        v_sub2_main_layout.addRow('Peak Location: ', self.stat_peak_location)
        v_sub2_main_layout.addRow('Peak Cup Current (nA): ', self.stat_peak_cup_current)
        v_sub2_main_layout.addRow(
            'Total Current at Peak (µA): ', self.stat_peak_total_current
        )
        v_sub2_main_layout.addRow('FWHM Area (mm²): ', self.stat_fwhm_area)
        v_sub2_main_layout.addRow('FWHM Max Diam (mm): ', self.stat_fwhm_max_diam)
        v_sub2_main_layout.addRow('FWHM Min Diam (mm): ', self.stat_fwhm_min_diam)
        v_sub2_main_layout.addRow('FWQM Area (mm²): ', self.stat_fwqm_area)
        v_sub2_main_layout.addRow('FWQM Max Diam (mm): ', self.stat_fwqm_max_diam)
        v_sub2_main_layout.addRow('FWQM Min Diam (mm): ', self.stat_fwqm_min_diam)

        # Create a vertical line
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.Shape.VLine)

        h_main_layout = QHBoxLayout()
        h_main_layout.addLayout(v_sub1_main_layout)
        h_main_layout.addWidget(vertical_line)
        h_main_layout.addLayout(v_sub2_main_layout)

        container = QWidget()
        container.setLayout(h_main_layout)

        self.setCentralWidget(container)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Event filter to capture mouse button press events and clear focus from the currently focused widget.

        This method listens for mouse button press events and, when such an event occurs,
        it checks which widget is currently focused. If a widget is focused, it clears
        the focus from that widget.

        Args:
            watched (QObject): The object being watched for events.
            event (QEvent): The event that is being processed, typically a mouse button press event.

        Returns:
            bool: Returns the result of the parent class's eventFilter method, which determines
                whether the event was handled or not.

        Notes:
            This filter is useful when you want to ensure that no widget retains focus
            after a mouse button press, which can help in resetting focus or preventing
            unintended focus-related behaviors in the UI.
        """
        if (
            isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            focused_widget = QApplication.focusWidget()
            if focused_widget is not None:
                focused_widget.clearFocus()
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        # Confirm the user wants to exit the application.
        reply = QMessageBox.question(
            self,
            'Confirmation',
            'Are you sure you want to close the window?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    # --- Slots ---

    @Slot()
    def update_ui(self, stats: dict) -> None:
        # Get rid of useless entries
        for key, value in stats.items():
            if value == 'nan' or value == '0.0':
                stats[key] = ''

        # Update stats
        self.stat_beam_supply_current.setText(stats['beam_supply_current'])
        self.stat_beam_voltage.setText(stats['beam_voltage'])
        self.stat_centroid_location.setText(
            f'({stats["centroid_x"]}, {stats["centroid_y"]})'
        )
        self.stat_datetime.setText(stats['datetime'])
        self.stat_ext_voltage.setText(stats['ext_voltage'])
        self.stat_fwhm_area.setText(stats['hm_contour_area'])
        self.stat_fwhm_max_diam.setText(stats['hm_max_diam'])
        self.stat_fwhm_min_diam.setText(stats['hm_min_diam'])
        self.stat_lens_voltage.setText(stats['lens_voltage'])
        self.stat_peak_cup_current.setText(stats['peak_cup_current'])
        self.stat_peak_location.setText(
            f'({stats["peak_location_x"]}, {stats["peak_location_y"]})'
        )
        self.stat_peak_total_current.setText(stats['peak_total_current'])
        self.stat_polarity.setText(stats['polarity'])
        self.stat_power.setText(stats['power'])
        self.stat_pressure.setText(stats['pressure'])
        self.stat_fwqm_area.setText(stats['qm_contour_area'])
        self.stat_fwqm_max_diam.setText(stats['qm_max_diam'])
        self.stat_fwqm_min_diam.setText(stats['qm_min_diam'])
        self.stat_resolution.setText(stats['resolution'])
        self.stat_serial_number.setText(stats['serial_number'])
        self.stat_step_size.setText(stats['step_size'])

        # Update editable fields
        self.serial_number_input.setText(stats['serial_number'])
        self.beam_voltage_input.setText(stats['beam_voltage'])
        self.ext_voltage_input.setText(stats['ext_voltage'])
        self.lens_voltage_input.setText(stats['lens_voltage'])
        self.solenoid_current_input.setText(stats['solenoid_current'])
        self.power_input.setText(stats['power'])
        self.test_stand_input.setText(stats['test_stand'])
        if stats['fcup_diam']:
            self.fcup_diameter_input.setText(stats['fcup_diam'])
        if stats['fcup_dist']:
            self.fcup_distance_input.setText(stats['fcup_dist'])

        # Activate the Plot Beam Scan button
        if not self.plot_button.isEnabled():
            self.plot_button.setEnabled(True)

        # Activate the export to csv option
        if not self.export_csv_option.isEnabled():
            self.export_csv_option.setEnabled(True)

        # Activate the override centroid option
        if not self.override_centroid_option.isEnabled():
            self.override_centroid_option.setEnabled(True)

        # Uncheck override centroid option when new scan data is loaded
        if self.override_centroid_option.isChecked():
            self.override_centroid_option.setChecked(False)

        # Activate the disable interpolation option
        if not self.enable_interp_option.isEnabled():
            self.enable_interp_option.setEnabled(True)

        # Uncheck disable interpolation option when new data is loaded
        if self.enable_interp_option.isChecked():
            self.enable_interp_option.setChecked(False)

        # Activate the save options
        if not self.save_3D_surface_option.isEnabled():
            self.save_3D_surface_option.setEnabled(True)
        if not self.save_heatmap_option.isEnabled():
            self.save_heatmap_option.setEnabled(True)
        if not self.save_xy_profiles_option.isEnabled():
            self.save_xy_profiles_option.setEnabled(True)
        if not self.save_i_prime_option.isEnabled():
            self.save_i_prime_option.setEnabled(True)
        if not self.save_all_html_option.isEnabled():
            self.save_all_html_option.setEnabled(True)
        if not self.save_all_png_option.isEnabled():
            self.save_all_png_option.setEnabled(True)

    @Slot()
    def receive_centroid_coords_sig(self, coords: list) -> None:
        x = float(coords[0])
        y = float(coords[1])
        self.centroid_coords = [x, y]

    @Slot()
    def receive_window_closed_without_input_sig(self) -> None:
        self.override_centroid_option.setChecked(False)

    # --- Popup Messages ---

    @Slot()
    def csv_load_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'Failed to load beam scan data.\n\nTry another csv file.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def save_html_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nUnable to save html files.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def save_png_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = (
            f'An error occurred.\n\nUnable to save png files.\n\n{error}\n\n{traceback}'
        )
        QMessageBox.critical(self, title, message)

    @Slot()
    def heatmap_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = (
            f'An error occurred.\n\nUnable to plot heatmap.\n\n{error}\n\n{traceback}'
        )
        QMessageBox.critical(self, title, message)

    @Slot()
    def surface_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nUnable to plot 3D surface.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def cross_sections_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nUnable to plot XY cross sections.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def i_prime_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nUnable to plot Angular Intensity cross sections.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def create_grid_failed_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nUnable to create meshgrid.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @Slot()
    def csv_export_error_message(self, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nCould not export CSV. Try selecting another beam scan csv file.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(self, title, message)

    @staticmethod
    def quick_start_guide_error_message(parent) -> None:
        title = 'Error'
        message = 'An error occurred.\n\nUnable to find quick start guide.'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def empty_fcup_inputs_error_message(parent, error, traceback) -> None:
        title = 'Error'
        message = f'An error occurred.\n\nInput distance to cup and cup diameter measurements.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)


if __name__ == '__main__':
    from src.model.beam_scan import BeamScan
    from src.model.model import Model

    app = QApplication([])
    beam_scan = BeamScan()
    model = Model(beam_scan)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())
