import sys
import numpy as np
from PySide6.QtCore import QEvent, Qt, QObject, QRegularExpression
from PySide6.QtGui import QIcon, QMouseEvent, QRegularExpressionValidator, QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QCheckBox
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QFrame, QFileDialog
from qt_material import apply_stylesheet

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.installEventFilter(self)
        self.create_gui()
        
    def create_gui(self) -> None:
        input_box_width = 130
        input_box_height = 28
        stat_label_width = 250
        stat_label_height = 28
        window_width = 550
        window_height = 500

        self.setFixedSize(window_width,window_height)
        if hasattr(sys, 'frozen'):  # Check if running from the bundled app
            icon_path = sys._MEIPASS + '/scan.ico'  # type: ignore
        else:
            icon_path = 'scan.ico'  # Use the local icon file in dev mode
        self.setWindowIcon(QIcon(icon_path))
        apply_stylesheet(self, theme='dark_lightgreen.xml', invert_secondary=True)
        self.setStyleSheet(self.styleSheet() + """QLineEdit, QTextEdit {color: lightgreen;}""")

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
        self.save_menu = self.menu_bar.addMenu('Save')
        self.help_menu = self.menu_bar.addMenu('Help')

        # Create the QAction objects for the menus
        self.export_csv_option = QAction('Export Scan Data', self)
        self.exit_option = QAction('Exit', self)
        self.save_3D_surface_option = QAction('Save 3D Surface', self)
        self.save_heatmap_option = QAction('Save Heatmap', self)
        self.save_xy_profiles_option = QAction('Save XY-Profiles', self)
        self.save_i_prime_option = QAction("Save I' vs Divergence Angle", self)
        self.open_quick_start_guide = QAction('Quick Start Guide', self)

        # Add the action objects to the menu bar items
        self.file_menu.addAction(self.export_csv_option)
        self.file_menu.addAction(self.exit_option)
        self.save_menu.addAction(self.save_3D_surface_option)
        self.save_menu.addAction(self.save_heatmap_option)
        self.save_menu.addAction(self.save_xy_profiles_option)
        self.save_menu.addAction(self.save_i_prime_option)
        self.help_menu.addAction(self.open_quick_start_guide)

        # Create data entry fields and labels
        self.serial_number_label = QLabel('Serial Number')
        self.serial_number_input = QLineEdit()
        self.serial_number_input.setFixedSize(input_box_width, input_box_height)
        self.beam_voltage_label = QLabel('Beam Voltage (kV)')
        self.beam_voltage_input = QLineEdit()
        self.beam_voltage_input.setFixedSize(input_box_width, input_box_height)
        self.beam_voltage_input.setValidator(validator)
        self.ext_voltage_label = QLabel('Extractor Votage (kV)')
        self.ext_voltage_input = QLineEdit()
        self.ext_voltage_input.setFixedSize(input_box_width, input_box_height)
        self.ext_voltage_input.setValidator(validator)
        self.solenoid_current_label = QLabel('Solenoid Current (A)')
        self.solenoid_current_input = QLineEdit()
        self.solenoid_current_input.setFixedSize(input_box_width, input_box_height)
        self.solenoid_current_input.setValidator(validator)
        self.test_stand_label = QLabel('Test Stand')
        self.test_stand_input = QLineEdit()
        self.test_stand_input.setFixedSize(input_box_width, input_box_height)
        self.upper_bound_label = QLabel('Set Max Z (µA)')
        self.upper_bound_input = QLineEdit()
        self.upper_bound_input.setFixedSize(input_box_width, input_box_height)
        self.upper_bound_input.setValidator(validator)
        self.lower_bound_label = QLabel('Set Min Z (µA)')
        self.lower_bound_input = QLineEdit()
        self.lower_bound_input.setFixedSize(input_box_width, input_box_height)
        self.lower_bound_input.setValidator(validator)
        self.fcup_distance_label = QLabel('Dist. to Cup (mm)')
        self.fcup_distance_input = QLineEdit('205')
        self.fcup_distance_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fcup_distance_input.setFixedSize(input_box_width, input_box_height)
        self.fcup_distance_input.setValidator(validator)
        self.fcup_diameter_label = QLabel('Cup Diam. (mm)')
        self.fcup_diameter_input = QLineEdit('2.5')
        self.fcup_diameter_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fcup_diameter_input.setFixedSize(input_box_width, input_box_height)
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
        self.stat_serial_number = QLabel('Serial Number: ')
        self.stat_serial_number.setFixedSize(stat_label_width, stat_label_height)
        self.stat_datetime = QLabel('Scan Timestamp: ')
        self.stat_datetime.setFixedSize(stat_label_width, stat_label_height)
        self.stat_resolution = QLabel('Resolution: ')
        self.stat_resolution.setFixedSize(stat_label_width, stat_label_height)
        self.stat_step_size = QLabel('Step Size (mm): ')
        self.stat_step_size.setFixedSize(stat_label_width, stat_label_height)
        self.stat_polarity = QLabel('Polarity: ')
        self.stat_polarity.setFixedSize(stat_label_width, stat_label_height)
        self.stat_pressure = QLabel('Pressure (mBar): ')
        self.stat_pressure.setFixedSize(stat_label_width, stat_label_height)
        self.stat_beam_voltage = QLabel('Beam Voltage (kV): ')
        self.stat_beam_voltage.setFixedSize(stat_label_width, stat_label_height)
        self.stat_ext_voltage = QLabel('Extractor Voltage (kV): ')
        self.stat_ext_voltage.setFixedSize(stat_label_width, stat_label_height)
        self.stat_beam_supply_current = QLabel('Beam Supply Current (µA): ')
        self.stat_beam_supply_current.setFixedSize(stat_label_width, stat_label_height)
        self.stat_peak_location = QLabel('Peak Location: ')
        self.stat_peak_location.setFixedSize(stat_label_width, stat_label_height)
        self.stat_peak_cup_current = QLabel('Peak Beam Current (nA): ')
        self.stat_peak_cup_current.setFixedSize(stat_label_width, stat_label_height)
        self.stat_peak_total_current = QLabel('Total Current at Peak (µA): ')
        self.stat_peak_total_current.setFixedSize(stat_label_width, stat_label_height)
        self.stat_fwhm_area = QLabel('FWHM Area (mm²): ')
        self.stat_fwhm_area.setFixedSize(stat_label_width, stat_label_height)
        self.stat_fwqm_area = QLabel('FWQM Area (mm²): ')
        self.stat_fwqm_area.setFixedSize(stat_label_width, stat_label_height)

        # Arrange widgets in window
        v_labels_layout = QVBoxLayout()
        v_labels_layout.addWidget(self.serial_number_label)
        v_labels_layout.addWidget(self.beam_voltage_label)
        v_labels_layout.addWidget(self.ext_voltage_label)
        v_labels_layout.addWidget(self.solenoid_current_label)
        v_labels_layout.addWidget(self.test_stand_label)
        v_labels_layout.addWidget(self.upper_bound_label)
        v_labels_layout.addWidget(self.lower_bound_label)

        v_inputs_layout = QVBoxLayout()
        v_inputs_layout.addWidget(self.serial_number_input)
        v_inputs_layout.addWidget(self.beam_voltage_input)
        v_inputs_layout.addWidget(self.ext_voltage_input)
        v_inputs_layout.addWidget(self.solenoid_current_input)
        v_inputs_layout.addWidget(self.test_stand_input)
        v_inputs_layout.addWidget(self.upper_bound_input)
        v_inputs_layout.addWidget(self.lower_bound_input)

        h_labels_and_inputs_layout = QHBoxLayout()
        h_labels_and_inputs_layout.addLayout(v_labels_layout)
        h_labels_and_inputs_layout.addLayout(v_inputs_layout)

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
        v_sub1_main_layout.addLayout(h_labels_and_inputs_layout)
        v_sub1_main_layout.addWidget(self.plot_button)
        v_sub1_main_layout.addLayout(g_checkboxes_layout)
        v_sub1_main_layout.addLayout(g_i_prime_layout)

        v_sub2_main_layout = QVBoxLayout()
        v_sub2_main_layout.addWidget(self.stat_serial_number)
        v_sub2_main_layout.addWidget(self.stat_datetime)
        v_sub2_main_layout.addWidget(self.stat_resolution)
        v_sub2_main_layout.addWidget(self.stat_step_size)
        v_sub2_main_layout.addWidget(self.stat_polarity)
        v_sub2_main_layout.addWidget(self.stat_pressure)
        v_sub2_main_layout.addWidget(self.stat_beam_voltage)
        v_sub2_main_layout.addWidget(self.stat_ext_voltage)
        v_sub2_main_layout.addWidget(self.stat_beam_supply_current)
        v_sub2_main_layout.addWidget(self.stat_peak_location)
        v_sub2_main_layout.addWidget(self.stat_peak_cup_current)
        v_sub2_main_layout.addWidget(self.stat_peak_total_current)
        v_sub2_main_layout.addWidget(self.stat_fwhm_area)
        v_sub2_main_layout.addWidget(self.stat_fwqm_area)

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
    
    def clear_stats(self) -> None:
        default_stat_labels = {
            self.stat_serial_number: 'Serial Number: ',
            self.stat_datetime: 'Scan Timestamp: ',
            self.stat_resolution: 'Resolution: ',
            self.stat_step_size: 'Step Size (mm): ',
            self.stat_polarity: 'Polarity: ',
            self.stat_pressure: 'Pressure (mBar): ',
            self.stat_beam_voltage: 'Beam Voltage (kV): ',
            self.stat_ext_voltage: 'Extractor Voltage (kV): ',
            self.stat_beam_supply_current: 'Beam Supply Current (µA): ',
            self.stat_peak_location: 'Peak Location: ',
            self.stat_peak_cup_current: 'Peak Beam Current (nA): ',
            self.stat_peak_total_current: 'Total Current at Peak (µA): ',
            self.stat_fwhm_area: 'FWHM Area (mm²): ',
            self.stat_fwqm_area: 'FWQM Area (mm²): '
        }
        for label, default_text in default_stat_labels.items():
            label.setText(default_text)

    def update_stat_label(self, label: QLabel, stat_value: tuple[float, float] | float | np.float64 | str | int, name: str) -> None:
        current_text = label.text()
        new_text = '#####'
        one_decimal_place_floats = ('peak_cup_current', 'beam_voltage', 'extractor_voltage')
        three_decimal_place_floats = ('step_size', 'peak_total_current', 'FWHM_area', 'FWQM_area')

        if type(stat_value) == str or type(stat_value) == int:
            new_text = current_text + f'{stat_value}'
        elif type(stat_value) == tuple:
            new_text = current_text + f'({stat_value[0]:.0f}, {stat_value[1]:.0f})'
        elif (type(stat_value) == float or type(stat_value) == np.float64) and name in one_decimal_place_floats:
            new_text = current_text + f'{stat_value:.1f}'
        elif (type(stat_value) == float or type(stat_value) == np.float64) and name in three_decimal_place_floats:
            new_text = current_text + f'{stat_value:.3f}'

        label.setText(new_text)

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
        if isinstance(event, QMouseEvent) and event.type() == QEvent.Type.MouseButtonPress:
            focused_widget = QApplication.focusWidget()
            if focused_widget is not None:
                focused_widget.clearFocus()
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        # Confirm the user wants to exit the application.
        reply = QMessageBox.question(self, 'Confirmation',
                                     'Are you sure you want to close the window?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def get_save_filename(parent, filename) -> str | None:
        filename, _ = QFileDialog.getSaveFileName(
            parent=parent,
            caption='Save CSV File',
            dir=filename,
            filter='CSV Files (*.csv);;All Files (*)'
        )

        if not filename:
            return None
        
        return filename

    @staticmethod
    def csv_load_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'Failed to load beam scan data.\n\nTry another csv file.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def heatmap_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'An error occurred.\n\nUnable to plot heatmap.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def surface_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'An error occurred.\n\nUnable to plot 3D surface.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def cross_sections_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'An error occurred.\n\nUnable to plot XY cross sections.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def i_prime_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'An error occurred.\n\nUnable to plot Angular Intensity cross sections.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def quick_start_guide_error_message(parent) -> None:
        title='Error'
        message='An error occurred.\n\nUnable to find quick start guide.'
        QMessageBox.critical(parent, title, message)
    
    @staticmethod
    def area_calculation_error_message(parent) -> None:
        title='Error'
        message='An error occurred.\n\nFWHM and FWQM extend beyond scan range limits.\nUnable to calculate areas.'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def fwqm_area_calculation_error_message(parent) -> None:
        title='Error'
        message='An error occurred.\n\nFWQM extends beyond scan range limits.\nUnable to calculate area.'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def csv_export_error_message(parent) -> None:
        title='Error'
        message='An error occurred.\n\nCould not export CSV. Try selecting another beam scan csv file.'
        QMessageBox.critical(parent, title, message)

    @staticmethod
    def empty_fcup_inputs_error_message(parent, error, traceback) -> None:
        title='Error'
        message=f'An error occurred.\n\nInput distance to cup and cup diameter measurements.\n\n{error}\n\n{traceback}'
        QMessageBox.critical(parent, title, message)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())