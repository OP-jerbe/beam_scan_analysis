import sys
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

from beam_scan_analysis import ScanData
from beam_scan_gui import MainWindow, OverrideCentroidWindow, QApplication
from beam_scan_plotting import Heatmap, IPrime, Plotter, Surface, XYCrossSections
from load_scan_data import CSVLoader

APP_VERSION = '1.10.3'
CSV_EXPORT_VERSION = '2'


class App:
    """
    A class for managing the application logic and GUI interactions.

    This class creates an instance of the PyQt application, sets up the user interface (UI),
    and handles interactions between the user and the beam scan analysis process. It is responsible
    for selecting and loading CSV data, configuring the scan analysis, and invoking plot generation
    through the `Surface` and `Heatmap` classes.

    Attributes:
        app (QApplication): The main PyQt application instance.
        gui (MainWindow): The main GUI window instance.
        z_scaled (tuple[int | float | None, int | float | None]): The z-axis scale range, or None if not set.
        csv_filepath (str): The file path of the selected CSV file containing scan data.
        scan_data (ScanData): The scan data object holding the loaded beam scan information.
        solenoid (str): The solenoid current from the GUI input.
        test_stand (str): The test stand identifier from the GUI input.

    Methods:
        __init__() -> None:
            Initializes the application and the GUI, sets up the event handlers for button clicks,
            and applies a dark theme to the GUI.

        select_csv_handler() -> None:
            Handles the selection of a CSV file, loads the scan data, and updates the GUI with the relevant information.

        analyze_beam_scan_handler() -> None:
            Handles the analysis of the beam scan data, including retrieving inputs from the GUI and invoking plot
            generation for the surface and heatmap.

        run() -> None:
            Starts the application event loop, initiating the GUI and processing user interactions.
    """

    def __init__(self) -> None:
        self.app = QApplication([])
        self.gui = MainWindow()
        self.gui.setWindowTitle(f'Beam Scan Analysis v{APP_VERSION}')
        self.csv_loader: CSVLoader = CSVLoader()
        self.z_scaled: list[int | float | None] = [None, None]
        self.csv_filepath: str
        self.scan_data: ScanData
        self.beam_voltage: str
        self.ext_voltage: str
        self.solenoid: str
        self.test_stand: str

        self.gui.select_csv_button.clicked.connect(self.select_csv_handler)
        self.gui.plot_button.clicked.connect(self.plot_beam_scan_handler)
        self.gui.export_csv_option.triggered.connect(self.export_to_csv)
        self.gui.override_centroid_option.triggered.connect(
            self.override_centroid_handler
        )
        self.gui.exit_option.triggered.connect(QApplication.quit)
        self.gui.save_3D_surface_option.triggered.connect(self.save_3d_surface_html)
        self.gui.save_heatmap_option.triggered.connect(self.save_heatmap_html)
        self.gui.save_xy_profiles_option.triggered.connect(
            self.save_x_cross_section_html
        )
        self.gui.save_i_prime_option.triggered.connect(self.save_i_prime_html)
        self.gui.save_all_html_option.triggered.connect(self.save_all_html)
        self.gui.save_all_png_option.triggered.connect(self.save_all_png)
        self.gui.open_quick_start_guide.triggered.connect(self.open_quick_start_guide)

        self.gui.show()

    def override_centroid_handler(self) -> None:
        if self.gui.override_centroid_option.isChecked():
            self.override_centroid_window = OverrideCentroidWindow()
            self.override_centroid_window.centroid_set.connect(
                self.receive_centroid_values
            )
            self.override_centroid_window.window_closed_without_input.connect(
                lambda: self.gui.override_centroid_option.setChecked(False)
            )
            self.override_centroid_window.show()

    def receive_centroid_values(self, x: float, y: float) -> None:
        self.Xc = x
        self.Yc = y

    def select_csv_handler(self) -> None:
        """
        Handle CSV file selection, load scan data, and update the GUI.

        This method allows the user to select a CSV file containing beam scan data using the `select_csv` function.
        If a valid file is selected, the scan data is loaded and stored in `self.scan_data`. The method then enables the
        "Plot" button and populates relevant fields in the GUI (e.g., serial number, beam voltage, and extractor voltage)
        with the corresponding values from the loaded scan data. If the csv file was a "well structured" csv file, then
        the solenoid current and test stand values will also be populated.

        Returns:
            None
        """
        self.csv_filepath = self.csv_loader.select_csv()
        if self.csv_filepath != '':
            try:
                self.scan_data = self.csv_loader.load_scan_data(self.csv_filepath)
                self.display_stats(self.scan_data)
                self.gui.plot_button.setDisabled(False)
                self.gui.serial_number_input.setText(self.scan_data.serial_num)
                self.gui.beam_voltage_input.setText(str(self.scan_data.beam_voltage))
                self.gui.ext_voltage_input.setText(
                    str(self.scan_data.extractor_voltage)
                )
                if self.scan_data.well_structured_csv:
                    self.gui.solenoid_current_input.setText(
                        str(self.scan_data.solenoid_current)
                    )
                    self.gui.test_stand_input.setText(str(self.scan_data.test_stand))
                if self.scan_data.fcup_diameter:
                    self.gui.fcup_diameter_input.setText(self.scan_data.fcup_diameter)
                if self.scan_data.fcup_distance:
                    self.gui.fcup_distance_input.setText(self.scan_data.fcup_distance)
                if self.gui.override_centroid_option.isChecked():
                    self.gui.override_centroid_option.setChecked(False)
            except Exception as e:
                full_traceback = traceback.format_exc()
                self.gui.csv_load_error_message(
                    parent=self.gui, error=e, traceback=full_traceback
                )

    def plot_beam_scan_handler(self) -> None:
        """
        Handle the analysis of the beam scan data and plotting of surface, heatmap, and x/y profiles.

        This method retrieves the serial number, beam voltage, and extractor voltage from the GUI inputs and stores them
        in the `self.scan_data` object. Additionally, it retrieves the solenoid current and test stand information from
        the GUI. Then, it creates instances of the `Surface`, `Heatmap`, `XCrossSection`, and `YCrossSection` classes, passing the necessary data, and calls
        their respective `plot_surface`, `plot_heatmap`, and `plot_cross_section` methods to visualize the data.

        Returns:
            None
        """
        self.scan_data.serial_num = self.gui.serial_number_input.text()
        self.solenoid: str = self.gui.solenoid_current_input.text()
        self.test_stand: str = self.gui.test_stand_input.text()
        self.beam_voltage: str = self.gui.beam_voltage_input.text()
        self.ext_voltage: str = self.gui.ext_voltage_input.text()
        try:
            self.fcup_distance: float = float(self.gui.fcup_distance_input.text())
            self.fcup_diameter: float = float(self.gui.fcup_diameter_input.text())
        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.empty_fcup_inputs_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )
            return

        try:
            self.scan_data.beam_voltage = float(self.beam_voltage)
            if self.scan_data.beam_voltage.is_integer():
                self.scan_data.beam_voltage = int(self.beam_voltage)
        except:
            self.scan_data.beam_voltage = None

        try:
            self.scan_data.extractor_voltage = float(self.ext_voltage)
            if float(self.ext_voltage).is_integer():
                self.scan_data.extractor_voltage = int(self.ext_voltage)
        except:
            self.scan_data.extractor_voltage = None

        self.z_scaled = [None, None]  # reset the scaling
        if self.gui.lower_bound_input.text():
            self.z_scaled[0] = float(self.gui.lower_bound_input.text())
        if self.gui.upper_bound_input.text():
            self.z_scaled[1] = float(self.gui.upper_bound_input.text())

        if self.gui.surface_cb.isChecked():
            try:
                surface = Surface(
                    self.scan_data, self.solenoid, self.test_stand, self.z_scaled
                )
                if self.gui.override_centroid_option.isChecked():
                    centroid = self.override_centroid_window.get_override_values()
                    if centroid is not None:
                        surface.centroid = centroid
                surface.plot_surface()
            except Exception as e:
                full_traceback = traceback.format_exc()
                self.gui.surface_error_message(
                    parent=self.gui, error=e, traceback=full_traceback
                )
        if self.gui.heatmap_cb.isChecked():
            try:
                heatmap = Heatmap(
                    self.scan_data, self.solenoid, self.test_stand, self.z_scaled
                )
                if self.gui.override_centroid_option.isChecked():
                    centroid = self.override_centroid_window.get_override_values()
                    if centroid is not None:
                        heatmap.centroid = centroid
                heatmap.plot_heatmap()
            except Exception as e:
                full_traceback = traceback.format_exc()
                self.gui.heatmap_error_message(
                    parent=self.gui, error=e, traceback=full_traceback
                )
        if self.gui.xy_profile_cb.isChecked():
            try:
                xy_cross_section = XYCrossSections(
                    self.scan_data, self.solenoid, self.test_stand, self.z_scaled
                )
                if self.gui.override_centroid_option.isChecked():
                    centroid = self.override_centroid_window.get_override_values()
                    if centroid is not None:
                        xy_cross_section.centroid = centroid
                xy_cross_section.plot_cross_sections()
            except Exception as e:
                full_traceback = traceback.format_exc()
                self.gui.cross_sections_error_message(
                    parent=self.gui, error=e, traceback=full_traceback
                )
        if self.gui.i_prime_cb.isChecked():
            try:
                i_prime = IPrime(
                    self.scan_data, self.solenoid, self.test_stand, self.z_scaled
                )
                if self.gui.override_centroid_option.isChecked():
                    centroid = self.override_centroid_window.get_override_values()
                    if centroid is not None:
                        i_prime.centroid = centroid
                i_prime.plot_i_prime(self.fcup_diameter, self.fcup_distance)
            except Exception as e:
                full_traceback = traceback.format_exc()
                self.gui.i_prime_error_message(
                    parent=self.gui, error=e, traceback=full_traceback
                )

    def display_stats(self, scan_data: ScanData) -> None:
        summary_keys_and_stat_labels = {
            'serial_number': self.gui.stat_serial_number,
            'scan_datetime': self.gui.stat_datetime,
            'resolution': self.gui.stat_resolution,
            'step_size': self.gui.stat_step_size,
            'polarity': self.gui.stat_polarity,
            'pressure': self.gui.stat_pressure,
            'beam_voltage': self.gui.stat_beam_voltage,
            'extractor_voltage': self.gui.stat_ext_voltage,
            'beam_supply_current': self.gui.stat_beam_supply_current,
            'peak_location': self.gui.stat_peak_location,
            'peak_cup_current': self.gui.stat_peak_cup_current,
            'peak_total_current': self.gui.stat_peak_total_current,
            'FWHM_area': self.gui.stat_fwhm_area,
            'FWQM_area': self.gui.stat_fwqm_area,
        }
        self.gui.clear_stats()
        stats: dict = scan_data.display_summary()
        for key, stat_label in summary_keys_and_stat_labels.items():
            self.gui.update_stat_label(stat_label, stats[key], key)

    def export_to_csv(self) -> None:
        try:  # check if csv data has been loaded
            scan_datetime: str = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            step_size: float = self.scan_data.step_size
            serial_number: str = self.gui.serial_number_input.text().strip()
            beam_voltage: str = self.gui.beam_voltage_input.text().strip()
            extractor_voltage: str = self.gui.ext_voltage_input.text().strip()
            solenoid_current: str = self.gui.solenoid_current_input.text().strip()
            test_stand: str = self.gui.test_stand_input.text().strip()
            beam_supply_current: str = self.scan_data.beam_supply_current
            pressure: str = self.scan_data.pressure
            fcup_distance: str = self.gui.fcup_distance_input.text().strip()
            fcup_diameter: str = self.gui.fcup_diameter_input.text().strip()
            default_filename = f'{scan_date} SN-{serial_number} @ {beam_voltage}_{extractor_voltage} kV & {solenoid_current} A on TS{test_stand}.csv'
            data = pd.DataFrame(
                {
                    'X': self.scan_data.x_location,
                    'Y': self.scan_data.y_location,
                    'cup_current': self.scan_data.cup_current,
                    'screen_current': self.scan_data.screen_current,
                    'total_current': self.scan_data.total_current,
                }
            )
        except:
            self.gui.csv_export_error_message(parent=self.gui)
            return None

        filename = self.gui.get_save_filename(
            parent=self.gui, filename=default_filename
        )
        if not filename:  # if the user cancels, return None
            return None

        with open(filename, 'w') as f:
            f.write(f'CSV export version,{CSV_EXPORT_VERSION}\n')
            f.write(f'Serial Number,{serial_number}\n')
            f.write(f'Scan Datetime,{scan_datetime}\n')
            f.write(f'Step Size (mm),{step_size}\n')
            f.write(f'Beam Voltage (kV),{beam_voltage}\n')
            f.write(f'Extractor Voltage (kV),{extractor_voltage}\n')
            f.write(f'Solenoid Current (A),{solenoid_current}\n')
            f.write(f'Test Stand,{test_stand}\n')
            f.write(f'Beam Supply Current (uA),{beam_supply_current}\n')
            f.write(f'Pressure (mBar),{pressure}\n')
            f.write(f'F-Cup Distance (mm),{fcup_distance}\n')
            f.write(f'F-Cup Diameter (mm),{fcup_diameter}\n')

        data.to_csv(filename, index=False, mode='a')  # append the data

    def save_3d_surface_html(self) -> None:
        try:
            scan_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand} 3D Surface.html'
            surface = Surface(self.scan_data, solenoid, test_stand, self.z_scaled)
            fig = surface.plot_surface(show=False)
            surface.save_as_html(fig, default_filename, parent=self.gui)
        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.surface_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def save_heatmap_html(self) -> None:
        try:
            scan_data_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_data_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand} Heatmap.html'
            heatmap = Heatmap(self.scan_data, solenoid, test_stand, self.z_scaled)
            fig = heatmap.plot_heatmap(show=False)
            heatmap.save_as_html(fig, default_filename, parent=self.gui)
        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.heatmap_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def save_x_cross_section_html(self) -> None:
        try:
            scan_data_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_data_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand} XY Cross Section.html'
            cross_section = XYCrossSections(
                self.scan_data, solenoid, test_stand, self.z_scaled
            )
            fig = cross_section.plot_cross_sections(show=False)
            cross_section.save_as_html(fig, default_filename, parent=self.gui)
        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.cross_sections_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def save_i_prime_html(self) -> None:
        try:
            scan_data_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_data_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            self.fcup_diameter = float(self.gui.fcup_diameter_input.text())
            self.fcup_distance = float(self.gui.fcup_distance_input.text())
            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand} Ang Int vs Divergence Angle.html'
            i_prime = IPrime(self.scan_data, solenoid, test_stand, self.z_scaled)
            fig = i_prime.plot_i_prime(
                self.fcup_diameter, self.fcup_distance, show=False
            )
            i_prime.save_as_html(fig, default_filename, parent=self.gui)
        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.i_prime_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def save_all_html(self) -> None:
        try:
            scan_data_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_data_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            self.fcup_diameter = float(self.gui.fcup_diameter_input.text())
            self.fcup_distance = float(self.gui.fcup_distance_input.text())
            surface = Surface(self.scan_data, solenoid, test_stand, self.z_scaled)
            surface_fig = surface.plot_surface(show=False)
            heatmap = Heatmap(self.scan_data, solenoid, test_stand, self.z_scaled)
            heatmap_fig = heatmap.plot_heatmap(show=False)
            cross_section = XYCrossSections(
                self.scan_data, solenoid, test_stand, self.z_scaled
            )
            cross_section_fig = cross_section.plot_cross_sections(show=False)
            i_prime = IPrime(self.scan_data, solenoid, test_stand, self.z_scaled)
            i_prime_fig = i_prime.plot_i_prime(
                self.fcup_diameter, self.fcup_distance, show=False
            )

            plots = {
                '3D Surface.html': surface_fig,
                'Heatmap.html': heatmap_fig,
                'XY Cross Section.html': cross_section_fig,
                'Ang Int vs Divergence Angle.html': i_prime_fig,
            }

            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand}'

            Plotter.save_all_as_html(
                plots,
                default_filename,
            )

        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.i_prime_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def save_all_png(self) -> None:
        try:
            scan_data_datetime = self.scan_data.scan_datetime
            date_obj = datetime.strptime(scan_data_datetime, '%m/%d/%Y %I:%M %p')
            scan_date = date_obj.strftime('%Y-%m-%d %H_%M')
            serial_num = self.gui.serial_number_input.text()
            solenoid = self.gui.solenoid_current_input.text()
            test_stand = self.gui.test_stand_input.text()
            beam_voltage = self.gui.beam_voltage_input.text()
            ext_voltage = self.gui.ext_voltage_input.text()
            self.fcup_diameter = float(self.gui.fcup_diameter_input.text())
            self.fcup_distance = float(self.gui.fcup_distance_input.text())
            surface = Surface(self.scan_data, solenoid, test_stand, self.z_scaled)
            surface_fig = surface.plot_surface(show=False)
            heatmap = Heatmap(self.scan_data, solenoid, test_stand, self.z_scaled)
            heatmap_fig = heatmap.plot_heatmap(show=False)
            cross_section = XYCrossSections(
                self.scan_data, solenoid, test_stand, self.z_scaled
            )
            cross_section_fig = cross_section.plot_cross_sections(show=False)
            i_prime = IPrime(self.scan_data, solenoid, test_stand, self.z_scaled)
            i_prime_fig = i_prime.plot_i_prime(
                self.fcup_diameter, self.fcup_distance, show=False
            )

            plots = {
                '3D Surface.png': surface_fig,
                'Heatmap.png': heatmap_fig,
                'XY Cross Section.png': cross_section_fig,
                'Ang Int vs Divergence Angle.png': i_prime_fig,
            }

            default_filename = f'{scan_date} SN-{serial_num} @ {beam_voltage}_{ext_voltage} kV & {solenoid} A on TS{test_stand}'

            Plotter.save_all_as_png(
                plots,
                default_filename,
            )

        except Exception as e:
            full_traceback = traceback.format_exc()
            self.gui.i_prime_error_message(
                parent=self.gui, error=e, traceback=full_traceback
            )

    def open_quick_start_guide(self) -> None:
        script_dir = Path(__file__).parent
        filepath = script_dir / 'quick_start_guide.html'

        if not filepath.is_file():
            self.gui.quick_start_guide_error_message(self)
            return

        webbrowser.open_new_tab(filepath.resolve().as_uri())

    def run(self) -> None:
        """
        Start the application event loop.

        This method initializes the application event loop by calling `exec()` on the app instance.
        Once the event loop completes, it exits the application with the corresponding exit code.

        Returns:
            None
        """
        exit_code: int = self.app.exec()
        sys.exit(exit_code)


if __name__ == '__main__':
    app = App()
    app.run()
