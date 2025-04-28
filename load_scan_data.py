import sys

import pandas as pd
from PySide6.QtWidgets import QFileDialog

from beam_scan_analysis import ScanData


class CSVLoader:
    def __init__(self): ...

    @staticmethod
    def select_csv() -> str:
        """
        Open a file dialog to select a CSV file.

        Returns:
            str: The path to the selected CSV file. If no file is selected, an empty string is returned.

        Notes:
            The file dialog starts in the Production History directory and filters for CSV files.
            If no file is selected, the function will return an empty string.
        """
        if hasattr(sys, 'frozen'):  # Check if running from the bundled app
            dir = r'C:\\teststand data'
        else:
            dir = r'\\opdata2\Company\PRODUCTION FOLDER\Production History'

        caption = 'Choose CSV Files'
        initial_dir = dir
        file_types = 'CSV Files (*.csv);;All Files (*)'
        # Open the file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            None,  # Parent widget, can be None
            caption,  # Dialog title
            initial_dir,  # Initial directory
            file_types,  # Filter for file types
        )

        return filepath

    @staticmethod
    def load_scan_data(filepath: str) -> ScanData:
        """
        Load scan data from a CSV file and return a ScanData object.

        Args:
            filepath (str): The path to the CSV file containing the scan data.

        Returns:
            ScanData: An object containing the parsed scan data, including serial number,
                    scan datetime, step size, resolution, x and y coordinates, cup
                    current, screen current, total current, polarity, beam voltage,
                    and extractor voltage.

        Notes:
            The function reads specific metadata from the first few rows of the CSV file
            and uses the rest of the data to populate the scan data. The beam voltage
            is used to determine the scan polarity ('NEG' or 'POS'), and the step size
            is used to determine the scan resolution (Ultra, High, or Med).
        """

        csv_loaded: bool = False

        try:
            # check if csv is exported from application as version 2
            df: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=12, usecols=[1])
            csv_export_version: str = str(df.iloc[0, 0])
            if csv_export_version != '2':
                raise CustomSkipError
            serial_number: str = str(df.iloc[1, 0])
            scan_datetime: str = str(df.iloc[2, 0])
            step_size: float = float(pd.to_numeric(df.iloc[3, 0]))
            beam_voltage_data: str = str(df.iloc[4, 0])
            extractor_voltage_data: str = str(df.iloc[5, 0])
            solenoid_current: str = str(df.iloc[6, 0])
            test_stand: str = str(df.iloc[7, 0])
            beam_supply_current_data: str = str(df.iloc[8, 0])
            pressure: str = str(df.iloc[9, 0])
            fcup_distance: str | None = str(df.iloc[10, 0])
            fcup_diameter: str | None = str(df.iloc[11, 0])

            beam_supply_current: str = str(beam_supply_current_data)
            beam_voltage = float(beam_voltage_data)
            extractor_voltage = float(extractor_voltage_data)

            if beam_voltage.is_integer():
                beam_voltage = int(beam_voltage_data)
            if extractor_voltage.is_integer():
                extractor_voltage = int(extractor_voltage_data)
            if solenoid_current == 'nan':
                solenoid_current = ''
            if test_stand == 'nan':
                test_stand = ''

            df: pd.DataFrame = pd.read_csv(filepath, skiprows=12)

            csv_loaded = True
            well_structured_csv = True

        except CustomSkipError:
            pass
        except Exception as e:
            print(f'Unexpected Error: {e}')

        if not csv_loaded:
            try:
                # check if csv is exported from application as version 1
                df: pd.DataFrame = pd.read_csv(
                    filepath, header=None, nrows=8, usecols=[1]
                )
                csv_export_version: str = str(df.iloc[0, 0])
                if csv_export_version != '1':
                    raise CustomSkipError
                serial_number: str = str(df.iloc[1, 0])
                scan_datetime: str = str(df.iloc[2, 0])
                step_size: float = float(pd.to_numeric(df.iloc[3, 0]))
                beam_voltage_data: str = str(df.iloc[4, 0])
                extractor_voltage_data: str = str(df.iloc[5, 0])
                solenoid_current: str = str(df.iloc[6, 0])
                test_stand: str = str(df.iloc[7, 0])
                beam_supply_current: str = ''
                pressure: str = ''
                fcup_distance: str | None = None
                fcup_diameter: str | None = None

                beam_voltage = float(beam_voltage_data)
                extractor_voltage = float(extractor_voltage_data)

                if beam_voltage.is_integer():
                    beam_voltage = int(beam_voltage_data)
                if extractor_voltage.is_integer():
                    extractor_voltage = int(extractor_voltage_data)
                if solenoid_current == 'nan':
                    solenoid_current = ''
                if test_stand == 'nan':
                    test_stand = ''

                df: pd.DataFrame = pd.read_csv(filepath, skiprows=8)

                csv_loaded: bool = True
                well_structured_csv = True

            except CustomSkipError:
                pass
            except Exception as e:
                print(f'Unexpected Error: {e}')

        if not csv_loaded:
            try:
                # load in csv as exported from LabVIEW application
                df: pd.DataFrame = pd.read_csv(
                    filepath, header=None, usecols=[0], nrows=7, skiprows=[3, 7]
                )

                scan_datetime_data: str = str(df.iloc[0, 0])
                if scan_datetime_data.split()[0] != '#DATE/TIME:':
                    raise Exception('CSV is not a beam scan csv.')
                scan_datetime: str = scan_datetime_data.replace(
                    '#DATE/TIME: ', ''
                ).strip()

                serial_data: str = str(df.iloc[1, 0])
                serial_number: str = serial_data.replace('#Module Number: ', '').strip()

                step_size_data: str = str(df.iloc[2, 0])
                step_size: float = float(
                    step_size_data.replace('#Step Size(mm): ', '').strip()
                )  # mm

                beam_voltage_data: str = str(df.iloc[3, 0])
                beam_voltage: float = (
                    round(
                        int(beam_voltage_data.replace('#Beam Voltage: ', '').strip()),
                        -2,
                    )
                    * 1e-3
                )  # kV
                if beam_voltage.is_integer():
                    beam_voltage = int(beam_voltage)

                extractor_voltage_data: str = str(df.iloc[4, 0])
                extractor_voltage: float = (
                    round(
                        int(
                            extractor_voltage_data.replace(
                                '#Extractor Voltage: ', ''
                            ).strip()
                        ),
                        -2,
                    )
                    * 1e-3
                )  # kV
                if extractor_voltage.is_integer():
                    extractor_voltage = int(extractor_voltage)

                beam_supply_current_data: str = str(df.iloc[5, 0])
                beam_supply_current: str = beam_supply_current_data.replace(
                    '#Beam Supply Current: ', ''
                ).strip()  # uA

                pressure_data: str = str(df.iloc[6, 0])
                pressure: str = pressure_data.replace(
                    '#Chamber Pressure:  ', ''
                )  # mBar

                solenoid_current: str = ''
                test_stand: str = ''
                fcup_distance: str | None = None
                fcup_diameter: str | None = None

                df: pd.DataFrame = pd.read_csv(
                    filepath, usecols=[0, 1, 2, 3], skiprows=9
                )

                # Rename columns
                df.rename(
                    columns={
                        ' Y Coordinate': 'X',
                        '#X Coordinate': 'Y',
                        ' Faraday Cup Current': 'cup_current',
                        ' Screen Current': 'screen_current',
                    },
                    inplace=True,
                )

                # Add total current column
                df['total_current'] = df['cup_current'] + df['screen_current']

                csv_loaded = True
                well_structured_csv = False

            except Exception as e:
                print(f'Unexpected Error: {e}')
                return ScanData(
                    serial_num='',
                    scan_datetime='',
                    step_size=0.0,
                    resolution='',
                    x_location=pd.Series(dtype=float),
                    y_location=pd.Series(dtype=float),
                    cup_current=pd.Series(dtype=float),
                    screen_current=pd.Series(dtype=float),
                    total_current=pd.Series(dtype=float),
                    polarity='',
                    beam_voltage=None,
                    extractor_voltage=None,
                    solenoid_current='',
                    beam_supply_current='',
                    pressure='',
                    test_stand=None,
                    fcup_distance=None,
                    fcup_diameter=None,
                    well_structured_csv=False,
                )

        # Determine whether the beam scan was taken in NEG mode or POS mode
        polarity: str = 'NEG'
        if beam_voltage > 0:  # type: ignore
            polarity = 'POS'

        # Determine resolution of the beam scan
        resolutions: dict[float, str] = {
            0.221: 'Highest',
            0.442: 'High',
            0.884: 'Med',
            1.767: 'Low',
        }
        resolution: str = resolutions[step_size]  # type: ignore

        return ScanData(
            serial_num=serial_number,  # type: ignore
            scan_datetime=scan_datetime,  # type: ignore
            step_size=step_size,  # type: ignore
            resolution=resolution,
            x_location=df['X'],  # type: ignore
            y_location=df['Y'],  # type: ignore
            cup_current=df['cup_current'],  # type: ignore
            screen_current=df['screen_current'],  # type: ignore
            total_current=df['total_current'],  # type: ignore
            polarity=polarity,
            beam_voltage=beam_voltage,  # type: ignore
            extractor_voltage=extractor_voltage,  # type: ignore
            solenoid_current=solenoid_current,  # type: ignore
            beam_supply_current=beam_supply_current,  # type: ignore
            pressure=pressure,  # type: ignore
            test_stand=test_stand,  # type: ignore
            fcup_distance=fcup_distance,  # type: ignore
            fcup_diameter=fcup_diameter,  # type: ignore
            well_structured_csv=well_structured_csv,  # type: ignore
        )


class CustomSkipError(Exception):
    """
    Custom exception used to skip processing in specific cases.
    """

    pass


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    QApplication([])
    filepath = CSVLoader.select_csv()
    scan_data = CSVLoader.load_scan_data(filepath)
    print(scan_data.display_summary())
