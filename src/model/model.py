import pandas as pd
from pandas import DataFrame

import helpers.helpers as h


class ScanData:
    def __init__(self) -> None:
        self.well_structured_csv: bool = False
        self.metadata: dict = {}
        self.data: DataFrame = DataFrame([])

    def load_scan_data(self) -> None:
        """
        Load scan data from a CSV file and return a ScanData object.

        Notes:
            The function reads specific metadata from the first few rows of the CSV file
            and uses the rest of the data to populate the scan data. The beam voltage
            is used to determine the scan polarity ('NEG' or 'POS'), and the step size
            is used to determine the scan resolution (Ultra, High, or Med).
        """

        csv_loaded: bool = False

        filepath = h.select_file()
        if not filepath:
            return

        while not csv_loaded:
            try:
                self.metadata, self.data = self._load_v3_csv(filepath)
                csv_loaded = True
                self.well_structured_csv = True
                break
            except CustomSkipError:
                pass

            try:
                self.metadata, self.data = self._load_v2_csv(filepath)
                csv_loaded = True
                self.well_structured_csv = True
                break
            except CustomSkipError:
                pass

            try:
                self.metadata, self.data = self._load_v1_csv(filepath)
                csv_loaded = True
                self.well_structured_csv = True
                break
            except CustomSkipError:
                pass

            try:
                self.metadata, self.data = self._load_labview_csv(filepath)
                csv_loaded = True
                break
            except Exception as e:
                # Put a pop up message here to let the user know data load failed.
                print(f'Could not load scan data.\n\n{str(e)}')
                raise

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # # Determine whether the beam scan was taken in NEG mode or POS mode
    # polarity: str = 'NEG'
    # if beam_voltage > 0:  # type: ignore
    #     polarity = 'POS'

    # self.serial_num = serial_number
    # self.scan_datetime = scan_datetime
    # self.step_size = step_size
    # self.resolution = self._get_resolution(step_size)
    # self.x_location = df['X']
    # self.y_location = df['Y']
    # self.cup_current = df['cup_current']
    # self.screen_current = df['screen_current']
    # self.total_current = df['total_current']
    # self.polarity = polarity
    # self.beam_voltage = beam_voltage
    # self.extractor_voltage = extractor_voltage
    # self.solenoid_current = solenoid_current
    # self.beam_supply_current = beam_supply_current
    # self.pressure = pressure
    # self.test_stand = test_stand
    # self.fcup_distance = fcup_distance
    # self.fcup_diameter = fcup_diameter
    # self.well_structured_csv = well_structured_csv

    @property
    def resolution(self) -> str:
        match self.metadata['step_size']:
            case 0.221:
                resolution = 'Highest'
            case 0.442:
                resolution = 'High'
            case 0.884:
                resolution = 'Med'
            case 1.767:
                resolution = 'Low'
            case _:
                resolution = ''

        return resolution

    @property
    def polarity(self) -> str:
        if self.metadata['beam_voltage'] < 0:
            return 'NEG'
        else:
            return 'POS'

    def _load_v3_csv(self, filepath: str) -> tuple[dict, DataFrame]:
        # Load in the metadata from the csv file.
        df: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=13, usecols=[1])

        # Check if csv is exported from application as version 2.
        csv_version: str = str(df.iloc[0, 0])
        if csv_version != '3':
            raise CustomSkipError

        serial_number = str(df.iloc[1, 0])
        scan_datetime = str(df.iloc[2, 0])
        step_size = float(pd.to_numeric(df.iloc[3, 0]))
        beam_voltage = float(str(df.iloc[4, 0]))
        extractor_voltage = float(str(df.iloc[5, 0]))
        solenoid_current = float(str(df.iloc[6, 0]))
        test_stand = str(df.iloc[7, 0]).replace('nan', '')
        beam_supply_current = float(pd.to_numeric(df.iloc[8, 0]))
        pressure = float(pd.to_numeric(df.iloc[9, 0]))
        fcup_distance = float(str(df.iloc[10, 0]))
        fcup_diameter = float(str(df.iloc[11, 0]))
        power = float(str(df.iloc[12, 0]))

        if beam_voltage.is_integer():
            beam_voltage = int(beam_voltage)
        if extractor_voltage.is_integer():
            extractor_voltage = int(extractor_voltage)
        if power.is_integer():
            power = int(power)

        # Get the scan data (y, x, cup current, screen current)
        data: pd.DataFrame = pd.read_csv(filepath, skiprows=13)

        metadata: dict = {
            'csv_version': csv_version,
            'serial_number': serial_number,
            'scan_datetime': scan_datetime,
            'step_size': step_size,
            'beam_voltage': beam_voltage,
            'extractor_voltage': extractor_voltage,
            'solenoid_current': solenoid_current,
            'test_stand': test_stand,
            'beam_supply_current': beam_supply_current,
            'pressure': pressure,
            'fcup_distance': fcup_distance,
            'fcup_diameter': fcup_diameter,
            'power': power,
        }

        return metadata, data

    def _load_v2_csv(self, filepath: str) -> tuple[dict, DataFrame]:
        # Load in the metadata from the csv file.
        df: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=12, usecols=[1])

        # Check if csv is exported from application as version 2.
        csv_version: str = str(df.iloc[0, 0])
        if csv_version != '2':
            raise CustomSkipError

        serial_number = str(df.iloc[1, 0])
        scan_datetime = str(df.iloc[2, 0])
        step_size = float(pd.to_numeric(df.iloc[3, 0]))
        beam_voltage = float(str(df.iloc[4, 0]))
        extractor_voltage = float(str(df.iloc[5, 0]))
        solenoid_current = float(str(df.iloc[6, 0]))
        test_stand = str(df.iloc[7, 0]).replace('nan', '')
        beam_supply_current = float(pd.to_numeric(df.iloc[8, 0]))
        pressure = float(pd.to_numeric(df.iloc[9, 0]))
        fcup_distance = float(str(df.iloc[10, 0]))
        fcup_diameter = float(str(df.iloc[11, 0]))
        power = float('nan')

        if beam_voltage.is_integer():
            beam_voltage = int(beam_voltage)
        if extractor_voltage.is_integer():
            extractor_voltage = int(extractor_voltage)

        # Get the scan data (y, x, cup current, screen current)
        data: pd.DataFrame = pd.read_csv(filepath, skiprows=12)

        metadata: dict = {
            'csv_version': csv_version,
            'serial_number': serial_number,
            'scan_datetime': scan_datetime,
            'step_size': step_size,
            'beam_voltage': beam_voltage,
            'extractor_voltage': extractor_voltage,
            'solenoid_current': solenoid_current,
            'test_stand': test_stand,
            'beam_supply_current': beam_supply_current,
            'pressure': pressure,
            'fcup_distance': fcup_distance,
            'fcup_diameter': fcup_diameter,
            'power': power,
        }

        return metadata, data

    def _load_v1_csv(self, filepath: str) -> tuple[dict, DataFrame]:
        """
        Check if csv is exported from application as version 1.
        (No Beam Supply Current, Pressure, fcup dist/diam data)
        """
        # Load in the metadata from the csv file.
        df: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=8, usecols=[1])

        # Check if csv is exported from application as version 1.
        csv_version: str = str(df.iloc[0, 0])
        if csv_version != '1':
            raise CustomSkipError

        serial_number = str(df.iloc[1, 0])
        scan_datetime = str(df.iloc[2, 0])
        step_size = float(pd.to_numeric(df.iloc[3, 0]))
        beam_voltage = float(str(df.iloc[4, 0]))
        extractor_voltage = float(str(df.iloc[5, 0]))
        solenoid_current = float(str(df.iloc[6, 0]))
        test_stand = str(df.iloc[7, 0]).replace('nan', '')
        beam_supply_current = float('nan')
        pressure = float('nan')
        fcup_distance = float('nan')
        fcup_diameter = float('nan')
        power = float('nan')

        if beam_voltage.is_integer():
            beam_voltage = int(beam_voltage)
        if extractor_voltage.is_integer():
            extractor_voltage = int(extractor_voltage)

        # Get the scan data (y, x, cup current, screen current)
        data: pd.DataFrame = pd.read_csv(filepath, skiprows=8)

        metadata: dict = {
            'csv_version': csv_version,
            'serial_number': serial_number,
            'scan_datetime': scan_datetime,
            'step_size': step_size,
            'beam_voltage': beam_voltage,
            'extractor_voltage': extractor_voltage,
            'solenoid_current': solenoid_current,
            'test_stand': test_stand,
            'beam_supply_current': beam_supply_current,
            'pressure': pressure,
            'fcup_distance': fcup_distance,
            'fcup_diameter': fcup_diameter,
            'power': power,
        }

        return metadata, data

    def _load_labview_csv(self, filepath: str) -> tuple[dict, DataFrame]:
        """Load in csv as exported from LabVIEW application."""
        data: pd.DataFrame = pd.read_csv(
            filepath, header=None, usecols=[0], nrows=7, skiprows=[3, 7]
        )

        csv_version: str = '0'
        scan_datetime_data: str = str(data.iloc[0, 0])
        if scan_datetime_data.split()[0] != '#DATE/TIME:':
            raise Exception('CSV is not a beam scan csv.')
        scan_datetime = scan_datetime_data.replace('#DATE/TIME: ', '').strip()
        serial_data = str(data.iloc[1, 0])
        serial_number = serial_data.replace('#Module Number: ', '').strip()
        step_size_data = str(data.iloc[2, 0])
        step_size = float(step_size_data.replace('#Step Size(mm): ', '').strip())  # mm

        beam_voltage_data: str = str(data.iloc[3, 0])
        beam_voltage: float = (
            round(
                int(beam_voltage_data.replace('#Beam Voltage: ', '').strip()),
                -2,
            )
            * 1e-3
        )  # kV
        if beam_voltage.is_integer():
            beam_voltage = int(beam_voltage)

        extractor_voltage_data: str = str(data.iloc[4, 0])
        extractor_voltage: float = (
            round(
                int(extractor_voltage_data.replace('#Extractor Voltage: ', '').strip()),
                -2,
            )
            * 1e-3
        )  # kV
        if extractor_voltage.is_integer():
            extractor_voltage = int(extractor_voltage)

        beam_supply_current_data: str = str(data.iloc[5, 0])
        beam_supply_current_data: str = beam_supply_current_data.replace(
            '#Beam Supply Current: ', ''
        ).strip()  # uA
        beam_supply_current: float = round(float(beam_supply_current_data), 2)

        pressure_data = str(data.iloc[6, 0])
        pressure_data = pressure_data.replace('#Chamber Pressure:  ', '')  # mBar
        pressure = float(pressure_data)

        solenoid_current = float('nan')
        test_stand = ''
        fcup_distance = float('nan')
        fcup_diameter = float('nan')
        power = float('nan')

        # Get the scan data (y, x, cup current, screen current)
        data: pd.DataFrame = pd.read_csv(filepath, usecols=[0, 1, 2, 3], skiprows=9)

        # Rename columns
        data.rename(
            columns={
                ' Y Coordinate': 'X',
                '#X Coordinate': 'Y',
                ' Faraday Cup Current': 'cup_current',
                ' Screen Current': 'screen_current',
            },
            inplace=True,
        )

        # Add total current column
        data['total_current'] = data['cup_current'] + data['screen_current']

        metadata: dict = {
            'csv_version': csv_version,
            'serial_number': serial_number,
            'scan_datetime': scan_datetime,
            'step_size': step_size,
            'beam_voltage': beam_voltage,
            'extractor_voltage': extractor_voltage,
            'solenoid_current': solenoid_current,
            'test_stand': test_stand,
            'beam_supply_current': beam_supply_current,
            'pressure': pressure,
            'fcup_distance': fcup_distance,
            'fcup_diameter': fcup_diameter,
            'power': power,
        }

        return metadata, data


class CustomSkipError(Exception):
    """
    Custom exception used to skip processing in specific cases.
    """

    pass


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    QApplication([])
    scan_data = ScanData()
    scan_data.load_scan_data()
    resolution = scan_data.resolution
    polarity = scan_data.polarity
    if scan_data.well_structured_csv:
        print('This csv was output by the app.')
    else:
        print('This csv was output by labview.')
    print(f'{scan_data.metadata = }')
    # print(f'{scan_data.data.head()}')
    print(f'{resolution = }')
    print(f'{polarity = }')
