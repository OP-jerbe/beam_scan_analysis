import numpy as np
import pandas as pd
from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import (
    measure,
)  # For find_contours <conda install -c conda-forge scikit-image>

import helpers.helpers as h


class ScanData:
    def __init__(self) -> None:
        self._metadata: dict = {}
        self._data: DataFrame = DataFrame([])
        self.interp_num: int = 500
        self.grid_x: NDArray[float64] = np.array([])
        self.grid_y: NDArray[float64] = np.array([])
        self.grid_z: NDArray[float64] = np.array([])

    def load_scan_data(self) -> None:
        """
        Load scan data from a CSV file and return a ScanData object.
        Args:
            interp_num (int): The interpolation number for creating the grid.
        Notes:
            The function reads specific metadata from the first few rows of the CSV file
            and uses the rest of the data to populate the scan data. The beam voltage
            is used to determine the scan polarity ('NEG' or 'POS'), and the step size
            is used to determine the scan resolution (Ultra, High, or Med).
        """

        filepath = h.select_file()
        if not filepath:
            return

        csv_version = self._check_version(filepath)
        match csv_version:
            case 3:
                self._metadata, self._data = self._load_v3_csv(filepath)
            case 2:
                self._metadata, self._data = self._load_v2_csv(filepath)
            case 1:
                self._metadata, self._data = self._load_v1_csv(filepath)
            case 0:
                self._metadata, self._data = self._load_v0_csv(filepath)

        # Generate the interpolated grid
        self.create_grid(self.interp_num)

    def create_grid(self, interp_num) -> None:
        """
        Create the interpolated meshgrid from the scan data.

        Args:
            interp_num (int): The number of points on each axis of the grid.
        """
        x = self.x_location.to_numpy()
        y = self.y_location.to_numpy()
        z = self.cup_current.to_numpy()

        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), interp_num),
            np.linspace(y.min(), y.max(), interp_num),
        )

        self.grid_z = griddata((x, y), z, (self.grid_x, self.grid_y), method='cubic')

    # --- Contour methods ---

    def _contour(
        self, level: float
    ) -> tuple[NDArray[float64], NDArray[float64]] | None:
        """
        Finds the contour of a 3D map at the specified level.

        Args:
            level (float): The height/elevation at which the contour should be calculated.

        Returns:
            tuple[NDArray[float64], NDArray[float64]]: The 2D array of the contour x-y coordinates if the contour exists
            None: if the contours do not exists
        """

        # Extract and scale the contour
        raw_contours = measure.find_contours(self.grid_z, level)

        if not raw_contours:
            return None

        contour = np.array(raw_contours[0])

        # Map from pixel indices (row, col) to data coordinates (y, x)
        x_contour = np.interp(
            contour[:, 1],
            [0, self.grid_z.shape[1] - 1],
            [self.grid_x.min(), self.grid_x.max()],
        )
        y_contour = np.interp(
            contour[:, 0],
            [0, self.grid_z.shape[0] - 1],
            [self.grid_y.min(), self.grid_y.max()],
        )

        return x_contour, y_contour

    def _contour_min_diameter(
        self, x_contour: NDArray[float64], y_contour: NDArray[float64]
    ) -> float:
        """
        Calculate minimum diameter of a contour using Convex Hull.

        Args:
            x_contour (NDArray[float64]): The x-coordinate points of the contour.
            y_contour (NDArray[float64]): The y-coordinate points of the contour.

        Returns:
            float: The minimum calculated diameter of the contour. Returns zero
            if it is impossible to calculate the convex hull.
        """
        # Check if we have enough points to even make a shape
        if len(x_contour) < 3:
            return 0.0

        # Create the x-y pairs
        points = np.column_stack([x_contour, y_contour])

        # Create the "Rubber Band" around the points
        # This ignores all internal points and gives us the outer boundary
        hull = ConvexHull(points)

        # Extract the actual X-Y coordinates of the hull vertices
        # hull.vertices provides the indices; points[hull.vertices] gives coordinates
        hull_points = points[hull.vertices]

        # Prepare to find the Minimum Diameter (Rotating Calipers logic)
        # We initialize with infinity so any real width will be smaller
        min_diameter = float('inf')

        # Iterate over every flat edge of the hull
        for i in range(len(hull_points)):
            # Define the edge using current point (p1) and next point (p2)
            # The % n ensures the last point connects back to the first
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            # Calculate the 'edge' vector and its magnitude
            edge = p2 - p1  # (p2x-p1x, p2y-p1y)
            edge_length = np.linalg.norm(edge)  # sqrt( edgex**2 + edgey**2 )

            # Safety check for duplicate points
            if edge_length == 0:
                continue

            # Create the "Ruler" (Unit Normal)
            # We rotate the edge 90 degrees CCW and scale it to a length of 1
            # This points perpendicular to the edge we are currently "resting" on
            unit_normal = np.array([-edge[1], edge[0]]) / edge_length

            # Project all points onto this ruler
            # The dot product tells us where each vertex "lands" on our normal axis
            projections = np.dot(hull_points, unit_normal)

            # Calculate the span (width) for this specific orientation
            # Max - Min gives the total thickness of the shape in this direction
            width = np.max(projections) - np.min(projections)

            # Update the "Minimum" if this orientation is narrower than previous ones
            min_diameter = min(min_diameter, width)

        min_diameter = float(min_diameter * 1e-3)

        return round(min_diameter, 3)

    def _contour_max_diameter(
        self, x_contour: NDArray[float64], y_contour: NDArray[float64]
    ) -> float:
        """
        Calculate max diameter of a contour using Convex Hull.

        Args:
            x_contour (NDArray[float64]): The x-coordinate points of the contour.
            y_contour (NDArray[float64]): The y-coordinate points of the contour.

        Returns:
            float: The maximum calculated diameter of the contour. Returns zero
            if it is impossible to calculate the convex hull.
        """
        # Check if we have enough points to even make a shape
        if len(x_contour) < 3:
            return 0.0

        # Create the x-y pairs
        points = np.column_stack([x_contour, y_contour])

        # Create the "Rubber Band" around the points
        # This ignores all internal points and gives us the outer boundary
        hull = ConvexHull(points)

        # Extract the actual X-Y coordinates of the hull vertices
        # hull.vertices provides the indices; points[hull.vertices] gives coordinates
        hull_points = points[hull.vertices]

        # Calculate the Maximum Diameter (Feret Max)
        # pdist finds the distance between every possible pair of hull points
        # np.max picks the longest one
        max_diameter = float(np.max(pdist(hull_points))) * 1e-3  # convert to mm

        return round(max_diameter, 3)

    def _contour_area(
        self, x_contour: NDArray[float64], y_contour: NDArray[float64]
    ) -> float:
        """
        Calculate the area of a enclosed by a contour.

        Args:
            x_contour (NDArray[float64]): The x-coordinate points of the contour.
            y_contour (NDArray[float64]): The y-coordinate points of the contour.

        Returns:
            float: The enclosed area calculated diameter of the contour.
            Returns zero if the contour is not closed.
        """
        # Check if contour is closed (first point (p1) equals last point (pn))
        p1 = [x_contour[0], y_contour[0]]
        pn = [x_contour[-1], y_contour[-1]]
        if not np.allclose(p1, pn):
            area = 0.0
        else:
            # Calculate Area (Shoelace Formula)
            area = 0.5 * np.abs(
                np.dot(x_contour, np.roll(y_contour, 1))
                - np.dot(y_contour, np.roll(x_contour, 1))
            )
        area = float(area)
        return round(area, 2)

    # --- Scan data properties ---

    @property
    def x_location(self) -> Series:
        """
        GETTER: Gets the x-location array of the faraday cup.

        Units are in micrometers
        """
        return self._data['X']

    @property
    def y_location(self) -> Series:
        """
        GETTER: Gets the y-location array of the faraday cup.

        Units are in micrometers
        """
        return self._data['Y']

    @property
    def cup_current(self) -> Series:
        """GETTER: Gets the cup current array in nanoamps."""
        return self._data['cup_current'] * 1e9

    @property
    def screen_current(self) -> Series:
        """GETTER: Gets the screen current array in microamps."""
        return self._data['screen_current'] * 1e6

    @property
    def total_current(self) -> Series:
        """GETTER: Gets the total current array in microamps."""
        return self._data['total_current']

    # --- Metadata properties ---

    @property
    def csv_version(self) -> int:
        """GETTER: Gets the csv version number."""
        return self._metadata['csv_version']

    @property
    def serial_number(self) -> str:
        """GETTER: Gets the serial number of the ion source."""
        return self._metadata['serial_number']

    @property
    def scan_datetime(self) -> str:
        """GETTER: Gets the timestamps of when the scan finished."""
        return self._metadata['scan_datetime']

    @property
    def step_size(self) -> float:
        """GETTER: Gets the step size of faraday cup movement during the scan."""
        return self._metadata['step_size']

    @property
    def beam_voltage(self) -> float:
        """GETTER: Gets the beam voltage setting in kilovolts."""
        return self._metadata['beam_voltage']

    @property
    def extractor_voltage(self) -> float:
        """GETTER: Gets the extractor voltage setting in kilovolts."""
        return self._metadata['extractor_voltage']

    @property
    def solenoid_current(self) -> float:
        """GETTER: Gets the solenoid current setting during the scan in amps."""
        return self._metadata['solenoid_current']

    @property
    def test_stand(self) -> str:
        """GETTER: Gets the test stand number the scan was performed on."""
        return self._metadata['test_stand']

    @property
    def beam_supply_current(self) -> float:
        """GETTER: Gets the beam supply current at the end of the scan in microamps."""
        return self._metadata['beam_supply_current']

    @property
    def pressure(self) -> float:
        """GETTER: Gets the pressure during the beam scan in millibar."""
        return self._metadata['pressure']

    @property
    def fcup_distance(self) -> float:
        """GETTER: Gets the faraday cup distance in millimeters."""
        return self._metadata['fcup_distance']

    @property
    def fcup_diameter(self) -> float:
        """GETTER: Gets the faraday cup diameter in millimeters."""
        return self._metadata['fcup_diameter']

    @property
    def power(self) -> float:
        """GETTER: Gets the power during the scan in watts."""
        return self._metadata['power']

    # --- Derived properties ---

    @property
    def resolution(self) -> str:
        match self._metadata['step_size']:
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
        if self._metadata['beam_voltage'] < 0:
            return 'NEG'
        else:
            return 'POS'

    @property
    def peak_cup_current(self) -> float:
        return self.cup_current[self._peak_idx]

    @property
    def peak_total_current(self) -> float:
        return self.total_current[self._peak_idx]

    @property
    def half_max(self) -> float:
        return self.peak_cup_current * 0.5

    @property
    def quarter_max(self) -> float:
        return self.peak_cup_current * 0.25

    @property
    def angular_intensity(self) -> NDArray[float64]:
        """
        Computes the angular intensity of the collected cup current based on the given distance (in mm) to the cup
        and the aperture diameter (in mm).

        The angular intensity is calculated as the cup current (converted to milliamps) divided by the solid angle
        subtended by the aperture. The solid angle is approximated using the formula for a small circular aperture.

        Returns:
            NDArray[np.float64]: Computed angular intensity values in milliamps per steradian.
        """

        cup_current = self.cup_current * 1000  # milliamps
        half_angle = np.tan(0.5 * self.fcup_diameter / self.fcup_distance)  # radians
        solid_angle = np.pi * half_angle**2  # steradians
        angular_intensity = cup_current / solid_angle  # mA/sr
        return angular_intensity

    @property
    def weighted_centroid(self) -> tuple[float, float]:
        """
        Compute the weighted centroid of the beam profile, handling negative currents.

        Returns:
            tuple: (Xc, Yc) - centroid coordinates.
        """
        # Zero out the cup current measurements that are below the threshold so
        # that the centroid is calculated from strong beam current readings only.
        # This gets rid of the contribution from the noise to find the centroid.
        threshold: float = self.half_max
        cup_current = np.where(abs(self.grid_z) < abs(threshold), 0, self.grid_z)

        # Add up all of the cup_current readings greater than half_max to get the total current
        total_current = float(np.sum(np.abs(cup_current)))  # Use absolute values

        # Compute a weighted average along the x and y directions to get the centroid coordinates.
        Xc = float(np.sum(self.grid_x * np.abs(cup_current)) / total_current)
        Yc = float(np.sum(self.grid_y * np.abs(cup_current)) / total_current)
        # print(f'Centroid = ({Xc:.1f}, {Yc:.1f})')

        return Xc, Yc

    @property
    def peak_location(self) -> tuple[float, float]:
        """GETTER: Get the (x,y) coordinate of the peak cup current."""
        peak_idx = self._peak_idx
        return self.x_location[peak_idx], self.y_location[peak_idx]

    @property
    def hm_contour(self) -> tuple[NDArray[float64], NDArray[float64]] | None:
        """GETTER: Gets the x-y coordiates of the half-max contour."""
        hm_contour = self._contour(self.half_max)
        return hm_contour

    @property
    def qm_contour(self) -> tuple[NDArray[float64], NDArray[float64]] | None:
        """GETTER: Gets the x-y coordiates of the quarter-max contour."""
        hm_contour = self._contour(self.quarter_max)
        return hm_contour

    # --- csv loading methods ---

    def _load_v3_csv(self, filepath: str) -> tuple[dict, DataFrame]:
        # Load in the metadata from the csv file.
        df: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=13, usecols=[1])

        csv_version: str = str(df.iloc[0, 0])
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

        csv_version: str = str(df.iloc[0, 0])
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

        csv_version: str = str(df.iloc[0, 0])
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

    def _load_v0_csv(self, filepath: str) -> tuple[dict, DataFrame]:
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

    def _check_version(self, filepath: str) -> int:
        # Read the first row
        df: pd.DataFrame = pd.read_csv(filepath, header=None, usecols=[1], nrows=1)
        csv_version: str = str(df.iloc[0, 0])
        match csv_version:
            case '3':
                return 3
            case '2':
                return 2
            case '1':
                return 1
            case _:
                return 0

    # --- Helpers ---

    @property
    def _peak_idx(self) -> int:
        peak_index = {
            'NEG': self.cup_current.idxmin(),
            'POS': self.cup_current.idxmax(),
        }
        return int(peak_index[self.polarity])


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    QApplication([])
    sd = ScanData()
    sd.load_scan_data()
    hm_contour = sd.hm_contour
    qm_contour = sd.qm_contour
    resolution = sd.resolution
    polarity = sd.polarity
    print(f'{sd.csv_version = }')
    if hm_contour:
        print('Half-max contour exists.')
    if qm_contour:
        print('Quarter-max contour exists.')
    # print(f'{sd._metadata = }')
    # print(f'{scan_data.data.head()}')
    # print(f'{resolution = }')
    # print(f'{polarity = }')
    # print(f'{sd.weighted_centroid = }')
