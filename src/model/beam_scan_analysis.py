from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy import float64
from numpy.typing import NDArray

# from PySide6.QtWidgets import QFileDialog
# import sys
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import (
    measure,
)  # For find_contours <conda install -c conda-forge scikit-image>


@dataclass
class ScanData:
    """
    A class to hold the beam scan data.
    """

    serial_num: str
    scan_datetime: str
    step_size: float
    resolution: str
    x_location: pd.Series
    y_location: pd.Series
    cup_current: pd.Series
    screen_current: pd.Series
    total_current: pd.Series
    polarity: str
    beam_voltage: int | float | None
    extractor_voltage: int | float | None
    solenoid_current: str | None
    beam_supply_current: str
    pressure: str
    test_stand: str | None
    fcup_distance: str | None
    fcup_diameter: str | None
    well_structured_csv: bool

    disable_interp_flag: bool = False

    interp_num: int = field(init=False)
    grid_x: NDArray[float64] = field(init=False)
    grid_y: NDArray[float64] = field(init=False)
    grid_z: NDArray[float64] = field(init=False)

    def __post_init__(self) -> None:
        # if the override option is not checked
        if not self.disable_interp_flag:
            self.interp_num = 500
            self.grid_x, self.grid_y, self.grid_z = self.create_grid(self.interp_num)
            return

        ##### NOT NEEDED FOR GUI BUT MAY BE USEFUL FOR LABVIEW IMPLEMENTATION -- DO NOT DELETE #####
        # if the override option is checked
        # match self.resolution:
        #     case 'Highest':
        #         interp_num = 65
        #     case 'High':
        #         interp_num = 33
        #     case 'Med':
        #         interp_num = 17
        #     case 'Low':
        #         interp_num = 9
        #     case _:
        #         interp_num = 500
        # self.grid_x, self.grid_y, self.grid_z = self.create_grid(interp_num)

    def _peak_idx(self) -> int | str:
        """
        Get the index of the peak value in `cup_current` based on polarity.

        Returns:
            int | str: The index of the minimum (`idxmin()`) if polarity is 'NEG',
                    or the maximum (`idxmax()`) if polarity is 'POS'.
        """
        indeces = {'NEG': self.cup_current.idxmin(), 'POS': self.cup_current.idxmax()}
        return indeces[self.polarity]

    def create_grid(
        self, interp_num: int
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        x: NDArray[float64] = self.x_location.to_numpy()
        y: NDArray[float64] = self.y_location.to_numpy()
        z: NDArray[float64] = self.cup_current.to_numpy()

        # Create grid for interpolation
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), interp_num),
            np.linspace(y.min(), y.max(), interp_num),
        )

        # Interpolate data to grid
        self.grid_z = griddata((x, y), z, (self.grid_x, self.grid_y), method='cubic')

        return self.grid_x, self.grid_y, self.grid_z

    def fwhm_area(self) -> float:
        half_max = self.half_max()

        # Calculate the area enclosed by the contour lines at half max of cup current
        try:
            fwhm_properties: dict = self.get_contour_properties(
                self.grid_x, self.grid_y, self.grid_z, half_max
            )
            fwhm_enclosed_area = fwhm_properties['area'] * 1e-6  # mm-sq
        except Exception as e:
            fwhm_enclosed_area: float = 0.0
            print(f'{e}')
            print('Could not calculate FWHM area.')

        return fwhm_enclosed_area

    def fwhm_diams(self) -> dict[str, float]:
        half_max = self.half_max()

        # Calculate the area enclosed by the contour lines at half max of cup current
        try:
            fwhm_properties: dict = self.get_contour_properties(
                self.grid_x, self.grid_y, self.grid_z, half_max
            )
            fwhm_max_diam = fwhm_properties['max_diameter']  # mm
            fwhm_min_diam = fwhm_properties['min_diameter']  # mm
            fwhm_diams = {'min': fwhm_min_diam, 'max': fwhm_max_diam}
        except Exception as e:
            fwhm_diams = {'min': 0.0, 'max': 0.0}
            print(f'{e}')
            print('Could not calculate FWHM area.')

        return fwhm_diams

    def fwqm_area(self) -> float:
        quarter_max = self.quarter_max()

        # Calculate the area enclosed by the contour lines at quarter max of cup current
        try:
            fwqm_properties = self.get_contour_properties(
                self.grid_x, self.grid_y, self.grid_z, quarter_max
            )
            fwqm_enclosed_area: float = fwqm_properties['area'] * 1e-6
        except Exception as e:
            fwqm_enclosed_area: float = 0.0
            print(f'{e}')
            print('Could not calculate FWQM area.')

        return fwqm_enclosed_area

    def fwqm_diams(self) -> dict[str, float]:
        quarter_max = self.quarter_max()

        # Calculate the area enclosed by the contour lines at quarter max of cup current
        try:
            fwhm_properties: dict = self.get_contour_properties(
                self.grid_x, self.grid_y, self.grid_z, quarter_max
            )
            fwqm_max_diam = fwhm_properties['max_diameter']  # mm
            fwqm_min_diam = fwhm_properties['min_diameter']  # mm
            fwqm_diams = {'min': fwqm_min_diam, 'max': fwqm_max_diam}
        except Exception as e:
            fwqm_diams = {'min': 0.0, 'max': 0.0}
            print(f'{e}')
            print('Could not calculate FWHM area.')

        return fwqm_diams

    def peak_location(self) -> tuple[float, float]:
        """
        Get the x and y coordinates of the peak value in `cup_current` based on polarity.

        Returns:
            tuple[float, float]: A tuple containing the x and y coordinates
                                of the peak value, determined by the peak index.
        """
        peak_idx = self._peak_idx()
        return (self.x_location[peak_idx], self.y_location[peak_idx])

    def peak_cup_current(self) -> float:
        """
        Get the peak cup current value based on polarity. Value is in amperes.

        Returns:
            float: The cup current value at the peak index, determined by the polarity.
        """
        peak_idx = self._peak_idx()
        return self.cup_current[peak_idx]

    def peak_total_current(self) -> float:
        """
        Get the peak total current value based on polarity. Value is in amperes.

        Returns:
            float: The total current value at the peak index, determined by the polarity.
        """
        peak_idx = self._peak_idx()
        return self.total_current[peak_idx]

    def half_max(self) -> float:
        """
        Calculate half of the peak cup current. Value is in amperes.

        Returns:
            float: Half of the peak cup current value.
        """
        return self.peak_cup_current() * 0.5

    def quarter_max(self) -> float:
        """
        Calculate one-quarter of the peak cup current. Value is in amperes.

        Returns:
            float: one-quarter of the peak cup current value.
        """
        return self.peak_cup_current() * 0.25

    def get_contour_properties(
        self, x: NDArray, y: NDArray, z: NDArray, level: int | float
    ) -> dict:
        """
        Calculate the area, max diameter, and min diameter of a contour.
        """

        # 1. Get the contours but check to make sure they exists first
        contours = self.get_contours(x, y, z, level)
        if not contours:
            return {'area': 0.0, 'max_diameter': 0.0, 'min_diameter': 0.0}
        x_contour, y_contour = contours

        # 2. Calculate the min and max diameter of the contour
        min_diameter, max_diameter = self.get_min_max_diameters(x_contour, y_contour)

        # Check if contour is closed (first point (p1) equals last point (pn))
        p1 = [x_contour[0], y_contour[0]]
        pn = [x_contour[-1], y_contour[-1]]
        if not np.allclose(p1, pn):
            area = 0.0
            min_diameter = 0.0
            max_diameter = 0.0
        else:
            # 3. Calculate Area (Shoelace Formula)
            area = 0.5 * np.abs(
                np.dot(x_contour, np.roll(y_contour, 1))
                - np.dot(y_contour, np.roll(x_contour, 1))
            )

        return {
            'area': area,
            'max_diameter': max_diameter,
            'min_diameter': min_diameter,
        }

    @staticmethod
    def get_contours(
        x: NDArray[float64],
        y: NDArray[float64],
        z: NDArray[float64],
        level: int | float,
    ) -> tuple[NDArray[float64], NDArray[float64]] | None:
        # Extract and scale the contour
        raw_contours = measure.find_contours(z, level)

        if not raw_contours:
            return None

        contour = np.array(raw_contours[0])

        # Map from pixel indices (row, col) to data coordinates (y, x)
        x_contour = np.interp(contour[:, 1], [0, z.shape[1] - 1], [x.min(), x.max()])
        y_contour = np.interp(contour[:, 0], [0, z.shape[0] - 1], [y.min(), y.max()])

        return x_contour, y_contour

    @staticmethod
    def get_min_max_diameters(x_contour, y_contour) -> tuple[float, float]:
        """Calculate Diameters using Convex Hull"""
        # Check if we have enough points to even make a shape
        if len(x_contour) < 3:
            return 0.0, 0.0

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
        max_diameter = np.max(pdist(hull_points))

        # Prepare to find the Minimum Diameter (Rotating Calipers logic)
        # We initialize with infinity so any real width will be smaller
        min_diameter = float('inf')
        n = len(hull_points)

        # Iterate over every flat edge of the hull
        for i in range(n):
            # Define the edge using current point (p1) and next point (p2)
            # The % n ensures the last point connects back to the first
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            # Calculate the 'edge' vector and its magnitude
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)

            # Safety check for duplicate points
            if edge_length == 0:
                continue

            # Create the "Ruler" (Unit Normal)
            # We rotate the edge 90 degrees and scale it to a length of 1
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
        max_diameter = float(max_diameter * 1e-3)

        return round(min_diameter, 3), round(max_diameter, 3)

    def compute_weighted_centroid(self) -> tuple[float, float]:
        """
        Compute the weighted centroid of the beam profile, handling negative currents.

        Returns:
            tuple: (Xc, Yc) - centroid coordinates.
        """

        # Zero out the cup current measurements that are below the threshold so that the centroid is calculated from strong beam current readings.
        # This gets rid of the contribution from the noise to find the centroid.
        threshold: float = self.half_max()
        cup_current = np.where(abs(self.grid_z) < abs(threshold), 0, self.grid_z)

        total_current = float(np.sum(np.abs(cup_current)))  # Use absolute values

        Xc = float(np.sum(self.grid_x * np.abs(cup_current)) / total_current)
        Yc = float(np.sum(self.grid_y * np.abs(cup_current)) / total_current)
        # print(f'Centroid = ({Xc:.1f}, {Yc:.1f})')

        return Xc, Yc

    def compute_angular_intensity(
        self, distance: int | float, diameter: int | float
    ) -> NDArray[float64]:
        """
        Computes the angular intensity of the collected cup current based on the given distance (in mm) to the cup
        and the aperture diameter (in mm).

        The angular intensity is calculated as the cup current (converted to milliamps) divided by the solid angle
        subtended by the aperture. The solid angle is approximated using the formula for a small circular aperture.

        Args:
            distance (int | float): Distance from the source aperture to the faraday cup screen aperture (same unit as diameter).
            diameter (int | float): Diameter of the cup aperture (same unit as distance).

        Returns:
            NDArray[np.float64]: Computed angular intensity values in milliamps per steradian.
        """

        cup_current_in_milliamps = self.grid_z * 1000  # milliamps
        half_angle = np.tan(0.5 * diameter / distance)  # radians
        solid_angle = np.pi * half_angle**2  # steradians
        return cup_current_in_milliamps / solid_angle  # milliamps/steradian

    def display_summary(self) -> dict[str, Any]:
        """
        Generate a summary of key scan parameters and peak measurements.

        Returns:
            dict[str, Any]: A dictionary containing the scan's serial number, datetime,
                            step size, resolution, polarity, beam and extractor voltages,
                            as well as the peak location, peak cup current, and peak
                            total current values.
        """

        return {
            'serial_number': self.serial_num,
            'scan_datetime': self.scan_datetime,
            'resolution': self.resolution,
            'step_size': self.step_size,
            'polarity': self.polarity,
            'beam_voltage': self.beam_voltage,
            'extractor_voltage': self.extractor_voltage,
            'beam_supply_current': self.beam_supply_current,
            'pressure': self.pressure,
            'peak_location': self.peak_location(),
            'peak_cup_current': self.peak_cup_current()
            * 1e9,  # returns value in nanoamps
            'peak_total_current': self.peak_total_current()
            * 1e6,  # returns value in microamps
            'FWHM_area': self.fwhm_area(),
            'FWQM_area': self.fwqm_area(),
            'solenoid_current': self.solenoid_current,
            'test_stand': self.test_stand,
            'centroid': self.compute_weighted_centroid(),
            'FWHM_min_diam': self.fwhm_diams()['min'],
            'FWHM_max_diam': self.fwhm_diams()['max'],
            'FWQM_min_diam': self.fwqm_diams()['min'],
            'FWQM_max_diam': self.fwqm_diams()['max'],
        }


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    from helpers.load_scan_data import CSVLoader

    QApplication([])
    filepath: str = CSVLoader.select_csv()
    scan_data: ScanData = CSVLoader.load_scan_data(filepath)
    print(scan_data.display_summary())
