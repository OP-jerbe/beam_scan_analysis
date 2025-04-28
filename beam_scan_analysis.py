from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy import float64
from numpy.typing import NDArray

# from PySide6.QtWidgets import QFileDialog
# import sys
from scipy.interpolate import griddata
from skimage import (
    measure,
)  # For find_contours <conda install -c conda-forge scikit-image>


@dataclass
class ScanData:
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

    def _peak_idx(self) -> int | str:
        """
        Get the index of the peak value in `cup_current` based on polarity.

        Returns:
            int | str: The index of the minimum (`idxmin()`) if polarity is 'NEG',
                    or the maximum (`idxmax()`) if polarity is 'POS'.
        """
        indeces = {'NEG': self.cup_current.idxmin(), 'POS': self.cup_current.idxmax()}
        return indeces[self.polarity]

    def create_grid(self) -> tuple[NDArray, NDArray, NDArray]:
        x: NDArray[float64] = self.x_location.to_numpy()
        y: NDArray[float64] = self.y_location.to_numpy()
        z: NDArray[float64] = self.cup_current.to_numpy()

        grid_x: NDArray[float64]
        grid_y: NDArray[float64]
        grid_z: NDArray[float64]

        # Create grid for interpolation
        interp_num: int = 1000
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), interp_num),
            np.linspace(y.min(), y.max(), interp_num),
        )

        # Interpolate data to grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        return grid_x, grid_y, grid_z

    def fwhm_area(self) -> float:
        x, y, z = self.create_grid()
        half_max = self.half_max()

        # Calculate the area enclosed by the contour lines at half max of cup current
        try:
            fwhm_enclosed_area: float = (
                self.area_enclosed_by_contour(x, y, z, half_max) * 1e-6
            )  # sq-mm
        except Exception as e:
            fwhm_enclosed_area: float = 0.0
            print(f'{e}')
            print('Could not calculate FWHM area.')

        return fwhm_enclosed_area

    def fwqm_area(self) -> float:
        x, y, z = self.create_grid()
        quarter_max = self.quarter_max()

        # Calculate the area enclosed by the contour lines at quarter max of cup current
        try:
            fwqm_enclosed_area: float = (
                self.area_enclosed_by_contour(x, y, z, quarter_max) * 1e-6
            )  # sq-mm
        except Exception as e:
            fwqm_enclosed_area: float = 0.0
            print(f'{e}')
            print('Could not calculate FWQM area.')

        return fwqm_enclosed_area

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

    def area_enclosed_by_contour(
        self, x: NDArray, y: NDArray, z: NDArray, level: int | float
    ) -> float:
        """
        Calculate the area enclosed by a contour at a specified level in a 2D array.

        Args:
            x (NDArray): The x-axis values corresponding to the columns of `z`.
            y (NDArray): The y-axis values corresponding to the rows of `z`.
            z (NDArray): A 2D array of values from which to extract the contour.
            level (int | float): The level at which to find the contour in `z`.

        Returns:
            float: The area enclosed by the contour at the specified level.

        Notes:
            Uses the Shoelace formula to compute the area within the closed contour.
            Ensures the contour is closed by appending the starting point to the end
            if necessary.
        """
        contour = np.array(measure.find_contours(z, level))[0]
        x_contour = np.interp(
            contour[:, 1], [0, z.shape[1] - 1], [x.min(), x.max()]
        )  # X-axis
        y_contour = np.interp(
            contour[:, 0], [0, z.shape[0] - 1], [y.min(), y.max()]
        )  # Y-axis
        contours = np.column_stack([x_contour, y_contour])
        if not np.allclose(contours[0], contours[-1]):
            # contours = np.vstack([contours, contours[0]])  # Append the first point to the end to ensure the contour is closed in order to calculate area
            return 0.0  # i.e. do not try to calculate area if contour goes off of the stage limits
        x_contour = contours[:, 1]
        y_contour = contours[:, 0]
        return 0.5 * np.abs(
            np.dot(x_contour, np.roll(y_contour, 1))
            - np.dot(y_contour, np.roll(x_contour, 1))
        )

    def compute_weighted_centroid(self) -> tuple[float, float]:
        """
        Compute the weighted centroid of the beam profile, handling negative currents.

        Returns:
            tuple: (Xc, Yc) - centroid coordinates.
        """

        x, y, cup_current = self.create_grid()

        # Zero out the cup current measurements that are below the threshold so that the centroid is calculated from strong beam current readings.
        # This gets rid of the contribution from the noise to find the centroid.
        threshold: float = self.half_max()
        cup_current = np.where(abs(cup_current) < abs(threshold), 0, cup_current)

        total_current = float(np.sum(np.abs(cup_current)))  # Use absolute values

        Xc = float(np.sum(x * np.abs(cup_current)) / total_current)
        Yc = float(np.sum(y * np.abs(cup_current)) / total_current)

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

        _, _, cup_current = self.create_grid()
        cup_current_in_milliamps = cup_current * 1000  # milliamps
        half_angle = np.tan(0.5 * diameter / (distance))  # radians
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
        }

    def to_csv(self, filename: str, index: bool = False) -> None:
        """
        Save scan data to a CSV file.

        Args:
            filename (str): The path of the CSV file to save the data to.
            index (bool, optional): Whether to write row names (index) to the file. Defaults to False.

        Returns:
            None: This function does not return any value. It saves the data to a CSV file.

        Notes:
            The data saved includes `x_location`, `y_location`, `cup_current`, `screen_current`,
            and `total_current` from the current object.
        """
        data = pd.DataFrame(
            {
                'x_location': self.x_location,
                'y_location': self.y_location,
                'cup_current': self.cup_current,
                'screen_current': self.screen_current,
                'total_current': self.total_current,
            }
        )
        data.to_csv(filename, index=index)


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    from load_scan_data import CSVLoader

    QApplication([])
    csv_loader: CSVLoader = CSVLoader()
    filepath: str = csv_loader.select_csv()
    scan_data: ScanData = csv_loader.load_scan_data(filepath)
    print(scan_data.display_summary())
