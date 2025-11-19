from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from numpy import float64
from numpy.typing import NDArray
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from PySide6.QtWidgets import QFileDialog

import heatmaps
import surface_figures
from beam_scan_analysis import ScanData

PNG_WIDTH = 700
PNG_HEIGHT = 500


class Plotter:
    """
    A class to handle the preparation and processing of scan data for visualization.

    This class extracts important data points from the provided `ScanData` object, such as
    the scan's polarity, peak location, and peak current values. It also performs interpolation
    on the scan data and computes areas enclosed by contour lines at half-max and quarter-max
    intensity values.

    Attributes:
        scan_data (ScanData): The scan data object containing the raw data from a beam scan.
        solenoid (str): The solenoid current in amps.
        test_stand (str | None): The test stand identifier, or None if not provided.
        serial_number (str): The serial number from the scan data.
        polarity (str): The polarity of the scan ('POS' or 'NEG').
        beam_voltage (int): The beam voltage from the scan data.
        extractor_voltage (int): The extractor voltage from the scan data.
        half_max (float): The half-max intensity value, used for calculating FWHM.
        quarter_max (float): The quarter-max intensity value, used for calculating FWQM.
        peak_location (tuple[float, float]): The (x, y) location of the peak intensity.
        peak_cup_current (float): The peak cup current value from the scan.
        peak_total_current (float): The peak total current value from the scan.
        z_scale (tuple[int | float | None, int | float | None]): The z-axis scale range (optional).
        grid_x (ndarray): The x-coordinates of the interpolated grid.
        grid_y (ndarray): The y-coordinates of the interpolated grid.
        grid_z (ndarray): The z-values (cup current) of the interpolated grid.
        fwhm_enclosed_area (float): The area enclosed by the contour at half-max intensity.
        fwqm_enclosed_area (float): The area enclosed by the contour at quarter-max intensity.
        color (str): The colormap used for plotting, based on polarity.
        levels (list): The contour levels used for plotting.
        centroid (tuple): The coordinates for the weighted geometric center of the beam profile.
        x_slice (pandas DataFrame): A slice of x-locations and cup currents at Y-centoid value for plotting the X-profile view.
        y_slice (pandas DataFrame): A slice of y-locations and cup currents at X-centoid value for plotting the Y-profile view.
        i_prime (ndarray): The cup current reading converted to angular intensity readings.
    """

    def __init__(
        self,
        scan_data: ScanData,
        solenoid: str,
        fcup_diam: float,
        fcup_dist: float,
        test_stand: str | None = None,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        self.scan_data: ScanData = scan_data
        self.solenoid: str = solenoid
        self.test_stand: str | None = test_stand
        self.serial_number: str = scan_data.serial_num
        self.polarity: str = scan_data.polarity
        self.beam_voltage: int | float | None = scan_data.beam_voltage
        self.extractor_voltage: int | float | None = scan_data.extractor_voltage
        self.half_max: float = scan_data.half_max()
        self.quarter_max: float = scan_data.quarter_max()
        self.peak_location: tuple[float, float] = scan_data.peak_location()
        self.peak_cup_current: float = scan_data.peak_cup_current()
        self.peak_total_current: float = scan_data.peak_total_current()
        self.fwhm_enclosed_area: float = scan_data.fwhm_area()
        self.fwqm_enclosed_area: float = scan_data.fwqm_area()
        self.centroid: tuple[float, float] = scan_data.compute_weighted_centroid()

        self.z_scale: list[int | float | None] = z_scale

        self.grid_x: NDArray[float64] = self.scan_data.grid_x
        self.grid_y: NDArray[float64] = self.scan_data.grid_y
        self.grid_z: NDArray[float64] = self.scan_data.grid_z

        self.y_idx: int = int(np.abs(self.grid_y[:, 0] - self.centroid[1]).argmin())
        self.x_idx: int = int(np.abs(self.grid_x[0, :] - self.centroid[0]).argmin())
        self.x_slice = pd.DataFrame(
            {
                'X Coordinate': self.grid_x[self.y_idx, :],
                'Faraday Cup Current': self.grid_z[self.y_idx, :],
            }
        )
        self.y_slice = pd.DataFrame(
            {
                'Y Coordinate': self.grid_y[:, self.x_idx],
                'Faraday Cup Current': self.grid_z[:, self.x_idx],
            }
        )

        # Set the plotting color based on polarity of beam scan
        colors: dict[str, str] = {'NEG': 'viridis_r', 'POS': 'viridis'}
        self.color: str = colors[self.polarity]

        # Sort the contour levels based on polarity of beam scan
        self.levels: list = sorted([self.quarter_max, self.half_max])

        # Set the default 3D surface renderer to be the user's browser
        pio.renderers.default = 'browser'

    @staticmethod
    def save_as_html(
        fig: Figure | None, default_filename: str | None = None, parent=None
    ) -> None:
        if not default_filename:
            default_filename = ''

        file_path, _ = QFileDialog.getSaveFileName(
            parent=parent,
            caption='Save figure',
            dir=default_filename,
            filter='HTML Files (*.html);;All Files (*)',
        )

        if fig and file_path:
            fig.write_html(file_path)
        else:
            pass

    @staticmethod
    def save_all_as_html(
        titles_and_figs: dict[str, Figure | None],
        filename: str,
        default_dir: str | None = None,
        parent=None,
    ) -> None:
        if default_dir is None:
            default_dir = ''

        folder_path = QFileDialog.getExistingDirectory(
            parent=parent,
            caption='Select folder to save figures',
            dir=default_dir,
            options=QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        folder = Path(folder_path)
        for title, fig in titles_and_figs.items():
            if fig is None:
                continue
            file_name = f'{filename} {title}'
            full_path = folder / file_name
            fig.write_html(str(full_path))

    @staticmethod
    def save_all_as_png(
        titles_and_figs: dict[str, Figure | None],
        filename: str,
        default_dir: str | None = None,
        parent=None,
    ) -> None:
        if default_dir is None:
            default_dir = ''

        folder_path = QFileDialog.getExistingDirectory(
            parent=parent,
            caption='Select folder to save figures',
            dir=default_dir,
            options=QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        folder = Path(folder_path)
        for title, fig in titles_and_figs.items():
            if fig is None:
                continue
            file_name = f'{filename} {title}'
            full_path = folder / file_name
            fig.write_image(str(full_path), width=PNG_WIDTH, height=PNG_HEIGHT)


class Surface(Plotter):
    def __init__(
        self,
        scan_data: ScanData,
        solenoid: str,
        fcup_diam: float,
        fcup_dist: float,
        test_stand: str | None = None,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(scan_data, solenoid, fcup_diam, fcup_dist, test_stand, z_scale)

    def plot_surface(self, show=True) -> None | Figure:
        fig = surface_figures.surface(
            self,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self.centroid,
            self.x_slice,
            self.y_slice,
        )
        if show is False:
            return fig

        fig.show()


class Heatmap(Plotter):
    def __init__(
        self,
        scan_data: ScanData,
        solenoid: str,
        fcup_diam: float,
        fcup_dist: float,
        test_stand: str | None = None,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(scan_data, solenoid, fcup_diam, fcup_dist, test_stand, z_scale)

    def plot_heatmap(self, show=True) -> None | Figure:
        heatmap = heatmaps.heatmap(
            self.grid_x, self.grid_y, self.grid_z, self.z_scale, self.color
        )
        contour_size = (max(self.levels) - min(self.levels)) - 1e-9

        # Create contour
        if contour_size > 0:
            contour = go.Contour(
                z=self.grid_z,
                x=np.linspace(
                    self.grid_x.min(), self.grid_x.max(), self.grid_z.shape[1]
                ),
                y=np.linspace(
                    self.grid_y.min(), self.grid_y.max(), self.grid_z.shape[0]
                ),
                contours=dict(
                    coloring='lines',
                    showlabels=False,
                    start=min(self.levels),
                    end=max(self.levels),
                    size=contour_size,
                ),
                line=dict(color='red', width=1),
                name='',
                showscale=False,  # do not show level on colorbar
            )

            # Create figure
            fig = go.Figure(data=[heatmap, contour])
        else:
            fig = go.Figure(data=[heatmap])

        # Add peak location annotation
        fig.add_trace(
            go.Scatter(
                x=[self.peak_location[0]],
                y=[self.peak_location[1]],
                mode='markers',
                textposition='bottom center',
                marker=dict(color='black', size=2.5, symbol='circle'),
                name='Peak',
            )
        )

        # Add X-axis cross-section line
        fig.add_trace(
            go.Scatter(
                x=self.x_slice['X Coordinate'],
                y=[self.centroid[1]] * len(self.x_slice),
                mode='lines',
                line=dict(color='black', width=1),
                name='X-Cross Section',
            )
        )

        # Add Y-axis cross-section line
        fig.add_trace(
            go.Scatter(
                x=[self.centroid[0]] * len(self.y_slice),
                y=self.y_slice['Y Coordinate'],
                mode='lines',
                line=dict(color='black', width=1),
                name='Y-Cross Section',
            )
        )

        # Add additional annotations (customize positions and text as needed)
        fig.add_annotation(
            # x=0.275, y=1, xref="paper", yref="paper",
            x=self.scan_data.x_location.max(),
            y=1,
            yref='paper',
            text=f'Cup current = {self.peak_cup_current * 1e9:.1f} nA <br>'
            f'Total current = {self.peak_total_current * 1e6:.3f} µA <br>'
            f'Settings = {self.beam_voltage}/{self.extractor_voltage} kV & {self.solenoid} A',
            showarrow=False,
            align='left',
            xanchor='left',
            yanchor='bottom',
        )
        fig.add_annotation(
            x=self.scan_data.x_location.min(),
            y=1,
            yref='paper',
            text=f'FWHM Area = {self.fwhm_enclosed_area:.3f} mm²<br>'
            f'FWQM Area = {self.fwqm_enclosed_area:.3f} mm²<br>'
            f'peak = ({self.peak_location[0]:.0f},{self.peak_location[1]:.0f})',
            showarrow=False,
            align='right',
            xanchor='right',
            yanchor='bottom',
        )

        # Set title and axis properties
        fig.update_layout(
            title=dict(
                text=f'{self.serial_number} on TS{self.test_stand}',
                x=0.475,
                xanchor='center',
            ),
            xaxis=dict(
                title='X Location',
                range=[self.grid_x.max(), self.grid_x.min()],
                scaleanchor='y',
                showgrid=True,
                autorange=False,
            ),
            yaxis=dict(
                title='Y Location',
                range=[self.grid_x.max(), self.grid_x.min()],
                scaleanchor='x',
                showgrid=True,
                autorange=False,
            ),
            showlegend=False,
        )

        # Show the plot
        if show is False:
            return fig

        fig.show()


class XYCrossSections(Plotter):
    def __init__(
        self,
        scan_data: ScanData,
        solenoid: str,
        fcup_diam: float,
        fcup_dist: float,
        test_stand: str | None = None,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(scan_data, solenoid, fcup_diam, fcup_dist, test_stand, z_scale)

    def plot_cross_sections(self, show=True) -> Figure | None:
        scaling_factor = 1e-6  # scale to microamps
        self.z_scale = [
            value * scaling_factor if value is not None else None
            for value in self.z_scale
        ]

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=['X Cross Section', 'Y Cross Section']
        )

        fig.add_trace(
            go.Scatter(
                x=self.x_slice['X Coordinate'],
                y=self.x_slice['Faraday Cup Current'],
                mode='lines',
                name='X-Axis Cross Section',
                line=dict(color='blue', width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.y_slice['Y Coordinate'],
                y=self.y_slice['Faraday Cup Current'],
                mode='lines',
                name='Y-Axis Cross Section',
                line=dict(color='red', width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            title_text=f'{self.serial_number} Beam Current Cross Sections',
        )

        if any(value is not None for value in self.z_scale):
            fig.update_xaxes(title_text='X Location', row=1, col=1)
            fig.update_xaxes(title_text='Y Location', row=1, col=2)
            fig.update_yaxes(
                title_text='Cup Current (A)', row=1, col=1, range=self.z_scale
            )
            fig.update_yaxes(
                row=1,
                col=2,
                range=self.z_scale,
                matches='y',
            )
        else:  # autoscale
            fig.update_xaxes(title_text='X Location', row=1, col=1)
            fig.update_xaxes(title_text='Y Location', row=1, col=2)
            fig.update_yaxes(title_text='Cup Current (A)', row=1, col=1)
            fig.update_yaxes(
                row=1,
                col=2,
                matches='y',
            )

        if show is False:
            return fig

        fig.show()


class IPrime(Plotter):
    def __init__(
        self,
        scan_data: ScanData,
        solenoid: str,
        fcup_diam: float,
        fcup_dist: float,
        test_stand: str | None = None,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(scan_data, solenoid, fcup_diam, fcup_dist, test_stand, z_scale)
        self.fcup_diameter = fcup_diam
        self.fcup_distance = fcup_dist

    def plot_i_prime(
        self,
        show=True,
    ) -> Figure | None:
        Xc = self.centroid[0]
        Yc = self.centroid[1]
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=['X Cross Section', 'Y Cross Section']
        )

        i_prime: NDArray[np.float64] = self.scan_data.compute_angular_intensity(
            self.fcup_distance, self.fcup_diameter
        )

        y_idx: int = int(np.abs(self.grid_y[:, 0] - Yc).argmin())
        x_idx: int = int(np.abs(self.grid_x[0, :] - Xc).argmin())
        self.x_slice = pd.DataFrame(
            {
                'X Coordinate': self.grid_x[y_idx, :],
                'Faraday Cup Current': self.grid_z[y_idx, :],
                'Angular Intensity': i_prime[self.y_idx, :],
            }
        )
        self.y_slice = pd.DataFrame(
            {
                'Y Coordinate': self.grid_y[:, x_idx],
                'Faraday Cup Current': self.grid_z[:, x_idx],
                'Angular Intensity': i_prime[:, self.x_idx],
            }
        )

        x_center: float = Xc  # equivalent to 0 radians on x-slice
        y_center: float = Yc  # equivalent to 0 radians on y-slice
        dist_from_x_center: NDArray[np.float64] = (
            np.asarray(self.x_slice['X Coordinate'] - x_center) / 1000
        )  # millimeters
        dist_from_y_center: NDArray[np.float64] = (
            np.asarray(self.y_slice['Y Coordinate'] - y_center) / 1000
        )  # millimeters
        x_angle: NDArray[np.float64] = (
            np.arctan(dist_from_x_center / self.fcup_distance) * 1000.0
        )  # milli-radians
        y_angle: NDArray[np.float64] = (
            np.arctan(dist_from_y_center / self.fcup_distance) * 1000.0
        )  # milli-radians

        fig.add_trace(
            go.Scatter(
                x=x_angle,
                y=self.x_slice['Angular Intensity'],
                mode='lines',
                name="X-Axis i'",
                line=dict(color='blue', width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=y_angle,
                y=self.y_slice['Angular Intensity'],
                mode='lines',
                name="Y-Axis i'",
                line=dict(color='red', width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            title_text=f'{self.serial_number} Angular Intensity vs Divergence Angle',
        )

        fig.update_xaxes(title_text='X Divergence Angle (mRad)', row=1, col=1)
        fig.update_xaxes(title_text='Y Divergence Angle (mRad)', row=1, col=2)
        fig.update_yaxes(title_text='Angular Intensity (mA/sr)', row=1, col=1)
        fig.update_yaxes(row=1, col=2, matches='y')

        if show is False:
            return fig

        fig.show()


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    from load_scan_data import CSVLoader

    QApplication([])
    filepath: str = CSVLoader.select_csv()
    scan_data: ScanData = CSVLoader.load_scan_data(filepath)
    if scan_data.polarity == 'NEG':
        z_scale: list[int | float | None] = [None, None]
        solenoid: str = '2.5'
    else:
        z_scale: list[int | float | None] = [None, None]
        solenoid: str = '0.3'
    fcup_diam = 2.5
    fcup_dist = 205
    surface = Surface(
        scan_data, solenoid, fcup_diam, fcup_dist, test_stand='4', z_scale=z_scale
    )
    surface.plot_surface()
    heatmap = Heatmap(
        scan_data, solenoid, fcup_diam, fcup_dist, test_stand='4', z_scale=z_scale
    )
    heatmap.plot_heatmap()
    xy_cross_sections = XYCrossSections(
        scan_data, solenoid, fcup_diam, fcup_dist, test_stand='4', z_scale=z_scale
    )
    xy_cross_sections.plot_cross_sections()
    i_prime = IPrime(
        scan_data, solenoid, fcup_diam, fcup_dist, test_stand='4', z_scale=z_scale
    )
    i_prime.plot_i_prime()
