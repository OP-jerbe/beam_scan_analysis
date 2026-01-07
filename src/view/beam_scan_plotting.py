from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import NDArray
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from PySide6.QtWidgets import QFileDialog

import src.view.heatmaps as heatmaps
import src.view.surface_figures as surface_figures
from src.model.beam_scan import BeamScan

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
        beam_scan: BeamScan,
        inputs: dict,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        self.bs = beam_scan
        self.inputs = inputs
        self.z_scale = z_scale
        self.angular_intensity = self.bs.angular_intensity(
            inputs['fcup_diam'], inputs['fcup_dist']
        )

        self.y_slice_idx: int = int(
            np.abs(self.bs.grid_y[:, 0] - self.inputs['centroid_y']).argmin()
        )
        self.x_slice_idx: int = int(
            np.abs(self.bs.grid_x[0, :] - self.inputs['centroid_x']).argmin()
        )
        self.x_slice = pd.DataFrame(
            {
                'X Coordinate': self.bs.grid_x[self.y_slice_idx, :],
                'Faraday Cup Current': self.bs.grid_z[self.y_slice_idx, :],
                'Angular Intensity': self.angular_intensity[self.y_slice_idx, :],
            }
        )
        self.y_slice = pd.DataFrame(
            {
                'Y Coordinate': self.bs.grid_y[:, self.x_slice_idx],
                'Faraday Cup Current': self.bs.grid_z[:, self.x_slice_idx],
                'Angular Intensity': self.angular_intensity[:, self.x_slice_idx],
            }
        )
        self.centroid_slice_x = pd.DataFrame({})
        self.centroid_slice_y = pd.DataFrame({})

        # Set the plotting color based on polarity of beam scan
        colors: dict[str, str] = {'NEG': 'viridis_r', 'POS': 'viridis'}
        self.color: str = colors[self.bs.polarity]

        # Sort the contour levels based on polarity of beam scan
        self.levels: list = sorted([self.bs.quarter_max, self.bs.half_max])

        # Set the default 3D surface renderer to be the user's browser
        pio.renderers.default = 'browser'


class Surface(Plotter):
    def __init__(
        self,
        beam_scan: BeamScan,
        inputs: dict,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(beam_scan, inputs, z_scale)

    def plot(self, show=True) -> None | Figure:
        fig = surface_figures.surface(
            self,
            self.bs,
            self.inputs,
            self.x_slice,
            self.y_slice,
        )
        if show is False:
            return fig

        fig.show()


class Heatmap(Plotter):
    def __init__(
        self,
        beam_scan: BeamScan,
        inputs: dict,
        z_scale: list[float | None] = [None, None],
    ) -> None:
        super().__init__(beam_scan, inputs, z_scale)

    def plot(self, show=True) -> None | Figure:
        heatmap = heatmaps.heatmap(
            self.bs.grid_x, self.bs.grid_y, self.bs.grid_z, self.z_scale, self.color
        )
        contour_size = (max(self.levels) - min(self.levels)) - 1e-9

        # Create contour
        if contour_size > 0:
            contour = go.Contour(
                z=self.bs.grid_z,
                x=np.linspace(
                    self.bs.grid_x.min(), self.bs.grid_x.max(), self.bs.grid_z.shape[1]
                ),
                y=np.linspace(
                    self.bs.grid_y.min(), self.bs.grid_y.max(), self.bs.grid_z.shape[0]
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
                x=[self.bs.peak_location[0]],
                y=[self.bs.peak_location[1]],
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
                y=[self.inputs['centroid_y']] * len(self.x_slice),
                mode='lines',
                line=dict(color='black', width=1),
                name='X-Cross Section',
            )
        )

        # Add Y-axis cross-section line
        fig.add_trace(
            go.Scatter(
                x=[self.inputs['centroid_x']] * len(self.y_slice),
                y=self.y_slice['Y Coordinate'],
                mode='lines',
                line=dict(color='black', width=1),
                name='Y-Cross Section',
            )
        )

        # Add additional annotations (customize positions and text as needed)
        fig.add_annotation(
            # x=0.275, y=1, xref="paper", yref="paper",
            x=self.bs.x_location.max(),
            y=1,
            yref='paper',
            text=f'Cup current = {self.bs.peak_cup_current:.1f} nA <br>'
            f'Total current = {self.bs.peak_total_current:.3f} µA <br>'
            f'Settings = {self.inputs["beam_voltage"]}/{self.inputs["ext_voltage"]} kV & {self.inputs["solenoid_current"]} A',
            showarrow=False,
            align='left',
            xanchor='left',
            yanchor='bottom',
        )
        fig.add_annotation(
            x=self.bs.x_location.min(),
            y=1,
            yref='paper',
            text=f'FWHM Area = {self.bs.hm_contour_area:.3f} mm²<br>'
            f'FWQM Area = {self.bs.qm_contour_area:.3f} mm²<br>'
            f'peak = ({self.bs.peak_location[0]:.0f},{self.bs.peak_location[1]:.0f})',
            showarrow=False,
            align='right',
            xanchor='right',
            yanchor='bottom',
        )

        # Set title and axis properties
        fig.update_layout(
            title=dict(
                text=f'{self.inputs["serial_number"]} on TS{self.inputs["test_stand"]}',
                x=0.475,
                xanchor='center',
            ),
            xaxis=dict(
                title='X Location',
                range=[self.bs.grid_x.max(), self.bs.grid_x.min()],
                scaleanchor='y',
                showgrid=True,
                autorange=False,
            ),
            yaxis=dict(
                title='Y Location',
                range=[self.bs.grid_x.max(), self.bs.grid_x.min()],
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
        beam_scan: BeamScan,
        inputs: dict,
        z_scale: list[int | float | None] = [None, None],
    ) -> None:
        super().__init__(beam_scan, inputs, z_scale)

    def plot(self, show=True) -> Figure | None:
        scaling_factor = 1e3  # scale to microamps
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
            title_text=f'{self.inputs["serial_number"]} Beam Current Cross Sections',
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
        beam_scan: BeamScan,
        inputs: dict,
    ) -> None:
        super().__init__(beam_scan, inputs)
        self.fcup_diam = self.inputs['fcup_diam']
        self.fcup_dist = self.inputs['fcup_dist']

    def plot(
        self,
        show=True,
    ) -> Figure | None:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=['X Cross Section', 'Y Cross Section']
        )

        dist_from_x_center: NDArray[np.float64] = (
            np.asarray(self.x_slice['X Coordinate'] - self.bs.weighted_centroid[0])
            / 1000
        )  # millimeters
        dist_from_y_center: NDArray[np.float64] = (
            np.asarray(self.y_slice['Y Coordinate'] - self.bs.weighted_centroid[1])
            / 1000
        )  # millimeters
        x_angle: NDArray[np.float64] = (
            np.arctan(dist_from_x_center / self.fcup_dist) * 1000.0
        )  # milli-radians
        y_angle: NDArray[np.float64] = (
            np.arctan(dist_from_y_center / self.fcup_dist) * 1000.0
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
            title_text=f'{self.bs.serial_number} Angular Intensity vs Divergence Angle',
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

    import src.helpers.helpers as h
    from src.model.beam_scan import BeamScan

    QApplication([])
    bs: BeamScan = BeamScan()
    filepath: str = h.select_file()
    bs.load_scan_data(filepath)
    if bs.polarity == 'NEG':
        z_scale: list[int | float | None] = [None, None]
    else:
        z_scale: list[int | float | None] = [None, None]
    inputs: dict = {
        'serial_number': '111',
        'test_stand': '4',
        'fcup_diam': 2.5,
        'fcup_dist': 205,
        'beam_voltage': '-13',
        'ext_voltage': '-9',
        'power': '800',
        'solenoid_current': '1.2',
    }
    surface = Surface(bs, inputs, z_scale)
    surface.plot()
    heatmap = Heatmap(bs, inputs, z_scale)
    heatmap.plot()
    xy_cross_sections = XYCrossSections(bs, inputs, z_scale)
    xy_cross_sections.plot()
    i_prime = IPrime(bs, inputs)
    i_prime.plot()
