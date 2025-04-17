import plotly.graph_objects as go
from pandas import DataFrame


def surface(
    self,
    grid_x,
    grid_y,
    grid_z,
    centroid: tuple[float, float],
    x_slice: DataFrame,
    y_slice: DataFrame,
) -> go.Figure:
    scaling_factor = 1e-6  # scale to micro-amps
    self.z_scale = [
        value * scaling_factor if value is not None else None for value in self.z_scale
    ]
    contour_size = (self.levels[1] - self.levels[0]) - 1e-9

    if any(value is not None for value in self.z_scale):
        if contour_size > 0:
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=grid_x,
                        y=grid_y,
                        z=grid_z,
                        colorscale=self.color,
                        cmin=self.z_scale[0],
                        cmax=self.z_scale[1],  # scale to micro-amps
                        lighting=dict(ambient=0.9),
                        opacity=0.8,
                        contours=dict(
                            x={
                                'show': True,
                                'start': grid_x.min(),
                                'end': grid_x.max(),
                                'size': int(1000),
                                'color': 'gray',
                            },
                            y={
                                'show': True,
                                'start': grid_y.min(),
                                'end': grid_y.max(),
                                'size': int(1000),
                                'color': 'gray',
                            },
                            z={
                                'show': True,
                                'start': self.levels[0],
                                'end': self.levels[1],
                                'size': (self.levels[1] - self.levels[0]) - 1e-9,
                                'color': 'red',
                            },
                        ),
                    )
                ]
            )
        else:  # something weird is going on, do not plot contours
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=grid_x,
                        y=grid_y,
                        z=grid_z,
                        colorscale=self.color,
                        cmin=self.z_scale[0],
                        cmax=self.z_scale[1],  # scale to micro-amps
                        lighting=dict(ambient=0.9),
                        opacity=0.8,
                    )
                ]
            )

        fig.add_trace(
            go.Scatter3d(
                x=x_slice['X Coordinate'],
                y=grid_y[0, :],
                z=x_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=4),
                name='X Profile',
            )
        )  # adds a line on the back wall of the plot to show the x profile

        fig.add_trace(
            go.Scatter3d(
                x=grid_x[:, 0],
                y=y_slice['Y Coordinate'],
                z=y_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=4),
                name='Y Profile',
            )
        )  # adds a line on the back wall of the plot to show the y profile

        fig.add_trace(
            go.Scatter3d(
                x=x_slice['X Coordinate'],
                y=[centroid[1]] * len(x_slice),
                z=x_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=2),
                name='X Profile',
            )
        )  # adds a line on the 3D surface at the centroid to show the x profile

        fig.add_trace(
            go.Scatter3d(
                x=[centroid[0]] * len(y_slice),
                y=y_slice['Y Coordinate'],
                z=y_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=2),
                name='Y Profile',
            )
        )  # adds a line on the 3D surface at the centroid to show the y profile

        fig.update_layout(
            title=f'{self.serial_number}; PEAK: cup current = {self.peak_cup_current * 1e9:.0f} nA; total current = {self.peak_total_current * 1e6:.3f} \u03bcA',
            scene=dict(zaxis=dict(range=[self.z_scale[0], self.z_scale[1]])),
            scene_camera=dict(
                center=dict(x=0, y=0, z=-0.1), eye=dict(x=1.5, y=1.5, z=0.7)
            ),
            showlegend=False,
        )

    else:  # autoscale
        if contour_size > 0:  # plot contours
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=grid_x,
                        y=grid_y,
                        z=grid_z,
                        colorscale=self.color,
                        lighting=dict(ambient=0.9),
                        opacity=0.8,
                        contours=dict(
                            x={
                                'show': True,
                                'start': grid_x.min(),
                                'end': grid_x.max(),
                                'size': int(1000),
                                'color': 'gray',
                            },
                            y={
                                'show': True,
                                'start': grid_y.min(),
                                'end': grid_y.max(),
                                'size': int(1000),
                                'color': 'gray',
                            },
                            z={
                                'show': True,
                                'start': self.levels[0],
                                'end': self.levels[1],
                                'size': (self.levels[1] - self.levels[0]) - 1e-9,
                                'color': 'red',
                            },
                        ),
                    )
                ]
            )
        else:  # something weird is going on, do not plot contours
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=grid_x,
                        y=grid_y,
                        z=grid_z,
                        colorscale=self.color,
                        lighting=dict(ambient=0.9),
                        opacity=0.8,
                    )
                ]
            )

        fig.add_trace(
            go.Scatter3d(
                x=x_slice['X Coordinate'],
                y=grid_y[0, :],
                z=x_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=4),
                name='X Profile',
            )
        )  # adds a line on the back wall of the plot to show the x profile

        fig.add_trace(
            go.Scatter3d(
                x=grid_x[:, 0],
                y=y_slice['Y Coordinate'],
                z=y_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=4),
                name='Y Profile',
            )
        )  # adds a line on the back wall of the plot to show the y profile

        fig.add_trace(
            go.Scatter3d(
                x=x_slice['X Coordinate'],
                y=[centroid[1]] * len(x_slice),
                z=x_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=2),
                name='X Profile',
            )
        )  # adds a line on the 3D surface at the centroid to show the x profile

        fig.add_trace(
            go.Scatter3d(
                x=[centroid[0]] * len(y_slice),
                y=y_slice['Y Coordinate'],
                z=y_slice['Faraday Cup Current'],
                mode='lines',
                line=dict(color='black', width=2),
                name='Y Profile',
            )
        )  # adds a line on the 3D surface at the centroid to show the y profile

        fig.update_layout(
            title=f'{self.serial_number}; PEAK: cup current = {self.peak_cup_current * 1e9:.0f} nA; total current = {self.peak_total_current * 1e6:.3f} \u03bcA',
            scene_camera=dict(
                center=dict(x=0, y=0, z=-0.1), eye=dict(x=1.5, y=1.5, z=0.7)
            ),
            showlegend=False,
        )

    return fig
