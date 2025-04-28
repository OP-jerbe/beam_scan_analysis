import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


def heatmap(
    grid_x: NDArray,
    grid_y: NDArray,
    grid_z: NDArray,
    z_scale: list[int | float | None],
    color: str,
) -> go.Heatmap:
    scaling_factor = 1e-6  # scale to micro-amps
    z_scale = [value and value * scaling_factor for value in z_scale]

    if any(value is not None for value in z_scale):
        heatmap = go.Heatmap(
            z=grid_z,
            x=np.linspace(grid_x.min(), grid_x.max(), grid_z.shape[1]),
            y=np.linspace(grid_y.min(), grid_y.max(), grid_z.shape[0]),
            colorscale=color,
            zauto=False,
            zmin=z_scale[0],
            zmax=z_scale[1],
            colorbar=dict(title=dict(text='Cup Current', side='right')),
        )
    else:  # Autoscale
        heatmap = go.Heatmap(
            z=grid_z,
            x=np.linspace(grid_x.min(), grid_x.max(), grid_z.shape[1]),
            y=np.linspace(grid_y.min(), grid_y.max(), grid_z.shape[0]),
            colorscale=color,
            colorbar=dict(title=dict(text='Cup Current', side='right')),
        )

    return heatmap
