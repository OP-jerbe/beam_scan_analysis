import plotly.graph_objects as go
import numpy as np

def heatmap(self, grid_x, grid_y, grid_z) -> go.Heatmap:
    scaling_factor = 1e-6 # scale to micro-amps
    self.z_scale = [value * scaling_factor if value is not None else None for value in self.z_scale]

    if any(value is not None for value in self.z_scale):
        heatmap = go.Heatmap(
                    z=grid_z,
                    x=np.linspace(grid_x.min(), grid_x.max(), grid_z.shape[1]),
                    y=np.linspace(grid_y.min(), grid_y.max(), grid_z.shape[0]),
                    colorscale=self.color,
                    zauto=False,
                    zmin=self.z_scale[0],
                    zmax=self.z_scale[1],
                    colorbar=dict(
                        title='Cup Current',
                        titleside='right'
                        ),
                )
    else: # Autoscale
        heatmap = go.Heatmap(
            z=self.grid_z,
            x=np.linspace(grid_x.min(), grid_x.max(), grid_z.shape[1]),
            y=np.linspace(grid_y.min(), grid_y.max(), grid_z.shape[0]),
            colorscale=self.color,
            colorbar=dict(
                title='Cup Current',
                titleside='right',
                ),
        )
    
    return heatmap