import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_beam_profile_with_slices(scan_data, Xc, Yc):
    """
    Plots the beam profile and cross-sectional slices along the centroid's X and Y axes.
    """

    x, y, z = scan_data.create_grid()
    
    y_idx = np.abs(y[:, 0] - Yc).argmin()
    x_idx = np.abs(x[0, :] - Xc).argmin()

    x_slice = pd.DataFrame({"X Coordinate": x[y_idx, :], "Faraday Cup Current": z[y_idx, :]})
    y_slice = pd.DataFrame({"Y Coordinate": y[:, x_idx], "Faraday Cup Current": z[:, x_idx]})


    # Main beam profile scatter plot
    heatmap = go.Heatmap(
                    z=z,
                    x=np.linspace(x.min(), x.max(), z.shape[1]),
                    y=np.linspace(y.min(), y.max(), z.shape[0]),
                    colorscale='viridis',
                    colorbar=dict(
                        title='Cup Current',
                        titleside='right'
                        ),
                )

    fig = go.Figure([heatmap])

    # Add centroid marker
    fig.add_trace(go.Scatter(
            x=[Xc],
            y=[Yc],
            mode='markers',
            textposition='bottom center',
            marker=dict(color='black', size=8, symbol='diamond'),
            name='Centroid'
        ))

    # Add X-axis cross-section line
    fig.add_trace(go.Scatter(
        x=x_slice["X Coordinate"],
        y=[Yc] * len(x_slice),  
        mode="lines",
        line=dict(color='black', width=1),
        name="X-Cross Section"
    ))

    # Add Y-axis cross-section line
    fig.add_trace(go.Scatter(
        x=[Xc] * len(y_slice),
        y=y_slice["Y Coordinate"],
        mode="lines",
        line=dict(color='black', width=1),
        name="Y-Cross Section"
    ))

    fig.update_layout(
    coloraxis_colorbar=dict(
        orientation="h",  # 'v' for vertical, 'h' for horizontal
        xanchor="center",
        yanchor="bottom"
    ),
    xaxis=dict(range=[-8000, 8000]),
    yaxis=dict(range=[-8000, 8000])
)

    # Create subplots for cross-sections
    x_cross_section_fig = go.Figure()
    y_cross_section_fig = go.Figure()

    # X Cross-Section
    x_cross_section_fig.add_trace(go.Scatter(
        x=x_slice["X Coordinate"],
        y=x_slice["Faraday Cup Current"],
        mode="lines",
        name="X-Axis Cross Section",
        line=dict(color="blue", width=2)
    ))

    # Y Cross-Section
    y_cross_section_fig.add_trace(go.Scatter(
        x=y_slice["Y Coordinate"],
        y=y_slice["Faraday Cup Current"],
        mode="lines",
        name="Y-Axis Cross Section",
        line=dict(color="red", width=2)
    ))

    x_cross_section_fig.update_layout(
        title="Beam Current X Cross-Sections",
        xaxis_title="Position",
        yaxis_title="Current (A)",
        legend=dict(x=0.1, y=1),
    )

    y_cross_section_fig.update_layout(
        title="Beam Current Y Cross-Sections",
        xaxis_title="Position",
        yaxis_title="Current (A)",
        legend=dict(x=0.1, y=1),
    )

    # Show both plots
    fig.show()
    x_cross_section_fig.show()
    y_cross_section_fig.show()

# Example usage with synthetic data
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    QApplication([])
    from beam_scan_analysis import ScanData
    from beam_scan_analysis import select_csv, load_scan_data
    filepath = select_csv()
    scan_data = load_scan_data(filepath)

    Xc, Yc = scan_data.compute_weighted_centroid()
    

    # Plot beam profile with slices
    plot_beam_profile_with_slices(scan_data, Xc, Yc)
