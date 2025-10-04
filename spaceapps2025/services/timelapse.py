"""Generate interactive timelapse visualizations."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd
import plotly.graph_objects as go
import xarray as xr


def create_timelapse(
    fields: Sequence[xr.DataArray],
    timestamps: Sequence[pd.Timestamp],
    title: str,
    units: str,
    colorscale: str = "Inferno",
) -> go.Figure:
    if not fields:
        raise ValueError("At least one field is required to build a timelapse")

    base_field = fields[0]
    x = base_field["longitude"].values
    y = base_field["latitude"].values

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=base_field.values,
            x=x,
            y=y,
            coloraxis="coloraxis",
            showscale=False,
        )
    )

    frames = []
    for field, timestamp in zip(fields, timestamps):
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=field.values,
                        x=field["longitude"].values,
                        y=field["latitude"].values,
                        coloraxis="coloraxis",
                        showscale=False,
                    )
                ],
                name=timestamp.strftime("%Y-%m-%d %H:%M"),
            )
        )

    figure.frames = frames
    figure.update_layout(
        title=title,
        coloraxis={"colorscale": colorscale},
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 30},
                "currentvalue": {"prefix": "Timestamp: "},
                "steps": [
                    {
                        "label": frame.name,
                        "method": "animate",
                        "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for frame in frames
                ],
            }
        ],
    )
    figure.update_layout(
        coloraxis_colorbar=dict(title=units),
    )
    return figure

