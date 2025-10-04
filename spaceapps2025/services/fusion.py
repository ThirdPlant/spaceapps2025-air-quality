"""Data fusion utilities for combining satellite, ground, and weather sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


@dataclass
class CollocatedObservation:
    location: str
    latitude: float
    longitude: float
    parameter: str
    ground_value: float
    satellite_value: float
    unit: str
    timestamp: pd.Timestamp


def sample_satellite_at_points(
    field: xr.DataArray,
    points: pd.DataFrame,
    method: str = "nearest",
) -> pd.DataFrame:
    """Sample satellite field at provided ground station coordinates."""
    sampled = []
    for _, row in points.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        try:
            value = float(field.sel(latitude=lat, longitude=lon, method=method))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to sample lat=%s lon=%s: %s", lat, lon, exc)
            value = np.nan
        sampled.append(value)
    result = points.copy()
    result["satellite_value"] = sampled
    return result


def merge_ground_and_satellite(
    ground: pd.DataFrame,
    field: xr.DataArray,
    parameter: str,
) -> pd.DataFrame:
    """Return merged dataset with collocated ground and satellite values."""
    sampled = sample_satellite_at_points(field, ground)
    sampled["parameter"] = parameter
    return sampled


def aggregate_by_region(
    data: pd.DataFrame,
    bin_size: float = 1.0,
) -> pd.DataFrame:
    """Aggregate measurements into latitude/longitude bins for mapping."""
    data = data.copy()
    data["lat_bin"] = (data["latitude"] / bin_size).round() * bin_size
    data["lon_bin"] = (data["longitude"] / bin_size).round() * bin_size
    grouped = (
        data.groupby(["parameter", "lat_bin", "lon_bin"])
        .agg({"value": "mean", "satellite_value": "mean"})
        .reset_index()
    )
    return grouped
