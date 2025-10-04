"""Access NASA POWER weather and surface data."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

import pandas as pd
import requests

from ..utils.cache import FileCache
from ..utils.config import get_cache_root
from ..utils.dates import to_ymd_compact

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily"
DEFAULT_PARAMETERS = ["T2M", "RH2M", "U2M", "V2M", "PS", "PRECTOT"]


class NasaPowerClient:
    def __init__(self, cache: Optional[FileCache] = None) -> None:
        self.cache = cache or FileCache(get_cache_root() / "nasa_power")

    def _request(self, endpoint: str, params: dict) -> dict:
        LOGGER.debug("Requesting NASA POWER endpoint=%s", endpoint)
        response = requests.get(endpoint, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def point_timeseries(
        self,
        latitude: float,
        longitude: float,
        start: date | datetime,
        end: date | datetime,
        parameters: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start": to_ymd_compact(start),
            "end": to_ymd_compact(end),
            "parameters": ",".join(parameters or DEFAULT_PARAMETERS),
            "community": "RE",
            "format": "JSON",
        }
        payload = self._request(f"{BASE_URL}/point", params)
        properties = payload.get("properties", {})
        parameter_data = properties.get("parameter", {})
        rows = []
        for yyyymmdd, values in parameter_data.get(next(iter(parameter_data)), {}).items():
            row = {"date": pd.to_datetime(yyyymmdd)}
            for key, series in parameter_data.items():
                row[key] = series.get(yyyymmdd)
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values("date")
        return frame

    def regional_grid(
        self,
        bbox: Sequence[float],
        start: date | datetime,
        end: date | datetime,
        parameters: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        params = {
            "bbox": ",".join(map(str, bbox)),
            "start": to_ymd_compact(start),
            "end": to_ymd_compact(end),
            "parameters": ",".join(parameters or DEFAULT_PARAMETERS),
            "community": "RE",
            "format": "JSON",
        }
        payload = self._request(f"{BASE_URL}/regional", params)
        properties = payload.get("properties", {})
        parameter_data = properties.get("parameter", {})
        rows = []
        for key, by_date in parameter_data.items():
            for yyyymmdd, value_grid in by_date.items():
                for entry in value_grid:
                    rows.append(
                        {
                            "date": pd.to_datetime(yyyymmdd),
                            "parameter": key,
                            "latitude": entry.get("latitude"),
                            "longitude": entry.get("longitude"),
                            "value": entry.get("value"),
                        }
                    )
        frame = pd.DataFrame(rows)
        return frame
