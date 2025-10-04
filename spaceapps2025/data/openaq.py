"""Access OpenAQ ground sensor data."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

from ..utils.cache import FileCache
from ..utils.config import get_cache_root
from ..utils.dates import to_ymd

LOGGER = logging.getLogger(__name__)

OPENAQ_BASE_URL = "https://api.openaq.org/v2"
MAX_PAGE_SIZE = 1000


class OpenAQClient:
    """Thin wrapper around the OpenAQ API with basic caching."""

    def __init__(self, cache: Optional[FileCache] = None) -> None:
        self.cache = cache or FileCache(get_cache_root() / "openaq")

    def _paged_get(self, endpoint: str, params: Dict[str, object]) -> Iterable[dict]:
        page = 1
        while True:
            query = dict(params)
            query.update({"page": page, "limit": MAX_PAGE_SIZE})
            LOGGER.debug("Requesting OpenAQ %s page=%s", endpoint, page)
            response = requests.get(f"{OPENAQ_BASE_URL}/{endpoint}", params=query, timeout=60)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not results:
                break
            for result in results:
                yield result
            found = payload.get("meta", {}).get("found", 0)
            if page * MAX_PAGE_SIZE >= found:
                break
            page += 1

    def measurements(
        self,
        parameter: str,
        start: datetime,
        end: datetime,
        bbox: Optional[Sequence[float]] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch measurements for the specified pollutant and window."""
        params: Dict[str, object] = {
            "parameter": parameter,
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "order_by": "date",
            "sort": "asc",
        }
        if bbox:
            params["bbox"] = ",".join(map(str, bbox))
        if country:
            params["country"] = country
        if city:
            params["city"] = city

        rows = list(self._paged_get("measurements", params))
        if not rows:
            return pd.DataFrame(columns=["datetime", "value", "unit", "location", "coordinates"])

        frame = pd.DataFrame(rows)
        frame["datetime"] = pd.to_datetime(frame["date"].apply(lambda d: d["utc"]))
        frame = frame[["datetime", "value", "unit", "location", "coordinates"]]
        coords = frame["coordinates"].apply(lambda c: (c.get("latitude"), c.get("longitude")))
        frame["latitude"], frame["longitude"] = zip(*coords)
        frame.drop(columns=["coordinates"], inplace=True)
        return frame

    def latest(
        self,
        parameter: str,
        bbox: Optional[Sequence[float]] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        params: Dict[str, object] = {
            "parameter": parameter,
            "order_by": "lastUpdated",
            "sort": "desc",
            "limit": limit,
        }
        if bbox:
            params["bbox"] = ",".join(map(str, bbox))
        if country:
            params["country"] = country
        if city:
            params["city"] = city

        response = requests.get(f"{OPENAQ_BASE_URL}/latest", params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows = []
        for result in payload.get("results", []):
            coords = result.get("coordinates", {})
            for measurement in result.get("measurements", []):
                rows.append(
                    {
                        "location": result.get("location"),
                        "city": result.get("city"),
                        "country": result.get("country"),
                        "parameter": measurement.get("parameter"),
                        "value": measurement.get("value"),
                        "unit": measurement.get("unit"),
                        "datetime": pd.to_datetime(measurement.get("lastUpdated")),
                        "latitude": coords.get("latitude"),
                        "longitude": coords.get("longitude"),
                    }
                )
        return pd.DataFrame(rows)
