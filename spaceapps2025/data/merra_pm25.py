"""Access MERRA-2 CNN HAQAST PM2.5 datasets."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import requests
import xarray as xr
from harmony import Client, Collection, Request
from harmony.config import Environment

from ..utils.cache import FileCache
from ..utils.config import get_cache_root, get_data_root, load_earthdata_credentials
from ..utils.dates import to_ymd, to_ymd_compact

LOGGER = logging.getLogger(__name__)

COLLECTION_ID = "C3094710982-GES_DISC"
CMR_GRANULE_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.json"
DEFAULT_VARIABLE = "MERRA2_CNN_Surface_PM25"


class MerraPM25Client:
    def __init__(self, env: Environment = Environment.PROD, cache: Optional[FileCache] = None) -> None:
        username, password = load_earthdata_credentials()
        self.client = Client(env=env, auth=(username, password))
        self.cache = cache or FileCache(get_cache_root() / "merra2")
        self.data_root = get_data_root() / "merra2"
        self.data_root.mkdir(parents=True, exist_ok=True)

    def latest_available_date(self) -> date:
        params = {
            "concept_id": COLLECTION_ID,
            "page_size": 1,
            "sort_key": "-start_date",
        }
        response = requests.get(CMR_GRANULE_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        entries = response.json().get("feed", {}).get("entry", [])
        if not entries:
            raise RuntimeError("No MERRA-2 CNN granules available")
        time_start = entries[0].get("time_start")
        if not time_start:
            raise RuntimeError("Granule missing time_start")
        return datetime.fromisoformat(time_start.replace("Z", "+00:00")).date()

    def find_granule(self, target_date: date | datetime) -> dict:
        params = {
            "concept_id": COLLECTION_ID,
            "temporal": f"{to_ymd(target_date)}T00:00:00Z,{to_ymd(target_date)}T23:59:59Z",
            "page_size": 1,
        }
        response = requests.get(CMR_GRANULE_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        entries = response.json().get("feed", {}).get("entry", [])
        if not entries:
            latest = self.latest_available_date()
            raise RuntimeError(
                f"No MERRA-2 CNN granule for {target_date}; latest available is {latest}"
            )
        return entries[0]

    def _download(self, granule_id: str, producer_id: str) -> Path:
        cache_key = f"{COLLECTION_ID}:{granule_id}"
        cached_path = self.cache.get_path(cache_key, ".nc4")
        if cached_path.exists():
            return cached_path
        request = Request(collection=Collection(id=COLLECTION_ID), granule_id=[granule_id])
        job_id = self.client.submit(request)
        LOGGER.debug("Harmony job id=%s", job_id)
        self.client.wait_for_processing(job_id, show_progress=True)
        results = self.client.download_all(job_id, directory=str(self.data_root))
        files = [Path(result.result()) for result in results]
        if not files:
            raise RuntimeError("Harmony returned no files")
        downloaded = files[0]
        downloaded.rename(cached_path)
        return cached_path

    def load_surface_pm25(
        self,
        target_date: date | datetime,
        variable: str = DEFAULT_VARIABLE,
    ) -> xr.Dataset:
        granule = self.find_granule(target_date)
        path = self._download(granule["id"], granule.get("producer_granule_id", granule["id"]))
        dataset = xr.open_dataset(path, engine="netcdf4")
        if variable not in dataset:
            raise RuntimeError(
                f"Variable {variable} not present; available: {list(dataset.data_vars)}"
            )
        return dataset

