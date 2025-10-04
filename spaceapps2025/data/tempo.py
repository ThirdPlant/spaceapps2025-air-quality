"""Access TEMPO Level-2 data via NASA Harmony."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests
import xarray as xr
from harmony import Client, Collection, Request
from harmony.config import Environment

from ..utils.cache import FileCache
from ..utils.config import get_cache_root, get_data_root, load_earthdata_credentials
from ..utils.dates import to_ymd

LOGGER = logging.getLogger(__name__)

CMR_GRANULE_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.json"


@dataclass(frozen=True)
class TempoProduct:
    name: str
    collection_id: str
    default_variable: str
    description: str


TEMPO_PRODUCTS: Dict[str, TempoProduct] = {
    "NO2": TempoProduct(
        name="NO2",
        collection_id="C2930725014-LARC_CLOUD",
        default_variable="vertical_column_troposphere",
        description="Tropospheric nitrogen dioxide column",
    ),
    "HCHO": TempoProduct(
        name="HCHO",
        collection_id="C2930730944-LARC_CLOUD",
        default_variable="vertical_column",
        description="Formaldehyde total column",
    ),
    "O3": TempoProduct(
        name="O3",
        collection_id="C2930726639-LARC_CLOUD",
        default_variable="total_ozone_column",
        description="Ozone total column",
    ),
}


class TempoDataClient:
    """Convenience wrapper to query CMR and download TEMPO granules via Harmony."""

    def __init__(
        self,
        product_key: str = "NO2",
        env: Environment = Environment.PROD,
        cache: Optional[FileCache] = None,
    ) -> None:
        if product_key not in TEMPO_PRODUCTS:
            raise ValueError(
                f"Unknown TEMPO product '{product_key}'. Available: {list(TEMPO_PRODUCTS)}"
            )
        self.product = TEMPO_PRODUCTS[product_key]
        username, password = load_earthdata_credentials()
        self.client = Client(env=env, auth=(username, password))
        self.cache = cache or FileCache(get_cache_root() / "tempo")
        self.data_root = get_data_root() / "tempo"
        self.data_root.mkdir(parents=True, exist_ok=True)

    def search_granules(
        self,
        start: date | datetime,
        end: Optional[date | datetime] = None,
        bounding_box: Optional[Sequence[float]] = None,
        limit: int = 5,
    ) -> List[dict]:
        """Search CMR for granules matching the query parameters."""
        params = {
            "collection_concept_id": self.product.collection_id,
            "temporal": f"{to_ymd(start)}T00:00:00Z,{to_ymd(end or start)}T23:59:59Z",
            "page_size": limit,
            "sort_key": "-start_date",
        }
        if bounding_box:
            params["bounding_box"] = ",".join(map(str, bounding_box))

        LOGGER.debug("Querying CMR with params=%s", params)
        response = requests.get(CMR_GRANULE_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        entries = response.json().get("feed", {}).get("entry", [])
        return entries

    def _download_via_harmony(self, granule_id: str, producer_id: str) -> Path:
        cache_key = f"{self.product.collection_id}:{granule_id}"
        cached_path = self.cache.get_path(cache_key, ".nc4")
        if cached_path.exists():
            LOGGER.info("Using cached TEMPO granule %s", cached_path.name)
            return cached_path

        request = Request(
            collection=Collection(id=self.product.collection_id),
            granule_id=[granule_id],
        )
        LOGGER.info("Submitting Harmony job for granule %s", granule_id)
        job_id = self.client.submit(request)
        LOGGER.debug("Harmony job id=%s", job_id)
        self.client.wait_for_processing(job_id, show_progress=True)
        urls = list(self.client.result_urls(job_id))
        if not urls:
            raise RuntimeError("Harmony completed but returned no result URLs")
        download_url = urls[0]
        LOGGER.info("Downloading %s", download_url)
        results = self.client.download_all(job_id, directory=str(self.data_root))
        files = [Path(result.result()) for result in results]
        if not files:
            raise RuntimeError("Harmony reported success but produced no files")
        downloaded_path = files[0]
        downloaded_path.rename(cached_path)
        return cached_path

    def fetch_dataset(
        self,
        granule: dict,
        group: str = "product",
        decode_times: bool = True,
    ) -> xr.Dataset:
        """Download a granule (if needed) and return it as an xarray dataset."""
        granule_id = granule["id"]
        producer_id = granule.get("producer_granule_id", granule_id)
        path = self._download_via_harmony(granule_id, producer_id)
        LOGGER.debug("Opening %s group=%s", path, group)
        return xr.open_dataset(path, engine="netcdf4", group=group, decode_times=decode_times)

    def list_available_granules(
        self,
        start: date | datetime,
        end: Optional[date | datetime] = None,
        bounding_box: Optional[Sequence[float]] = None,
        limit: int = 10,
    ) -> List[dict]:
        entries = self.search_granules(start=start, end=end, bounding_box=bounding_box, limit=limit)
        LOGGER.info("Found %d TEMPO granules", len(entries))
        return entries


def load_geolocated_field(
    dataset: xr.Dataset,
    geolocation: xr.Dataset,
    var_name: Optional[str] = None,
) -> xr.Dataset:
    """Merge product values with geolocation coordinates for mapping."""
    var_name = var_name or next(iter(dataset.data_vars))
    field = dataset[var_name]
    merged = field.assign_coords(
        latitude=("y", geolocation["latitude"].values),
        longitude=("x", geolocation["longitude"].values),
    )
    return merged
