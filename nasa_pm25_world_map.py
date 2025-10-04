#!/usr/bin/env python3
"""Download NASA MERRA-2 CNN HAQAST PM2.5 data via Harmony and plot a world map."""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from harmony import Client, Collection, Request
from harmony.config import Environment

COLLECTION_ID = "C3094710982-GES_DISC"
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "nasa_data"
DEFAULT_OUTPUT = BASE_DIR / "output" / "global_pm25_map.png"
ENV_PATH = BASE_DIR / ".env"
CMR_COLLECTION_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.json"
MAX_REDIRECTS = 8
DEFAULT_QUALITY_THRESHOLD = 3
PM25_VARIABLE = "MERRA2_CNN_Surface_PM25"
QFLAG_VARIABLE = "QFLAG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a global PM2.5 map using the NASA MERRA-2 CNN HAQAST dataset."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD). Defaults to the latest available day (yesterday UTC).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File path for the rendered figure (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help=f"Directory to store downloaded NetCDF files (default: {DOWNLOAD_DIR}).",
    )
    parser.add_argument(
        "--quality-min",
        type=int,
        default=DEFAULT_QUALITY_THRESHOLD,
        help="Minimum QFLAG score (1-4) to keep when averaging (default: 3).",
    )
    return parser.parse_args()


def load_credentials(env_path: Path) -> Tuple[str, str]:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if username and password:
        return username, password

    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "EARTHDATA_USERNAME" and not username:
                username = value
            elif key == "EARTHDATA_PASSWORD" and not password:
                password = value

    if not (username and password):
        raise RuntimeError(
            "Earthdata credentials not found. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )

    return username, password


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def fetch_latest_available_date() -> date:
    """Query CMR for the most recent granule date available."""
    params = {
        "short_name": "MERRA2_CNN_HAQAST_PM25",
        "provider": "GES_DISC",
        "page_size": 1,
        "sort_key": "-start_date",
    }
    response = requests.get(CMR_COLLECTION_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    if not entries:
        raise RuntimeError(
            "Unable to determine latest available date; CMR returned no granules for the collection."
        )
    time_start = entries[0].get("time_start") or entries[0].get("start_time")
    if not time_start:
        raise RuntimeError("Latest granule metadata missing time_start field.")
    return datetime.fromisoformat(time_start.replace("Z", "+00:00")).date()


def default_date() -> date:
    latest = fetch_latest_available_date()
    yesterday = datetime.utcnow().date() - timedelta(days=1)
    return latest if latest <= yesterday else yesterday


def parse_date(value: str | None) -> date:
    if value is None:
        return default_date()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise RuntimeError("--date must be in YYYY-MM-DD format") from exc


def find_granule_metadata(target_date: date) -> Tuple[str, str, str]:
    temporal = f"{target_date}T00:00:00Z,{target_date}T23:59:59Z"
    params = {
        "short_name": "MERRA2_CNN_HAQAST_PM25",
        "temporal": temporal,
        "page_size": 1,
        "sort_key": "-start_date",
        "provider": "GES_DISC",
    }
    response = requests.get(CMR_COLLECTION_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    entries = data.get("feed", {}).get("entry", [])
    if not entries:
        latest_available = fetch_latest_available_date()
        if target_date > latest_available:
            raise RuntimeError(
                f"No granules found for {target_date}. The latest available date is {latest_available}."
            )
        raise RuntimeError(
            f"No granules found for {target_date}. Check the date or dataset availability."
        )
    entry = entries[0]
    granule_id = entry["id"]
    producer_id = entry.get("producer_granule_id")
    download_url = None
    for link in entry.get("links", []):
        rel = link.get("rel", "")
        href = link.get("href")
        if rel.endswith("/data#") and href and href.endswith(".nc4"):
            download_url = href
            break
    if not download_url:
        raise RuntimeError("Could not locate an HTTPS download link for the granule.")
    return granule_id, producer_id, download_url


def stage_granule_with_harmony(client: Client, granule_id: str) -> str:
    request = Request(collection=Collection(id=COLLECTION_ID), granule_id=[granule_id])
    job_id = client.submit(request)
    print(f"Submitted Harmony job: {job_id}")
    client.wait_for_processing(job_id, show_progress=True)
    urls = list(client.result_urls(job_id))
    if not urls:
        raise RuntimeError("Harmony job completed without providing result URLs.")
    return urls[0]


def resolve_earthdata_download(session: requests.Session, url: str) -> requests.Response:
    current_url = url
    for _ in range(MAX_REDIRECTS):
        response = session.get(current_url, stream=True, allow_redirects=False)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not location:
                raise RuntimeError(
                    f"Received redirect without Location header when fetching {current_url}."
                )
            current_url = urljoin(current_url, location)
            continue
        if response.status_code == 401:
            login_url = response.headers.get("Location") or response.url
            parsed = urlparse(login_url)
            query = parse_qs(parsed.query)
            resolution = query.get("resolution_url")
            if resolution:
                raise RuntimeError(
                    "Earthdata Login reports that the GES DISC application is not authorized "
                    "for this account. Visit the following URL once in a browser to approve it:\n"
                    f"  {resolution[0]}"
                )
            auth_response = session.get(
                login_url, auth=session.auth, allow_redirects=False, stream=True
            )
            if auth_response.status_code == 401:
                raise RuntimeError(
                    "Earthdata credentials were rejected (HTTP 401). Verify the username, "
                    "password, and that the account is active."
                )
            if auth_response.status_code in {301, 302, 303, 307, 308}:
                location = auth_response.headers.get("Location")
                if not location:
                    raise RuntimeError(
                        "Authentication redirect missing Location header; cannot continue."
                    )
                current_url = urljoin(login_url, location)
                continue
            # Fall through to retry original download after successful authentication
            current_url = url
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {current_url}: HTTP {response.status_code}")
        return response
    raise RuntimeError("Exceeded redirect attempts while negotiating Earthdata authentication.")


def download_granule(
    session: requests.Session, url: str, destination: Path
) -> Path:
    response = resolve_earthdata_download(session, url)
    with destination.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=524_288):
            if chunk:
                file_obj.write(chunk)
    return destination


def load_pm25_dataset(netcdf_path: Path) -> xr.Dataset:
    return xr.open_dataset(netcdf_path, engine="netcdf4")


def compute_daily_pm25(
    dataset: xr.Dataset, quality_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    if PM25_VARIABLE not in dataset:
        available = ", ".join(dataset.data_vars)
        raise RuntimeError(
            f"Expected variable '{PM25_VARIABLE}' not present. Found: {available}"
        )
    pm25 = dataset[PM25_VARIABLE]

    if QFLAG_VARIABLE in dataset:
        qflag = dataset[QFLAG_VARIABLE]
        mask = qflag >= quality_min
        pm25 = pm25.where(mask)
    elif quality_min > 1:
        print(
            "Warning: QFLAG variable missing; quality filter cannot be applied. Proceeding with all data."
        )

    if "time" in pm25.dims:
        pm25 = pm25.mean(dim="time", skipna=True)

    lats = np.array(dataset["lat"].values, copy=True)
    lons = np.array(dataset["lon"].values, copy=True)
    values = np.ma.masked_invalid(pm25.to_masked_array(copy=True))

    return lons, lats, values


def determine_color_limits(values: np.ma.MaskedArray) -> Tuple[float, float]:
    masked = np.ma.masked_invalid(np.ma.array(values, copy=True))
    if masked.count() == 0:
        raise RuntimeError("The PM2.5 array contains no valid data after quality filtering.")
    filled = masked.filled(np.nan)
    vmin = 0.0
    vmax = float(np.nanpercentile(filled, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(filled))
    return vmin, vmax


def plot_global_pm25(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title_suffix: str,
) -> None:
    vmin, vmax = determine_color_limits(values)
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", edgecolor="0.3", linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="#dce9f6", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="dimgray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":")

    mesh = ax.pcolormesh(
        lons,
        lats,
        values,
        cmap="inferno",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("Surface PM$_{2.5}$ [ug m$^{-3}$]")

    grid = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        linestyle="--",
        color="0.4",
        alpha=0.35,
    )
    grid.top_labels = False
    grid.right_labels = False
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER

    ax.set_title(
        "NASA MERRA-2 CNN HAQAST Surface PM$_{2.5}$" + title_suffix,
        pad=16,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target_date = parse_date(args.date)
    username, password = load_credentials(ENV_PATH)

    ensure_directories([args.download_dir, args.output.parent])

    granule_id, producer_id, cmr_url = find_granule_metadata(target_date)
    print(f"Selected granule: {producer_id} ({granule_id})")

    client = Client(env=Environment.PROD, auth=(username, password))
    harmony_url = stage_granule_with_harmony(client, granule_id)
    print(f"Harmony returned URL: {harmony_url}")

    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({"User-Agent": "nasa-pm25-map/1.0"})

    destination = args.download_dir / producer_id
    download_url = harmony_url or cmr_url
    if download_url.startswith("s3://"):
        raise RuntimeError(
            "Harmony returned an S3 URL. Update the script to obtain temporary credentials via "
            "client.aws_credentials(job_id) before retrying."
        )

    print(f"Downloading granule from {download_url}")
    netcdf_path = download_granule(session, download_url, destination)

    print(f"Opening dataset {netcdf_path}")
    dataset = load_pm25_dataset(netcdf_path)
    try:
        lons, lats, values = compute_daily_pm25(dataset, args.quality_min)
    finally:
        dataset.close()

    title_suffix = f"\n{target_date.isoformat()} (QFLAG >= {args.quality_min})"
    plot_global_pm25(lons, lats, values, args.output, title_suffix)
    print(f"World map saved to {args.output}")


if __name__ == "__main__":
    main()

