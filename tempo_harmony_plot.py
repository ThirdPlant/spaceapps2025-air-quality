#!/usr/bin/env python3
"""Download TEMPO NO2 data via NASA Harmony and generate a Matplotlib map."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from harmony import Client, Collection, Request
from harmony.config import Environment

COLLECTION_ID = "C2930725014-LARC_CLOUD"
GRANULE_NAMES = ["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"]
DOWNLOAD_DIR = Path("tempo_data")
OUTPUT_PATH = Path("output/tempo_no2_map.png")
ENV_PATH = Path(".env")


def load_credentials(env_path: Path) -> Tuple[str, str]:
    """Load Earthdata credentials from the environment or a .env file."""
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
    """Ensure that the given directories exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def download_tempo_granules(client: Client, download_dir: Path) -> Path:
    """Submit a Harmony request and download the first available granule."""
    request = Request(
        collection=Collection(id=COLLECTION_ID),
        granule_name=GRANULE_NAMES,
    )

    job_id = client.submit(request)
    client.wait_for_processing(job_id, show_progress=True)

    results = client.download_all(job_id, directory=str(download_dir))
    files = [Path(task.result()) for task in results]

    if not files:
        raise RuntimeError("Harmony request completed but no files were downloaded.")

    return files[0]


def load_datasets(netcdf_path: Path) -> Tuple[xr.Dataset, xr.Dataset]:
    """Load product and geolocation groups from the NetCDF granule."""
    with xr.open_dataset(netcdf_path, engine="netcdf4", group="product") as product:
        product_ds = product.load()
    with xr.open_dataset(netcdf_path, engine="netcdf4", group="geolocation") as geo:
        geo_ds = geo.load()
    return product_ds, geo_ds


def plot_no2(product_ds: xr.Dataset, geo_ds: xr.Dataset, output_path: Path) -> None:
    """Render a NO2 map using Matplotlib and Cartopy."""
    no2 = product_ds["vertical_column_troposphere"].where(
        product_ds.get("main_data_quality_flag", 0) == 0
    )

    lon = geo_ds["longitude"].values
    lat = geo_ds["latitude"].values
    data = np.ma.masked_invalid(no2.values)

    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gridlines = ax.gridlines(draw_labels=["left", "bottom"], alpha=0.4)
    gridlines.xformatter = LONGITUDE_FORMATTER
    gridlines.yformatter = LATITUDE_FORMATTER

    mesh = ax.pcolormesh(
        lon,
        lat,
        data,
        cmap="viridis",
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
    unit = no2.attrs.get("units", "")
    cbar.set_label(f"NO2 Tropospheric Column {f'[{unit}]' if unit else ''}")

    title = product_ds.attrs.get("title") or "TEMPO NO2 Tropospheric Column"
    acquisition = product_ds.attrs.get("time_coverage_start")
    if acquisition:
        title = f"{title}\n{acquisition}"
    ax.set_title(title, pad=18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    username, password = load_credentials(ENV_PATH)
    ensure_directories([DOWNLOAD_DIR, OUTPUT_PATH.parent])

    client = Client(env=Environment.PROD, auth=(username, password))
    granule_path = download_tempo_granules(client, DOWNLOAD_DIR)

    product_ds, geo_ds = load_datasets(granule_path)
    try:
        plot_no2(product_ds, geo_ds, OUTPUT_PATH)
    finally:
        product_ds.close()
        geo_ds.close()

    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
