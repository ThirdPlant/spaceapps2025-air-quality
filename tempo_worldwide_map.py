#!/usr/bin/env python3
"""Generate a world map of TEMPO NO2 air quality using NASA Harmony."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from harmony import Client, Collection, Request
from harmony.config import Environment

COLLECTION_ID = "C2930725014-LARC_CLOUD"
DEFAULT_GRANULES = ["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"]
# Resolve paths relative to this script's directory to avoid VS Code CWD issues
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "tempo_data"
DEFAULT_OUTPUT = BASE_DIR / "output" / "tempo_no2_world.png"
ENV_PATH = BASE_DIR / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TEMPO NO2 data and render it on a global map."
    )
    parser.add_argument(
        "--granule",
        dest="granules",
        action="append",
        help="Specific TEMPO granule name to download (can be passed multiple times).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save the rendered plot (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help=f"Directory to store downloaded granules (default: {DOWNLOAD_DIR}).",
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


def download_tempo_granule(
    client: Client, download_dir: Path, granule_names: Sequence[str]
) -> Path:
    if not granule_names:
        raise ValueError("At least one granule name is required.")

    request = Request(
        collection=Collection(id=COLLECTION_ID),
        granule_name=list(granule_names),
    )

    job_id = client.submit(request)
    client.wait_for_processing(job_id, show_progress=True)

    results = client.download_all(job_id, directory=str(download_dir))
    files = [Path(task.result()) for task in results]

    if not files:
        raise RuntimeError("Harmony request completed but no files were downloaded.")

    return files[0]


def load_datasets(netcdf_path: Path) -> Tuple[xr.Dataset, xr.Dataset]:
    with xr.open_dataset(netcdf_path, engine="netcdf4", group="product") as product:
        product_ds = product.load()
    with xr.open_dataset(netcdf_path, engine="netcdf4", group="geolocation") as geo:
        geo_ds = geo.load()
    return product_ds, geo_ds


def prepare_coordinates(
    longitudes: np.ndarray, latitudes: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_wrapped = ((longitudes + 180.0) % 360.0) - 180.0
    sort_index = np.argsort(lon_wrapped, axis=1)

    lon_sorted = np.take_along_axis(lon_wrapped, sort_index, axis=1)
    lat_sorted = np.take_along_axis(latitudes, sort_index, axis=1)
    val_sorted = np.take_along_axis(values, sort_index, axis=1)

    return lon_sorted, lat_sorted, val_sorted


def determine_color_scale(data: np.ndarray) -> Tuple[float, float]:
    compressed = np.ma.masked_invalid(data).compressed()
    if compressed.size == 0:
        raise RuntimeError("No valid NO2 values available for plotting.")
    high = float(np.percentile(compressed, 99))
    if not np.isfinite(high) or high <= 0.0:
        high = float(np.max(compressed))
    return 0.0, high


def plot_worldwide_no2(
    product_ds: xr.Dataset, geo_ds: xr.Dataset, output_path: Path
) -> None:
    no2 = product_ds["vertical_column_troposphere"]
    quality_flag = product_ds.get("main_data_quality_flag")
    if quality_flag is not None:
        no2 = no2.where(quality_flag == 0)

    lon = np.asarray(geo_ds["longitude"])
    lat = np.asarray(geo_ds["latitude"])
    data = np.asarray(no2)

    lon_sorted, lat_sorted, data_sorted = prepare_coordinates(lon, lat, data)
    data_masked = np.ma.masked_invalid(data_sorted)

    vmin, vmax = determine_color_scale(data_masked)

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#dceeff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", edgecolor="gray")

    mesh = ax.pcolormesh(
        lon_sorted,
        lat_sorted,
        data_masked,
        cmap="plasma",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.68, pad=0.04)
    unit = no2.attrs.get("units", "")
    unit_suffix = f" [{unit}]" if unit else ""
    cbar.set_label(f"TEMPO NO2 Tropospheric Column{unit_suffix}")

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        linestyle="--",
        alpha=0.35,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    title = product_ds.attrs.get("title") or "TEMPO NO2 Tropospheric Column"
    acquisition = product_ds.attrs.get("time_coverage_start")
    if acquisition:
        title = f"{title}\n{acquisition}"
    ax.set_title(f"{title}\nDisplayed on a global map (TEMPO coverage highlighted)", pad=18)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    username, password = load_credentials(ENV_PATH)

    ensure_directories([args.download_dir, args.output.parent])

    client = Client(env=Environment.PROD, auth=(username, password))
    granule_path = download_tempo_granule(client, args.download_dir, args.granules or DEFAULT_GRANULES)

    product_ds, geo_ds = load_datasets(granule_path)
    try:
        plot_worldwide_no2(product_ds, geo_ds, args.output)
    finally:
        product_ds.close()
        geo_ds.close()

    print(f"World map saved to {args.output}")


if __name__ == "__main__":
    main()
