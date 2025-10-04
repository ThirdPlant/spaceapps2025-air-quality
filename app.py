#!/usr/bin/env python3
"""Space Apps 2025 Air Quality Intelligence Hub."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import xarray as xr

from spaceapps2025.data.merra_pm25 import MerraPM25Client
from spaceapps2025.data.nasa_power import NasaPowerClient
from spaceapps2025.data.openaq import OpenAQClient
from spaceapps2025.data.tempo import TEMPO_PRODUCTS, TempoDataClient, load_geolocated_field
from spaceapps2025.services.alerts import assess_air_quality
from spaceapps2025.services.fusion import merge_ground_and_satellite
from spaceapps2025.services.forecast import ForecastResult, generate_forecast
from spaceapps2025.services.insights import summarize_ground_data
from spaceapps2025.services.timelapse import create_timelapse
from spaceapps2025.utils.logging import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

REGION_PRESETS: Dict[str, Sequence[float]] = {
    "North America": (-130, 15, -60, 60),
    "CONUS": (-125, 24, -66, 50),
    "Global": (-180, -60, 180, 75),
    "Custom": (-100, 20, -80, 40),
}

POLLUTANT_OPTIONS = ["NO2", "HCHO", "O3", "PM2.5"]
DEFAULT_GROUND_PARAMETER = {
    "NO2": "no2",
    "HCHO": "hcho",
    "O3": "o3",
    "PM2.5": "pm25",
}


def daterange(start: date, end: date) -> List[date]:
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days + 1)]


@st.cache_data(show_spinner=False)
def load_tempo_frames(
    product_key: str,
    dates: List[date],
    bbox: Sequence[float],
    max_granules: int,
) -> List[xr.DataArray]:
    client = TempoDataClient(product_key=product_key)
    frames: List[xr.DataArray] = []
    for target_date in dates:
        granules = client.list_available_granules(target_date, limit=max_granules, bounding_box=bbox)
        for granule in granules:
            product_ds = client.fetch_dataset(granule, group="product")
            geo_ds = client.fetch_dataset(granule, group="geolocation")
            field = load_geolocated_field(
                product_ds,
                geo_ds,
                var_name=client.product.default_variable,
            )
            product_ds.close()
            geo_ds.close()
            frames.append(field)
    return frames


@st.cache_data(show_spinner=False)
def load_openaq_data(
    parameter: str,
    start_dt: datetime,
    end_dt: datetime,
    bbox: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    return OpenAQClient().measurements(parameter=parameter, start=start_dt, end=end_dt, bbox=bbox)


@st.cache_data(show_spinner=False)
def load_latest_ground(parameter: str, bbox: Optional[Sequence[float]] = None) -> pd.DataFrame:
    return OpenAQClient().latest(parameter=parameter, bbox=bbox)


@st.cache_data(show_spinner=False)
def load_weather_data(bbox: Sequence[float], start: date, end: date) -> pd.DataFrame:
    client = NasaPowerClient()
    return client.regional_grid(bbox=bbox, start=start, end=end)


@st.cache_data(show_spinner=False)
def load_pm25_surface(target_date: date) -> xr.Dataset:
    client = MerraPM25Client()
    return client.load_surface_pm25(target_date)


def build_satellite_map(field: xr.DataArray, title: str, units: str) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=field.values,
                x=field["longitude"].values,
                y=field["latitude"].values,
                colorscale="Inferno",
                colorbar=dict(title=units),
            )
        ]
    )
    figure.update_layout(title=title, xaxis_title="Longitude", yaxis_title="Latitude")
    return figure


def build_ground_map(ground: pd.DataFrame, parameter: str) -> go.Figure:
    if ground.empty:
        return go.Figure()
    figure = px.scatter_mapbox(
        ground,
        lat="latitude",
        lon="longitude",
        color="value",
        size="value",
        hover_name="location",
        hover_data={"city": True, "country": True, "value": True, "datetime": True},
        color_continuous_scale="Turbo",
        title=f"Ground Observations ({parameter})",
        height=500,
    )
    figure.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
    return figure


def plot_forecast(result: ForecastResult, title: str, unit: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.history.index, y=result.history.values, name="History", mode="lines"))
    figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast.values, name="Forecast", mode="lines"))
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([result.forecast.index, result.forecast.index[::-1]]),
            y=np.concatenate([result.upper.values, result.lower.values[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",
            line=dict(color="rgba(255,0,0,0)"),
            hoverinfo="skip",
            name="Confidence",
        )
    )
    figure.update_layout(title=title, xaxis_title="Time", yaxis_title=f"Concentration ({unit})")
    return figure


def main() -> None:
    st.set_page_config(page_title="Space Apps 2025: Cleaner, Safer Skies", layout="wide")
    st.title("Space Apps 2025 - Cleaner, Safer Skies")
    st.caption(
        "Integrating TEMPO satellite data, ground networks, and weather intelligence to forecast air quality."
    )

    st.sidebar.header("Scenario Controls")
    region_choice = st.sidebar.selectbox("Focus Region", list(REGION_PRESETS))
    bbox = list(REGION_PRESETS[region_choice])
    if region_choice == "Custom":
        lon_min = st.sidebar.number_input("Lon min", value=bbox[0], min_value=-180.0, max_value=180.0)
        lat_min = st.sidebar.number_input("Lat min", value=bbox[1], min_value=-90.0, max_value=90.0)
        lon_max = st.sidebar.number_input("Lon max", value=bbox[2], min_value=-180.0, max_value=180.0)
        lat_max = st.sidebar.number_input("Lat max", value=bbox[3], min_value=-90.0, max_value=90.0)
        bbox = [lon_min, lat_min, lon_max, lat_max]

    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=2)
    date_range = st.sidebar.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    pollutant = st.sidebar.selectbox("Primary Pollutant", POLLUTANT_OPTIONS, index=0)
    include_satellite = st.sidebar.checkbox("Include TEMPO satellite data", value=True)
    include_ground = st.sidebar.checkbox("Include ground sensor data (OpenAQ)", value=True)
    include_weather = st.sidebar.checkbox("Include NASA POWER weather fields", value=True)
    include_pm25 = st.sidebar.checkbox("Add global PM2.5 context (MERRA-2)", value=(pollutant == "PM2.5"))

    timelapse_enabled = st.sidebar.checkbox("Generate satellite timelapse", value=True)
    max_timelapse_granules = st.sidebar.slider("Max granules per day", 1, 5, 3)
    forecast_hours = st.sidebar.slider("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)

    tabs = st.tabs(["Overview", "Satellite", "Ground", "Forecast", "Weather", "Timelapse"])

    satellite_frames: List[xr.DataArray] = []
    ground_data = pd.DataFrame()
    weather_data = pd.DataFrame()
    pm25_dataset: Optional[xr.Dataset] = None

    if include_satellite and pollutant != "PM2.5":
        with st.spinner("Fetching TEMPO granules..."):
            satellite_frames = load_tempo_frames(
                pollutant,
                dates=daterange(start_date, end_date),
                bbox=bbox,
                max_granules=max_timelapse_granules,
            )

    ground_error: Optional[str] = None
    if include_ground:
        try:
            with st.spinner("Fetching ground sensor data from OpenAQ..."):
                ground_data = load_openaq_data(
                    parameter=DEFAULT_GROUND_PARAMETER.get(pollutant, pollutant.lower()),
                    start_dt=datetime.combine(start_date, datetime.min.time()),
                    end_dt=datetime.combine(end_date, datetime.max.time()),
                    bbox=bbox,
                )
        except requests.HTTPError as exc:
            response = exc.response
            status = getattr(response, "status_code", "unknown")
            detail = ""
            if response is not None:
                try:
                    detail = response.json().get("message")  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    detail = (response.text or "").strip()[:200]
            if detail:
                ground_error = f"OpenAQ request failed (status {status}): {detail}"
            else:
                ground_error = f"OpenAQ request failed with status {status}."
            ground_data = pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            ground_error = f"Unable to load OpenAQ measurements: {exc}"
            ground_data = pd.DataFrame()

    weather_error: Optional[str] = None
    if include_weather:
        try:
            with st.spinner("Fetching NASA POWER weather data..."):
                weather_data = load_weather_data(bbox=bbox, start=start_date, end=end_date)
        except requests.HTTPError as exc:
            response = exc.response
            status = getattr(response, "status_code", "unknown")
            detail = ""
            if response is not None:
                try:
                    detail = response.json().get("messages", [""])[0]  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    detail = (response.text or "").strip()[:200]
            if detail:
                weather_error = f"NASA POWER request failed (status {status}): {detail}"
            else:
                weather_error = f"NASA POWER request failed with status {status}."
            weather_data = pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            weather_error = f"Unable to load NASA POWER data: {exc}"
            weather_data = pd.DataFrame()

    if include_pm25:
        with st.spinner("Loading MERRA-2 PM2.5 analysis..."):
            pm25_dataset = load_pm25_surface(end_date)

    with tabs[0]:
        st.subheader("Mission Overview")
        st.markdown(
            """
            - **TEMPO integration** delivers geostationary insights for NO2, HCHO, and O3.
            - **OpenAQ** provides hyper-local ground validation and historical trends.
            - **NASA POWER** contextualizes atmospheric conditions (wind, humidity, precipitation).
            - **MERRA-2 CNN** offers global surface PM2.5 for health-sensitive comparisons.
            - Built-in alerts and forecasting highlight risk windows for sensitive groups.
            """
        )
        stats = []
        if satellite_frames:
            stats.append(("Satellite frames", len(satellite_frames)))
        if not ground_data.empty:
            stats.append(("Ground samples", len(ground_data)))
        if not weather_data.empty:
            stats.append(("Weather grid points", len(weather_data)))
        cols = st.columns(len(stats) or 1)
        for col, (label, value) in zip(cols, stats):
            col.metric(label, f"{value:,}")
        if include_ground and not ground_data.empty:
            insights = summarize_ground_data(ground_data, pollutant)
            st.info(
                f"Latest {pollutant} reading at **{insights['latest_location']}**: "
                f"{insights['latest_value']:.1f} {ground_data['unit'].iloc[0]} "
                f"({insights['latest_category']}). {insights['latest_guidance']}"
            )

    with tabs[1]:
        st.subheader("Satellite Intelligence")
        if satellite_frames:
            latest_field = satellite_frames[-1]
            units = TEMPO_PRODUCTS[pollutant].default_variable
            st.plotly_chart(
                build_satellite_map(
                    latest_field,
                    title=f"TEMPO {pollutant} | {latest_field.attrs.get('title', 'latest granule')}",
                    units=units,
                ),
                use_container_width=True,
            )
            st.caption("Satellite map derives from NASA Harmony-processed TEMPO Level-2 data.")
        elif pollutant == "PM2.5" and pm25_dataset is not None:
            field = pm25_dataset["MERRA2_CNN_Surface_PM25"].isel(time=0)
            st.plotly_chart(
                build_satellite_map(field, "MERRA-2 CNN Surface PM2.5", "ug m^-3"),
                use_container_width=True,
            )
        else:
            st.warning("Enable TEMPO data or select a supported pollutant to view satellite layers.")

    with tabs[2]:
        st.subheader("Ground Networks")
        if ground_error:
            st.warning(ground_error)

        if not ground_data.empty:
            st.plotly_chart(build_ground_map(ground_data, pollutant), use_container_width=True)
            with st.expander("Raw data"):
                st.dataframe(ground_data)
        elif not ground_error:
            st.info("No ground measurements returned for the selected window and region.")

    with tabs[3]:
        st.subheader("Forecasting & Alerts")
        if not ground_data.empty:
            location_options = ground_data["location"].unique().tolist()
            selected_location = st.selectbox("Forecast location", location_options)
            location_series = ground_data[ground_data["location"] == selected_location]
            try:
                result = generate_forecast(location_series, steps=forecast_hours)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Forecast failed: {exc}")
            else:
                unit = location_series["unit"].iloc[0]
                st.plotly_chart(
                    plot_forecast(result, f"{pollutant} outlook for {selected_location}", unit),
                    use_container_width=True,
                )
                band = assess_air_quality(pollutant, result.forecast.iloc[-1])
                st.warning(
                    f"Projected {pollutant} after {forecast_hours} hours: {result.forecast.iloc[-1]:.1f} {unit}"
                    f" ({band.name}). {band.guidance}"
                )
        else:
            st.info("Ground data required to compute localized forecasts.")

    with tabs[4]:
        st.subheader("Weather Context")
        if weather_error:
            st.warning(weather_error)

        if not weather_data.empty:
            parameter = st.selectbox("Weather parameter", sorted(weather_data["parameter"].unique()))
            subset = weather_data[weather_data["parameter"] == parameter]
            pivot = subset.pivot_table(index="date", values="value", aggfunc="mean")
            st.line_chart(pivot)
            st.caption("Daily mean from NASA POWER regional grid.")
        elif not weather_error:
            st.info("Weather data not requested.")

    with tabs[5]:
        st.subheader("Satellite Timelapse")
        if timelapse_enabled and satellite_frames:
            timestamps: List[pd.Timestamp] = []
            for frame in satellite_frames:
                ts = frame.attrs.get("time_coverage_start")
                timestamps.append(pd.to_datetime(ts) if ts else pd.Timestamp.now())
            try:
                figure = create_timelapse(
                    satellite_frames,
                    timestamps,
                    title=f"TEMPO {pollutant} Timelapse",
                    units=TEMPO_PRODUCTS.get(pollutant, TEMPO_PRODUCTS["NO2"]).default_variable,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Timelapse rendering failed: {exc}")
            else:
                st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Enable TEMPO data and timelapse to animate satellite swaths.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built for the **NASA Space Apps Challenge 2025** -- From EarthData to Action: Predicting Cleaner, Safer Skies."
    )


if __name__ == "__main__":
    main()

