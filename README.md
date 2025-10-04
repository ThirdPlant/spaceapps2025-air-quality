# Space Apps 2025 - Cleaner, Safer Skies Toolkit

An end-to-end prototype for the NASA Space Apps Challenge 2025 (Challenge 15: **From EarthData to Action: Cloud Computing with Earth Observation Data for Predicting Cleaner, Safer Skies**). The toolkit combines satellite, ground, and weather intelligence to forecast air quality, surface emerging hazards, and deliver interactive timelapse visualizations.

## Capabilities

- **TEMPO Level-2 ingestion** via NASA Harmony for NO2, HCHO, and O3 swaths (with caching).
- **Ground validation** through OpenAQ historical and latest measurements.
- **Weather context** from NASA POWER regional grids (wind, humidity, precipitation, and temperature proxies).
- **Global PM2.5 baseline** using the MERRA-2 CNN HAQAST surface product.
- **Data fusion & forecasting**: collocates ground stations with satellite pixels, provides health-band alerts, and generates hourly forecasts through Holt-Winters Smoothing.
- **Interactive web app** (Streamlit) with:
  - Region presets and custom spatial bounds.
  - Timescale controls and multi-day granule sampling.
  - Satellite heatmaps, ground sensor mapbox overlay, and weather trend charts.
  - Timelapse player with play/pause slider for satellite evolution.
  - Health guidance messaging aligned with EPA AQI breakpoints.
  - Export-ready tables for downstream analytics.

## Repository Layout

```
spaceapps2025/         Modular data + services package
  data/               Remote data access modules (TEMPO, OpenAQ, NASA POWER, MERRA-2)
  services/           Analytics: fusion, forecasting, alerts, timelapse helpers
  utils/              Config, caching, logging helpers
app.py                 Streamlit mission control UI
nasa_pm25_world_map.py Legacy PM2.5 mapping script (still available for CLI workflows)
requirements.txt       Python dependencies
```

## Quick Start

> **Prerequisites**
> - Python 3.9+
> - NASA Earthdata Login credentials stored as environment variables or in `.env`:
>   ```bash
>   EARTHDATA_USERNAME=your_username
>   EARTHDATA_PASSWORD=your_password
>   ```
> - Network access to NASA Harmony, OpenAQ, and NASA POWER endpoints.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the mission control hub:

```bash
streamlit run app.py
```

The app opens in your browser with sidebar controls for pollutant, date range, region, and optional modules (ground, weather, timelapse). Select a location from the ground tab to unlock localized forecasts and health messaging.

## Key Workflows

1. **Situational Awareness** - Choose a region (e.g., North America), enable TEMPO and ground layers, and review the Overview tab for headline metrics and guidance.
2. **Cross-Domain Validation** - Switch to the Ground tab to compare OpenAQ measurements, then sample collocated satellite values through the Satellite tab.
3. **Forecasting & Alerts** - On the Forecast tab pick a station and review the 6-72 hour Holt-Winters projection; generated alerts follow EPA AQI risk levels.
4. **Meteorological Context** - Inspect NASA POWER wind, humidity, or precipitation trends to understand dispersion factors.
5. **Timelapse Storytelling** - Activate the Timelapse tab to animate successive TEMPO granules and export visuals for outreach decks.

## Extensibility Ideas

- Add machine-learning ensembles (e.g., gradient boosting) that leverage weather regressors for fine-grained forecasts.
- Integrate citizen-science feeds (e.g., EPA AirNow or PurpleAir) through modular plug-ins in `spaceapps2025/data`.
- Deploy in the cloud (Streamlit Community Cloud, HuggingFace Spaces, Docker on AWS Fargate) and wire alerts to messaging APIs.
- Incorporate socio-economic vulnerability indices to prioritize high-risk communities.

## Acknowledgements

- NASA TEMPO science team for pioneering geostationary air quality monitoring.
- NASA POWER project for long-standing surface meteorology datasets.
- OpenAQ community for democratizing ground-based air quality data.
- NASA Earth Science Division, Booz Allen Hamilton, Mindgrub, and SecondMuse for stewarding Space Apps.

*Developed for Space Apps 2025 hackathons - ready for rapid iteration and storytelling.*
