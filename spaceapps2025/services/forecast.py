"""Forecasting utilities for air quality time-series."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series


def prepare_series(
    frame: pd.DataFrame,
    value_column: str = "value",
    freq: str = "1H",
) -> pd.Series:
    series = frame.set_index("datetime")[value_column].sort_index()
    series = series.asfreq(freq)
    series = series.interpolate(method="time").ffill().bfill()
    return series


def generate_forecast(
    frame: pd.DataFrame,
    steps: int = 24,
    seasonal_periods: Optional[int] = 24,
    value_column: str = "value",
) -> ForecastResult:
    series = prepare_series(frame, value_column=value_column)
    LOGGER.debug("Training ExponentialSmoothing on %d points", len(series))
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
    )
    fit = model.fit(optimized=True, remove_bias=True)
    forecast_index = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(series.index.freq),
        periods=steps,
        freq=series.index.freq,
    )
    forecast_values = fit.forecast(steps)
    stderr = np.sqrt(fit.sse / len(series))
    conf_interval = 1.96 * stderr
    lower = forecast_values - conf_interval
    upper = forecast_values + conf_interval
    return ForecastResult(
        history=series,
        forecast=pd.Series(forecast_values.values, index=forecast_index),
        lower=pd.Series(lower.values, index=forecast_index),
        upper=pd.Series(upper.values, index=forecast_index),
    )

