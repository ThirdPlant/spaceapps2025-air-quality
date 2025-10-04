"""Generate narrative insights from fused datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .alerts import assess_air_quality


def summarize_ground_data(frame: pd.DataFrame, pollutant: str) -> Dict[str, object]:
    if frame.empty:
        return {"message": "No ground measurements available for the selected filters."}
    latest = frame.sort_values("datetime", ascending=False).iloc[0]
    worst = frame.sort_values("value", ascending=False).iloc[0]
    band = assess_air_quality(pollutant, latest["value"])
    return {
        "latest_location": latest["location"],
        "latest_value": float(latest["value"]),
        "latest_unit": latest.get("unit", ""),
        "latest_time": latest["datetime"],
        "latest_category": band.name,
        "latest_guidance": band.guidance,
        "peak_location": worst["location"],
        "peak_value": float(worst["value"]),
        "record_count": int(len(frame)),
    }
