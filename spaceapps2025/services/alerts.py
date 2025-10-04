"""Health risk categorization for pollutants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AirQualityBand:
    name: str
    min_value: float
    max_value: float
    guidance: str


AIR_QUALITY_THRESHOLDS: Dict[str, List[AirQualityBand]] = {
    "PM2.5": [
        AirQualityBand("Good", 0, 12, "Air quality is satisfactory."),
        AirQualityBand("Moderate", 12.1, 35.4, "Sensitive groups should monitor symptoms."),
        AirQualityBand("Unhealthy for Sensitive Groups", 35.5, 55.4, "Consider reducing prolonged outdoor exertion."),
        AirQualityBand("Unhealthy", 55.5, 150.4, "Everyone may begin to experience health effects."),
        AirQualityBand("Very Unhealthy", 150.5, 250.4, "Health alert: serious effects for sensitive groups."),
        AirQualityBand("Hazardous", 250.5, float("inf"), "Emergency conditions: avoid outdoor activities."),
    ],
    "NO2": [
        AirQualityBand("Good", 0, 53, "Air quality is satisfactory."),
        AirQualityBand("Moderate", 54, 100, "Unusually sensitive individuals should limit outdoor exertion."),
        AirQualityBand("Unhealthy for Sensitive Groups", 101, 360, "Sensitive groups should reduce outdoor exertion."),
        AirQualityBand("Unhealthy", 361, 649, "Sensitive groups avoid outdoor exertion; others limit prolonged exertion."),
        AirQualityBand("Very Unhealthy", 650, 1249, "General population should avoid outdoor exertion."),
        AirQualityBand("Hazardous", 1250, float("inf"), "Everyone should stay indoors."),
    ],
    "O3": [
        AirQualityBand("Good", 0, 54, "Air quality is satisfactory."),
        AirQualityBand("Moderate", 55, 70, "Unusually sensitive individuals should consider limiting outdoor exertion."),
        AirQualityBand("Unhealthy for Sensitive Groups", 71, 85, "Sensitive groups reduce prolonged or heavy outdoor exertion."),
        AirQualityBand("Unhealthy", 86, 105, "General public limit heavy outdoor exertion."),
        AirQualityBand("Very Unhealthy", 106, 200, "Avoid outdoor exertion."),
        AirQualityBand("Hazardous", 201, float("inf"), "Stay indoors."),
    ],
}


def assess_air_quality(pollutant: str, value: float) -> AirQualityBand:
    bands = AIR_QUALITY_THRESHOLDS.get(pollutant.upper()) or AIR_QUALITY_THRESHOLDS.get(pollutant)
    if not bands:
        raise ValueError(f"Unsupported pollutant {pollutant}")
    for band in bands:
        if band.min_value <= value <= band.max_value:
            return band
    return bands[-1]

