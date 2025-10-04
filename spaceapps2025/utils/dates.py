"""Date utilities."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, List


def daterange(start: date, end: date, step: timedelta = timedelta(days=1)) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += step


def to_ymd(dt: date | datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def to_ymd_compact(dt: date | datetime) -> str:
    return dt.strftime("%Y%m%d")
