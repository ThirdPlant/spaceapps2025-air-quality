"""Configuration helpers for credentials and defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import dotenv_values


def load_earthdata_credentials(env_path: Path | None = None) -> Tuple[str, str]:
    """Load Earthdata credentials from environment variables or a .env file."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if username and password:
        return username, password

    env_path = env_path or Path(".env")
    if env_path.exists():
        values = dotenv_values(str(env_path))
        username = username or values.get("EARTHDATA_USERNAME")
        password = password or values.get("EARTHDATA_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Earthdata credentials must be set in environment variables or .env"
        )
    return username, password


def get_data_root() -> Path:
    """Return the directory where downloads and cache files should live."""
    root = Path(os.environ.get("SPACEAPPS_DATA_ROOT", "spaceapps_data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_cache_root() -> Path:
    """Return directory to store cached API responses."""
    root = Path(os.environ.get("SPACEAPPS_CACHE_ROOT", "spaceapps_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root
